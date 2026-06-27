"""
MediMap AI — Hybrid Dual-Stream Late-Fusion Model
==================================================
Architecture overview
─────────────────────
  Stream A (Tabular) ──► MLP Feature Extractor ──┐
                                                   ├──► Fusion Head ──► Softmax ──► Disease Label
  Stream B (Vision) ───► CNN Feature Extractor ──┘

Dataset targets
───────────────
  • Tabular  : Kaggle "Disease Symptom Description" dataset
               (one-hot encoded symptom vector → disease label)
  • Vision   : Chest X-ray (NIH / Kaggle) + skin-lesion datasets
               (JPEG/PNG → disease label)

CUDA policy
───────────
  • Tensors and model parameters are moved to GPU when available.
  • torch.cuda.amp is used for mixed-precision training.
  • Peak GPU memory is monitored and logged after each epoch.

Author : MediMap AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Device resolution ────────────────────────────────────────────────────────
FORCE_CPU: bool = bool(int(os.getenv("FORCE_CPU", "0")))
DEVICE: torch.device = (
    torch.device("cpu")
    if FORCE_CPU or not torch.cuda.is_available()
    else torch.device("cuda")
)
logger.info("MediMap AI — compute device: %s", DEVICE)


# =============================================================================
# 1.  STREAM A — TABULAR MLP
# =============================================================================

class TabularMLP(nn.Module):
    """
    Multi-Layer Perceptron for one-hot encoded symptom vectors.

    Parameters
    ----------
    input_dim : int
        Length of the flattened symptom feature vector.
    hidden_dims : tuple[int, ...]
        Sizes of each hidden layer.
    embedding_dim : int
        Output dimensionality of the feature embedding (before fusion).
    dropout_rate : float
        Dropout probability applied after every hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        embedding_dim: int = 128,
        dropout_rate: float = 0.35,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout_rate),
                ]
            )
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, input_dim) — one-hot symptom vector.

        Returns
        -------
        torch.Tensor
            Shape (batch_size, embedding_dim) — symptom feature embedding.
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"[TabularMLP] Expected input_dim={self.input_dim}, "
                f"got {x.shape[-1]}."
            )
        return self.encoder(x)


# =============================================================================
# 2.  STREAM B — VISION CNN (Transfer Learning)
# =============================================================================

class VisionCNN(nn.Module):
    """
    Transfer-learning vision encoder built on MobileNetV2 or ResNet50.

    The backbone is optionally frozen for the first N epochs (gradual
    unfreezing strategy).  A lightweight projection head maps the
    backbone output to ``embedding_dim``.

    Parameters
    ----------
    backbone : str
        One of ``"mobilenet_v2"`` or ``"resnet50"``.
    embedding_dim : int
        Output dimensionality of the projection head (before fusion).
    pretrained : bool
        Whether to load ImageNet-pretrained weights.
    freeze_backbone : bool
        If True, backbone weights are frozen initially.
    """

    _SUPPORTED_BACKBONES = ("mobilenet_v2", "resnet50")

    def __init__(
        self,
        backbone: str = "mobilenet_v2",
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if backbone not in self._SUPPORTED_BACKBONES:
            raise ValueError(
                f"backbone must be one of {self._SUPPORTED_BACKBONES}; got '{backbone}'."
            )

        self.backbone_name = backbone
        self.embedding_dim = embedding_dim

        weights_arg = "IMAGENET1K_V1" if pretrained else None

        if backbone == "mobilenet_v2":
            base = models.mobilenet_v2(weights=weights_arg)
            feature_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()  # Strip original head
        else:  # resnet50
            base = models.resnet50(weights=weights_arg)
            feature_dim = base.fc.in_features
            base.fc = nn.Identity()  # Strip original head

        self.backbone = base

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, embedding_dim),
        )

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (call after warm-up epochs)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("[VisionCNN] Backbone '%s' unfrozen.", self.backbone_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, 3, H, W) — pre-processed medical image tensor.

        Returns
        -------
        torch.Tensor
            Shape (batch_size, embedding_dim) — image feature embedding.
        """
        features = self.backbone(x)  # (B, feature_dim)
        return self.projector(features)  # (B, embedding_dim)


# =============================================================================
# 3.  FUSION HEAD
# =============================================================================

class FusionHead(nn.Module):
    """
    Late-Fusion concatenation block.

    Accepts embeddings from TabularMLP and VisionCNN, concatenates them,
    and passes through a classifier that outputs disease logits.

    Parameters
    ----------
    tabular_dim : int
        Embedding dimension from TabularMLP.
    vision_dim : int
        Embedding dimension from VisionCNN.
    num_classes : int
        Number of target disease categories.
    fusion_hidden : int
        Width of the intermediate fusion layer.
    dropout_rate : float
        Dropout probability in the fusion block.
    """

    def __init__(
        self,
        tabular_dim: int = 128,
        vision_dim: int = 128,
        num_classes: int = 41,
        fusion_hidden: int = 256,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        combined_dim = tabular_dim + vision_dim

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden // 2, num_classes),
        )

    def forward(
        self,
        tab_embed: torch.Tensor,
        vis_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        tab_embed : torch.Tensor
            Shape (batch_size, tabular_dim).
        vis_embed : torch.Tensor
            Shape (batch_size, vision_dim).

        Returns
        -------
        torch.Tensor
            Shape (batch_size, num_classes) — raw logits.

        Raises
        ------
        ValueError
            If batch sizes of the two streams don't match.
        """
        if tab_embed.shape[0] != vis_embed.shape[0]:
            raise ValueError(
                f"[FusionHead] Batch-size mismatch: tabular={tab_embed.shape[0]}, "
                f"vision={vis_embed.shape[0]}."
            )
        fused = torch.cat([tab_embed, vis_embed], dim=1)  # (B, combined_dim)
        return self.fusion(fused)  # (B, num_classes)


# =============================================================================
# 4.  FULL HYBRID MODEL
# =============================================================================

class MediMapHybridModel(nn.Module):
    """
    End-to-end Hybrid Dual-Stream Late-Fusion model for MediMap AI.

    Combines TabularMLP, VisionCNN, and FusionHead into a single trainable
    module.  Outputs both raw logits and Softmax confidence scores.

    Parameters
    ----------
    tabular_input_dim : int
        Number of symptom features (columns in one-hot encoded dataset).
    num_classes : int
        Number of disease categories.
    vision_backbone : str
        CNN backbone identifier (``"mobilenet_v2"`` or ``"resnet50"``).
    embedding_dim : int
        Shared embedding size for both streams (default 128).
    """

    def __init__(
        self,
        tabular_input_dim: int,
        num_classes: int,
        vision_backbone: str = "mobilenet_v2",
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.tab_stream = TabularMLP(
            input_dim=tabular_input_dim,
            embedding_dim=embedding_dim,
        )
        self.vis_stream = VisionCNN(
            backbone=vision_backbone,
            embedding_dim=embedding_dim,
        )
        self.fusion = FusionHead(
            tabular_dim=embedding_dim,
            vision_dim=embedding_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        symptoms: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Parameters
        ----------
        symptoms : torch.Tensor
            Shape (batch_size, tabular_input_dim).
        image : torch.Tensor
            Shape (batch_size, 3, 224, 224).

        Returns
        -------
        logits : torch.Tensor
            Raw class scores, shape (batch_size, num_classes).
        probabilities : torch.Tensor
            Softmax-normalised confidence scores, shape (batch_size, num_classes).
        """
        tab_embed = self.tab_stream(symptoms)
        vis_embed = self.vis_stream(image)
        logits = self.fusion(tab_embed, vis_embed)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def predict(
        self,
        symptoms: torch.Tensor,
        image: torch.Tensor,
        label_names: Optional[list[str]] = None,
    ) -> dict:
        """
        Inference convenience method.

        Returns a dictionary with the top-1 predicted class, confidence,
        and the full probability distribution.

        Parameters
        ----------
        symptoms : torch.Tensor
            One-hot symptom vector (unsqueezed to batch of 1 if needed).
        image : torch.Tensor
            Pre-processed image tensor (unsqueezed to batch of 1 if needed).
        label_names : list[str], optional
            Human-readable class names aligned with model output indices.

        Returns
        -------
        dict
            Keys: ``predicted_class``, ``confidence``, ``probabilities``.
        """
        self.eval()
        with torch.no_grad():
            if symptoms.dim() == 1:
                symptoms = symptoms.unsqueeze(0)
            if image.dim() == 3:
                image = image.unsqueeze(0)
            symptoms = symptoms.to(DEVICE)
            image = image.to(DEVICE)

            _, probs = self.forward(symptoms, image)

            # ── Rare-disease confidence boost (1.3×) ──────────────────────
            # These diseases have very few unique symptoms in the dataset and
            # tend to be drowned out by common diseases like Malaria/Typhoid.
            # Multiplying their raw softmax probability by 1.3 before argmax
            # ensures they surface when the user provides the right symptoms.
            _RARE_DISEASE_BOOST: dict[str, float] = {
                "AIDS":                              1.3,
                "Dengue":                            1.3,
                "Hepatitis A":                       1.3,
                "Hepatitis B":                       1.3,
                "Hepatitis C":                       1.3,
                "Hepatitis D":                       1.3,
                "Hepatitis E":                       1.3,
                "Jaundice":                          1.3,
                "Chronic kidney disease":            1.3,
                "Paralysis (brain hemorrhage)":      1.3,
                "Dimorphic hemmorhoids(piles)":      1.3,
                "Osteoarthritis":                    1.3,
                "Cervical spondylosis":              1.3,
            }
            if label_names:
                boosted = probs[0].clone()
                for disease, multiplier in _RARE_DISEASE_BOOST.items():
                    if disease in label_names:
                        idx = label_names.index(disease)
                        boosted[idx] = boosted[idx] * multiplier
                # Re-normalise so probabilities still sum to 1
                boosted = boosted / boosted.sum()
                probs = boosted.unsqueeze(0)
            # ──────────────────────────────────────────────────────────────

            top1_idx = probs.argmax(dim=1).item()
            confidence = probs[0, top1_idx].item()

        predicted_label = (
            label_names[top1_idx] if label_names else str(top1_idx)
        )
        return {
            "predicted_class": predicted_label,
            "class_index": top1_idx,
            "confidence": round(confidence * 100, 2),
            "probabilities": probs[0].cpu().numpy(),
        }


# =============================================================================
# 5.  IMAGE PRE-PROCESSING TRANSFORMS
# =============================================================================

def get_image_transforms(split: str = "val") -> transforms.Compose:
    """
    Return a torchvision transform pipeline for medical images.

    Parameters
    ----------
    split : str
        ``"train"`` applies heavy augmentations; ``"val"``/``"test"``
        applies only normalisation.

    Returns
    -------
    transforms.Compose
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


# =============================================================================
# 6.  TRAINING LOOP
# =============================================================================

def train_one_epoch(
    model: MediMapHybridModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    epoch: int,
) -> float:
    """
    Train the model for one epoch with mixed-precision (AMP).

    Parameters
    ----------
    model : MediMapHybridModel
    loader : DataLoader
        Yields (symptoms, images, labels) batches.
    optimizer : torch.optim.Optimizer
    criterion : nn.Module
        Cross-entropy loss.
    scaler : GradScaler
        AMP gradient scaler.
    epoch : int
        Current epoch index (for logging).

    Returns
    -------
    float
        Mean training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for step, (symptoms, images, labels) in enumerate(loader):
        symptoms = symptoms.to(DEVICE, non_blocking=True)
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu"):
            logits, _ = model(symptoms, images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if step % 50 == 0:
            logger.info(
                "Epoch %d | Step %d/%d | Loss: %.4f",
                epoch,
                step,
                len(loader),
                loss.item(),
            )

    # Log peak GPU memory
    if DEVICE.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6
        torch.cuda.reset_peak_memory_stats(DEVICE)
        logger.info("Epoch %d | Peak GPU memory: %.1f MB", epoch, peak_mb)

    return running_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: MediMapHybridModel,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Evaluate model on a validation / test split.

    Returns
    -------
    Tuple[float, float]
        (mean_loss, accuracy_percent)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for symptoms, images, labels in loader:
            symptoms = symptoms.to(DEVICE, non_blocking=True)
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits, _ = model(symptoms, images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total if total > 0 else 0.0
    return total_loss / len(loader), acc


# =============================================================================
# 7.  MODEL PERSISTENCE UTILITIES
# =============================================================================

def save_checkpoint(
    model: MediMapHybridModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    save_dir: str = "models/saved",
) -> Path:
    """
    Serialise full model checkpoint to disk.

    Parameters
    ----------
    model : MediMapHybridModel
    optimizer : torch.optim.Optimizer
    epoch : int
    val_loss : float
    save_dir : str
        Target directory (created if absent).

    Returns
    -------
    Path
        Path to the saved checkpoint file.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    ckpt_file = save_path / f"medimap_epoch{epoch:03d}_loss{val_loss:.4f}.pth"
    torch.save(checkpoint, ckpt_file)
    logger.info("Checkpoint saved → %s", ckpt_file)
    return ckpt_file


def load_checkpoint(
    model: MediMapHybridModel,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """
    Load model (and optionally optimizer) state from a checkpoint file.

    Parameters
    ----------
    model : MediMapHybridModel
        Uninitialised model instance with correct architecture.
    checkpoint_path : str
        Path to the ``.pth`` checkpoint file.
    optimizer : torch.optim.Optimizer, optional
        If provided, its state is restored from the checkpoint.

    Returns
    -------
    int
        Epoch number stored in the checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", 0)
    logger.info("Loaded checkpoint from '%s' (epoch %d).", checkpoint_path, epoch)
    return epoch


# =============================================================================
# 8.  ENTRY-POINT — QUICK SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    logger.info("Running architecture smoke test …")

    # Synthetic data
    BATCH = 4
    N_SYMPTOMS = 131     # Kaggle disease-symptom dataset has 131 symptom cols
    N_CLASSES = 41       # 41 disease categories

    dummy_symptoms = torch.randn(BATCH, N_SYMPTOMS)
    dummy_images = torch.randn(BATCH, 3, 224, 224)
    dummy_labels = torch.randint(0, N_CLASSES, (BATCH,))

    model = MediMapHybridModel(
        tabular_input_dim=N_SYMPTOMS,
        num_classes=N_CLASSES,
        vision_backbone="mobilenet_v2",
    ).to(DEVICE)

    dummy_symptoms = dummy_symptoms.to(DEVICE)
    dummy_images = dummy_images.to(DEVICE)

    logits, probs = model(dummy_symptoms, dummy_images)
    logger.info("Logits  shape : %s", logits.shape)
    logger.info("Probs   shape : %s", probs.shape)
    logger.info("Sum(probs[0]) : %.6f (should be ~1.0)", probs[0].sum().item())
    logger.info("Smoke test PASSED ✔")
