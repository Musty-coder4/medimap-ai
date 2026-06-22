"""
MediMap AI — Training Entry-Point
===================================
Orchestrates end-to-end training of the Hybrid Dual-Stream model using
the Kaggle Disease Symptom dataset (tabular) paired with a medical image
dataset (vision).

Usage
-----
    python models/train.py --epochs 30 --batch_size 32 --backbone mobilenet_v2

Author : MediMap AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.hybrid_fusion import (
    DEVICE,
    MediMapHybridModel,
    evaluate,
    get_image_transforms,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# =============================================================================
# DATASET HELPERS
# =============================================================================

def load_tabular_data(
    csv_path: str,
    symptom_columns_out: str = "data/processed/symptom_columns.json",
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, list[str]]:
    """
    Load and preprocess the Kaggle Disease Symptom dataset.

    Parameters
    ----------
    csv_path : str
        Path to ``dataset.csv`` (Disease, Symptom_1 … Symptom_17 format).
    symptom_columns_out : str
        Where to persist the list of one-hot symptom column names.

    Returns
    -------
    X : np.ndarray
        Float32 one-hot symptom matrix (n_samples, n_features).
    y : np.ndarray
        Integer-encoded disease labels (n_samples,).
    le : LabelEncoder
        Fitted label encoder.
    symptom_cols : list[str]
        List of one-hot column names (used at inference time).
    """
    from utils.data_preprocessor import load_and_one_hot_encode
    X_df, y_series, symptom_list = load_and_one_hot_encode(csv_path)
    
    X = X_df.values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(y_series.astype(str))

    # Persist column list for inference
    Path(symptom_columns_out).parent.mkdir(parents=True, exist_ok=True)
    with open(symptom_columns_out, "w") as fh:
        json.dump(symptom_list, fh, indent=2)
    logger.info(
        "Tabular data: %d samples | %d symptoms | %d diseases",
        X.shape[0], X.shape[1], len(le.classes_),
    )
    return X, y, le, symptom_list


from PIL import Image
from torchvision import transforms
import random

class HybridImageDataset(Dataset):
    """
    True multi-modal image dataset. Returns actual images for classes
    that have them in data/raw/images, and falls back to a blank tensor
    if no image is available for that disease.
    """

    def __init__(self, sym: np.ndarray, lbl: np.ndarray, le: LabelEncoder, img_dir="data/raw/images", transform=None) -> None:
        self.sym = sym
        self.lbl = lbl
        self.le = le
        self.img_dir = Path(img_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Cache image paths by class name to speed up lookup
        self.img_cache = {}
        for class_name in self.le.classes_:
            xray_dir = self.img_dir / "xray" / class_name
            skin_dir = self.img_dir / "skin" / class_name
            
            paths = []
            if xray_dir.exists():
                paths.extend(list(xray_dir.glob("*.jpg")) + list(xray_dir.glob("*.jpeg")))
            if skin_dir.exists():
                paths.extend(list(skin_dir.glob("*.jpg")) + list(skin_dir.glob("*.jpeg")))
            
            self.img_cache[class_name] = paths

    def __len__(self) -> int:
        return len(self.sym)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, torch.Tensor, int]:
        x_tab = self.sym[idx]
        y_val = self.lbl[idx]
        
        class_name = self.le.inverse_transform([y_val.item()])[0]
        paths = self.img_cache.get(class_name, [])
        
        if paths:
            # Load actual image if available
            img_path = random.choice(paths)
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img)
            except Exception:
                img_tensor = torch.zeros(3, 224, 224)
        else:
            img_tensor = torch.zeros(3, 224, 224)
            
        return x_tab, img_tensor, y_val


# =============================================================================
# MAIN TRAINING ROUTINE
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """Run the full training pipeline."""
    # ── 1. Load tabular data ─────────────────────────────────────────────────
    X, y, le, symptom_cols = load_tabular_data(
        args.tabular_csv,
        symptom_columns_out="data/processed/symptom_columns.json",
    )
    n_symptoms = X.shape[1]
    n_classes = len(le.classes_)

    if args.subset > 0:
        indices = np.random.choice(len(X), min(args.subset, len(X)), replace=False)
        X = X[indices]
        y = y[indices]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 2. Tensors ───────────────────────────────────────────────────────────
    sym_train = torch.from_numpy(X_train)
    sym_val = torch.from_numpy(X_val)
    lbl_train = torch.from_numpy(y_train).long()
    lbl_val = torch.from_numpy(y_val).long()

    train_ds = HybridImageDataset(sym_train, lbl_train, le)
    val_ds = HybridImageDataset(sym_val, lbl_val, le)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # ── 3. Build model ───────────────────────────────────────────────────────
    model = MediMapHybridModel(
        tabular_input_dim=n_symptoms,
        num_classes=n_classes,
        vision_backbone=args.backbone,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(DEVICE.type, enabled=(DEVICE.type == "cuda"))

    # ── 4. Resume from checkpoint if given ───────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, args.resume, optimizer)

    # ── 5. Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Unfreeze vision backbone after warm-up
        if epoch == args.unfreeze_epoch:
            model.vis_stream.unfreeze_backbone()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        logger.info(
            "Epoch %d/%d | TrainLoss=%.4f | ValLoss=%.4f | ValAcc=%.2f%%",
            epoch, args.epochs, train_loss, val_loss, val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, "models/saved")

    # ── 6. Persist label encoder & symptom columns ───────────────────────────
    Path("models/saved").mkdir(parents=True, exist_ok=True)
    joblib.dump(le, "models/saved/label_encoder.pkl")
    logger.info("Label encoder saved → models/saved/label_encoder.pkl")
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediMap AI — Train Hybrid Model")
    parser.add_argument(
        "--tabular_csv",
        type=str,
        default="data/raw/tabular/dataset.csv",
        help="Path to Kaggle disease-symptom CSV file.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "resnet50"],
    )
    parser.add_argument(
        "--unfreeze_epoch",
        type=int,
        default=5,
        help="Epoch at which to unfreeze the CNN backbone.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=0,
        help="Limit number of samples for fast iteration (0 for all).",
    )
    args = parser.parse_args()
    main(args)
