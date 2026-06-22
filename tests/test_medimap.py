"""
MediMap AI — Unit Tests
========================
Smoke-tests and unit tests for the core modules.
Run with: pytest tests/ -v --tb=short

Author : MediMap AI Engineering Team
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ── Ensure project root is on path ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.hybrid_fusion import (
    DEVICE,
    FusionHead,
    MediMapHybridModel,
    TabularMLP,
    VisionCNN,
)
from utils.geo_recommender import disease_to_specialty, DISEASE_SPECIALTY_MAP


# =============================================================================
# FIXTURES
# =============================================================================

N_SYMPTOMS = 131
N_CLASSES = 41
BATCH = 4


@pytest.fixture(scope="module")
def tabular_mlp() -> TabularMLP:
    return TabularMLP(input_dim=N_SYMPTOMS, embedding_dim=128).to(DEVICE)


@pytest.fixture(scope="module")
def vision_cnn() -> VisionCNN:
    return VisionCNN(backbone="mobilenet_v2", embedding_dim=128, pretrained=False).to(DEVICE)


@pytest.fixture(scope="module")
def fusion_head() -> FusionHead:
    return FusionHead(tabular_dim=128, vision_dim=128, num_classes=N_CLASSES).to(DEVICE)


@pytest.fixture(scope="module")
def full_model() -> MediMapHybridModel:
    return MediMapHybridModel(
        tabular_input_dim=N_SYMPTOMS,
        num_classes=N_CLASSES,
        vision_backbone="mobilenet_v2",
    ).to(DEVICE)


# =============================================================================
# STREAM A — TabularMLP
# =============================================================================

class TestTabularMLP:
    def test_output_shape(self, tabular_mlp):
        x = torch.randn(BATCH, N_SYMPTOMS).to(DEVICE)
        out = tabular_mlp(x)
        assert out.shape == (BATCH, 128), f"Expected (4,128), got {out.shape}"

    def test_wrong_input_dim_raises(self, tabular_mlp):
        x = torch.randn(BATCH, 50).to(DEVICE)
        with pytest.raises(ValueError, match="input_dim"):
            tabular_mlp(x)

    def test_gradient_flows(self, tabular_mlp):
        x = torch.randn(BATCH, N_SYMPTOMS, requires_grad=True).to(DEVICE)
        out = tabular_mlp(x)
        out.sum().backward()
        assert x.grad is not None


# =============================================================================
# STREAM B — VisionCNN
# =============================================================================

class TestVisionCNN:
    def test_output_shape(self, vision_cnn):
        x = torch.randn(BATCH, 3, 224, 224).to(DEVICE)
        out = vision_cnn(x)
        assert out.shape == (BATCH, 128), f"Expected (4,128), got {out.shape}"

    def test_invalid_backbone_raises(self):
        with pytest.raises(ValueError, match="backbone"):
            VisionCNN(backbone="vgg19")

    def test_backbone_freeze_unfreeze(self, vision_cnn):
        for p in vision_cnn.backbone.parameters():
            assert not p.requires_grad
        vision_cnn.unfreeze_backbone()
        for p in vision_cnn.backbone.parameters():
            assert p.requires_grad


# =============================================================================
# FUSION HEAD
# =============================================================================

class TestFusionHead:
    def test_output_shape(self, fusion_head):
        tab = torch.randn(BATCH, 128).to(DEVICE)
        vis = torch.randn(BATCH, 128).to(DEVICE)
        out = fusion_head(tab, vis)
        assert out.shape == (BATCH, N_CLASSES)

    def test_batch_size_mismatch_raises(self, fusion_head):
        tab = torch.randn(4, 128).to(DEVICE)
        vis = torch.randn(3, 128).to(DEVICE)
        with pytest.raises(ValueError, match="Batch-size mismatch"):
            fusion_head(tab, vis)


# =============================================================================
# FULL MODEL
# =============================================================================

class TestMediMapHybridModel:
    def test_forward_shapes(self, full_model):
        sym = torch.randn(BATCH, N_SYMPTOMS).to(DEVICE)
        img = torch.randn(BATCH, 3, 224, 224).to(DEVICE)
        logits, probs = full_model(sym, img)
        assert logits.shape == (BATCH, N_CLASSES)
        assert probs.shape == (BATCH, N_CLASSES)

    def test_probabilities_sum_to_one(self, full_model):
        sym = torch.randn(BATCH, N_SYMPTOMS).to(DEVICE)
        img = torch.randn(BATCH, 3, 224, 224).to(DEVICE)
        _, probs = full_model(sym, img)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH).to(DEVICE), atol=1e-5)

    def test_predict_returns_dict(self, full_model):
        sym = torch.randn(N_SYMPTOMS).to(DEVICE)
        img = torch.randn(3, 224, 224).to(DEVICE)
        result = full_model.predict(sym, img)
        assert "predicted_class" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 100


# =============================================================================
# GEO RECOMMENDER
# =============================================================================

class TestGeoRecommender:
    def test_known_disease_mapping(self):
        assert disease_to_specialty("Pneumonia") == "Pulmonologist"
        assert disease_to_specialty("Diabetes") == "Endocrinologist"
        assert disease_to_specialty("Migraine") == "Neurologist"

    def test_unknown_disease_fallback(self):
        result = disease_to_specialty("Unknown Rare Condition XYZ")
        assert result == "General Practitioner"

    def test_all_diseases_have_mapping(self):
        """Every disease in the specialty map should resolve correctly."""
        for disease, expected_specialty in DISEASE_SPECIALTY_MAP.items():
            assert disease_to_specialty(disease) == expected_specialty
