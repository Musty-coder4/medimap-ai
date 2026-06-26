"""
MediMap AI — Inference Engine
==============================
Standalone inference pipeline for serving predictions from saved
model checkpoints.  Used by the Streamlit app and REST API endpoints.

Author : MediMap AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class MediMapInferenceEngine:
    """
    A self-contained inference wrapper that loads model checkpoints,
    label encoder, and symptom column definitions at construction time.

    Designed to be instantiated once and cached (e.g. via
    ``@st.cache_resource``) for low-latency inference.

    Parameters
    ----------
    model_checkpoint : str
        Path to the trained ``.pth`` checkpoint.
    label_encoder_path : str
        Path to the ``label_encoder.pkl`` (joblib-persisted LabelEncoder).
    symptom_columns_path : str
        Path to ``symptom_columns.json`` (list of feature names).
    vision_backbone : str
        CNN backbone name matching the checkpoint architecture.
    device : torch.device, optional
        Inference device. Defaults to CUDA if available.
    """

    def __init__(
        self,
        model_checkpoint: str,
        label_encoder_path: str,
        symptom_columns_path: str,
        vision_backbone: str = "mobilenet_v2",
        device: Optional[torch.device] = None,
    ) -> None:
        from models.hybrid_fusion import MediMapHybridModel, get_image_transforms

        self._transform = get_image_transforms(split="val")

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() and not int(os.getenv("FORCE_CPU", "0"))
                else "cpu"
            )
        self.device = device

        # ── Symptom columns ───────────────────────────────────────────────
        with open(symptom_columns_path) as fh:
            self.symptom_columns: list[str] = json.load(fh)

        # ── Label encoder ─────────────────────────────────────────────────
        self.le = joblib.load(label_encoder_path)
        self.label_names: list[str] = list(self.le.classes_)

        # ── Model ─────────────────────────────────────────────────────────
        self.model = MediMapHybridModel(
            tabular_input_dim=len(self.symptom_columns),
            num_classes=len(self.label_names),
            vision_backbone=vision_backbone,
        ).to(self.device)

        ckpt = torch.load(model_checkpoint, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info(
            "InferenceEngine initialised | device=%s | classes=%d | symptoms=%d",
            self.device, len(self.label_names), len(self.symptom_columns),
        )

    def predict_from_inputs(
        self,
        selected_symptoms: list[str],
        image: Optional[Image.Image] = None,
    ) -> dict:
        """
        Run a full prediction from user-facing inputs.

        Parameters
        ----------
        selected_symptoms : list[str]
            Symptom names selected by the user.
        image : PIL.Image.Image, optional
            Medical image (X-ray / skin scan). If None, a zero tensor is used.

        Returns
        -------
        dict
            Keys: ``predicted_class``, ``confidence``, ``probabilities``,
            ``class_index``.
        """
        # Build symptom vector
        vec = np.zeros(len(self.symptom_columns), dtype=np.float32)
        selected_lower = {s.strip().lower() for s in selected_symptoms}
        for idx, col in enumerate(self.symptom_columns):
            if col.lower() in selected_lower:
                vec[idx] = 1.0
        symptom_tensor = torch.from_numpy(vec).unsqueeze(0).to(self.device)

        # Build image tensor
        if image is not None:
            if image.mode != "RGB":
                image = image.convert("RGB")
            img_tensor = self._transform(image).unsqueeze(0).to(self.device)
        else:
            img_tensor = torch.zeros(1, 3, 224, 224).to(self.device)

        res = self.model.predict(symptom_tensor, img_tensor, self.label_names)
        probs = np.array(res["probabilities"])
        
        # Apply Bayesian Priors (Prevalence Weighting)
        disease_priors = {
            "Malaria": 1.5,
            "Common Cold": 1.5,
            "Typhoid": 1.5,
            "Dengue": 1.5,
            "Fungal infection": 1.5,
            "Allergy": 1.5,
            "Gastroenteritis": 1.5,
            "Acne": 1.5,
            "Paralysis (brain hemorrhage)": 0.1,
            "AIDS": 0.1,
            "Heart attack": 0.1,
        }
        
        weights = np.ones(len(self.label_names), dtype=np.float32)
        for i, name in enumerate(self.label_names):
            if name in disease_priors:
                weights[i] = disease_priors[name]
                
        calibrated_probs = probs * weights
        calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
        
        idx = int(np.argmax(calibrated_probs))
        
        return {
            "predicted_class": self.label_names[idx],
            "confidence": float(calibrated_probs[idx] * 100),
            "probabilities": calibrated_probs.tolist(),
            "class_index": idx
        }

    def get_clarifying_questions(self, current_symptoms: list[str], top_n: int = 5) -> list[str]:
        """
        Dynamically probe the model to find which unselected symptoms would most 
        increase the confidence of the leading hypothesis.
        
        Parameters
        ----------
        current_symptoms : list[str]
            Symptoms the user has already confirmed.
        top_n : int
            Number of clarifying symptoms to return.
            
        Returns
        -------
        list[str]
            The top_n symptom names that maximize the leading disease's probability.
        """
        baseline_res = self.predict_from_inputs(current_symptoms)
        top_idx = baseline_res["class_index"]
        
        # Access the raw probability vector for fast operations.
        # We need the base probability of the top class.
        base_probs = baseline_res["probabilities"]
        top_class_name = baseline_res["predicted_class"]
        base_prob = base_probs[top_class_name]
        
        unselected = [s for s in self.symptom_columns if s not in current_symptoms]
        scores = []
        
        for sym in unselected:
            # Test what happens if we add this symptom
            test_syms = current_symptoms + [sym]
            res = self.predict_from_inputs(test_syms)
            
            # How much does this symptom boost the top hypothesis?
            new_prob = res["probabilities"][top_class_name]
            shift = new_prob - base_prob
            
            # We only care about positive shifts (symptoms that confirm the hypothesis)
            if shift > 0:
                scores.append((sym, shift))
                
        # Load disease symptom mappings to check if we've exhausted all symptoms for the top class
        try:
            import json, os
            map_path = os.path.join(os.path.dirname(__file__), 'disease_symptoms.json')
            with open(map_path, 'r') as f:
                disease_map = json.load(f)
            top_disease_symptoms = disease_map.get(top_class_name, [])
            
            # If the user has already selected every single symptom belonging to this disease
            if set(top_disease_symptoms).issubset(set(current_symptoms)):
                return []  # Exception: Do not run fallback, skip survey!
        except Exception:
            pass

        # Sort by highest positive shift
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # FALLBACK: If no symptoms increase the top hypothesis, return the ones that cause 
        # the largest absolute change across all classes to disambiguate.
        if not scores:
            for sym in unselected:
                test_syms = current_symptoms + [sym]
                res = self.predict_from_inputs(test_syms)
                shift = sum(abs(res["probabilities"][c] - base_probs[c]) for c in base_probs)
                scores.append((sym, shift))
            scores.sort(key=lambda x: x[1], reverse=True)
            
        return [sym for sym, _ in scores[:top_n]]
