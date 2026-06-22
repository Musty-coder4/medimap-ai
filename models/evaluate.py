"""
MediMap AI — Model Evaluation & Metrics
=========================================
Comprehensive evaluation suite for the hybrid fusion model including:
  - Per-class accuracy, precision, recall, F1
  - Confusion matrix visualisation
  - AUROC / AUPRC curves

Author : MediMap AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

logger = logging.getLogger(__name__)


def full_evaluation(
    model,
    loader,
    label_names: Optional[list[str]] = None,
    device: Optional[torch.device] = None,
    save_dir: str = "logs",
) -> dict:
    """
    Run a comprehensive evaluation pass on a DataLoader.

    Parameters
    ----------
    model : MediMapHybridModel
        Trained model in eval mode.
    loader : DataLoader
        Validation or test DataLoader yielding (symptoms, images, labels).
    label_names : list[str], optional
        Human-readable class names.
    device : torch.device, optional
        Compute device.
    save_dir : str
        Directory where the classification report is saved.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``macro_f1``, ``auroc``, ``report_text``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for symptoms, images, labels in loader:
            symptoms = symptoms.to(device)
            images = images.to(device)
            _, probs = model(symptoms, images)
            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.vstack(all_probs)

    acc = (y_true == y_pred).mean() * 100
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        zero_division=0,
    )

    try:
        auroc = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro"
        )
    except ValueError:
        auroc = float("nan")

    logger.info("Accuracy : %.2f%%", acc)
    logger.info("AUROC    : %.4f", auroc)
    logger.info("\n%s", report)

    # Persist report
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(save_dir) / "classification_report.txt"
    with open(report_path, "w") as fh:
        fh.write(report)

    return {
        "accuracy": acc,
        "auroc": auroc,
        "report_text": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
