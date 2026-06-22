"""
MediMap AI — Image & Tabular Data Preprocessing Utilities
==========================================================
Shared helpers for both training and inference pipelines.

Author : MediMap AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


# =============================================================================
# TABULAR PREPROCESSING
# =============================================================================

def load_and_one_hot_encode(
    csv_path: str,
    symptom_col_prefix: str = "Symptom",
    label_col: str = "Disease",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Load the dataset and produce a binary symptom matrix.
    Supports both the old Kaggle format and the new SympScan binary matrix format.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Auto-detect label column if default is not found
    if label_col not in df.columns:
        if "diseases" in df.columns:
            label_col = "diseases"
        else:
            label_col = df.columns[0]
            
    df[label_col] = df[label_col].astype(str).str.strip()

    sym_cols = [c for c in df.columns if c.startswith(symptom_col_prefix)]
    
    if not sym_cols:
        # SympScan Binary Matrix Format
        raw_symptom_names = sorted([c for c in df.columns if c != label_col])
        X = df[raw_symptom_names].astype(np.float32)
        
        # Clean column names (replace spaces, drop weird chars)
        import re
        cleaned_names = [re.sub(r'[^a-zA-Z0-9_]', '', c.replace(" ", "_").replace("-", "_")).lower() for c in raw_symptom_names]
        X.columns = cleaned_names
        symptom_names = cleaned_names
    else:
        # Old Kaggle Format
        all_symptoms: set[str] = set()
        for col in sym_cols:
            vals = df[col].dropna().str.strip().str.lower()
            all_symptoms.update(vals.unique())
        all_symptoms.discard("")
        
        # Clean names
        import re
        cleaned_symptoms = {sym: re.sub(r'[^a-zA-Z0-9_]', '', sym.replace(" ", "_").replace("-", "_")) for sym in all_symptoms}
        symptom_names = sorted(list(set(cleaned_symptoms.values())))

        # Build binary matrix
        X = pd.DataFrame(0, index=df.index, columns=symptom_names, dtype=np.float32)
        for col in sym_cols:
            normalised = df[col].str.strip().str.lower().fillna("")
            for old_sym, new_sym in cleaned_symptoms.items():
                X.loc[normalised == old_sym, new_sym] = 1.0

    logger.info(
        "Loaded %d samples, %d symptoms, %d disease classes.",
        len(df), len(symptom_names), df[label_col].nunique(),
    )
    return X, df[label_col], symptom_names


def validate_symptom_vector(
    vec: np.ndarray,
    expected_dim: int,
    name: str = "input",
) -> None:
    """
    Assert that a symptom vector has the expected dimensionality.

    Parameters
    ----------
    vec : np.ndarray
        The symptom vector to validate.
    expected_dim : int
        Expected number of features.
    name : str
        Identifier for error messages.

    Raises
    ------
    ValueError
        If the shape does not match.
    """
    if vec.ndim != 1 or vec.shape[0] != expected_dim:
        raise ValueError(
            f"[{name}] Expected shape ({expected_dim},), got {vec.shape}."
        )


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def load_medical_image(
    path: str,
    target_size: tuple[int, int] = (224, 224),
    grayscale_to_rgb: bool = True,
) -> Image.Image:
    """
    Load and resize a medical image from disk.

    Handles grayscale X-rays (mode L / I) by converting to RGB so that
    torchvision models receive the expected 3-channel input.

    Parameters
    ----------
    path : str
        Filesystem path to the image.
    target_size : tuple[int, int]
        ``(width, height)`` to resize to.
    grayscale_to_rgb : bool
        If True, single-channel images are converted to RGB.

    Returns
    -------
    PIL.Image.Image
        Loaded (and possibly converted) image.
    """
    img = Image.open(path)
    if grayscale_to_rgb and img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    elif img.mode == "RGBA":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return img


def compute_file_hash(file_path: str) -> str:
    """
    Compute MD5 hash of a file (for de-duplication in datasets).

    Parameters
    ----------
    file_path : str

    Returns
    -------
    str
        Hex-encoded MD5 digest.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# =============================================================================
# DATA DIRECTORY SCAFFOLDING
# =============================================================================

DATA_DIRS = [
    "data/raw/tabular",
    "data/raw/images/xray",
    "data/raw/images/skin",
    "data/processed",
    "data/processed/splits",
    "models/saved",
    "models/checkpoints",
    "logs",
    "mlruns",
]


def scaffold_directories(base_path: str = ".") -> None:
    """
    Create all required project directories if they don't exist.

    Parameters
    ----------
    base_path : str
        Project root directory.
    """
    root = Path(base_path)
    for rel_path in DATA_DIRS:
        target = root / rel_path
        target.mkdir(parents=True, exist_ok=True)
        # Add .gitkeep so empty dirs are tracked
        gitkeep = target / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    logger.info("Project directories scaffolded at '%s'.", base_path)
