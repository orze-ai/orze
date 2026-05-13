"""Competition diagnosis — automated dataset analysis for research guidance.

CALLING SPEC:
    diagnose_dataset(train_path, test_path=None, sample_submission_path=None)
        -> CompetitionDiagnosis
        Analyzes a competition dataset and produces structured recommendations
        for CV strategy, sampling, augmentation, and model families.

    format_diagnosis(diagnosis: CompetitionDiagnosis) -> str
        Renders diagnosis as YAML text for injection into research/engineer prompts.

    load_cached_diagnosis(results_dir, competition_id) -> CompetitionDiagnosis | None
        Loads a previously computed diagnosis from results/_diagnoses/.

    save_diagnosis(results_dir, competition_id, diagnosis) -> Path
        Saves diagnosis to results/_diagnoses/{competition_id}.yaml.

This module runs dataset-level analysis ONCE per competition (or on plateau)
and produces actionable guidance that feeds into the research agent's prompt.
It catches common pitfalls that LLM agents miss:
  - Class imbalance → recommend rebalancing
  - Group columns → recommend GroupKFold
  - Image data → recommend augmentation + backbone diversity
  - High cardinality categoricals → recommend encoding strategy
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("orze.diagnosis")


@dataclass
class CompetitionDiagnosis:
    """Structured analysis of a competition dataset."""
    competition_id: str = ""

    # Data shape
    n_train: int = 0
    n_test: int = 0
    n_features: int = 0
    target_column: str = ""
    target_type: str = ""  # binary, multiclass, regression, multilabel

    # Class distribution (classification only)
    class_distribution: Dict[str, float] = field(default_factory=dict)
    is_imbalanced: bool = False
    imbalance_ratio: float = 1.0

    # Data types
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    text_columns: List[str] = field(default_factory=list)
    image_columns: List[str] = field(default_factory=list)
    group_columns: List[str] = field(default_factory=list)

    # Recommendations
    recommended_cv: str = "StratifiedKFold"
    recommended_sampling: str = "none"
    recommended_augmentation: List[str] = field(default_factory=list)
    recommended_model_families: List[str] = field(default_factory=list)
    recommended_preprocessing: List[str] = field(default_factory=list)

    # Pitfalls
    pitfalls: List[str] = field(default_factory=list)

    # Metadata
    has_images: bool = False
    has_text: bool = False
    has_tabular: bool = True


def diagnose_dataset(
    train_path: str,
    test_path: Optional[str] = None,
    sample_submission_path: Optional[str] = None,
    competition_id: str = "",
) -> CompetitionDiagnosis:
    """Analyze a competition dataset and produce structured recommendations."""
    import pandas as pd
    import numpy as np

    diag = CompetitionDiagnosis(competition_id=competition_id)

    try:
        train = pd.read_csv(train_path, nrows=50000)
    except Exception as e:
        logger.error("Failed to read train.csv: %s", e)
        return diag

    diag.n_train = len(train)
    diag.n_features = len(train.columns) - 1

    # Detect target column
    target_candidates = ["target", "label", "class", "y", "is_duplicate",
                         "label_group", "Score", "score"]
    for col in target_candidates:
        if col in train.columns:
            diag.target_column = col
            break
    if not diag.target_column and "id" in train.columns:
        diag.target_column = train.columns[-1]

    if diag.target_column and diag.target_column in train.columns:
        target = train[diag.target_column]

        # Determine target type
        n_unique = target.nunique()
        if n_unique == 2:
            diag.target_type = "binary"
        elif n_unique <= 100:
            diag.target_type = "multiclass"
        else:
            diag.target_type = "regression"

        # Class distribution
        if diag.target_type in ("binary", "multiclass"):
            vc = target.value_counts(normalize=True)
            diag.class_distribution = {str(k): float(v) for k, v in vc.items()}
            min_class = vc.min()
            max_class = vc.max()
            diag.imbalance_ratio = float(max_class / min_class) if min_class > 0 else float("inf")
            diag.is_imbalanced = diag.imbalance_ratio > 10

    # Detect column types
    for col in train.columns:
        if col in (diag.target_column, "id", "Id", "ID"):
            continue

        dtype = train[col].dtype
        n_unique = train[col].nunique()

        # Check for group columns
        col_lower = col.lower()
        if any(g in col_lower for g in ["patient", "user", "group", "session",
                                         "subject", "study", "case"]):
            diag.group_columns.append(col)

        # Check for image paths
        if train[col].dtype == object and n_unique > 100:
            sample = str(train[col].iloc[0])
            if any(ext in sample.lower() for ext in [".jpg", ".png", ".dcm", ".tif", ".jpeg"]):
                diag.image_columns.append(col)
                diag.has_images = True
                continue

        # Check for text
        if train[col].dtype == object and n_unique > 100:
            avg_len = train[col].astype(str).str.len().mean()
            if avg_len > 50:
                diag.text_columns.append(col)
                diag.has_text = True
                continue

        if np.issubdtype(dtype, np.number):
            diag.numeric_columns.append(col)
        elif dtype == object or str(dtype) == "category":
            diag.categorical_columns.append(col)

    # Test set size
    if test_path and os.path.exists(test_path):
        try:
            test = pd.read_csv(test_path, nrows=5)
            diag.n_test = sum(1 for _ in open(test_path)) - 1
        except Exception:
            pass

    # Generate recommendations
    _generate_recommendations(diag)

    return diag


def _generate_recommendations(diag: CompetitionDiagnosis) -> None:
    """Fill in recommendations based on detected properties."""

    # CV strategy
    if diag.group_columns:
        diag.recommended_cv = f"GroupKFold(groups='{diag.group_columns[0]}')"
        diag.pitfalls.append(f"GROUP LEAKAGE RISK: Use GroupKFold on '{diag.group_columns[0]}' "
                            "to prevent data leakage across folds")
    elif diag.is_imbalanced:
        diag.recommended_cv = "StratifiedKFold"
    else:
        diag.recommended_cv = "KFold"

    # Sampling strategy
    if diag.is_imbalanced:
        if diag.imbalance_ratio > 50:
            diag.recommended_sampling = f"WeightedRandomSampler(pos_weight~{diag.imbalance_ratio:.0f}:1)"
            diag.pitfalls.append(f"SEVERE CLASS IMBALANCE: {diag.imbalance_ratio:.0f}:1 ratio. "
                                "Models will collapse to majority class without rebalancing.")
        elif diag.imbalance_ratio > 10:
            diag.recommended_sampling = f"WeightedRandomSampler(pos_weight~{diag.imbalance_ratio:.0f}:1)"
            diag.pitfalls.append(f"CLASS IMBALANCE: {diag.imbalance_ratio:.0f}:1 ratio. "
                                "Use weighted sampling or focal loss.")

    # Model families
    if diag.has_images:
        diag.recommended_model_families = [
            "efficientnet (proven baseline)",
            "convnext (modern CNN, different inductive bias)",
            "swin/vit (vision transformer, global attention)",
            "dino/eva (self-supervised pretrained)",
        ]
        diag.recommended_augmentation = ["HorizontalFlip", "VerticalFlip", "RandomCrop",
                                         "ColorJitter", "MixUp/CutMix"]
        diag.recommended_preprocessing = [
            "Multi-resolution training (384, 512, 640)",
            "TTA (4-pass: original + H-flip + V-flip + HV-flip)",
            "Metadata fusion head (if auxiliary features exist)",
        ]
        diag.pitfalls.append("ARCHITECTURE DIVERSITY: Train at least 3 different backbone "
                            "families for ensemble diversity.")
    elif diag.has_text:
        diag.recommended_model_families = [
            "bert/deberta (masked language model)",
            "distilbert (faster inference)",
            "lgbm/xgboost (with TF-IDF features)",
        ]
    else:
        diag.recommended_model_families = [
            "lightgbm (fast, good default)",
            "xgboost (alternative gradient boosting)",
            "catboost (handles categoricals natively)",
            "neural network (if >100K samples)",
            "tabnet (attention-based tabular)",
        ]

    if diag.n_train > 1_000_000:
        diag.pitfalls.append(f"LARGE DATASET: {diag.n_train:,} rows. Consider subsampling "
                            "for initial exploration, full training for final submission.")


def format_diagnosis(diagnosis: CompetitionDiagnosis) -> str:
    """Render diagnosis as YAML text for prompt injection."""
    d = {
        "competition": diagnosis.competition_id,
        "data": {
            "train_size": diagnosis.n_train,
            "test_size": diagnosis.n_test,
            "features": diagnosis.n_features,
            "target": {
                "column": diagnosis.target_column,
                "type": diagnosis.target_type,
            },
        },
        "class_distribution": diagnosis.class_distribution if diagnosis.class_distribution else "N/A",
        "imbalanced": diagnosis.is_imbalanced,
        "imbalance_ratio": round(diagnosis.imbalance_ratio, 1) if diagnosis.is_imbalanced else "N/A",
        "group_columns": diagnosis.group_columns or "none detected",
        "recommendations": {
            "cv_strategy": diagnosis.recommended_cv,
            "sampling": diagnosis.recommended_sampling,
            "model_families": diagnosis.recommended_model_families,
            "augmentation": diagnosis.recommended_augmentation or "N/A",
            "preprocessing": diagnosis.recommended_preprocessing or "N/A",
        },
        "pitfalls": diagnosis.pitfalls or "none detected",
    }
    return yaml.dump(d, default_flow_style=False, sort_keys=False)


def load_cached_diagnosis(
    results_dir: Path,
    competition_id: str,
) -> Optional[CompetitionDiagnosis]:
    """Load a previously computed diagnosis."""
    p = Path(results_dir) / "_diagnoses" / f"{competition_id}.yaml"
    if not p.exists():
        return None
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        diag = CompetitionDiagnosis(competition_id=competition_id)
        diag.n_train = data.get("data", {}).get("train_size", 0)
        diag.target_type = data.get("data", {}).get("target", {}).get("type", "")
        diag.is_imbalanced = data.get("imbalanced", False)
        diag.pitfalls = data.get("pitfalls", [])
        return diag
    except Exception:
        return None


def save_diagnosis(
    results_dir: Path,
    competition_id: str,
    diagnosis: CompetitionDiagnosis,
) -> Path:
    """Save diagnosis to results/_diagnoses/."""
    out_dir = Path(results_dir) / "_diagnoses"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{competition_id}.yaml"
    out_path.write_text(format_diagnosis(diagnosis), encoding="utf-8")
    return out_path
