"""Ensemble diversity tracker — measures model correlation and suggests next architecture.

CALLING SPEC:
    compute_diversity(predictions: dict[str, np.ndarray],
                      scores: dict[str, float],
                      families: dict[str, str] | None = None)
        -> DiversityReport
        predictions: model_name -> 1D prediction array (probabilities or logits)
        scores: model_name -> scalar metric (higher = better assumed unless sort_ascending)
        families: model_name -> architecture family string (e.g. "efficientnet", "vit", "xgboost")
        Returns DiversityReport with correlation matrix, family distribution, and suggestions.

    suggest_next_family(report: DiversityReport,
                        all_families: list[str] | None = None)
        -> list[str]
        Given a DiversityReport, returns a ranked list of architecture families
        most likely to improve ensemble diversity (lowest mean correlation with
        existing high-scoring models).

    blend_optimize(predictions: dict[str, np.ndarray],
                   grade_fn: Callable[[np.ndarray], float],
                   method: str = "nelder_mead",
                   top_n: int = 8)
        -> (float, dict[str, float])
        Finds optimal weighted blend of top_n models using grade_fn as objective.
        Returns (best_score, {model: weight}).

    format_diversity_context(report: DiversityReport) -> str
        Renders the diversity report as markdown text suitable for injection
        into a research agent prompt.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("orze.diversity")

KNOWN_FAMILIES = [
    "efficientnet", "convnext", "swin", "vit", "resnet", "resnext",
    "densenet", "maxvit", "eva", "dino", "nfnet", "regnet",
    "xgboost", "lightgbm", "catboost",
    "transformer", "lstm", "mlp", "tabnet",
]


def _infer_family(model_name: str) -> str:
    """Best-effort architecture family from model name."""
    name = model_name.lower().replace("-", "_").replace(" ", "_")
    for fam in KNOWN_FAMILIES:
        if fam in name:
            return fam
    if "eff" in name and ("b5" in name or "b6" in name or "b7" in name or "b8" in name or "v2" in name):
        return "efficientnet"
    if "swin" in name:
        return "swin"
    if "convnext" in name or "convnet" in name:
        return "convnext"
    if "vit" in name or "deit" in name:
        return "vit"
    if "lgb" in name or "lgbm" in name:
        return "lightgbm"
    if "xgb" in name:
        return "xgboost"
    if "cat" in name and "boost" in name:
        return "catboost"
    return "unknown"


@dataclass
class DiversityReport:
    """Structured report on ensemble model diversity."""
    model_names: List[str]
    scores: Dict[str, float]
    families: Dict[str, str]
    correlation_matrix: Any  # np.ndarray, but avoid type dep at import
    family_distribution: Dict[str, int]
    mean_correlation: float
    diversity_score: float  # 1 - mean_correlation
    dominant_family: Optional[str]
    dominant_fraction: float
    suggested_families: List[str]
    blend_score: Optional[float] = None
    blend_weights: Optional[Dict[str, float]] = None


def compute_diversity(
    predictions: Dict[str, np.ndarray],
    scores: Dict[str, float],
    families: Optional[Dict[str, str]] = None,
    top_n: int = 20,
) -> DiversityReport:
    """Compute diversity metrics across a set of model predictions."""
    from scipy.stats import spearmanr

    if families is None:
        families = {m: _infer_family(m) for m in predictions}

    sorted_models = sorted(scores, key=lambda m: scores[m], reverse=True)[:top_n]
    sorted_models = [m for m in sorted_models if m in predictions]

    if len(sorted_models) < 2:
        return DiversityReport(
            model_names=sorted_models,
            scores=scores,
            families=families,
            correlation_matrix=np.array([[1.0]]),
            family_distribution={families.get(sorted_models[0], "unknown"): 1} if sorted_models else {},
            mean_correlation=1.0,
            diversity_score=0.0,
            dominant_family=families.get(sorted_models[0], "unknown") if sorted_models else None,
            dominant_fraction=1.0,
            suggested_families=["vit", "convnext", "swin"],
        )

    n = len(sorted_models)
    corr = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pi = predictions[sorted_models[i]]
            pj = predictions[sorted_models[j]]
            min_len = min(len(pi), len(pj))
            rho, _ = spearmanr(pi[:min_len], pj[:min_len])
            corr[i, j] = corr[j, i] = rho

    mask = np.ones_like(corr, dtype=bool)
    np.fill_diagonal(mask, False)
    mean_corr = float(corr[mask].mean()) if mask.sum() > 0 else 1.0

    fam_dist: Dict[str, int] = {}
    for m in sorted_models:
        f = families.get(m, "unknown")
        fam_dist[f] = fam_dist.get(f, 0) + 1

    dominant = max(fam_dist, key=fam_dist.get) if fam_dist else None
    dominant_frac = fam_dist.get(dominant, 0) / len(sorted_models) if dominant else 0.0

    present_families = set(fam_dist.keys())
    suggested = [f for f in KNOWN_FAMILIES if f not in present_families][:5]

    return DiversityReport(
        model_names=sorted_models,
        scores=scores,
        families=families,
        correlation_matrix=corr,
        family_distribution=fam_dist,
        mean_correlation=mean_corr,
        diversity_score=1.0 - mean_corr,
        dominant_family=dominant,
        dominant_fraction=dominant_frac,
        suggested_families=suggested,
    )


def suggest_next_family(
    report: DiversityReport,
    all_families: Optional[List[str]] = None,
) -> List[str]:
    """Rank architecture families by expected diversity contribution."""
    if all_families is None:
        all_families = KNOWN_FAMILIES

    present = set(report.family_distribution.keys())
    absent = [f for f in all_families if f not in present]

    if report.dominant_fraction > 0.5 and report.dominant_family:
        suggestion = f"WARNING: {report.dominant_fraction:.0%} of models are {report.dominant_family}. "
        suggestion += f"Strongly recommend trying: {', '.join(absent[:3])}"
        logger.warning(suggestion)

    return absent if absent else list(report.family_distribution.keys())


def blend_optimize(
    predictions: Dict[str, np.ndarray],
    grade_fn: Callable[[np.ndarray], float],
    method: str = "nelder_mead",
    top_n: int = 8,
    n_trials: int = 10,
    maxiter: int = 150,
) -> Tuple[float, Dict[str, float]]:
    """Find optimal weighted blend of models."""
    from scipy.optimize import minimize
    from scipy.special import expit, logit

    models = list(predictions.keys())[:top_n]
    if len(models) < 2:
        if models:
            return grade_fn(predictions[models[0]]), {models[0]: 1.0}
        return 0.0, {}

    logits = {}
    for m in models:
        p = np.clip(predictions[m], 1e-6, 1 - 1e-6)
        logits[m] = logit(p)

    logit_arr = np.array([logits[m] for m in models])

    best_score = -1.0
    best_weights = np.ones(len(models)) / len(models)

    def neg_score(w):
        w = np.abs(w)
        s = w.sum()
        if s < 1e-10:
            return 0.0
        w = w / s
        blend = expit((w[:, None] * logit_arr).sum(axis=0))
        return -grade_fn(blend)

    for trial in range(n_trials):
        w0 = np.random.dirichlet(np.ones(len(models)))
        try:
            res = minimize(neg_score, w0, method="Nelder-Mead",
                           options={"maxiter": maxiter, "xatol": 0.001})
            score = -res.fun
            if score > best_score:
                best_score = score
                w = np.abs(res.x)
                best_weights = w / w.sum()
        except Exception:
            continue

    weight_dict = {m: float(best_weights[i]) for i, m in enumerate(models)}
    weight_dict = {m: w for m, w in weight_dict.items() if w > 0.01}
    total = sum(weight_dict.values())
    weight_dict = {m: w / total for m, w in weight_dict.items()}

    return best_score, weight_dict


def format_diversity_context(report: DiversityReport) -> str:
    """Render diversity report as text for research agent prompt injection."""
    lines = []
    lines.append(f"### Ensemble Diversity ({len(report.model_names)} models)")
    lines.append(f"- Mean pairwise correlation: **{report.mean_correlation:.3f}** "
                 f"(diversity score: {report.diversity_score:.3f})")

    if report.dominant_family:
        lines.append(f"- Dominant family: **{report.dominant_family}** "
                     f"({report.dominant_fraction:.0%} of models)")

    lines.append(f"- Family distribution: {dict(report.family_distribution)}")

    if report.mean_correlation > 0.9:
        lines.append("")
        lines.append("⚠️ **HIGH CORRELATION WARNING**: Models are too similar. "
                     "Blending provides diminishing returns.")
        lines.append(f"**DO NOT propose another {report.dominant_family} variant.**")
        lines.append(f"Instead, propose models from: {', '.join(report.suggested_families[:3])}")

    if report.blend_score is not None:
        lines.append(f"- Best blend score: {report.blend_score:.6f}")

    if report.blend_weights:
        top_w = sorted(report.blend_weights.items(), key=lambda x: -x[1])[:5]
        lines.append(f"- Top blend weights: " +
                     ", ".join(f"{m}={w:.2f}" for m, w in top_w))

    return "\n".join(lines)
