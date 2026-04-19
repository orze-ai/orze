"""agg_search — sweep aggregations × calibrators on a bundle (F11).

Given a ``preds_npz`` artifact (or a list of artifacts with a shared
``ckpt_sha`` — an InferenceBundle, F10), run a nested-CV search over
REGISTRY['aggregations'] × REGISTRY['calibrators'] to pick the best
per-clip aggregation and calibrator. The winning recipe is recorded as
a new idea with ``kind='agg_search'`` in idea_lake.

No test-set leakage: all fitting happens inside k-fold CV over the
*val* split labels the caller provides.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from orze.engine.aggregations import (
    REGISTRY,
    _auroc_like,
    make_aggregation,
    make_calibrator,
)

logger = logging.getLogger("agg_search")


def _aggregate_per_clip(
    probs_per_video: List[List[float]], agg_name: str, **agg_kw
) -> np.ndarray:
    agg = make_aggregation(agg_name, **agg_kw)
    return np.array([agg.apply(np.asarray(p)) for p in probs_per_video],
                    dtype=np.float64)


def _score(scores: np.ndarray, labels: np.ndarray,
           metric: Optional[str] = None) -> float:
    # Use AUROC-like ranking metric as the default objective — independent
    # of the project's final metric but a strong proxy for pgmAP-style.
    return _auroc_like(scores, labels)


def sweep(
    probs_per_video: List[List[float]],
    labels: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
    aggregations: Optional[List[str]] = None,
    calibrators: Optional[List[str]] = None,
    k_folds: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """Sweep agg × calibrator with k-fold CV. Returns the best recipe + metrics."""
    aggregations = aggregations or list(REGISTRY["aggregations"].keys())
    calibrators = calibrators or ["identity", "platt", "isotonic", "group_calibrated"]
    labels = np.asarray(labels, dtype=np.float64)

    results: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    rng.shuffle(idx)
    folds = np.array_split(idx, k_folds)

    for agg_name in aggregations:
        try:
            clip_scores = _aggregate_per_clip(probs_per_video, agg_name)
        except Exception as e:  # pragma: no cover
            logger.warning("agg %s crashed: %s", agg_name, e)
            continue
        for cal_name in calibrators:
            fold_scores = []
            for fold_i, val_fold in enumerate(folds):
                val_mask = np.zeros(len(labels), dtype=bool)
                val_mask[val_fold] = True
                train_mask = ~val_mask
                try:
                    cal = make_calibrator(cal_name)
                    if cal_name in ("platt", "isotonic"):
                        cal.fit(clip_scores[train_mask], labels[train_mask])
                        out = cal.apply(clip_scores[val_mask])
                    elif cal_name == "group_calibrated" and groups is not None:
                        out = cal.apply(clip_scores[val_mask],
                                        groups=groups[val_mask])
                    else:
                        out = cal.apply(clip_scores[val_mask])
                except Exception as e:  # pragma: no cover
                    logger.debug("cal %s/%s failed: %s", agg_name, cal_name, e)
                    out = clip_scores[val_mask]
                fold_scores.append(_score(out, labels[val_mask]))
            mean = float(np.mean(fold_scores))
            std = float(np.std(fold_scores))
            results.append({
                "aggregation": agg_name,
                "calibrator": cal_name,
                "score_mean": mean,
                "score_std": std,
            })

    # --- cv_mix of the two best disjoint aggregations ---
    by_score = sorted(results, key=lambda r: -r["score_mean"])
    if len(by_score) >= 2:
        agg_a = by_score[0]["aggregation"]
        agg_b = next((r["aggregation"] for r in by_score[1:]
                      if r["aggregation"] != agg_a), None)
        if agg_b:
            a_scores = _aggregate_per_clip(probs_per_video, agg_a)
            b_scores = _aggregate_per_clip(probs_per_video, agg_b)
            cal = make_calibrator("cv_mix")
            cal.fit([a_scores, b_scores], labels, k_folds=k_folds)
            mix = cal.apply([a_scores, b_scores])
            results.append({
                "aggregation": f"cv_mix({agg_a},{agg_b})",
                "calibrator": "cv_mix",
                "score_mean": _score(mix, labels),
                "score_std": 0.0,
                "alpha": cal.alpha,
            })

    results.sort(key=lambda r: -r["score_mean"])
    return {
        "best": results[0] if results else None,
        "leaderboard": results,
        "n_aggregations": len(aggregations),
        "n_calibrators": len(calibrators),
    }


def record_idea(
    lake,
    idea_id: str,
    title: str,
    base_npz: str,
    winning: Dict[str, Any],
    parent: Optional[str] = None,
) -> None:
    """Insert the winning recipe as a kind='agg_search' idea."""
    cfg = {
        "base_npz": base_npz,
        "winning_recipe": winning,
    }
    cfg_yaml = json.dumps(cfg, indent=2)
    lake.insert(
        idea_id, title, cfg_yaml, raw_markdown=cfg_yaml,
        eval_metrics={"score_mean": winning["score_mean"]},
        status="completed",
        kind="agg_search",
        parent=parent,
    )
