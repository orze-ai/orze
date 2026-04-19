"""InferenceBundle + bundle_combiner (F10).

A *bundle* is a collection of ``preds_npz`` / ``tta_preds`` artifacts that
all share the same ``ckpt_sha`` — multiple TTA views of ONE trained model,
not an ensemble of different models. Ensembles are *forbidden* here by
design (rule: single-model discipline).

Given a bundle, the combiner searches over:

    * k in 1..len(bundle)                       # choose how many views to avg
    * every subset of size k                    # which views to pick
    * REGISTRY['aggregations']                  # per-frame → per-clip
    * REGISTRY['calibrators']                   # post-aggregation calibration

and records the best recipe as a ``kind='bundle_combine'`` idea in
idea_lake.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from orze.engine.aggregations import (
    REGISTRY,
    _auroc_like,
    make_aggregation,
    make_calibrator,
)

logger = logging.getLogger("bundle_combiner")


@dataclass
class InferenceBundle:
    """A bundle of prediction NPZs sharing a single ckpt_sha."""
    ckpt_sha: str
    view_paths: List[str]
    # probs[v] is a list[list[float]] — per-video, per-frame probs for view v.
    probs: List[List[List[float]]]

    def __post_init__(self):
        n = len(self.probs)
        if n != len(self.view_paths):
            raise ValueError("view_paths and probs length mismatch")
        if n == 0:
            raise ValueError("bundle is empty")

    @property
    def size(self) -> int:
        return len(self.view_paths)

    def mean_over_views(self, indices: Sequence[int]) -> List[List[float]]:
        """Per-clip, per-frame probs averaged over the selected views.

        Frame counts are assumed equal across views of the same clip (true
        for views derived from the same dataset pass). If they differ we
        clip to the shortest length.
        """
        chosen = [self.probs[i] for i in indices]
        n_videos = len(chosen[0])
        out: List[List[float]] = []
        for vid in range(n_videos):
            # min frame count across the chosen views for this video
            min_len = min(len(v[vid]) for v in chosen)
            if min_len == 0:
                out.append([])
                continue
            stack = np.stack(
                [np.asarray(v[vid][:min_len], dtype=np.float64) for v in chosen],
                axis=0,
            )
            out.append(stack.mean(axis=0).tolist())
        return out


def load_bundle_from_catalog(
    catalog,
    ckpt_sha: str,
    *,
    loader=None,
) -> InferenceBundle:
    """Build an InferenceBundle from ArtifactCatalog rows.

    ``loader`` is a callable ``(path) -> List[List[float]]`` that returns
    per-video per-frame probability lists. Defaults to an .npz loader
    expecting keys 'probs_per_video' or ('video_ids', 'per_frame_probs').
    """
    rows = catalog.bundle(ckpt_sha)
    if not rows:
        raise ValueError(f"no bundle artifacts for ckpt_sha={ckpt_sha!r}")
    shas = {r["ckpt_sha"] for r in rows if r.get("ckpt_sha")}
    if shas - {ckpt_sha}:
        raise ValueError(
            f"InferenceBundle mixes ckpt_shas {shas}; "
            "ensembles are not allowed — use the same ckpt for all views."
        )
    loader = loader or _default_npz_loader
    probs = [loader(r["path"]) for r in rows]
    return InferenceBundle(
        ckpt_sha=ckpt_sha,
        view_paths=[r["path"] for r in rows],
        probs=probs,
    )


def _default_npz_loader(path: str) -> List[List[float]]:
    data = np.load(path, allow_pickle=True)
    if "probs_per_video" in data.files:
        obj = data["probs_per_video"]
        # Could be an object array of variable-length arrays
        try:
            return [list(np.asarray(v, dtype=np.float64)) for v in obj]
        except Exception:
            return [list(map(float, v)) for v in obj]
    if "per_frame_probs" in data.files:
        return [list(np.asarray(v, dtype=np.float64))
                for v in data["per_frame_probs"]]
    raise ValueError(
        f"don't know how to load per-frame probs from {path!r}; "
        "expected key 'probs_per_video' or 'per_frame_probs'"
    )


# ---------------------------------------------------------------------- #
# Search                                                                 #
# ---------------------------------------------------------------------- #


def search(
    bundle: InferenceBundle,
    labels: np.ndarray,
    *,
    aggregations: Optional[List[str]] = None,
    calibrators: Optional[List[str]] = None,
    k_range: Optional[Sequence[int]] = None,
    max_subsets_per_k: int = 32,
    metric=None,
) -> Dict[str, Any]:
    """Sweep k × subsets × aggregations × calibrators; return best recipe."""
    aggregations = aggregations or ["last", "max", "mean", "late_k2",
                                    "late_k3", "top_k_mean", "noisy_or"]
    calibrators = calibrators or ["identity", "platt", "isotonic"]
    metric = metric or _auroc_like
    k_range = k_range or list(range(1, bundle.size + 1))
    labels = np.asarray(labels, dtype=np.float64)

    results: List[Dict[str, Any]] = []
    for k in k_range:
        if k < 1 or k > bundle.size:
            continue
        subsets = list(itertools.combinations(range(bundle.size), k))
        if len(subsets) > max_subsets_per_k:
            # deterministic sample for reproducibility
            subsets = subsets[:max_subsets_per_k]
        for subset in subsets:
            avg_probs = bundle.mean_over_views(subset)
            for agg_name in aggregations:
                agg = make_aggregation(agg_name)
                clip_scores = np.array([agg.apply(np.asarray(p))
                                        for p in avg_probs], dtype=np.float64)
                for cal_name in calibrators:
                    try:
                        cal = make_calibrator(cal_name)
                        if cal_name in ("platt", "isotonic"):
                            cal.fit(clip_scores, labels)
                        out = cal.apply(clip_scores)
                    except Exception:  # pragma: no cover
                        out = clip_scores
                    s = metric(out, labels)
                    results.append({
                        "k": k,
                        "subset": list(subset),
                        "subset_paths": [bundle.view_paths[i] for i in subset],
                        "aggregation": agg_name,
                        "calibrator": cal_name,
                        "score": float(s),
                    })

    if not results:
        return {"best": None, "leaderboard": []}
    results.sort(key=lambda r: -r["score"])
    return {
        "best": results[0],
        "leaderboard": results,
        "n_candidates": len(results),
        "ckpt_sha": bundle.ckpt_sha,
    }


def record_idea(
    lake,
    idea_id: str,
    title: str,
    bundle: InferenceBundle,
    winning: Dict[str, Any],
    *,
    eval_metrics: Optional[Dict[str, Any]] = None,
    parent: Optional[str] = None,
) -> None:
    """Insert the winning bundle recipe as a kind='bundle_combine' idea."""
    cfg = {
        "ckpt_sha": bundle.ckpt_sha,
        "view_paths": bundle.view_paths,
        "winning_recipe": {
            "subset_paths": winning["subset_paths"],
            "aggregation": winning["aggregation"],
            "calibrator": winning["calibrator"],
        },
    }
    cfg_yaml = json.dumps(cfg, indent=2)
    metrics = dict(eval_metrics or {})
    metrics.setdefault("score", winning["score"])
    lake.insert(
        idea_id, title, cfg_yaml, raw_markdown=cfg_yaml,
        eval_metrics=metrics,
        status="completed",
        kind="bundle_combine",
        parent=parent,
    )
