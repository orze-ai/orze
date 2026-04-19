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
    splits: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sweep k × subsets × aggregations × calibrators; return best recipe.

    If ``splits`` is given it must be::

        {
          "val":    bool-mask,                  # tuning set (labels OK)
          "report": {"all": None, "public": mask, "private": mask, ...},
        }

    Hyperparameters (and any calibrator fit) are selected on ``val`` only
    and then scored independently on each ``report`` mask (``None`` means
    "the full population"). The returned ``best`` entry carries a
    ``reports`` sub-dict with one score per report split, and the
    ``honest`` flag is set to ``True`` when tuning and reporting are
    disjoint.

    Without ``splits`` the legacy single-score behavior is preserved
    (``honest`` flag is ``False`` in that case).
    """
    aggregations = aggregations or ["last", "max", "mean", "late_k2",
                                    "late_k3", "top_k_mean", "noisy_or"]
    calibrators = calibrators or ["identity", "platt", "isotonic"]
    metric = metric or _auroc_like
    k_range = k_range or list(range(1, bundle.size + 1))
    labels = np.asarray(labels, dtype=np.float64)

    if splits is not None:
        val_mask = np.asarray(splits["val"], dtype=bool)
        reports = splits.get("report") or {"all": None}
        # Leak guard: there must be at least one report split that is NOT
        # identical to val_mask, otherwise tuning and reporting are the
        # same set (classic CV-OOF-on-test leak). Reporting the tuning
        # metric alongside is fine — that is just a readout.
        has_heldout = False
        for rm in reports.values():
            if rm is None:
                has_heldout = True
                break
            rm_arr = np.asarray(rm, dtype=bool)
            if rm_arr.shape != val_mask.shape or not np.array_equal(rm_arr, val_mask):
                has_heldout = True
                break
        if not has_heldout:
            raise ValueError(
                "bundle_combiner.search: every report split is identical "
                "to splits['val'] — that is a leak (tuning and reporting "
                "on the same rows). Add at least one disjoint report mask "
                "or use None to report on the full population.")
        honest = True
    else:
        val_mask = np.ones_like(labels, dtype=bool)
        reports = {"all": None}
        honest = False

    def _score(scores: np.ndarray, mask) -> float:
        if mask is None:
            return float(metric(scores, labels))
        mask = np.asarray(mask, dtype=bool)
        return float(metric(scores[mask], labels[mask]))

    results: List[Dict[str, Any]] = []
    for k in k_range:
        if k < 1 or k > bundle.size:
            continue
        subsets = list(itertools.combinations(range(bundle.size), k))
        if len(subsets) > max_subsets_per_k:
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
                            # Fit on val_mask only (honest contract).
                            cal.fit(clip_scores[val_mask], labels[val_mask])
                        out = cal.apply(clip_scores)
                    except Exception:  # pragma: no cover
                        out = clip_scores
                    out = np.asarray(out, dtype=np.float64)
                    val_score = _score(out, val_mask)
                    report_scores = {
                        name: _score(out, mask)
                        for name, mask in reports.items()
                    }
                    results.append({
                        "k": k,
                        "subset": list(subset),
                        "subset_paths": [bundle.view_paths[i] for i in subset],
                        "aggregation": agg_name,
                        "calibrator": cal_name,
                        "score": val_score,   # tuning metric (on val)
                        "reports": report_scores,
                        "honest": honest,
                    })

    if not results:
        return {"best": None, "leaderboard": [], "honest": honest}
    results.sort(key=lambda r: -r["score"])
    return {
        "best": results[0],
        "leaderboard": results,
        "n_candidates": len(results),
        "ckpt_sha": bundle.ckpt_sha,
        "honest": honest,
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
