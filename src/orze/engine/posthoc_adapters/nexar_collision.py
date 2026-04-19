"""Nexar-collision post-hoc adapter (orze v3.7.0).

Thin adapter: reads a list of per-frame probability NPZs (the nexar TTA
schema: keys ``video_ids``, ``frame_centers``, ``probs``), averages
across TTA views that share a ``ckpt_sha``, then delegates aggregation
and calibration to the :mod:`orze.engine.aggregations` registry and
:mod:`orze.engine.bundle_combiner` honest split-aware search.

Honest-eval protocol (default):

* tune α / aggregation on ``Usage == 'Public'`` rows of ``solution.csv``
* report:
    - ``pgmAP_Public``  (tuning metric; not the headline)
    - ``pgmAP_Private`` (held-out, headline)
    - ``pgmAP_ALL_honest`` (Public-tuned α, applied unchanged to ALL)

No hand-written consumer scripts are imported — this module is
adapter-thin.
"""

from __future__ import annotations

import collections
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from orze.engine.aggregations import per_group_rank_normalize
from orze.engine.posthoc_runner import register_adapter, subprocess_adapter

logger = logging.getLogger("posthoc_adapters.nexar_collision")


# --------------------------------------------------------------------- #
# NPZ loading                                                           #
# --------------------------------------------------------------------- #


def _load_per_vid_keyed(path: str) -> Dict[str, Dict[int, float]]:
    d = np.load(path, allow_pickle=True)
    out: Dict[str, Dict[int, float]] = collections.defaultdict(dict)
    for vid, fc, p in zip(d["video_ids"], d["frame_centers"], d["probs"]):
        out[str(vid).zfill(5)][int(fc)] = float(p)
    return out


def _avg_views(npz_paths: List[str]) -> Dict[str, np.ndarray]:
    """Average per-(vid, frame_center) probs across views; intersect keys."""
    dicts = [_load_per_vid_keyed(p) for p in npz_paths]
    vids = set(dicts[0].keys())
    for d in dicts[1:]:
        vids &= set(d.keys())
    per: Dict[str, np.ndarray] = {}
    for v in sorted(vids):
        fcs = set(dicts[0][v].keys())
        for d in dicts[1:]:
            fcs &= set(d[v].keys())
        if not fcs:
            continue
        sf = sorted(fcs)
        per[v] = np.array(
            [float(np.mean([d[v][fc] for d in dicts])) for fc in sf],
            dtype=np.float64,
        )
    return per


# --------------------------------------------------------------------- #
# Metric                                                                #
# --------------------------------------------------------------------- #


def _pgmAP(scores: np.ndarray, labels: np.ndarray,
           groups: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    from sklearn.metrics import average_precision_score
    if mask is not None:
        scores = scores[mask]
        labels = labels[mask]
        groups = groups[mask]
    vals = []
    for gg in np.unique(groups):
        m = groups == gg
        if labels[m].sum() == 0:
            continue
        vals.append(average_precision_score(labels[m], scores[m]))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------- #
# Native honest bundle_combine                                          #
# --------------------------------------------------------------------- #


def _load_solution(path: Path) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
    labels, groups, usages = {}, {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("id"):
                continue
            vid = str(r["id"]).zfill(5)
            labels[vid] = int(r["target"])
            groups[vid] = int(r["group"])
            usages[vid] = r["Usage"]
    return labels, groups, usages


def _top_k_mean(x: np.ndarray, k: int = 6) -> float:
    kk = min(k, x.size)
    return float(np.sort(x)[-kk:].mean())


def _run_native_bundle_combine(cfg: Dict[str, Any], idea_dir: Path) -> Dict[str, Any]:
    """Honest, split-aware bundle_combine for nexar_collision.

    Tunes α for the ``last1_max + top6_mean`` mixture on the Public subset
    of ``solution.csv`` only, and reports pgmAP on Public, Private, ALL
    with that single Public-tuned α.
    """
    from orze.engine.aggregations import make_calibrator

    bundle_paths: List[str] = [str(p) for p in (cfg.get("bundle") or [])]
    if not bundle_paths:
        raise ValueError("bundle_combine requires cfg['bundle'] (list of .npz)")

    solution_csv = cfg.get("solution_csv")
    if not solution_csv:
        raise ValueError(
            "nexar_collision adapter requires cfg['solution_csv'] "
            "(set posthoc_defaults.solution_csv in orze.yaml)")
    solution_csv = Path(solution_csv)
    if not solution_csv.exists():
        raise FileNotFoundError(f"solution_csv not found: {solution_csv}")

    per = _avg_views(bundle_paths)
    labels_all, groups_all, usages_all = _load_solution(solution_csv)
    vids = sorted(set(per) & set(labels_all))
    if not vids:
        raise ValueError("no overlap between bundle and solution ids")

    y = np.array([labels_all[v] for v in vids], dtype=np.float64)
    g = np.array([groups_all[v] for v in vids], dtype=int)
    u = np.array([usages_all[v] for v in vids])
    public_mask = (u == "Public")
    private_mask = (u == "Private")

    # Deterministic aggregations → per-clip scores
    A = np.array([per[v][-1] for v in vids], dtype=np.float64)  # last1_max
    B = np.array([_top_k_mean(per[v], 6) for v in vids])         # top6_mean

    # Per-group rank-normalize (unsupervised; leak-free)
    rA = per_group_rank_normalize(A, g)
    rB = per_group_rank_normalize(B, g)

    # Honest α search on Public only, metric = pgmAP on Public
    def pgmAP_on(mask, scores):
        return _pgmAP(scores, y, g, mask)

    cal = make_calibrator("cv_mix_honest")
    cal.fit([rA, rB], y, fit_mask=public_mask, groups=g,
            metric=lambda s, yy, gg: _pgmAP(s, yy, gg, None))
    s = cal.apply([rA, rB])

    ap_public = pgmAP_on(public_mask, s)
    ap_private = pgmAP_on(private_mask, s)
    ap_all = _pgmAP(s, y, g, None)

    # Also report the deterministic baselines for visibility
    variants: Dict[str, Dict[str, float]] = {}
    for name, scores in [
        ("last", A),
        ("top6_mean", B),
        ("CVmix_honest_last+top6", s),
    ]:
        variants[name] = {
            "pgmAP_Public": _pgmAP(scores, y, g, public_mask),
            "pgmAP_Private": _pgmAP(scores, y, g, private_mask),
            "pgmAP_ALL": _pgmAP(scores, y, g, None),
        }

    metrics: Dict[str, Any] = {
        "honest": True,
        "protocol": "tune-public_report-private-all",
        "pgmAP_Public": ap_public,
        "pgmAP_Private": ap_private,
        "pgmAP_ALL_honest": ap_all,
        "pgmAP_ALL": ap_all,  # alias for legacy harvesters
        "alpha_tuned_on_public": cal.alpha,
        "best_variant": "CVmix_honest_last+top6",
        "variants": variants,
        "n_views": len(bundle_paths),
        "n_vids": len(vids),
        "_adapter": "nexar_collision",
        "_kind": cfg.get("kind", "bundle_combine"),
        "_bundle": bundle_paths,
    }
    idea_dir.mkdir(parents=True, exist_ok=True)
    (idea_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# --------------------------------------------------------------------- #
# Adapter entry                                                         #
# --------------------------------------------------------------------- #


@register_adapter("nexar_collision")
def run(idea_id: str, cfg: Dict[str, Any], idea_dir: Path) -> Dict[str, Any]:
    """Dispatch based on cfg['kind'].

    Supported kinds:
      * ``bundle_combine`` — honest split-aware mix over a TTA bundle.
      * ``agg_search``     — single-view degenerate bundle (size 1).

    Required cfg keys (merged from ``posthoc_defaults`` in orze.yaml):
      * ``solution_csv``   — labels + Public/Private/group assignment.
      * ``bundle``         — list of .npz paths (bundle_combine) or
                             ``base_npz`` for agg_search.
    """
    kind = cfg.get("kind", "bundle_combine")
    if bool(cfg.get("dry_run")):
        return cfg.get("canned_metrics", {"pgmAP_ALL": 0.0, "dry_run": True})

    if kind == "bundle_combine":
        return _run_native_bundle_combine(cfg, idea_dir)

    if kind == "agg_search":
        inner = dict(cfg)
        base_npz = inner.get("base_npz") or (inner.get("bundle") or [None])[0]
        if not base_npz:
            raise ValueError("agg_search requires 'base_npz' or a 1-element 'bundle'")
        inner["bundle"] = [str(base_npz)]
        inner["kind"] = "bundle_combine"
        return _run_native_bundle_combine(inner, idea_dir)

    raise ValueError(
        f"nexar_collision adapter does not handle kind={kind!r}; "
        "supported: bundle_combine, agg_search")


__all__ = ["run", "_run_native_bundle_combine"]
