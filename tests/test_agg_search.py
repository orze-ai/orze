"""Tests for orze.engine.agg_search (F11)."""

import numpy as np

from orze.engine.agg_search import sweep, record_idea
from orze.idea_lake import IdeaLake


def _synth(n=100, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n).astype(float)
    # per-clip probs: signal concentrated at the end of each clip
    probs_per_video = []
    for y in labels:
        nframes = rng.integers(3, 12)
        base = rng.uniform(0, 0.5, nframes)
        if y > 0.5:
            base[-1] = rng.uniform(0.6, 1.0)
        probs_per_video.append(base.tolist())
    return probs_per_video, labels


def test_sweep_returns_best_recipe():
    probs, labels = _synth()
    out = sweep(probs, labels, k_folds=3)
    assert out["best"] is not None
    assert "aggregation" in out["best"]
    assert "calibrator" in out["best"]
    # leaderboard is sorted descending
    scores = [r["score_mean"] for r in out["leaderboard"]]
    assert scores == sorted(scores, reverse=True)


def test_sweep_includes_cv_mix_candidate():
    probs, labels = _synth()
    out = sweep(probs, labels, k_folds=3)
    assert any(r["calibrator"] == "cv_mix" for r in out["leaderboard"])


def test_record_idea_marks_agg_search_kind(tmp_path):
    probs, labels = _synth(n=60)
    out = sweep(probs, labels, k_folds=3)
    lake = IdeaLake(tmp_path / "l.db")
    record_idea(lake, "idea-agg-1", "sweep test",
                base_npz="/tmp/preds.npz", winning=out["best"])
    row = lake.get("idea-agg-1")
    assert row["kind"] == "agg_search"
    assert row["status"] == "completed"


def test_sweep_restricted_to_given_subset():
    probs, labels = _synth()
    out = sweep(probs, labels, k_folds=3,
                aggregations=["last", "max"],
                calibrators=["identity"])
    # Plus cv_mix may be appended if 2 aggs → leaderboard should include
    # at least (last, identity) and (max, identity).
    pairs = {(r["aggregation"], r["calibrator"]) for r in out["leaderboard"]}
    assert ("last", "identity") in pairs
    assert ("max", "identity") in pairs
