"""Tests for the v3.7.0 honest-eval protocol.

* split-aware bundle_combiner.search reports each split separately
* cv_mix_honest never sees labels outside fit_mask
* search raises when val_mask and any report_mask are identical (leak)
* nexar_collision adapter produces pgmAP_Public/Private/ALL_honest keys
* champion_guard blocks promotion when metrics.json.honest is False
"""

from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path

from orze.engine.aggregations import (
    make_calibrator,
    per_group_rank_normalize,
)
from orze.engine.bundle_combiner import InferenceBundle, search


def _synthetic_bundle(n_vids=60, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n_vids).astype(float)
    groups = rng.integers(0, 3, n_vids)
    probs_views = []
    for v in range(3):
        view = []
        for y in labels:
            p = rng.uniform(0, 0.4, 6)
            if y > 0.5:
                p[-1] = rng.uniform(0.6, 1.0)
            p = np.clip(p + rng.normal(0, 0.05, 6), 0, 1)
            view.append(p.tolist())
        probs_views.append(view)
    bundle = InferenceBundle(
        ckpt_sha="SHA_TEST",
        view_paths=[f"/tmp/v{v}.npz" for v in range(3)],
        probs=probs_views,
    )
    return bundle, labels, groups


def test_cv_mix_honest_fit_respects_fit_mask():
    rng = np.random.default_rng(0)
    n = 40
    a = rng.uniform(0, 1, n); b = rng.uniform(0, 1, n)
    y_fit = rng.integers(0, 2, n).astype(float)
    fit_mask = np.zeros(n, dtype=bool); fit_mask[:20] = True
    y_corrupted = y_fit.copy()
    y_corrupted[20:] = 1 - y_corrupted[20:]  # flip labels outside fit_mask

    cal1 = make_calibrator("cv_mix_honest")
    cal1.fit([a, b], y_fit, fit_mask=fit_mask)
    cal2 = make_calibrator("cv_mix_honest")
    cal2.fit([a, b], y_corrupted, fit_mask=fit_mask)

    # Corrupting labels outside the fit_mask must NOT change alpha.
    assert cal1.alpha == cal2.alpha


def test_cv_mix_honest_empty_mask_raises():
    cal = make_calibrator("cv_mix_honest")
    with pytest.raises(ValueError, match="fit_mask is empty"):
        cal.fit([np.zeros(4), np.zeros(4)], np.zeros(4),
                fit_mask=np.zeros(4, dtype=bool))


def test_search_splits_reports_three():
    bundle, labels, groups = _synthetic_bundle(seed=1)
    n = len(labels)
    val = np.zeros(n, dtype=bool); val[: n // 2] = True
    report_private = ~val
    out = search(
        bundle, labels,
        aggregations=["last", "top_k_mean"],
        calibrators=["identity"],
        k_range=[1, 2],
        splits={
            "val": val,
            "report": {
                "all": None,
                "public": val,
                "private": report_private,
            },
        },
    )
    assert out["honest"] is True
    b = out["best"]
    assert "reports" in b
    assert set(b["reports"]) == {"all", "public", "private"}


def test_search_splits_leak_detection():
    bundle, labels, _ = _synthetic_bundle(seed=2)
    n = len(labels)
    val = np.zeros(n, dtype=bool); val[: n // 2] = True
    # every report mask identical to val → leak
    with pytest.raises(ValueError, match="identical"):
        search(
            bundle, labels,
            aggregations=["last"], calibrators=["identity"], k_range=[1],
            splits={"val": val, "report": {"same": val, "same2": val}},
        )


def test_search_without_splits_is_flagged_non_honest():
    bundle, labels, _ = _synthetic_bundle(seed=3)
    out = search(bundle, labels, aggregations=["last"],
                 calibrators=["identity"], k_range=[1])
    assert out["honest"] is False


def test_per_group_rank_normalize_range():
    x = np.array([5, 1, 7, 3, 9, 2], dtype=float)
    g = np.array([0, 0, 0, 1, 1, 1])
    r = per_group_rank_normalize(x, g)
    # ranks are strictly within (0, 1]
    assert r.min() > 0 and r.max() <= 1.0


def _write_npz(path: Path, vids, fcs_per_vid, probs_per_vid):
    video_ids, frame_centers, probs = [], [], []
    for v, fcs, ps in zip(vids, fcs_per_vid, probs_per_vid):
        for fc, p in zip(fcs, ps):
            video_ids.append(v)
            frame_centers.append(fc)
            probs.append(p)
    np.savez(path,
             video_ids=np.array(video_ids),
             frame_centers=np.array(frame_centers),
             probs=np.array(probs, dtype=np.float64))


def test_nexar_adapter_produces_honest_metrics(tmp_path):
    from orze.engine.posthoc_adapters.nexar_collision import _run_native_bundle_combine

    n_vids = 40
    rng = np.random.default_rng(7)
    vids = [f"{i:05d}" for i in range(n_vids)]
    labels = rng.integers(0, 2, n_vids)
    groups = np.tile([0, 1, 2], (n_vids // 3 + 1))[:n_vids]
    usages = np.array(["Public" if i % 2 == 0 else "Private" for i in range(n_vids)])

    # solution.csv
    sol = tmp_path / "solution.csv"
    with open(sol, "w") as f:
        f.write("id,target,Usage,group\n")
        for i, v in enumerate(vids):
            f.write(f"{v},{labels[i]},{usages[i]},{groups[i]}\n")

    # 3 NPZ views (signal in last frame when label=1)
    bundle_paths = []
    for view_idx in range(3):
        p = tmp_path / f"view_{view_idx}.npz"
        fcs = [list(range(8)) for _ in range(n_vids)]
        probs = []
        for y in labels:
            base = rng.uniform(0, 0.3, 8)
            if y > 0.5:
                base[-1] = rng.uniform(0.6, 1.0)
            probs.append(np.clip(base + rng.normal(0, 0.05, 8), 0, 1).tolist())
        _write_npz(p, vids, fcs, probs)
        bundle_paths.append(str(p))

    idea_dir = tmp_path / "idea"
    metrics = _run_native_bundle_combine(
        {"bundle": bundle_paths, "solution_csv": str(sol)},
        idea_dir,
    )
    assert metrics["honest"] is True
    for k in ("pgmAP_Public", "pgmAP_Private", "pgmAP_ALL_honest",
              "alpha_tuned_on_public"):
        assert k in metrics
    assert (idea_dir / "metrics.json").exists()


def test_champion_guard_blocks_dishonest_metrics(tmp_path):
    from orze.engine.champion_guard import check_promotion
    idea_id = "idea-x"
    idir = tmp_path / idea_id
    idir.mkdir()
    (idir / "metrics.json").write_text(json.dumps(
        {"pgmAP_ALL": 0.99, "honest": False}), encoding="utf-8")
    # seed enough history so other paths are exercised cleanly
    allowed, info = check_promotion(tmp_path, idea_id, 0.99, cfg={})
    assert allowed is False
    assert info["blocked"] is True
    assert info.get("honest") is False


def test_champion_guard_allows_honest_metrics(tmp_path):
    from orze.engine.champion_guard import check_promotion
    idea_id = "idea-ok"
    idir = tmp_path / idea_id
    idir.mkdir()
    (idir / "metrics.json").write_text(json.dumps(
        {"pgmAP_ALL": 0.85, "honest": True}), encoding="utf-8")
    allowed, info = check_promotion(tmp_path, idea_id, 0.85, cfg={})
    assert allowed is True
    assert info.get("honest") is True
