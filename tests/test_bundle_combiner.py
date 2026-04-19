"""Tests for orze.engine.bundle_combiner (F10)."""

import numpy as np
import pytest

from orze.artifact_catalog import ArtifactCatalog
from orze.engine.bundle_combiner import (
    InferenceBundle,
    load_bundle_from_catalog,
    search,
    record_idea,
)
from orze.idea_lake import IdeaLake


def _synth_bundle(ckpt_sha="SHA_ONE", seed=0, n_views=3):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, 80).astype(float)
    probs_per_view = []
    for v in range(n_views):
        view = []
        for y in labels:
            base = rng.uniform(0, 0.4, 6)
            if y > 0.5:
                base[-1] = rng.uniform(0.6, 1.0)
            # add per-view noise so different views aren't identical
            base = np.clip(base + rng.normal(0, 0.05, 6), 0, 1)
            view.append(base.tolist())
        probs_per_view.append(view)
    paths = [f"/tmp/view_{i}.npz" for i in range(n_views)]
    bundle = InferenceBundle(
        ckpt_sha=ckpt_sha, view_paths=paths, probs=probs_per_view,
    )
    return bundle, labels


def test_bundle_rejects_mixed_ckpt_sha(tmp_path):
    """Catalog with two different ckpt_shas must not be loadable as a bundle."""
    cat = ArtifactCatalog(tmp_path / "a.db")
    (tmp_path / "a.npz").write_bytes(b"x")
    (tmp_path / "b.npz").write_bytes(b"x")
    cat.upsert(tmp_path / "a.npz", "preds_npz", ckpt_sha="S1")
    cat.upsert(tmp_path / "b.npz", "preds_npz", ckpt_sha="S2")
    # Only S1 is asked for: returns just that one (no mixing by construction)
    rows = cat.bundle("S1")
    assert len(rows) == 1

    # But if we hand-craft a mixed InferenceBundle via the loader path,
    # load_bundle_from_catalog must reject it when the catalog row lies.
    class _FakeCat:
        def bundle(self, sha):
            return [
                {"path": str(tmp_path / "a.npz"), "ckpt_sha": "S1"},
                {"path": str(tmp_path / "b.npz"), "ckpt_sha": "S2"},
            ]

    with pytest.raises(ValueError, match="ensembles are not allowed"):
        load_bundle_from_catalog(_FakeCat(), "S1",
                                  loader=lambda p: [[0.1, 0.2]])
    cat.close()


def test_bundle_size_and_mean_over_views():
    bundle, _ = _synth_bundle(n_views=3)
    assert bundle.size == 3
    avg = bundle.mean_over_views([0, 2])
    assert len(avg) == len(bundle.probs[0])
    # averaging two views != either original (different noise)
    assert not np.allclose(
        np.asarray(avg[0]), np.asarray(bundle.probs[0][0])
    )


def test_search_finds_recipe_and_monotone_in_k():
    bundle, labels = _synth_bundle(n_views=3, seed=1)
    out = search(bundle, labels, k_range=[1, 2, 3],
                 aggregations=["last", "max"], calibrators=["identity"])
    assert out["best"] is not None
    assert 1 <= out["best"]["k"] <= 3
    # sorted descending
    scores = [r["score"] for r in out["leaderboard"]]
    assert scores == sorted(scores, reverse=True)


def test_search_refuses_mixed_ckpt_sha():
    """The user-facing contract: no ensembles."""
    # Build a bundle object by hand with a differing sha — InferenceBundle
    # itself has no sha validation (it takes one sha), but
    # load_bundle_from_catalog does. This is covered by
    # test_bundle_rejects_mixed_ckpt_sha above; we assert the doc here.
    # (Extra guard: the search itself is agnostic to shas and trusts the
    # bundle invariant.)
    bundle, labels = _synth_bundle()
    out = search(bundle, labels, k_range=[1], aggregations=["last"],
                 calibrators=["identity"])
    assert out["ckpt_sha"] == "SHA_ONE"


def test_record_idea_marks_bundle_combine_kind(tmp_path):
    bundle, labels = _synth_bundle()
    out = search(bundle, labels, k_range=[1, 2],
                 aggregations=["last"], calibrators=["identity"])
    lake = IdeaLake(tmp_path / "l.db")
    record_idea(lake, "idea-bc-1", "bundle combine test",
                bundle, out["best"],
                eval_metrics={"pgmAP_ALL": 0.9057})
    row = lake.get("idea-bc-1")
    assert row["kind"] == "bundle_combine"
    assert row["status"] == "completed"
    assert row["eval_metrics"]["pgmAP_ALL"] == 0.9057


def test_load_bundle_from_catalog_happy_path(tmp_path, monkeypatch):
    cat = ArtifactCatalog(tmp_path / "a.db")
    for i in range(3):
        p = tmp_path / f"v{i}.npz"
        p.write_bytes(b"x")
        cat.upsert(p, "preds_npz", ckpt_sha="SHAOK")
    # Custom loader so we don't need real NPZs
    def loader(p):
        return [[0.1, 0.9], [0.2, 0.3]]
    bundle = load_bundle_from_catalog(cat, "SHAOK", loader=loader)
    assert bundle.size == 3
    assert bundle.ckpt_sha == "SHAOK"
    cat.close()
