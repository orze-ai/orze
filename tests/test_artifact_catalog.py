"""Tests for orze.artifact_catalog (F9)."""

import os
import sqlite3
from pathlib import Path

import pytest

from orze.artifact_catalog import (
    ALLOWED_KINDS,
    ArtifactCatalog,
    hash_ckpt,
    hash_inference_config,
)


def _mkfile(p: Path, size: int = 1024, seed: bytes = b"x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(seed * size)
    return p


def test_schema_is_created(tmp_path):
    cat = ArtifactCatalog(tmp_path / "a.db")
    # table + indexes exist
    rows = {r[0] for r in cat.conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
    )}
    assert "artifacts" in rows
    assert "idx_artifacts_ckpt_sha" in rows
    assert "idx_artifacts_kind" in rows
    assert cat.conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    cat.close()


def test_upsert_rejects_unknown_kind(tmp_path):
    cat = ArtifactCatalog(tmp_path / "a.db")
    with pytest.raises(ValueError):
        cat.upsert("/tmp/x.pt", "not_a_real_kind")
    cat.close()


def test_upsert_idempotent_merge(tmp_path):
    cat = ArtifactCatalog(tmp_path / "a.db")
    p = _mkfile(tmp_path / "foo.pt")
    cat.upsert(p, "ckpt", ckpt_sha="abc", metric_val=0.8)
    cat.upsert(p, "ckpt", metric_test=0.9)  # merge, don't wipe metric_val
    row = cat.get(p)
    assert row["ckpt_sha"] == "abc"
    assert row["metric_val"] == 0.8
    assert row["metric_test"] == 0.9
    assert cat.count() == 1
    cat.close()


def test_upsert_inference_config_hashed_and_json(tmp_path):
    cat = ArtifactCatalog(tmp_path / "a.db")
    p = _mkfile(tmp_path / "preds.npz")
    cfg = {"tta": "hflip", "frame_stride": 4}
    cat.upsert(p, "preds_npz", inference_config=cfg)
    row = cat.get(p)
    assert row["inference_config"] == cfg
    assert row["inference_config_hash"] == hash_inference_config(cfg)
    cat.close()


def test_by_ckpt_sha_and_bundle_semantics(tmp_path):
    cat = ArtifactCatalog(tmp_path / "a.db")
    ckpt = _mkfile(tmp_path / "m/best_model.pt")
    npz1 = _mkfile(tmp_path / "m/preds_a.npz")
    npz2 = _mkfile(tmp_path / "m/preds_b.npz")
    other = _mkfile(tmp_path / "n/preds_c.npz")
    cat.upsert(ckpt, "ckpt", ckpt_sha="S1")
    cat.upsert(npz1, "preds_npz", ckpt_sha="S1")
    cat.upsert(npz2, "tta_preds", ckpt_sha="S1")
    cat.upsert(other, "preds_npz", ckpt_sha="S2")

    all_s1 = cat.by_ckpt_sha("S1")
    assert {r["path"] for r in all_s1} == {str(ckpt), str(npz1), str(npz2)}

    bundle = cat.bundle("S1")
    # bundle excludes the ckpt itself — only view artifacts
    assert {r["path"] for r in bundle} == {str(npz1), str(npz2)}

    assert cat.by_ckpt_sha("missing") == []
    cat.close()


def test_scan_discovers_ckpts_and_npzs(tmp_path):
    # Build a fake results/ tree
    run = tmp_path / "results" / "run1"
    _mkfile(run / "best_model.pt", size=64, seed=b"a")
    _mkfile(run / "clip_preds_dense_end.npz", size=32, seed=b"b")
    _mkfile(run / "clip_preds_tta_hflip.npz", size=32, seed=b"c")
    # cache files must be ignored
    _mkfile(run / "riskprop_cache.npz", size=16, seed=b"d")
    _mkfile(tmp_path / "results" / "run2" / "best_ema_model.pt", size=64, seed=b"e")

    cat = ArtifactCatalog(tmp_path / "a.db")
    counts = cat.scan(tmp_path / "results")
    assert counts["ckpt"] == 2
    assert counts["preds_npz"] == 1
    assert counts["tta_preds"] == 1
    assert counts["features"] == 0

    # ckpt_sha computed and inherited by sibling npz
    row_ckpt = cat.get(run / "best_model.pt")
    assert row_ckpt["ckpt_sha"]
    row_npz = cat.get(run / "clip_preds_dense_end.npz")
    assert row_npz["ckpt_sha"] == row_ckpt["ckpt_sha"]

    # idempotency: second scan doesn't double
    counts2 = cat.scan(tmp_path / "results")
    assert counts2["ckpt"] == 2
    assert cat.count() == 4  # 2 ckpts + 2 preds
    cat.close()


def test_hash_ckpt_stable_and_size_sensitive(tmp_path):
    p = _mkfile(tmp_path / "x.pt", size=1024, seed=b"A")
    h1 = hash_ckpt(p)
    h2 = hash_ckpt(p)
    assert h1 == h2
    # changing content → different hash
    p.write_bytes(b"B" * 1024)
    assert hash_ckpt(p) != h1


def test_allowed_kinds_contract():
    assert ALLOWED_KINDS == {"ckpt", "preds_npz", "features", "tta_preds"}


def test_delete(tmp_path):
    cat = ArtifactCatalog(tmp_path / "a.db")
    p = _mkfile(tmp_path / "z.pt")
    cat.upsert(p, "ckpt")
    assert cat.get(p) is not None
    cat.delete(p)
    assert cat.get(p) is None
    cat.close()
