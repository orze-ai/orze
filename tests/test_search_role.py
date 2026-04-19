"""Tests for the F15 tight-loop search role."""

from pathlib import Path

import pytest

from orze.agents.search_role import enumerate_work
from orze.artifact_catalog import ArtifactCatalog
from orze.idea_lake import IdeaLake


def _seed_catalog(tmp_path, paths_per_sha):
    cat_db = tmp_path / "art.db"
    cat = ArtifactCatalog(cat_db)
    for sha, paths in paths_per_sha.items():
        ckpt = tmp_path / f"best_{sha}.pt"
        ckpt.write_bytes(b"x")
        cat.upsert(ckpt, "ckpt", ckpt_sha=sha)
        for p in paths:
            pp = tmp_path / p
            pp.write_bytes(b"y")
            kind = "tta_preds" if "tta" in p else "preds_npz"
            cat.upsert(pp, kind, ckpt_sha=sha)
    cat.close()
    return cat_db


def test_proposes_agg_and_bundle_for_multi_view_sha(tmp_path):
    cat_db = _seed_catalog(tmp_path, {
        "SHA1": ["preds_a.npz", "tta_hflip.npz", "tta_fs3.npz"],
    })
    lake_db = tmp_path / "lake.db"
    out = enumerate_work(tmp_path, artifact_db=cat_db, lake_db=lake_db)

    kinds = [i["kind"] for i in out["proposed"]]
    assert "agg_search" in kinds
    assert "bundle_combine" in kinds
    # They all land in the lake as pending
    lake = IdeaLake(lake_db)
    for idea in out["proposed"]:
        row = lake.get(idea["idea_id"])
        assert row is not None
        assert row["status"] == "pending"
        assert row["kind"] == idea["kind"]


def test_single_view_sha_has_no_bundle_proposal(tmp_path):
    cat_db = _seed_catalog(tmp_path, {"SHAONLY": ["preds_only.npz"]})
    out = enumerate_work(tmp_path, artifact_db=cat_db,
                         lake_db=tmp_path / "lake.db")
    kinds = [i["kind"] for i in out["proposed"]]
    assert "agg_search" in kinds
    assert "bundle_combine" not in kinds


def test_idempotent_on_second_run(tmp_path):
    cat_db = _seed_catalog(tmp_path, {
        "SHA2": ["preds_a.npz", "tta_hflip.npz"],
    })
    lake_db = tmp_path / "lake.db"
    first = enumerate_work(tmp_path, artifact_db=cat_db, lake_db=lake_db)
    assert first["proposed"]
    second = enumerate_work(tmp_path, artifact_db=cat_db, lake_db=lake_db)
    assert second["proposed"] == []
    assert second["skipped_existing"] >= len(first["proposed"])


def test_max_new_ideas_cap(tmp_path):
    # 4 distinct ckpts → up to 8 candidates; cap at 3.
    paths_per_sha = {
        f"S{i}": [f"preds_{i}.npz", f"tta_{i}.npz"] for i in range(4)
    }
    cat_db = _seed_catalog(tmp_path, paths_per_sha)
    out = enumerate_work(tmp_path, artifact_db=cat_db,
                         lake_db=tmp_path / "l.db", max_new_ideas=3)
    assert len(out["proposed"]) <= 3


def test_cli_entrypoint(tmp_path, capsys):
    import json
    cat_db = _seed_catalog(tmp_path, {"SHA9": ["preds_x.npz"]})
    from orze.agents.search_role import main
    rc = main([
        "--results-dir", str(tmp_path),
        "--artifact-db", str(cat_db),
        "--lake-db", str(tmp_path / "l.db"),
    ])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "proposed" in data
