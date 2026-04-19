"""Tests for F16 retroactive champion ingest."""

import json
from pathlib import Path

from orze.agents.ingest_champion import ingest
from orze.artifact_catalog import ArtifactCatalog
from orze.idea_lake import IdeaLake


def _synth_champion_cfg():
    return {
        "checkpoint": "results/model/best_model.pt",
        "predictions_files": [
            "results/clip_preds_A.npz",
            "results/clip_preds_tta_hflip.npz",
        ],
        "pgmAP_ALL": 0.9057,
        "pgmAP_Public": 0.9199,
        "pgmAP_Private": 0.8929,
        "per_group": {"G0": 0.9165, "G1": 0.9115, "G2": 0.8893},
        "reproducer": "scripts/eval_champion_0905_final.py",
        "notes": "4-view TTA bundle",
    }


def test_ingest_inserts_bundle_combine_idea(tmp_path):
    project_root = tmp_path
    results_dir = project_root / "results"
    results_dir.mkdir()
    (project_root / "results" / "model").mkdir()
    (project_root / "results" / "model" / "best_model.pt").write_bytes(b"x" * 128)
    for n in ("clip_preds_A.npz", "clip_preds_tta_hflip.npz"):
        (results_dir / n).write_bytes(b"y" * 64)

    info = ingest(results_dir, idea_id="idea-champion-test",
                  champion_cfg=_synth_champion_cfg(),
                  project_root=project_root)
    assert info["pgmAP_ALL"] == 0.9057
    assert info["npz_registered"] == 2

    lake = IdeaLake(results_dir / "idea_lake.db")
    row = lake.get("idea-champion-test")
    assert row is not None
    assert row["kind"] == "bundle_combine"
    assert row["status"] == "completed"
    assert row["eval_metrics"]["pgmAP_ALL"] == 0.9057
    assert row["eval_metrics"]["group"]["G0"] == 0.9165


def test_ingest_registers_artifacts_in_catalog(tmp_path):
    project_root = tmp_path
    results_dir = project_root / "results"
    results_dir.mkdir()
    (project_root / "results" / "model").mkdir()
    (project_root / "results" / "model" / "best_model.pt").write_bytes(b"x" * 128)
    for n in ("clip_preds_A.npz", "clip_preds_tta_hflip.npz"):
        (results_dir / n).write_bytes(b"y" * 64)

    info = ingest(results_dir, idea_id="idea-c2",
                  champion_cfg=_synth_champion_cfg(),
                  project_root=project_root)

    cat = ArtifactCatalog(results_dir / "idea_lake_artifacts.db")
    bundle = cat.by_ckpt_sha(info["ckpt_sha"])
    assert len(bundle) >= 2  # 1 ckpt + 2 preds
    names = [Path(r["path"]).name for r in bundle]
    assert "clip_preds_tta_hflip.npz" in names
    cat.close()


def test_ingest_updates_orze_state_best_idea(tmp_path):
    project_root = tmp_path
    results_dir = project_root / "results"
    results_dir.mkdir()
    (project_root / "results" / "model").mkdir()
    (project_root / "results" / "model" / "best_model.pt").write_bytes(b"x")
    (results_dir / "clip_preds_A.npz").write_bytes(b"y")
    (results_dir / "clip_preds_tta_hflip.npz").write_bytes(b"y")

    # Pre-existing state with a different champion
    (results_dir / "_orze_state.json").write_text(
        json.dumps({"best_idea_id": "idea-old"})
    )
    ingest(results_dir, idea_id="idea-champ",
           champion_cfg=_synth_champion_cfg(),
           project_root=project_root)

    state = json.loads((results_dir / "_orze_state.json").read_text())
    assert state["best_idea_id"] == "idea-champ"
    assert state["previous_best_idea_id"] == "idea-old"


def test_ingest_idempotent(tmp_path):
    project_root = tmp_path
    results_dir = project_root / "results"
    results_dir.mkdir()
    (project_root / "results" / "model").mkdir()
    (project_root / "results" / "model" / "best_model.pt").write_bytes(b"x")
    for n in ("clip_preds_A.npz", "clip_preds_tta_hflip.npz"):
        (results_dir / n).write_bytes(b"y")

    ingest(results_dir, idea_id="idea-c3", champion_cfg=_synth_champion_cfg(),
           project_root=project_root)
    ingest(results_dir, idea_id="idea-c3", champion_cfg=_synth_champion_cfg(),
           project_root=project_root)

    lake = IdeaLake(results_dir / "idea_lake.db")
    rows = list(lake.conn.execute(
        "SELECT idea_id FROM ideas WHERE idea_id='idea-c3'"))
    assert len(rows) == 1  # single row, upserted


def test_ingest_from_config_file(tmp_path):
    project_root = tmp_path
    results_dir = project_root / "results"
    results_dir.mkdir()
    (project_root / "results" / "model").mkdir()
    (project_root / "results" / "model" / "best_model.pt").write_bytes(b"x")
    (results_dir / "clip_preds_A.npz").write_bytes(b"y")
    (results_dir / "clip_preds_tta_hflip.npz").write_bytes(b"y")

    (results_dir / "_champion_config.json").write_text(json.dumps({
        "primary_champion": _synth_champion_cfg(),
    }))
    info = ingest(results_dir, idea_id="idea-c4",
                  project_root=project_root)
    assert info["pgmAP_ALL"] == 0.9057
