"""Tests for orze.engine.posthoc_runner (F12)."""

import json
from pathlib import Path

import pytest

from orze.engine.posthoc_runner import (
    get_adapter,
    list_adapters,
    register_adapter,
    run_posthoc,
    subprocess_adapter,
)


def test_null_adapter_returns_canned_metrics(tmp_path):
    cfg = {"canned_metrics": {"pgmAP_ALL": 0.9057, "foo": "bar"},
           "kind": "bundle_combine"}
    metrics = run_posthoc("idea-x", cfg, tmp_path / "idea-x")
    assert metrics["pgmAP_ALL"] == 0.9057
    assert metrics["_source"].startswith("posthoc_runner:")
    # metrics.json written to idea_dir
    assert (tmp_path / "idea-x" / "metrics.json").exists()


def test_run_posthoc_registers_outputs_in_catalog(tmp_path):
    from orze.artifact_catalog import ArtifactCatalog

    idea_dir = tmp_path / "idea-y"
    idea_dir.mkdir()
    # Pretend adapter wrote a prediction file
    npz = idea_dir / "clip_preds_idea-y.npz"
    npz.write_bytes(b"\x00" * 64)

    db = tmp_path / "artifacts.db"
    cfg = {"canned_metrics": {"pgmAP_ALL": 0.88}, "kind": "posthoc_eval",
           "ckpt_sha": "SHA_FAKE", "ops": {"aggregation": "last"}}
    run_posthoc("idea-y", cfg, idea_dir, artifact_catalog_db=db)
    cat = ArtifactCatalog(db)
    rows = cat.by_ckpt_sha("SHA_FAKE")
    assert any(r["path"].endswith("clip_preds_idea-y.npz") for r in rows)
    row = [r for r in rows if r["path"].endswith("clip_preds_idea-y.npz")][0]
    assert row["idea_id"] == "idea-y"
    assert row["metric_val"] == 0.88
    cat.close()


def test_custom_adapter_registration(tmp_path):
    @register_adapter("stub_test_adapter")
    def _stub(idea_id, cfg, idea_dir):
        return {"stub_score": 0.42, "saw": cfg.get("marker")}

    assert "stub_test_adapter" in list_adapters()
    metrics = run_posthoc(
        "idea-stub",
        {"adapter": "stub_test_adapter", "marker": "hi", "kind": "posthoc_eval"},
        tmp_path / "idea-stub",
    )
    assert metrics["stub_score"] == 0.42
    assert metrics["saw"] == "hi"


def test_get_adapter_raises_on_unknown():
    with pytest.raises(KeyError):
        get_adapter("definitely_not_a_real_adapter")


def test_subprocess_adapter_runs_ok(tmp_path):
    import sys
    # Emits metrics as JSON on stdout
    code = "import json; print(json.dumps({'ok': True, 'v': 1}))"
    out = subprocess_adapter([sys.executable, "-c", code])
    assert out == {"ok": True, "v": 1}


def test_launcher_dispatches_posthoc_for_nontrain_kind(tmp_path):
    """Smoke test of launcher.launch() for kind=posthoc_eval."""
    import yaml
    from orze.engine import launcher

    idea_id = "idea-pt"
    idea_dir = tmp_path / idea_id
    idea_dir.mkdir()
    # Write idea_config.yaml marking this as posthoc with the null adapter.
    (idea_dir / "idea_config.yaml").write_text(yaml.safe_dump({
        "kind": "posthoc_eval",
        "adapter": "null",
        "canned_metrics": {"pgmAP_ALL": 0.77},
    }))

    cfg = {"posthoc_adapter": "null",
           "idea_lake_db": str(tmp_path / "empty.db")}
    tp = launcher.launch(idea_id, gpu=-1, results_dir=tmp_path, cfg=cfg)
    # Wait for subprocess to finish (posthoc is fast)
    rc = tp.process.wait(timeout=30)
    assert rc == 0, (tp.log_path.read_text() if tp.log_path.exists() else "")
    metrics = json.loads((idea_dir / "metrics.json").read_text())
    assert metrics["pgmAP_ALL"] == 0.77
