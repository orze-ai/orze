"""New native-source gates and actual CPU READY/GO consumer integration.

The training script/adapter only writes a temporary marker and sleeps. GPU
allocation and telemetry are explicitly replaced by the inherited CPU fixture;
claims, resolver execution, supervision, SQLite and receipt publication are real.
No scientific results, models, providers or accelerator workloads are involved.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import launcher, native_posthoc, native_pre_script
from orze.engine import posthoc_attempts, process, training_attempts
from orze.engine.artifact_preflight_receipts import capture_preflight_source
from orze.engine.supervised_process import SupervisionUncertain, prepare_supervised
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_posthoc_tree_completion import cpu_posthoc, cpu_training, native_case
from test_posthoc_ready_go_boundary import _adapter


def _resolver(c, tmp_path, monkeypatch):
    c.cfg.pop("artifact_contract", None)
    c.cfg["idea_lake_db"] = str(c.lake.db_path)
    resolver = tmp_path / "resolver.py"
    resolver.write_text("print('resolved synthetic input')\n", encoding="utf-8")
    c.cfg["artifact_preflight"] = {
        "enabled": True, "script": str(resolver), "network": "inherit", "timeout": 5,
    }
    monkeypatch.setattr(process, "prepare_supervised", prepare_supervised)
    return resolver


@pytest.mark.parametrize("phase", ["training", "posthoc"])
@pytest.mark.parametrize("change", ["unchanged", "resolver_script", "network_policy"])
def test_actual_consumer_go_requires_captured_preflight_source(
        request, tmp_path, monkeypatch, phase, change):
    c = request.getfixturevalue("cpu_training" if phase == "training" else "cpu_posthoc")
    resolver = _resolver(c, tmp_path, monkeypatch)
    if phase == "training":
        marker = tmp_path / "training-marker"
        Path(c.cfg["train_script"]).write_text(
            "from pathlib import Path\nimport time\n"
            + "Path(" + repr(str(marker)) + ").write_text('executed')\n"
            + "time.sleep(30)\n", encoding="utf-8")
        (c.folder / "idea_config.yaml").write_text("seed: 13\n", encoding="utf-8")
    else:
        marker = _adapter(c, tmp_path, pause=True)
    claim_before = (c.folder / "claim.json").read_bytes()
    result = process.run_artifact_preflight(c.idea, c.results, c.cfg, lake=c.lake)
    assert result
    source = capture_preflight_source(c.lake, c.folder, c.cfg).source
    assert source["attempt_ref"] == asdict(result.attempt_ref)
    assert source["claim_sha256"] == hashlib.sha256(claim_before).hexdigest()
    original_source = current_attempt(c.lake.conn, c.idea, "artifact_preflight")
    real_prepare = launcher.prepare_supervised
    handles = []

    def ready(*args, **kwargs):
        child = real_prepare(*args, **kwargs)
        handles.append(child)
        c.pidfds.append((child.pid, os.pidfd_open(child.pid)))
        assert not marker.exists()
        if change == "resolver_script":
            resolver.write_text("print('changed after READY')\n", encoding="utf-8")
        elif change == "network_policy":
            c.cfg["artifact_preflight"]["network"] = "offline"
        return child

    monkeypatch.setattr(launcher, "prepare_supervised", ready)
    try:
        if change == "unchanged":
            tp = launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
            c.handles.append(tp)
            import time
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.read_text() == "executed"
            assert tp.process is handles[0]
        else:
            with pytest.raises(TerminationUnconfirmed):
                launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
            assert not marker.exists()
            assert handles[0].closure_receipt()["stop_requested"] is True
        assert len(handles) == 1
        row = current_attempt(c.lake.conn, c.idea, phase)
        assert row["state"] == "RUNNING" and row["terminal"] is None
        assert row["binding"]["artifact_preflight_source"] == source
        assert row["binding"]["claim_sha256"] == source["claim_sha256"]
        assert row["attempt_id"] == source["claim_attempt_id"] != result.attempt_ref.attempt_id
        claim_after = (c.folder / "claim.json").read_bytes()
        assert claim_after != claim_before
        assert json.loads(claim_after)["trainer_pid"] == handles[0].pid
        assert c.lake.get_fsm_state(c.idea) == "IN_PROGRESS"
        assert current_attempt(c.lake.conn, c.idea, "artifact_preflight") == original_source
        assert len(list(c.folder.glob("_compute_receipts/*/start.json"))) == 1
        assert not list(c.folder.glob("_compute_receipts/*/terminal.json"))
        assert not (c.folder / "metrics.json").exists()
    finally:
        for child in handles:
            assert child.stop(timeout=3)
            assert child.closure_receipt()["wait_proof"] == "ECHILD_WALL"


@pytest.mark.parametrize("entry", [
    "disabled", "missing_lake", "erased_routes", "pre_script", "native_pre_script",
    "launch", "training_begin", "posthoc_launch", "posthoc_begin",
])
def test_pending_resolver_cannot_be_erased_or_bypassed(
        cpu_training, tmp_path, monkeypatch, entry):
    from types import SimpleNamespace
    c = cpu_training
    _resolver(c, tmp_path, monkeypatch)
    calls = []

    def uncertain(*args, **kwargs):
        calls.append(True)
        raise SupervisionUncertain("explicit test-only unknown preflight READY")

    monkeypatch.setattr(process, "prepare_supervised", uncertain)
    with pytest.raises(TerminationUnconfirmed):
        process.run_artifact_preflight(c.idea, c.results, c.cfg, lake=c.lake)
    source = current_attempt(c.lake.conn, c.idea, "artifact_preflight")
    assert source["state"] == "LAUNCHING"
    claim = json.loads((c.folder / "claim.json").read_bytes())
    handle = SimpleNamespace(idea_id=c.idea, gpu=0, process=None,
                             attempt_id=claim["attempt_id"])
    forbidden = []
    monkeypatch.setattr(launcher, "_assert_campaign_evidence_authorized",
                        lambda *a: forbidden.append("campaign"))
    c.cfg["artifact_preflight"]["enabled"] = False
    if entry == "erased_routes":
        (c.folder / "_execution_catalog.json").unlink()
        claim.pop("lifecycle_db")
        (c.folder / "claim.json").write_text(json.dumps(claim), encoding="utf-8")
    with pytest.raises(TerminationUnconfirmed):
        if entry == "disabled":
            process.run_artifact_preflight(c.idea, c.results, c.cfg, lake=c.lake)
        elif entry in ("missing_lake", "erased_routes"):
            process.run_artifact_preflight(c.idea, c.results, c.cfg)
        elif entry == "pre_script":
            process.run_pre_script(c.idea, 0, c.cfg, c.results, lake=c.lake)
        elif entry == "native_pre_script":
            native_pre_script.run_native_pre_script(
                c.idea, 0, c.results, c.cfg, c.lake, ["never-executed"], 5, {})
        elif entry == "launch":
            launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
        elif entry == "training_begin":
            training_attempts.begin(c.lake, handle, c.folder, cfg=c.cfg)
        elif entry == "posthoc_launch":
            native_posthoc.launch(c.idea, 0, c.results, c.cfg, kind="test",
                                  idea_cfg_path=c.folder / "idea_config.yaml", lake=c.lake)
        else:
            posthoc_attempts.begin(c.lake, handle, c.folder, launch_inputs={}, cfg=c.cfg)
    assert calls == [True] and forbidden == []
    assert current_attempt(c.lake.conn, c.idea, "artifact_preflight") == source
    assert current_attempt(c.lake.conn, c.idea, "training") is None
    assert current_attempt(c.lake.conn, c.idea, "posthoc") is None
    assert current_attempt(c.lake.conn, c.idea, "pre_script") is None
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
    assert not (c.folder / "artifact_preflight.json").exists()
    assert not (c.folder / "_compute_receipts").exists()
