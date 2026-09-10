"""Independent B2 native training-to-evaluation acceptance mechanisms.

Real launch, B1 snapshots, retry, lifecycle, result copies and SQLite records
are used. Only process/GPU and precise I/O fault boundaries are substituted.
These are newly declared mechanisms, not historical missing-API regressions.
"""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt, get_artifact
from orze.core.research_observations import observations_for_attempt
from orze.engine import evaluator, launcher
from orze.engine.evaluation_retry import request_evaluation_retry
from orze.engine.termination_hold import TerminationUnconfirmed
from test_artifact_snapshot_contract import (
    project as artifact_project, native_case, _launch as launch_training,
    _output as training_output,
)


def _envelope(observations):
    return json.dumps({"schema": 1, "observations": observations},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _observations():
    return [
        {"name": "zero", "values": {"quality": 0, "aux": -1.25},
         "validation": {"status": "valid", "reason_code": "adapter_valid"},
         "comparison_scope": "protocol-fixture-v1"},
        {"name": "negative", "values": {"quality": -3},
         "validation": {"status": "invalid", "reason_code": "adapter_invalid"},
         "comparison_scope": None},
    ]


@pytest.fixture
def project(artifact_project, monkeypatch):
    c = artifact_project
    eval_script = Path(c.cfg["_project_root"]) / "evaluate.py"
    eval_script.write_text("# External evaluator boundary; never executed.\n", encoding="utf-8")
    c.cfg.update({
        "eval_script": str(eval_script), "eval_output": "assessment.json",
        "eval_checkpoint": "best_model.pt", "eval_timeout": 60,
        "report": {"primary_metric": "quality", "sort": "ascending",
                   "columns": [{"key": "quality", "source": "assessment.json:quality"}]},
        "observation_contract": {
            "version": 1, "adapter": "orze.json_observations.v1",
            "protocol_id": "bounded-fixture-v1", "inputs": ["checkpoint"],
            "output": {"path": "measurement.json", "max_bytes": 65536},
        },
    })
    c.training, c.folder = launch_training(c)
    training_output(c.training, c.folder)
    events = launcher.check_active({0: c.training}, c.results, c.cfg, {}, lake=c.lake)
    assert len(events) == 1 and events[0].attempt_ref == c.training.attempt_ref
    c.source_event = events[0]
    c.training_records = artifacts_for_attempt(c.lake.conn, c.training.attempt_ref)
    c.input_records = [r for r in c.training_records if r["logical_name"] == "checkpoint"]
    assert len(c.input_records) == 1
    c.training_before = current_attempt(c.lake.conn, c.idea, "training")
    c.metrics_before = (c.folder / "metrics.json").read_bytes()
    c.checkpoint_before = (c.folder / "best_model.pt").read_bytes()
    c.eval_calls, c.eval_logs = [], []

    class Child:
        pid = None
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("evaluate.py", timeout)
            return self.returncode

    def popen(cmd, **kwargs):
        child = Child()
        c.eval_calls.append((deepcopy(cmd), deepcopy(kwargs.get("env", {})), child))
        c.eval_logs.append(kwargs["stdout"])
        return child

    def no_training(*args, **kwargs):
        raise AssertionError("Evaluation retry must not launch training")

    monkeypatch.setattr(evaluator.subprocess, "Popen", popen)
    monkeypatch.setattr(evaluator, "gpu_execution_lease", lambda *a, **k: nullcontext(()))
    monkeypatch.setattr(evaluator, "_verify_gpu_free", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "launch", no_training)
    try:
        yield c
    finally:
        for handle in c.eval_logs:
            handle.close()


def _launch(c):
    previous_calls = len(c.eval_calls)
    ep = evaluator.launch_eval(c.idea, 0, c.results, c.cfg, lake=c.lake,
                               source_event=c.source_event)
    assert ep is not None and len(c.eval_calls) == previous_calls + 1
    env = c.eval_calls[-1][1]
    root = c.folder / "_evaluation_attempts" / ep.attempt_id
    manifest_path = Path(env["ORZE_EVALUATION_INPUT_MANIFEST"])
    work = Path(env["ORZE_EVALUATION_OUTPUT_DIR"])
    output = Path(env["ORZE_EVALUATION_OUTPUT_PATH"])
    assert manifest_path == root / "input_manifest.json"
    assert work == root / "work"
    assert output == work / c.cfg["observation_contract"]["output"]["path"]
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["schema"] == 1
    assert manifest["task_id"] == c.idea and manifest["attempt_id"] == ep.attempt_id
    assert manifest["source_ref"] == asdict(c.training.attempt_ref)
    assert manifest["inputs"] == c.input_records
    assert manifest["output_path"] == str(output)
    for record in manifest["inputs"]:
        path = Path(record["path"])
        assert path.stat().st_mode & 0o222 == 0
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["content_sha256"]
    return ep, manifest_path, manifest, output


def _finish(c, ep, output, observations, code=0):
    output.write_bytes(_envelope(observations))
    ep.process.returncode = code
    active = {0: ep}
    events = evaluator.check_active_evals(active, c.results, c.cfg, lake=c.lake)
    assert events == [(c.idea, 0)] and events[0].attempt_ref == ep.attempt_ref
    assert active == {}
    return current_attempt(c.lake.conn, c.idea, "evaluation")


def _accepted(c, ep, manifest, output, authored):
    row = current_attempt(c.lake.conn, c.idea, "evaluation")
    assert row["state"] == "TERMINAL" and row["terminal"]["outcome"] == "completed"
    artifacts = artifacts_for_attempt(c.lake.conn, ep.attempt_ref)
    observations = observations_for_attempt(c.lake.conn, ep.attempt_ref)
    assert len(artifacts) == 1 and len(observations) == len(authored)
    assert set(row["terminal"]["artifact_ids"]) == {a["artifact_id"] for a in artifacts}
    assert set(row["terminal"]["observation_ids"]) == {o["observation_id"] for o in observations}
    result = artifacts[0]
    accepted_path = Path(result["path"])
    expected = _envelope(authored)
    assert accepted_path.read_bytes() == expected
    assert result["content_sha256"] == hashlib.sha256(expected).hexdigest()
    assert result["size_bytes"] == len(expected)
    assert accepted_path.stat().st_nlink == 1 and accepted_path.stat().st_mode & 0o222 == 0
    assert (accepted_path.stat().st_dev, accepted_path.stat().st_ino) != (
        output.stat().st_dev, output.stat().st_ino)
    assert get_artifact(c.lake.conn, result["artifact_id"]) == result
    by_name = {entry["name"]: entry for entry in authored}
    for observation in observations:
        source = by_name[observation["name"]]
        assert observation["evaluator"] == asdict(ep.attempt_ref)
        assert observation["scope"] == str(c.results)
        assert observation["protocol_fingerprint"] == manifest["protocol_fingerprint"]
        assert observation["adapter_id"] == "orze.json_observations.v1"
        assert observation["input_artifact_ids"] == [a["artifact_id"] for a in c.input_records]
        assert observation["result_artifact_ids"] == [result["artifact_id"]]
        assert observation["spec_fingerprint"] == c.input_records[0]["spec_fingerprint"]
        assert observation["values"] == source["values"]
        assert observation["validation"] == source["validation"]
        assert observation["comparison_scope"] == source["comparison_scope"]
    assert current_attempt(c.lake.conn, c.idea, "training") == c.training_before
    assert (c.folder / "metrics.json").read_bytes() == c.metrics_before
    assert (c.folder / "best_model.pt").read_bytes() == c.checkpoint_before
    return artifacts, observations


def test_failed_evaluation_retry_keeps_inputs_and_rejects_late_fd_and_callback(project):
    c = project
    old, old_manifest_path, old_manifest, old_output = _launch(c)
    old_output.write_bytes(_envelope(_observations()))
    old_fd = os.open(old_output, os.O_RDWR)
    new_fd = None
    try:
        old.process.returncode = 1
        assert evaluator.check_active_evals({0: old}, c.results, c.cfg, lake=c.lake) == [(c.idea, 0)]
        assert observations_for_attempt(c.lake.conn, old.attempt_ref) == []
        before_manifest = old_manifest_path.read_bytes()
        assert request_evaluation_retry(c.idea, c.results, c.cfg, c.lake)["status"] == "evaluation_retry_pending"
        current, manifest_path, manifest, output = _launch(c)
        assert current.attempt_id != old.attempt_id and output != old_output
        assert manifest["inputs"] == old_manifest["inputs"]
        assert manifest["source_ref"] == old_manifest["source_ref"]
        assert manifest["protocol_fingerprint"] == old_manifest["protocol_fingerprint"]
        assert old_manifest_path.read_bytes() == before_manifest
        new_manifest = manifest_path.read_bytes()
        input_bytes = {r["artifact_id"]: Path(r["path"]).read_bytes() for r in manifest["inputs"]}
        output.write_bytes(_envelope(_observations()))
        new_fd = os.open(output, os.O_RDWR)
        os.lseek(old_fd, 0, os.SEEK_SET)
        os.write(old_fd, b"late old evaluator output")
        os.ftruncate(old_fd, len(b"late old evaluator output"))
        os.fsync(old_fd)
        before = current_attempt(c.lake.conn, c.idea, "evaluation")
        assert evaluator.check_active_evals({0: old}, c.results, c.cfg, lake=c.lake) == []
        assert current_attempt(c.lake.conn, c.idea, "evaluation") == before
        assert output.read_bytes() == _envelope(_observations())
        assert manifest_path.read_bytes() == new_manifest
        assert {r["artifact_id"]: Path(r["path"]).read_bytes() for r in manifest["inputs"]} == input_bytes
        _finish(c, current, output, _observations())
        records, observations = _accepted(c, current, manifest, output, _observations())
        os.lseek(new_fd, 0, os.SEEK_SET)
        os.write(new_fd, b"late current evaluator output")
        os.ftruncate(new_fd, len(b"late current evaluator output"))
        os.fsync(new_fd)
        assert Path(records[0]["path"]).read_bytes() == _envelope(_observations())
        assert observations_for_attempt(c.lake.conn, current.attempt_ref) == observations
        assert evaluator.check_active_evals({0: current}, c.results, c.cfg, lake=c.lake) == []
        assert len(c.eval_calls) == 2
    finally:
        if new_fd is not None:
            os.close(new_fd)
        os.close(old_fd)


@pytest.mark.parametrize("authored", [[], _observations()], ids=["zero-observations", "multiple-zero-negative"])
def test_result_snapshot_preserves_zero_or_multiple_adapter_reported_observations(project, authored):
    c = project
    ep, _, manifest, output = _launch(c)
    _finish(c, ep, output, authored)
    _accepted(c, ep, manifest, output, authored)


@pytest.mark.parametrize("fault", ["terminal_sql", "compute_receipt_io"])
def test_terminal_fault_rolls_back_all_observations_and_result_registration(project, monkeypatch, fault):
    c = project
    ep, _, _, output = _launch(c)
    output.write_bytes(_envelope(_observations()))
    ep.process.returncode = 0
    hits = []
    if fault == "terminal_sql":
        c.lake.conn.execute(
            "CREATE TRIGGER ignore_eval_terminal BEFORE UPDATE ON execution_attempts "
            "WHEN NEW.phase='evaluation' AND NEW.state='TERMINAL' "
            "BEGIN SELECT RAISE(IGNORE); END")
        c.lake.conn.commit()
    else:
        original_write = os.write
        target = c.folder / "_compute_receipts" / ep.attempt_id / "terminal.json"

        def fail_receipt(fd, data):
            if Path(os.readlink(f"/proc/self/fd/{fd}")) == target:
                hits.append(True)
                raise OSError(errno.ENOSPC, "synthetic receipt capacity fault")
            return original_write(fd, data)

        monkeypatch.setattr(os, "write", fail_receipt)
    active = {0: ep}
    with pytest.raises(TerminationUnconfirmed):
        evaluator.check_active_evals(active, c.results, c.cfg, lake=c.lake)
    if fault == "compute_receipt_io":
        assert hits == [True]
    assert active.get(0) is ep
    assert current_attempt(c.lake.conn, c.idea, "evaluation")["state"] == "RUNNING"
    assert c.lake.get_stage_state(c.idea, "training") == "COMPLETE"
    assert c.lake.get_stage_state(c.idea, "evaluation") == "IN_PROGRESS"
    assert observations_for_attempt(c.lake.conn, ep.attempt_ref) == []
    assert artifacts_for_attempt(c.lake.conn, ep.attempt_ref) == []
    assert (c.folder / "_attempt_effect.lock").is_dir()
    assert not (c.folder / "_execution_effects" / ep.attempt_id / "committed.json").exists()
    assert current_attempt(c.lake.conn, c.idea, "training") == c.training_before
    assert (c.folder / "metrics.json").read_bytes() == c.metrics_before


def test_legacy_off_evaluation_does_not_infer_observations(project):
    c = project
    c.cfg["observation_contract"] = None
    ep = evaluator.launch_eval(c.idea, 0, c.results, c.cfg, lake=c.lake,
                               source_event=c.source_event)
    assert ep is not None
    env = c.eval_calls[-1][1]
    assert not any(name in env for name in (
        "ORZE_EVALUATION_INPUT_MANIFEST", "ORZE_EVALUATION_OUTPUT_DIR", "ORZE_EVALUATION_OUTPUT_PATH"))
    (c.folder / "assessment.json").write_bytes(b'{"status":"COMPLETED","quality":0}')
    ep.process.returncode = 0
    assert evaluator.check_active_evals({0: ep}, c.results, c.cfg, lake=c.lake) == [(c.idea, 0)]
    assert observations_for_attempt(c.lake.conn, ep.attempt_ref) == []
    assert artifacts_for_attempt(c.lake.conn, ep.attempt_ref) == []
    row = current_attempt(c.lake.conn, c.idea, "evaluation")
    assert "observation_ids" not in row["terminal"]
    assert not (c.folder / "_evaluation_attempts").exists()
    assert c.lake.conn.execute(
        "SELECT name FROM sqlite_master WHERE name='research_observations'").fetchall() == []
