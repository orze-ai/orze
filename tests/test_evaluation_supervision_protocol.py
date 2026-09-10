"""C2a publisher mechanisms with real blocked/closed CPU workers.

The imported fixture retains real B1/B2 SQLite and filesystem authority.
Only training/GPU boundaries are simulated; the supervisor and evaluator
workers are real. These are new protocol requirements, not old API reds.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import AttemptRef, current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.core.research_observations import observations_for_attempt
from orze.engine import evaluator, native_evaluation, observation_publication
from orze.engine.accounting import record_compute_start
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.supervised_process import prepare_supervised
from test_native_eval_tree_completion import (
    cpu_project, b2_project, artifact_project, native_case, _launch,
)


def _publication_state(c, ep):
    return {
        "attempt": current_attempt(c.lake.conn, c.idea, "evaluation"),
        "artifacts": artifacts_for_attempt(c.lake.conn, ep.attempt_ref),
        "observations": observations_for_attempt(c.lake.conn, ep.attempt_ref),
        "files": {str(path.relative_to(c.folder)): path.read_bytes()
                  for path in c.folder.rglob("*") if path.is_file()},
    }


def _write_binding(c, attempt_id, binding):
    raw = json.dumps(binding, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False)
    assert c.lake.conn.execute(
        "UPDATE main.execution_attempts SET binding_json=? WHERE attempt_id=? COLLATE BINARY",
        (raw, attempt_id)).rowcount == 1
    c.lake.conn.commit()


def test_direct_observation_publisher_cannot_replace_real_worker_seven_with_zero(cpu_project, tmp_path):
    c = cpu_project
    ep, _ = _launch(c, tmp_path, detached=False, code=7)
    assert ep.process.closure_receipt()["worker_returncode"] == 7
    before = _publication_state(c, ep)

    with pytest.raises(AttemptEffectBusy):
        observation_publication.finish_evaluation(c.lake, ep, c.folder, c.cfg, 0)

    assert _publication_state(c, ep) == before
    assert observation_publication.finish_evaluation(c.lake, ep, c.folder, c.cfg, 7) == (c.idea, 0)
    row = current_attempt(c.lake.conn, c.idea, "evaluation")
    assert row["state"] == "TERMINAL" and row["terminal"]["outcome"] == "failed"
    assert row["terminal"]["return_code"] == 7
    assert row["terminal"]["process_tree"] == ep.process.closure_receipt()
    assert observations_for_attempt(c.lake.conn, ep.attempt_ref) == []
    compute = c.folder / "_compute_receipts" / ep.attempt_id
    start = json.loads((compute / "start.json").read_bytes())
    terminal = json.loads((compute / "terminal.json").read_bytes())
    assert start["process_pid"] == terminal["process_pid"] == ep.process.pid
    assert terminal["return_code"] == 7


def test_closed_worker_does_not_authorize_changed_persistent_supervision_nonce(cpu_project, tmp_path):
    c = cpu_project
    ep, _ = _launch(c, tmp_path, detached=False, code=0)
    row = current_attempt(c.lake.conn, c.idea, "evaluation")
    binding = row["binding"]
    original = binding["supervision"]["nonce_sha256"]
    binding["supervision"]["nonce_sha256"] = ("a" if original != "a" * 64 else "b") * 64
    _write_binding(c, ep.attempt_id, binding)
    before = _publication_state(c, ep)

    with pytest.raises(AttemptEffectBusy):
        native_evaluation.finish(c.lake, ep, c.folder, c.cfg, 0)

    assert _publication_state(c, ep) == before
    assert before["attempt"]["state"] == "RUNNING"
    assert not before["artifacts"] and not before["observations"]


def test_ready_worker_start_fsync_failure_closes_without_executing_user_code(cpu_project, tmp_path, monkeypatch):
    c = cpu_project
    marker = tmp_path / "user-executed"
    Path(c.cfg["eval_script"]).write_text(
        "from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text('executed')\n",
        encoding="utf-8")
    c.cfg["eval_args"] = [str(marker)]
    real_prepare, real_fsync = evaluator.prepare_supervised, os.fsync
    handles, failures = [], []

    def prepare(*args, **kwargs):
        process = real_prepare(*args, **kwargs)
        handles.append(process)
        assert process.poll() is None and not marker.exists()
        return process

    def fail_first_start(fd):
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if (not failures and path.name == "start.json"
                and "_compute_receipts" in path.parts and str(c.folder) in str(path)):
            failures.append(str(path))
            raise OSError("selected start receipt fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(evaluator, "prepare_supervised", prepare)
    monkeypatch.setattr(os, "fsync", fail_first_start)
    assert evaluator.launch_eval(c.idea, 0, c.results, c.cfg, lake=c.lake,
                                 source_event=c.source_event) is None

    assert len(handles) == len(failures) == 1
    process = handles[0]
    closure = process.closure_receipt()
    assert closure["stop_requested"] is True and closure["wait_proof"] == "ECHILD_WALL"
    assert process.poll() == closure["worker_returncode"]
    assert not marker.exists(), "READY must precede any user-program execution"
    row = current_attempt(c.lake.conn, c.idea, "evaluation")
    assert row["state"] == "TERMINAL" and row["terminal"]["outcome"] == "failed"
    assert row["terminal"]["reason_code"] == "evaluation_launch_initialization_failed"
    assert row["terminal"]["process_tree"] == closure
    ref = AttemptRef(c.idea, "evaluation", row["attempt_id"], row["generation"])
    assert observations_for_attempt(c.lake.conn, ref) == []
    assert not list(c.folder.glob("_evaluation_attempts/*/work/measurement.json"))
    terminal = json.loads((c.folder / "_compute_receipts" / row["attempt_id"] / "terminal.json").read_bytes())
    assert terminal["process_pid"] == process.pid != process.supervisor_pid


def test_started_cannot_enroll_historical_launching_without_supervision_protocol(cpu_project, tmp_path):
    c = cpu_project
    ref = native_evaluation.begin(c.lake, c.folder, "historical-launching", 0,
                                  source_event=c.source_event)
    row = current_attempt(c.lake.conn, c.idea, "evaluation")
    binding = row["binding"]
    binding.pop("process_supervision_protocol")
    _write_binding(c, ref.attempt_id, binding)
    marker = tmp_path / "historical-user-executed"
    process = prepare_supervised(
        [sys.executable, "-c", "from pathlib import Path; Path(" + repr(str(marker)) + ").touch()"],
        identity={"attempt_ref": asdict(ref), "scope": str(c.folder.absolute())},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ep = SimpleNamespace(idea_id=c.idea, gpu=0, attempt_id=ref.attempt_id,
                         attempt_ref=ref, process=process, start_time=time.time())
    before = _publication_state(c, ep)
    try:
        with pytest.raises(AttemptEffectBusy):
            native_evaluation.started(c.lake, ep, c.folder, record_compute_start)
        assert _publication_state(c, ep) == before
        assert not (c.folder / "_compute_receipts" / ref.attempt_id / "start.json").exists()
        assert not marker.exists()
    finally:
        assert process.stop(timeout=3) is True
