"""New durable-stop mechanism through real recovery/launch consumers.

No old-API red claim: stop evidence is produced by terminate_execution itself.
Files, lifecycle, accounting and retry preparation are real temporary state;
only OS process observation and forbidden compute/provider boundaries are mocked.
"""
import json
import os
import socket
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.engine import accounting, evaluator, failure, launcher, lifecycle, scheduler
from orze.engine.evaluation_retry import EvaluationRetryError, request_evaluation_retry
from orze.engine.termination_hold import (
    TerminationUnconfirmed, require_no_unconfirmed_stop, terminate_execution,
)
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    folder = results / "idea-stop"
    folder.mkdir(parents=True)
    database = tmp_path / "ideas.db"
    cfg = {
        "results_dir": str(results), "idea_lake_db": str(database),
        "_project_root": str(tmp_path), "_orze_dir": str(tmp_path / ".orze"),
        "eval_script": str(tmp_path / "evaluate.py"),
        "eval_output": "assessment.json", "eval_checkpoint": "checkpoint.pt",
        "train_script": str(tmp_path / "train.py"),
        "base_config": str(tmp_path / "base.yaml"),
        "ideas_file": str(tmp_path / "ideas.md"),
        "report": {"primary_metric": "score", "sort": "ascending",
                   "columns": [{"key": "score", "source": "assessment.json:score"}]},
    }
    for filename in ("train.py", "evaluate.py", "base.yaml", "ideas.md"):
        (tmp_path / filename).write_text("# never executed\n", encoding="utf-8")
    files = {
        "metrics.json": b'{"status":"COMPLETED","score":0}',
        "assessment.json": b'{"status":"COMPLETED","score":-2}',
        "checkpoint.pt": b"synthetic checkpoint, not a model",
        "train_output.log": b"training complete\n",
        "claim.json": json.dumps({
            "attempt_id": "training-original", "claimed_by": socket.gethostname(),
            "pid": 987654302, "owner_start_ticks": 12345, "gpu": 4,
        }).encode(),
    }
    for name, data in files.items():
        (folder / name).write_bytes(data)
        os.utime(folder / name, (1, 1))
    lake = IdeaLake(database)
    lake.insert(folder.name, "Fixture", "seed: 1\n", "notes", status="queued")
    assert lake.record_state_transition(folder.name, "QUEUED", "CLAIMED")
    assert lake.record_state_transition(folder.name, "CLAIMED", "IN_PROGRESS")
    assert lake.record_stage_transition(folder.name, "training", "IN_PROGRESS", "COMPLETE", "train_done")
    assert lake.record_stage_transition(folder.name, "evaluation", "PENDING", "IN_PROGRESS", "eval_started")
    tp = SimpleNamespace(
        idea_id=folder.name, gpu=4, attempt_id="evaluation-original",
        start_time=time.time() - 10,
        process=SimpleNamespace(pid=987654303, poll=lambda: 0),
    )
    accounting.record_compute_start(tp, folder, phase="evaluation")
    tripwires = []
    for owner, name in (
        (subprocess, "Popen"), (subprocess, "run"),
        (launcher, "_verify_gpu_free"), (evaluator, "_verify_gpu_free"),
    ):
        boundary = Mock(side_effect=AssertionError("stop admission must not run compute"))
        monkeypatch.setattr(owner, name, boundary)
        tripwires.append(boundary)
    monkeypatch.setattr("orze.extensions.get_extension", lambda name: None)
    monkeypatch.setattr(lifecycle, "_running_idea_pids", lambda: set())
    monkeypatch.setattr(lifecycle, "process_is_running", lambda *args: False)
    monkeypatch.setattr(scheduler, "process_is_running", lambda *args: False)
    p = SimpleNamespace(results=results, folder=folder, database=database,
                        cfg=cfg, lake=lake, tp=tp)
    try:
        yield p
    finally:
        p.lake.close()
        for boundary in tripwires:
            boundary.assert_not_called()


def _snapshot(p):
    files = {str(path.relative_to(p.folder)): path.read_bytes()
             for path in p.folder.rglob("*") if path.is_file()}
    tables = {}
    for table in ("ideas", "idea_state", "idea_stage_state", "idea_transitions", "idea_stage_transitions"):
        tables[table] = [tuple(row) for row in p.lake.conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid")]
    return files, tables


def _stopped(p, *, confirmed):
    reaper = Mock(return_value=confirmed)
    if confirmed:
        assert terminate_execution(p.tp, p.folder, phase="evaluation", reaper=reaper) == 0
    else:
        with pytest.raises(TerminationUnconfirmed):
            terminate_execution(p.tp, p.folder, phase="evaluation", reaper=reaper)
    reaper.assert_called_once()
    # Dispose of the original handle and Lake; no in-memory latch is available.
    attempt_id = p.tp.attempt_id
    p.tp = None
    p.lake.close()
    p.lake = IdeaLake(p.database)
    return SimpleNamespace(
        idea_id=p.folder.name, gpu=4, attempt_id=attempt_id,
        start_time=time.time() - 10,
        process=SimpleNamespace(pid=987654303, poll=lambda: 0),
    )


@pytest.mark.parametrize("entry", [
    "reset", "claim", "cleanup", "startup_reconcile", "dead_pid_reconcile",
    "accounting", "launch", "launch_eval",
])
def test_lost_handle_cannot_bypass_persisted_unconfirmed_stop(project, entry):
    p = project
    if entry == "claim":
        # Exercise the recoverable-directory claim path, not its ordinary
        # existing claim/metrics refusal control.
        (p.folder / "claim.json").unlink()
        (p.folder / "metrics.json").unlink()
        assert p.lake.set_status(p.folder.name, "queued")
    tp = _stopped(p, confirmed=False)
    before = _snapshot(p)
    observed = None
    try:
        if entry == "reset":
            observed = failure._reset_idea_for_retry(p.folder, release_claim=True)
        elif entry == "claim":
            observed = scheduler.claim(p.folder.name, p.results, 4, lake=p.lake)
        elif entry == "cleanup":
            observed = scheduler.cleanup_orphans(p.results, 0.001, lake=p.lake)
        elif entry == "startup_reconcile":
            observed = lifecycle.reconcile_stale_running(p.cfg)
        elif entry == "dead_pid_reconcile":
            observed = lifecycle.reconcile_running_dead_pids(p.cfg)
        elif entry == "accounting":
            observed = accounting.record_compute_terminal(
                tp, p.folder, "completed", "cannot_confirm_from_leader",
                phase="evaluation", return_code=0)
        elif entry == "launch":
            observed = launcher.launch(p.folder.name, 4, p.results, p.cfg, lake=p.lake)
        else:
            observed = evaluator.launch_eval(p.folder.name, 4, p.results, p.cfg, lake=p.lake)
    except TerminationUnconfirmed:
        pass
    assert _snapshot(p) == before, f"{entry} must retain all files and lifecycle under HOLD"
    assert observed in (None, False, 0)
    with pytest.raises(TerminationUnconfirmed):
        require_no_unconfirmed_stop(p.folder)


def test_confirmed_stop_and_terminal_receipt_allow_explicit_reset_then_claim(project):
    p = project
    tp = _stopped(p, confirmed=True)
    require_no_unconfirmed_stop(p.folder)
    receipt = accounting.record_compute_terminal(
        tp, p.folder, "interrupted", "confirmed_control", phase="evaluation", return_code=0)
    assert receipt["event"] == "terminal"
    stop_files = {str(path.relative_to(p.folder)): path.read_bytes()
                  for path in (p.folder / "_execution_stops").rglob("*") if path.is_file()}
    failure._reset_idea_for_retry(p.folder, release_claim=True)
    assert not (p.folder / "claim.json").exists()
    assert not (p.folder / "metrics.json").exists()
    assert p.lake.set_status(p.folder.name, "queued")
    assert scheduler.claim(p.folder.name, p.results, 4, lake=p.lake)
    assert {str(path.relative_to(p.folder)): path.read_bytes()
            for path in (p.folder / "_execution_stops").rglob("*") if path.is_file()} == stop_files


def test_evaluation_retry_cannot_archive_even_confirmed_stop_authority(project):
    p = project
    tp = _stopped(p, confirmed=True)
    accounting.record_compute_terminal(
        tp, p.folder, "failed", "confirmed_control", phase="evaluation", return_code=0)
    assert p.lake.record_state_transition(p.folder.name, "IN_PROGRESS", "FAILED", "eval_failed")
    p.cfg["eval_output"] = "_execution_stops/evaluation-original/requested.json"
    before = _snapshot(p)
    with pytest.raises(EvaluationRetryError, match="protected_artifact"):
        request_evaluation_retry(p.folder.name, p.results, p.cfg, p.lake)
    assert _snapshot(p) == before
    assert not (p.folder / "_evaluation_retries").exists()
