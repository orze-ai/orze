"""D2 public behavior: old completion cannot own a newer evaluation.

The Lake, launch/retry/completion functions, metric validation, files and
compute receipts are real. Only GPU/process observation boundaries are fake.
No new attempt-authority API is assumed by these frozen tests. Direct writes
by a surviving external worker to canonical output remain outside this scope.
"""
import json
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.engine import evaluator
from orze.engine.evaluation_retry import request_evaluation_retry
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "ideas.db")
    cfg = {
        "_project_root": str(tmp_path), "results_dir": str(results),
        "idea_lake_db": str(lake.db_path), "eval_script": "unused-evaluator.py",
        "eval_output": "assessment.json", "eval_checkpoint": "checkpoint.pt",
        "eval_timeout": 60,
        "report": {"primary_metric": "quality", "sort": "ascending",
                   "columns": [{"key": "quality", "source": "assessment.json:quality"}]},
    }
    p = SimpleNamespace(results=results, lake=lake, cfg=cfg, logs=[], processes=[])

    class OwnedProcess:
        pid = None
        returncode = None
        on_poll = None

        def poll(self):
            callback, self.on_poll = self.on_poll, None
            if callback is not None:
                callback()
            return self.returncode

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("unused-evaluator.py", timeout)
            return self.returncode

    def popen(*args, **kwargs):
        process = OwnedProcess()
        p.processes.append(process)
        p.logs.append(kwargs["stdout"])
        return process

    p.popen = Mock(side_effect=popen)
    p.lease = Mock(side_effect=lambda *args, **kwargs: nullcontext(()))
    p.gpu_check = Mock()
    p.reaper = Mock(side_effect=AssertionError("Exited-process tests must not signal any process"))
    monkeypatch.setattr(evaluator.subprocess, "Popen", p.popen)
    monkeypatch.setattr(evaluator, "gpu_execution_lease", p.lease)
    monkeypatch.setattr(evaluator, "_verify_gpu_free", p.gpu_check)
    monkeypatch.setattr(evaluator, "_terminate_and_reap", p.reaper)
    try:
        yield p
    finally:
        for handle in p.logs:
            handle.close()
        lake.close()


def _prepare(p, idea_id):
    folder = p.results / idea_id
    folder.mkdir()
    (folder / "metrics.json").write_bytes(b'{"status":"COMPLETED","quality":999}')
    (folder / "checkpoint.pt").write_bytes(b"successful-training\x00checkpoint\xff")
    p.lake.insert(idea_id, "bounded evaluation", "seed: 13", "", status="queued")
    assert p.lake.reconcile_training_complete(idea_id, "reconcile_test_training")
    return folder


def _launch(p, idea_id):
    ep = evaluator.launch_eval(idea_id, 0, p.results, p.cfg, lake=p.lake)
    assert ep is not None
    assert p.lake.get_fsm_state(idea_id) == "IN_PROGRESS"
    assert p.lake.get_stage_state(idea_id, "evaluation") == "IN_PROGRESS"
    return ep


def _exit(p, ep, code, *, valid=True):
    (p.results / ep.idea_id / "assessment.json").write_text(
        json.dumps({"status": "COMPLETED", "quality": 0 if valid else "invalid-number"}),
        encoding="utf-8")
    ep.process.returncode = code


def _files(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in sorted(folder.rglob("*")) if path.is_file()}


def _lifecycle(lake):
    return {
        table: [tuple(row) for row in lake.conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        for table in ("ideas", "idea_state", "idea_stage_state", "idea_transitions", "idea_stage_transitions")
    }


@pytest.mark.parametrize("old_exit", [1, 0], ids=["nonzero_exit", "zero_exit_invalid_evidence"])
def test_old_attempt_callback_cannot_change_the_running_retry_or_deliver_finished(project, old_exit):
    p = project
    folder = _prepare(p, "idea-stale")
    old = _launch(p, folder.name)
    # An exit-zero evaluator can fail its evidence contract. Its OS exit code
    # remains zero on replay; the fixture never changes an exited process code.
    _exit(p, old, old_exit, valid=old_exit != 0)
    assert evaluator.check_active_evals({0: old}, p.results, p.cfg, lake=p.lake) == [(folder.name, 0)]
    assert p.lake.get_fsm_state(folder.name) == "FAILED"
    assert request_evaluation_retry(folder.name, p.results, p.cfg, p.lake)["status"] == "evaluation_retry_pending"
    current = _launch(p, folder.name)
    assert current.attempt_id != old.attempt_id
    assert current.process.returncode is None
    before_files, before_lifecycle = _files(folder), _lifecycle(p.lake)

    finished = evaluator.check_active_evals({0: old}, p.results, p.cfg, lake=p.lake)

    assert _lifecycle(p.lake) == before_lifecycle, "a stale callback changed the newer attempt's lifecycle"
    assert _files(folder) == before_files, "a stale callback rewrote metrics, diagnostics, or receipts"
    assert finished == [], "only a newly accepted current-attempt terminal may be delivered"
    assert current.process.returncode is None
    assert p.lake.get_stage_state(folder.name, "training") == "COMPLETE"
    p.reaper.assert_not_called()


@pytest.mark.parametrize("exit_code", [0, 1], ids=["completed", "failed"])
def test_duplicate_callback_for_same_terminal_attempt_does_not_repeat_finished(project, exit_code):
    p = project
    folder = _prepare(p, "idea-duplicate")
    ep = _launch(p, folder.name)
    _exit(p, ep, exit_code)
    assert evaluator.check_active_evals({0: ep}, p.results, p.cfg, lake=p.lake) == [(folder.name, 0)]
    before_files, before_lifecycle = _files(folder), _lifecycle(p.lake)

    duplicate = evaluator.check_active_evals({0: ep}, p.results, p.cfg, lake=p.lake)

    assert duplicate == [], "immutable receipt idempotence does not authorize a second finished event"
    assert _lifecycle(p.lake) == before_lifecycle
    assert _files(folder) == before_files
    p.reaper.assert_not_called()


def test_old_completed_slot_owner_cannot_delete_a_new_process_registered_during_poll(project):
    p = project
    folder_a = _prepare(p, "idea-slot-a")
    folder_b = _prepare(p, "idea-slot-b")
    old = _launch(p, folder_a.name)
    _exit(p, old, 0)
    # A has physically exited before B starts; only A's monitor bookkeeping is
    # delayed. The GPU boundary is fake, not a claim of real GPU multiplexing.
    current = _launch(p, folder_b.name)
    active = {0: old}
    replacements = []

    def register_current():
        replacements.append("registered newer slot owner")
        active[0] = current

    old.process.on_poll = register_current
    finished = evaluator.check_active_evals(active, p.results, p.cfg, lake=p.lake)

    assert replacements == ["registered newer slot owner"]
    assert active.get(0) is current, "old completion blindly removed the current GPU-slot object"
    assert finished == [(folder_a.name, 0)]
    assert current.process.returncode is None
    assert p.lake.get_fsm_state(folder_b.name) == "IN_PROGRESS"
    assert not (folder_b / "_compute_receipts" / current.attempt_id / "terminal.json").exists()
    p.reaper.assert_not_called()


def test_current_success_is_delivered_once_and_an_empty_tick_is_a_noop(project):
    p = project
    folder = _prepare(p, "idea-current")
    ep = _launch(p, folder.name)
    _exit(p, ep, 0)
    active = {0: ep}
    assert evaluator.check_active_evals(active, p.results, p.cfg, lake=p.lake) == [(folder.name, 0)]
    assert active == {}
    before_files, before_lifecycle = _files(folder), _lifecycle(p.lake)
    assert evaluator.check_active_evals(active, p.results, p.cfg, lake=p.lake) == []
    assert _files(folder) == before_files
    assert _lifecycle(p.lake) == before_lifecycle
    assert p.lake.get_fsm_state(folder.name) == "COMPLETE"
    assert p.lake.get_stage_state(folder.name, "evaluation") == "COMPLETE"
    p.popen.assert_called_once()
    p.reaper.assert_not_called()
