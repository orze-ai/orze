"""D1: an exited leader is not proof its evaluation writers have stopped.

Exercise public evaluator/retry APIs with real Lake, artifacts and receipts.
Only GPU/process operations and one actual start-receipt fsync are doubled.
No provider, child process, signal, or non-temporary result path is accessed.
These tests do not claim attempt-generation fencing or output isolation (D2).
"""
import errno
import json
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.engine import evaluator
from orze.engine.evaluation_retry import EvaluationRetryError, request_evaluation_retry
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    folder = results / "idea-termination"
    folder.mkdir(parents=True)
    training = b'{"status":"COMPLETED","quality":999}'
    checkpoint = b"completed-training\x00checkpoint\xff"
    (folder / "metrics.json").write_bytes(training)
    (folder / "checkpoint.pt").write_bytes(checkpoint)
    lake = IdeaLake(tmp_path / "ideas.db")
    lake.insert(folder.name, "termination authority", "seed: 13", "", status="queued")
    assert lake.reconcile_training_complete(folder.name, "reconcile_test_training")
    cfg = {
        "_project_root": str(tmp_path), "results_dir": str(results),
        "idea_lake_db": str(lake.db_path), "eval_script": "unused-evaluator.py",
        "eval_output": "assessment.json", "eval_checkpoint": "checkpoint.pt",
        "eval_timeout": 60,
        "report": {"primary_metric": "quality", "sort": "ascending",
                   "columns": [{"key": "quality", "source": "assessment.json:quality"}]},
    }
    p = SimpleNamespace(
        results=results, folder=folder, lake=lake, cfg=cfg, idea_id=folder.name,
        training=training, checkpoint=checkpoint, processes=[], logs=[],
        confirmed=False, wait_error=False, start_write_failures=[],
    )

    class OwnedProcess:
        pid = None
        returncode = None
        poll_error = False

        def poll(self):
            if self.poll_error:
                self.poll_error = False
                raise OSError("test-only process observation failure")
            return self.returncode

        def wait(self, timeout=None):
            if p.wait_error:
                raise OSError("test-only wait failure")
            if self.returncode is None:
                raise subprocess.TimeoutExpired("unused-evaluator.py", timeout)
            return self.returncode

    def popen(*args, **kwargs):
        proc = OwnedProcess()
        p.processes.append(proc)
        p.logs.append(kwargs["stdout"])
        return proc

    def reap(proc, *args, **kwargs):
        # False models an exited leader with an unconfirmed/live descendant,
        # exactly the boolean contract of the real _terminate_and_reap.
        proc.returncode = -9
        return p.confirmed

    p.popen = Mock(side_effect=popen)
    p.reap = Mock(side_effect=reap)
    p.lease = Mock(side_effect=lambda *args, **kwargs: nullcontext(()))
    p.gpu_check = Mock()
    monkeypatch.setattr(evaluator.subprocess, "Popen", p.popen)
    monkeypatch.setattr(evaluator, "_terminate_and_reap", p.reap)
    monkeypatch.setattr(evaluator, "gpu_execution_lease", p.lease)
    monkeypatch.setattr(evaluator, "_verify_gpu_free", p.gpu_check)
    try:
        yield p
    finally:
        for handle in p.logs:
            handle.close()
        p.lake.close()


def _fail_actual_start_fsync_once(p, monkeypatch):
    original_open, original_fsync = os.open, os.fsync
    start_fds = set()

    def observing_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        target = Path(path)
        if target.name == "start.json" and p.folder / "_compute_receipts" in target.parents:
            start_fds.add(descriptor)
        return descriptor

    def fail_fsync(descriptor):
        if descriptor in start_fds and not p.start_write_failures:
            p.start_write_failures.append("start_receipt_fsync")
            raise OSError(errno.ENOSPC, "test-only start receipt durability failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(os, "open", observing_open)
    monkeypatch.setattr(os, "fsync", fail_fsync)


def _invoke(p, mode, monkeypatch, active):
    if mode == "sync_wait_error":
        p.wait_error = True
        evaluator.run_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
        return
    if mode == "launch_start_storage_error":
        _fail_actual_start_fsync_once(p, monkeypatch)
        evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
        return
    ep = evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
    assert ep is not None
    active[0] = ep
    if mode == "async_timeout":
        now = ep.start_time + 120
        monkeypatch.setattr(evaluator.time, "time", lambda: now)
    elif mode == "async_poll_error":
        ep.process.poll_error = True
    else:
        raise AssertionError(mode)
    evaluator.check_active_evals(active, p.results, p.cfg, lake=p.lake)


def _assert_training_unchanged(p):
    assert p.lake.get_stage_state(p.idea_id, "training") == "COMPLETE"
    assert (p.folder / "metrics.json").read_bytes() == p.training
    assert (p.folder / "checkpoint.pt").read_bytes() == p.checkpoint


def _terminal_paths(p):
    return list((p.folder / "_compute_receipts").glob("*/terminal.json"))


@pytest.mark.parametrize("mode", [
    "async_timeout", "async_poll_error", "sync_wait_error", "launch_start_storage_error",
])
@pytest.mark.parametrize("confirmed", [False, True], ids=["unconfirmed", "confirmed"])
def test_all_public_error_entries_require_whole_evaluation_termination(
        project, monkeypatch, mode, confirmed):
    p = project
    p.confirmed = confirmed
    active = {}
    observed_error = None
    try:
        _invoke(p, mode, monkeypatch, active)
    except RuntimeError as exc:
        observed_error = str(exc)
    p.popen.assert_called_once()
    p.reap.assert_called_once()
    assert p.processes[0].returncode == -9
    assert len(list((p.folder / "_compute_receipts").glob("*/start.json"))) == 1
    if mode == "launch_start_storage_error":
        assert p.start_write_failures == ["start_receipt_fsync"]
    _assert_training_unchanged(p)
    if not confirmed:
        assert p.lake.get_fsm_state(p.idea_id) == "IN_PROGRESS"
        assert p.lake.get_stage_state(p.idea_id, "evaluation") == "IN_PROGRESS"
        assert _terminal_paths(p) == []
        assert not (p.folder / "assessment.json").exists()
        assert observed_error == "evaluation_termination_unconfirmed"
        if mode.startswith("async_"):
            assert list(active) == [0]
        with pytest.raises(EvaluationRetryError):
            request_evaluation_retry(p.idea_id, p.results, p.cfg, p.lake)
    else:
        assert observed_error is None
        assert p.lake.get_fsm_state(p.idea_id) == "FAILED"
        assert p.lake.get_stage_state(p.idea_id, "evaluation") == "FAILED"
        terminals = _terminal_paths(p)
        assert len(terminals) == 1
        terminal = json.loads(terminals[0].read_text(encoding="utf-8"))
        assert terminal["return_code"] == -9
        assert terminal["outcome"] == ("interrupted" if mode == "async_timeout" else "failed")
        assert request_evaluation_retry(p.idea_id, p.results, p.cfg, p.lake)["status"] == "evaluation_retry_pending"


def test_next_poll_does_not_reinterpret_unconfirmed_integer_exit_as_closed(project, monkeypatch):
    p = project
    active = {}
    try:
        _invoke(p, "async_timeout", monkeypatch, active)
    except RuntimeError as exc:
        assert str(exc) == "evaluation_termination_unconfirmed"
    assert p.processes[0].returncode == -9
    # Next ordinary monitor tick observes the integer exit code, not another
    # timeout; prior unconfirmed process-tree cleanup must remain authoritative.
    try:
        evaluator.check_active_evals(active, p.results, p.cfg, lake=p.lake)
    except RuntimeError as exc:
        assert str(exc) == "evaluation_termination_unconfirmed"
    assert list(active) == [0]
    assert _terminal_paths(p) == []
    assert p.lake.get_fsm_state(p.idea_id) == "IN_PROGRESS"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "IN_PROGRESS"
    with pytest.raises(EvaluationRetryError):
        request_evaluation_retry(p.idea_id, p.results, p.cfg, p.lake)
    _assert_training_unchanged(p)


def test_restart_cannot_reconcile_late_output_across_an_unclosed_evaluation_start(project):
    p = project
    assert evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake) is not None
    db_path = p.lake.db_path
    p.lake.close()
    p.lake = IdeaLake(db_path)
    # The controller has no EvalProcess after restart. Valid-looking output
    # arriving from the previous worker does not close its missing receipt.
    (p.folder / "assessment.json").write_text(
        '{"status":"COMPLETED","quality":-99}', encoding="utf-8")
    before = (p.folder / "assessment.json").read_bytes()
    try:
        result = evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
        assert result is None
    except RuntimeError as exc:
        assert str(exc) == "evaluation_termination_unconfirmed"
    p.popen.assert_called_once()
    assert p.lake.get_fsm_state(p.idea_id) == "IN_PROGRESS"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "IN_PROGRESS"
    assert _terminal_paths(p) == []
    assert (p.folder / "assessment.json").read_bytes() == before
    _assert_training_unchanged(p)


def test_unconfirmed_timeout_cannot_enable_retry_and_accept_a_late_old_writer(project, monkeypatch):
    p = project
    active = {}
    try:
        _invoke(p, "async_timeout", monkeypatch, active)
    except RuntimeError as exc:
        assert str(exc) == "evaluation_termination_unconfirmed"
    accepted = None
    try:
        accepted = request_evaluation_retry(p.idea_id, p.results, p.cfg, p.lake)
    except EvaluationRetryError:
        pass
    late_writer_result = None
    if accepted is not None:
        # Preserve the full old public failure chain as diagnostic evidence,
        # rather than replacing admission, validation, or receipts with mocks.
        new = evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
        assert new is not None
        (p.folder / "assessment.json").write_text(
            '{"status":"COMPLETED","quality":-99}', encoding="utf-8")
        new.process.returncode = 0  # New evaluator produced no output itself.
        evaluator.check_active_evals({0: new}, p.results, p.cfg, lake=p.lake)
        late_writer_result = p.lake.get_fsm_state(p.idea_id)
    assert accepted is None, {"retry": accepted, "late_old_output_outcome": late_writer_result}
    p.popen.assert_called_once()
    assert _terminal_paths(p) == []
    assert p.lake.get_fsm_state(p.idea_id) == "IN_PROGRESS"
    _assert_training_unchanged(p)
