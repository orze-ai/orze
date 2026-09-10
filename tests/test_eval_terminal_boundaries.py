"""V1-01G1 fault boundaries around the real evaluator completion entry points.

All artifacts and SQLite state live under tmp_path. Popen, GPU hardware
admission/lease, and process termination are test doubles: no GPU inspection,
provider call, host process signal, or actual evaluator is permitted.
"""

import hashlib
import json
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.engine import accounting, evaluator
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.evaluation_retry import request_evaluation_retry
from orze.engine.sealed import write_sealed_manifest
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    folder = results / "idea-terminal-boundary"
    folder.mkdir(parents=True)
    metrics = folder / "metrics.json"
    metrics.write_text('{"status":"COMPLETED","quality":1}', encoding="utf-8")
    checkpoint = folder / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable-completed-training\x00\xff")
    lake = IdeaLake(tmp_path / "ideas.db")
    lake.insert(folder.name, "terminal boundary", "{}", "", status="queued")
    assert lake.reconcile_training_complete(folder.name, "reconcile_test_training_finished")
    cfg = {
        "_project_root": str(tmp_path), "idea_lake_db": str(lake.db_path),
        "eval_script": "unused-evaluator.py", "eval_output": "assessment.json",
        "eval_timeout": 60,
        "report": {"primary_metric": "quality", "sort": "ascending",
                   "columns": [{"key": "quality", "source": "assessment.json:quality"}]},
    }
    p = SimpleNamespace(
        results=results, folder=folder, idea_id=folder.name, lake=lake, cfg=cfg,
        training_bytes=metrics.read_bytes(), checkpoint_bytes=checkpoint.read_bytes(),
        exit_code=0, behavior="exit", on_exit=lambda: None, emit_assessment=True,
        emitted=False,
    )

    class FakeProcess:
        pid = None  # Even an unintended helper path cannot signal a host PID.
        returncode = None

        def finish(self):
            if not p.emitted:
                if p.emit_assessment:
                    (folder / "assessment.json").write_text(
                        '{"status":"COMPLETED","quality":0}', encoding="utf-8")
                p.on_exit()
                p.emitted = True
            self.returncode = p.exit_code
            return self.returncode

        def wait(self, timeout=None):
            if p.behavior == "wait_timeout":
                raise subprocess.TimeoutExpired("unused-evaluator.py", timeout)
            if p.behavior == "wait_error":
                raise OSError("injected wait failure")
            return self.finish()

        def poll(self):
            if self.returncode is not None:
                return self.returncode
            if p.behavior == "poll_error":
                raise OSError("injected poll failure")
            if p.behavior in ("wait_error", "wait_timeout"):
                return None
            return self.finish()

    p.process = FakeProcess()
    p.popen = Mock(return_value=p.process)
    p.gpu_check = Mock()
    p.lease = Mock(side_effect=lambda *args, **kwargs: nullcontext(()))

    def terminate(proc, *args, **kwargs):
        assert proc is p.process
        proc.returncode = -15
        return True

    p.terminate = Mock(side_effect=terminate)
    monkeypatch.setattr(evaluator.subprocess, "Popen", p.popen)
    monkeypatch.setattr(evaluator, "_verify_gpu_free", p.gpu_check)
    monkeypatch.setattr(evaluator, "gpu_execution_lease", p.lease)
    monkeypatch.setattr(evaluator, "_terminate_and_reap", p.terminate)
    # Spy, not a replacement implementation: receipt creation and conflict
    # detection still execute the production accounting function on disk.
    p.terminal_writer = Mock(wraps=accounting.record_compute_terminal)
    monkeypatch.setattr(accounting, "record_compute_terminal", p.terminal_writer)
    monkeypatch.setattr(evaluator, "record_compute_terminal", p.terminal_writer)
    try:
        yield p
    finally:
        lake.close()


def _launch(p):
    ep = evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
    assert ep is not None, "An existing training artifact is not completed evaluation"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "IN_PROGRESS"
    return ep


def _complete(p, mode):
    if mode == "sync":
        evaluator.run_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
    else:
        active = {0: _launch(p)}
        assert evaluator.check_active_evals(
            active, p.results, p.cfg, lake=p.lake) == [(p.idea_id, 0)]
        assert active == {}
    p.popen.assert_called_once()
    p.gpu_check.assert_called_once()
    p.lease.assert_called_once_with(0, require_idle=True)


def _assert_terminal(p, outcome, *, metrics_removed=False):
    expected = "COMPLETE" if outcome == "completed" else "FAILED"
    assert p.lake.get_fsm_state(p.idea_id) == expected
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == expected
    assert p.lake.get_stage_state(p.idea_id, "training") == "COMPLETE"
    assert (p.folder / "checkpoint.pt").read_bytes() == p.checkpoint_bytes
    if metrics_removed:
        assert not (p.folder / "metrics.json").exists()
    else:
        assert (p.folder / "metrics.json").read_bytes() == p.training_bytes
    starts = list((p.folder / "_compute_receipts").glob("*/start.json"))
    terminals = list((p.folder / "_compute_receipts").glob("*/terminal.json"))
    assert len(starts) == len(terminals) == 1
    start = json.loads(starts[0].read_text(encoding="utf-8"))
    terminal = json.loads(terminals[0].read_text(encoding="utf-8"))
    assert start["attempt_id"] == terminal["attempt_id"]
    assert start["phase"] == terminal["phase"] == "evaluation"
    assert terminal["outcome"] == outcome
    assert terminal["return_code"] == p.process.returncode
    return terminals[0]


@pytest.mark.parametrize("exit_code,outcome", [(0, "completed"), (1, "failed")])
def test_blocking_completion_invokes_real_terminal_writer_exactly_once(
        project, exit_code, outcome):
    p = project
    p.exit_code = exit_code

    _complete(p, "sync")

    p.terminal_writer.assert_called_once()
    _assert_terminal(p, outcome)


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_sealed_file_changed_during_evaluation_rejects_both_paths(project, mode):
    p = project
    sealed = p.results / "sealed-evaluation-policy.json"
    sealed.write_bytes(b'{"split":"held-out"}')
    write_sealed_manifest(p.results, {
        str(sealed): hashlib.sha256(sealed.read_bytes()).hexdigest()})
    p.cfg["sealed_files"] = [str(sealed)]
    p.on_exit = lambda: sealed.write_bytes(b'{"split":"changed"}')

    _complete(p, mode)

    p.terminal_writer.assert_called_once()
    _assert_terminal(p, "failed")


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_training_metrics_removed_during_evaluation_cannot_complete(project, mode):
    p = project
    p.on_exit = lambda: (p.folder / "metrics.json").unlink()

    _complete(p, mode)

    p.terminal_writer.assert_called_once()
    _assert_terminal(p, "failed", metrics_removed=True)


@pytest.mark.parametrize("behavior,outcome", [
    ("wait_timeout", "interrupted"), ("wait_error", "failed"),
])
def test_wait_exception_reaps_process_before_one_terminal_receipt(
        project, behavior, outcome):
    p = project
    p.behavior = behavior

    _complete(p, "sync")

    p.terminate.assert_called_once()
    assert p.process.returncode == -15
    p.terminal_writer.assert_called_once()
    _assert_terminal(p, outcome)


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_output_alias_to_training_file_still_runs_eval_and_preserves_training_on_failure(
        project, mode):
    p = project
    p.cfg["eval_output"] = "./metrics.json"
    p.cfg["report"]["columns"] = [
        {"key": "quality", "source": "metrics.json:quality"}]
    p.emit_assessment = False
    p.exit_code = 1

    _complete(p, mode)

    p.terminal_writer.assert_called_once()
    _assert_terminal(p, "failed")


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_symlink_to_training_output_is_rejected_before_launch_without_changing_training(
        project, mode):
    p = project
    # Existing source safety rejects symlinks, including targets inside the
    # idea directory. Do not weaken that rule to handle normalized aliases.
    (p.folder / "training-report.json").symlink_to("metrics.json")
    p.cfg["eval_output"] = "training-report.json"
    history_before = p.lake.get_stage_history(p.idea_id)

    if mode == "sync":
        evaluator.run_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
    else:
        assert evaluator.launch_eval(
            p.idea_id, 0, p.results, p.cfg, lake=p.lake) is None

    p.popen.assert_not_called()
    p.gpu_check.assert_not_called()
    p.lease.assert_not_called()
    p.terminal_writer.assert_not_called()
    assert p.lake.get_fsm_state(p.idea_id) == "IN_PROGRESS"
    assert p.lake.get_stage_state(p.idea_id, "training") == "COMPLETE"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "PENDING"
    assert p.lake.get_stage_history(p.idea_id) == history_before
    assert (p.folder / "metrics.json").read_bytes() == p.training_bytes
    assert (p.folder / "checkpoint.pt").read_bytes() == p.checkpoint_bytes
    audits = [json.loads(line) for line in
              (p.folder / "_eval_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["reason"] == "evaluation_output_path_invalid" for row in audits)


def test_terminal_receipt_io_failure_holds_until_explicit_resolution(project):
    p = project
    ep = _launch(p)
    active = {0: ep}
    terminal_path = p.folder / "_compute_receipts" / ep.attempt_id / "terminal.json"
    # Real filesystem failure, not a mocked accounting/qualification result.
    terminal_path.mkdir()
    with pytest.raises(AttemptEffectInDoubt):
        evaluator.check_active_evals(active, p.results, p.cfg, lake=p.lake)
    assert p.lake.get_fsm_state(p.idea_id) == "IN_PROGRESS"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "IN_PROGRESS"
    assert active == {0: ep}
    prepared = p.folder / "_execution_effects" / ep.attempt_id / "prepared.json"
    committed = prepared.with_name("committed.json")
    before = prepared.read_bytes()
    assert not committed.exists()
    terminal_path.rmdir()

    # D2 contract migration: removing the incidental I/O obstacle does not
    # resolve a publication whose prepared intent may already have effects.
    with pytest.raises(AttemptEffectInDoubt):
        evaluator.check_active_evals(active, p.results, p.cfg, lake=p.lake)
    with pytest.raises(AttemptEffectInDoubt):
        request_evaluation_retry(p.idea_id, p.results, p.cfg, p.lake)
    with pytest.raises(AttemptEffectInDoubt):
        evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)

    assert active == {0: ep}
    assert p.lake.get_fsm_state(p.idea_id) == "IN_PROGRESS"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "IN_PROGRESS"
    assert p.lake.get_stage_state(p.idea_id, "training") == "COMPLETE"
    assert (p.folder / "metrics.json").read_bytes() == p.training_bytes
    assert (p.folder / "checkpoint.pt").read_bytes() == p.checkpoint_bytes
    assert prepared.read_bytes() == before
    assert not committed.exists()
    assert not terminal_path.exists()
    p.terminal_writer.assert_called_once()
    p.popen.assert_called_once()


def test_async_poll_error_reaps_and_records_failure_instead_of_abandoning_attempt(project):
    p = project
    p.behavior = "poll_error"

    _complete(p, "async")

    p.terminate.assert_called_once()
    p.terminal_writer.assert_called_once()
    _assert_terminal(p, "failed")
