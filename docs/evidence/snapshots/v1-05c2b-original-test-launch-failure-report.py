"""Real phase error delivery must retain the failed launch's exact identity."""
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import failure, phases, scheduler, training_attempts
from orze.engine.termination_hold import TerminationUnconfirmed
from test_training_launch_termination_handoff import scenario


def _fail_initialization(monkeypatch, child):
    error = RuntimeError("synthetic confirmed native initialization failure")
    def reject(*args, **kwargs):
        raise error
    def stop(proc, *args, **kwargs):
        assert proc is child
        child.returncode = -15
        return True
    monkeypatch.setattr("orze.core.model_lineage.receive_model_lineage_attestation", reject)
    monkeypatch.setattr("orze.engine.launcher._terminate_and_reap", stop)
    return error


def _phase(runner):
    phases.OrzePhaseMixin._launch_training(runner, ["idea-handoff"], True,
        {"idea-handoff": {"title": "Fixture", "config": {"seed": 1}}})


def _files(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def test_confirmed_native_launch_failure_reports_once_without_inline_provider(scenario, monkeypatch):
    runner, child, popen, fixer = scenario
    _fail_initialization(monkeypatch, child)
    _phase(runner)
    assert popen.call_count == 1
    fixer.assert_not_called()
    assert runner.lake.get_fsm_state("idea-handoff") == "FAILED"
    assert runner.failure_counts == {"idea-handoff": 1}
    report = current_attempt(runner.lake.conn, "idea-handoff", "launch_failure_report")
    assert report["state"] == "TERMINAL"
    assert report["terminal"]["repair_status"] == "pending_explicit_action"


def test_late_a_launch_error_cannot_report_or_spend_budget_for_new_b(scenario, monkeypatch):
    runner, child, popen, fixer = scenario
    _fail_initialization(monkeypatch, child)
    folder = runner.results_dir / "idea-handoff"
    original_open = builtins.open
    captured = {}

    def rollover():
        failure._reset_idea_for_retry(folder, release_claim=True, lake=runner.lake)
        assert runner.lake.record_state_transition("idea-handoff", "CLAIMED", "QUEUED")
        assert scheduler.claim("idea-handoff", runner.results_dir, 4, lake=runner.lake)
        claim = json.loads((folder / "claim.json").read_text())
        pending = SimpleNamespace(idea_id="idea-handoff", attempt_id=claim["attempt_id"], gpu=4)
        ref = training_attempts.begin(runner.lake, pending, folder)
        (folder / "metrics.json").write_text('{"status":"IN_PROGRESS","owner":"B"}')
        captured.update(ref=ref, files=_files(folder), history=runner.lake.get_fsm_history("idea-handoff"))

    class ClosingBoundary:
        def __init__(self, stream):
            self.stream = stream
        def __getattr__(self, name):
            return getattr(self.stream, name)
        def close(self):
            self.stream.close()
            if not captured:
                rollover()

    def open_boundary(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if Path(path) == folder / "train_output.log" and args and args[0] == "w":
            return ClosingBoundary(stream)
        return stream

    monkeypatch.setattr(builtins, "open", open_boundary)
    try:
        _phase(runner)
    except TerminationUnconfirmed:
        pass
    assert captured, "the actual log-close boundary must advance B before A error delivery"
    assert popen.call_count == 1
    assert runner.failure_counts == {}
    fixer.assert_not_called()
    assert _files(folder) == captured["files"]
    assert runner.lake.get_fsm_history("idea-handoff") == captured["history"]
    row = current_attempt(runner.lake.conn, "idea-handoff", "training")
    assert row["attempt_id"] == captured["ref"].attempt_id and row["state"] == "LAUNCHING"
