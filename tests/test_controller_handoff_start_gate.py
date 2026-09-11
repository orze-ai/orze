"""Handoff entrance wiring; resource/commit doubles are not closure proofs.

Invalid admission tests use the real strong validator. The positive gate and
Orze ordering tests explicitly isolate its dispatch and resource boundaries;
real grant/handshake acceptance belongs to the independent product tests.
"""
from types import SimpleNamespace

import pytest

from orze.core.control_outcome import (
    ControllerStopHOLD, StopOutcome, require_controller_start_allowed,
)
from orze.engine import controller_control, controller_handoff, orchestrator


@pytest.mark.parametrize("admission", [None, True, StopOutcome("confirmed", "label_only")])
def test_namespace_never_accepts_absent_boolean_or_label_admission(tmp_path, monkeypatch, admission):
    marker = tmp_path / "_controller_registration.lock.source-lock"
    marker.write_bytes(b"permanent namespace")
    before = marker.stat()
    monkeypatch.setattr(controller_control, "current_controller", lambda: None)
    monkeypatch.setattr(controller_handoff, "_ADMISSION", admission)
    with pytest.raises(ControllerStopHOLD, match="blocked_by_registration"):
        require_controller_start_allowed(tmp_path)
    assert marker.read_bytes() == b"permanent namespace"
    assert marker.stat().st_ino == before.st_ino


@pytest.mark.parametrize("old_stop", [False, True])
def test_pending_validator_delegation_still_checks_old_stop_markers(tmp_path, monkeypatch, old_stop):
    marker = tmp_path / "_controller_registration.lock.source-lock"
    marker.write_bytes(b"namespace")
    stop = tmp_path / ".orze_stop_all"
    if old_stop:
        stop.write_bytes(b"not closure")
    calls = []
    monkeypatch.setattr(controller_control, "current_controller", lambda: None)
    # Explicit successful-validation boundary, not a fabricated strong grant.
    monkeypatch.setattr(controller_handoff, "require_pending_start", lambda scope: calls.append(scope))
    if old_stop:
        with pytest.raises(ControllerStopHOLD, match="blocked_by_sentinel"):
            require_controller_start_allowed(tmp_path)
        assert stop.read_bytes() == b"not closure"
    else:
        assert require_controller_start_allowed(tmp_path) is None
    assert calls == [tmp_path.absolute()]
    assert marker.read_bytes() == b"namespace"


@pytest.mark.parametrize("mode", ["successor", "refused", "fresh"])
def test_actual_profile_caller_pins_resources_then_marks_before_first_probe(tmp_path, monkeypatch, mode):
    events = []
    error = controller_control.ControllerHOLD("controlled_commit_refusal")
    leases = object()
    runner = orchestrator.Orze.__new__(orchestrator.Orze)
    runner.results_dir, runner.gpu_ids = tmp_path, [2, 4]

    def write_pid():
        runner._pid_file = tmp_path / ".orze.pid"
        runner._pid_file.write_text("explicit PID boundary", encoding="utf-8")
        events.append("pid")

    class Admission:
        def mark_started(self, session):
            assert runner._controller_session is session
            assert runner._gpu_leases is leases
            assert session.pid is runner._pid_file
            assert session.leases is leases
            events.append("mark_started")
            if mode == "refused":
                raise error

    class Session:
        def __init__(self, host):
            assert host is runner
            self._admission = None if mode == "fresh" else Admission()
            events.append("session")

        def start(self):
            events.append("start")

        def bind_pid_file(self, path):
            self.pid = path
            events.append("bind_pid")

        def bind_gpu_leases(self, value):
            self.leases = value
            events.append("bind_gpu")

        def fail(self, exc):
            assert exc is error
            events.append("fail")

        def finish(self):
            events.append("finish")
            return "explicit finalizer boundary"

    def acquire(gpus):
        assert gpus == [2, 4]
        events.append("acquire")
        return leases

    ctx = SimpleNamespace(check_admission=lambda: events.append("check"))
    monkeypatch.setattr(controller_control, "current_controller", lambda: ctx)
    monkeypatch.setattr("orze.engine.controller_session.ControllerSession", Session)
    monkeypatch.setattr(orchestrator, "acquire_gpu_leases", acquire)
    monkeypatch.setattr(orchestrator, "assert_gpu_scope_idle", lambda gpus: events.append("probe"))
    runner._write_pid_file = write_pid
    runner._run_leased = lambda: events.append("run")
    prefix = ["session", "start", "check", "pid", "bind_pid", "check", "acquire", "bind_gpu", "check"]
    if mode == "refused":
        with pytest.raises(controller_control.ControllerHOLD) as caught:
            runner._run_controller_profile()
        assert caught.value is error
        assert events == prefix + ["mark_started", "fail"]
    else:
        assert runner._run_controller_profile() == "explicit finalizer boundary"
        assert events == prefix + ([] if mode == "fresh" else ["mark_started"]) + ["probe", "run", "finish"]
