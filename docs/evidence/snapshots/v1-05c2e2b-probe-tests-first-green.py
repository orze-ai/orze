"""New probe-adapter mechanisms, not old-source or controller-ACK evidence.

The context and action-settlement boundary are explicit spies. CPU cases use
real Linux supervisors/blocked workers/closure; they do not claim persistent
controller enrollment (covered by the shared member tests). No GPU or remote
process is executed. Fault cleanup uses only captured owned pidfds.
"""
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.engine import controller_probe as probe
from orze.engine.supervised_process import prepare_supervised
from orze.engine.supervisor_worker import canonical


@pytest.fixture
def owned(tmp_path, monkeypatch):
    state = SimpleNamespace(scope=tmp_path, admissions=0, polls=0, holds=[],
                            handles=[], pidfds=[], settled=[], pipes_closed=False)
    def admit():
        state.admissions += 1
    def poll_control():
        state.polls += 1
    state.check_admission, state.poll_control, state.hold = admit, poll_control, state.holds.append
    monkeypatch.setattr(probe, "_current_controller", lambda: state)
    original_close = probe._ProbeStreams.close
    def close(streams):
        original_close(streams)
        state.pipes_closed = True
    monkeypatch.setattr(probe._ProbeStreams, "close", close)
    def prepare(*args, **kwargs):
        process = prepare_supervised(*args, **kwargs)
        state.handles.append(process)
        state.pidfds.append((os.pidfd_open(process.pid), os.pidfd_open(process.supervisor_pid)))
        return process
    monkeypatch.setattr(probe, "prepare_supervised", prepare)
    state.prepare = prepare
    def settle(process, **metadata):
        assert state.pipes_closed, "settlement requires pipe closure, not only leader return"
        closure = process.closure_receipt()
        assert closure["event"] == "TREE_CLOSED" and closure["wait_proof"] == "ECHILD_WALL"
        state.settled.append((closure, metadata))
    monkeypatch.setattr(probe, "_settle", settle)
    try:
        yield state
    finally:
        for process, (worker_fd, supervisor_fd) in zip(state.handles, state.pidfds):
            try:
                signal.pidfd_send_signal(worker_fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process._supervisor.wait(timeout=3)
            except subprocess.TimeoutExpired:
                signal.pidfd_send_signal(supervisor_fd, signal.SIGKILL)
                process._supervisor.wait(timeout=3)
            finally:
                os.close(worker_fd)
                os.close(supervisor_fd)
                process._close_descriptors()


def _run(state, code, **kwargs):
    return probe.run_probe([sys.executable, "-I", "-c", code], cwd=state.scope,
        capture_output=True, timeout=kwargs.pop("timeout", 5), **kwargs)


def _output_hash(stdout, stderr):
    record = {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
              for name, raw in (("stdout", stdout), ("stderr", stderr))}
    return hashlib.sha256(canonical(record)).hexdigest()


def test_no_controller_delegates_unmodified_arguments_result_and_exception(monkeypatch):
    monkeypatch.setattr(probe, "_current_controller", lambda: None)
    sentinel = object()
    run = Mock(return_value=sentinel)
    monkeypatch.setattr(probe.subprocess, "run", run)
    assert probe.run_probe("untouched command", shell=True, input=b"input", check=True) is sentinel
    run.assert_called_once_with("untouched command", shell=True, input=b"input", check=True)
    error = PermissionError("original run exception")
    run.side_effect = error
    with pytest.raises(PermissionError) as caught:
        probe.run_probe(args=["not-executed"])
    assert caught.value is error


@pytest.mark.parametrize("text", [False, True])
def test_real_complete_output_and_hashes_settle_after_tree_and_pipe_close(owned, text):
    stdout, stderr = "雪\r\n".encode(), b"error\r"
    result = _run(owned, f"import os; os.write(1,{stdout!r}); os.write(2,{stderr!r})", text=text)
    assert type(result) is subprocess.CompletedProcess and result.returncode == 0
    assert result.stdout == ("雪\n" if text else stdout)
    assert result.stderr == ("error\n" if text else stderr)
    assert len(owned.settled) == 1 and not owned.holds
    tree, metadata = owned.settled[0]
    assert tree["binding"]["identity"]["kind"] == "controller_probe"
    assert tree["binding"]["identity"]["scope"] == str(owned.scope)
    assert metadata == {"outcome": "completed", "output_sha256": _output_hash(stdout, stderr),
                        "output_bytes": len(stdout) + len(stderr)}


@pytest.mark.parametrize("extra", [0, 1])
def test_combined_stream_limit_is_exact_and_excess_is_not_a_normal_probe(owned, extra):
    stdout, stderr = b"a" * 32768, b"b" * (32768 + extra)
    code = f"import os; os.write(1,b'a'*32768); os.write(2,b'b'*{len(stderr)})"
    if extra:
        with pytest.raises(probe.ControllerProbeHOLD, match="output_limit"):
            _run(owned, code)
    else:
        result = _run(owned, code)
        assert result.stdout == stdout and result.stderr == stderr
    assert owned.settled[0][1]["output_bytes"] == 65536 + extra
    assert owned.settled[0][1]["output_sha256"] == _output_hash(stdout, stderr)
    assert owned.settled[0][1]["outcome"] == ("interrupted" if extra else "completed")


def test_real_nonzero_check_keeps_completed_process_exception_semantics(owned):
    with pytest.raises(subprocess.CalledProcessError) as caught:
        _run(owned, "import sys; print('known failure'); sys.exit(7)", text=True, check=True)
    assert caught.value.returncode == 7
    assert caught.value.stdout == "known failure\n" and caught.value.stderr == ""
    assert owned.settled[0][1]["outcome"] == "completed"
    assert not owned.holds


def test_actual_timeout_stop_zero_is_interrupted_not_normal_success(owned, monkeypatch):
    stops = []
    def prepare(*args, **kwargs):
        process = owned.prepare(*args, **kwargs)
        original = process.stop
        def stop(*a, **k):
            stops.append(True)
            return original(*a, **k)
        process.stop = stop
        return process
    monkeypatch.setattr(probe, "prepare_supervised", prepare)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        _run(owned, "import signal,sys,time; signal.signal(signal.SIGTERM,lambda *a:sys.exit(0)); "
             "print('ready',flush=True); time.sleep(30)", timeout=0.4)
    assert caught.value.returncode == 0
    assert caught.value.output == b"ready\n"
    assert stops == [True] and owned.settled[0][1]["outcome"] == "interrupted"
    assert owned.settled[0][0]["stop_requested"] is True


def test_quiesce_false_uses_actual_no_go_stop_and_never_returns_normal_result(owned, monkeypatch):
    marker = owned.scope / "should-not-execute"
    def prepare(*args, **kwargs):
        process = owned.prepare(*args, **kwargs)
        def refused_start():
            assert process.stop(timeout=3) is True
            return False
        process.start = refused_start
        return process
    monkeypatch.setattr(probe, "prepare_supervised", prepare)
    with pytest.raises(probe.ControllerProbeHOLD, match="stopped"):
        _run(owned, f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')")
    assert not marker.exists()
    assert owned.handles[0]._started is False
    assert owned.settled[0][1]["outcome"] == "interrupted"
    assert not owned.holds, "known interruption is not an invented closure uncertainty"


def test_actual_ready_uncertainty_is_held_without_second_stop_or_settlement(owned, monkeypatch):
    stops = []
    def prepare(*args, **kwargs):
        process = owned.prepare(*args, **kwargs)
        process.stop = lambda *a, **k: stops.append(True)
        process._fail("injected_probe_ready_confirmation_loss")
    monkeypatch.setattr(probe, "prepare_supervised", prepare)
    with pytest.raises(probe.ControllerProbeHOLD, match="execution_unconfirmed"):
        _run(owned, "pass")
    assert len(owned.handles) == 1 and owned.handles[0]._started is False
    assert stops == [] and owned.settled == []
    assert owned.holds == ["controller_probe_execution_unconfirmed"]


def test_invalid_utf8_never_becomes_empty_usable_inventory(owned):
    with pytest.raises(probe.ControllerProbeHOLD, match="execution_unconfirmed"):
        _run(owned, "import os; os.write(1,b'\\xff')", text=True)
    assert owned.settled == []
    assert owned.holds == ["controller_probe_execution_unconfirmed"]


@pytest.mark.parametrize("options", [{"shell": True}, {"timeout": None}, {"input": b"data"}])
def test_unsupported_registered_options_refuse_before_any_process(owned, options):
    with pytest.raises((TypeError, ValueError)):
        _run(owned, "pass", **options)
    assert owned.handles == [] and owned.settled == []


def test_remote_fleet_probe_is_refused_before_ssh_in_controller_context(owned, monkeypatch):
    from orze.engine import gpu_slots
    run = Mock(side_effect=AssertionError("no SSH process permitted"))
    monkeypatch.setattr(gpu_slots.subprocess, "run", run)
    with pytest.raises(probe.ControllerProbeHOLD, match="remote_probe"):
        gpu_slots.poll_fleet(["remote-node.invalid"])
    run.assert_not_called()
    assert owned.holds == ["controller_remote_probe_unsupported"]
