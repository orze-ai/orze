"""New real-CPU bounded-fixer consumer mechanisms, not historical reds.

The existing five behavior tests are unchanged. This file observes real helper
results/errors and supervised STOP through transparent instance wrappers; it
does not substitute Popen, tree proof, pipe reads, or provider output.
"""
import os
from pathlib import Path
import signal
import subprocess

import pytest

from orze.engine import bounded_executor, failure
from orze.engine.bounded_executor import BoundedExecutorHOLD, MAX_OUTPUT_BYTES
from test_executor_fix_tree_completion import (
    cpu_executor, _start, _finish, _alive, _wait_dead, CLI_SOURCE, DAEMON_SOURCE,
)


def _capture(c, monkeypatch, after_ready=None):
    prepare = bounded_executor.prepare_supervised
    c.stop_events, c.completed_outputs, c.confirmed_timeouts = [], [], []

    def captured_prepare(*args, **kwargs):
        child = prepare(*args, **kwargs)
        c.handles.append(child)
        for pid in (child.pid, child.supervisor_pid):
            c.pidfds.append((pid, os.pidfd_open(pid)))
        assert child.poll() is None
        stop = child.stop

        def observed_stop(*args, **kwargs):
            c.stop_events.append("requested")
            result = stop(*args, **kwargs)
            c.stop_events.append(result)
            return result

        child.stop = observed_stop
        if after_ready is not None:
            after_ready(child)
        return child

    run = failure._run_bounded_executor

    def observed_run(*args, **kwargs):
        try:
            result = run(*args, **kwargs)
        except subprocess.TimeoutExpired as exc:
            c.confirmed_timeouts.append(exc)
            raise
        c.completed_outputs.append(result)
        return result

    monkeypatch.setattr(bounded_executor, "prepare_supervised", captured_prepare)
    monkeypatch.setattr(failure, "_run_bounded_executor", observed_run)


def _script(c, *, daemon=DAEMON_SOURCE, source=CLI_SOURCE):
    path = Path(c.cfg["executor_fix"]["claude_bin"])
    shebang = path.read_text().splitlines()[0]
    path.write_text(shebang + "\nDAEMON_SOURCE = " + repr(daemon) + "\n" + source)


def test_retained_stdout_is_drained_with_a_live_deadline(
        cpu_executor, tmp_path, monkeypatch):
    c = cpu_executor
    daemon = DAEMON_SOURCE.replace("channel = socket.socket",
        "os.write(1, b'synthetic-retained-stdout\\n')\nchannel = socket.socket", 1)
    source = CLI_SOURCE.replace("for stream_fd in (0, 1, 2):", "for stream_fd in (0,):")
    _script(c, daemon=daemon, source=source)
    c.cfg["executor_fix"]["timeout"] = 0.5
    _capture(c, monkeypatch)
    _start(c, tmp_path, monkeypatch, mode="escaped")
    assert c.done.wait(5), "retained stdout bypassed the configured execution deadline"
    assert not _alive(c.daemon_pidfd)
    _finish(c)
    assert c.errors == []
    assert c.result is False
    assert c.stop_events == ["requested", True]
    assert len(c.confirmed_timeouts) == 1
    error = c.confirmed_timeouts[0]
    assert error.returncode == 0
    assert error.stopped is True
    assert error.process_tree["wait_proof"] == "ECHILD_WALL"
    assert "synthetic-retained-stdout" in error.output
    assert c.counts == {c.idea: 1}


@pytest.mark.parametrize("continuous", [False, True])
def test_output_flood_cannot_authorize_fix_from_an_incomplete_tail(
        cpu_executor, tmp_path, monkeypatch, continuous):
    c = cpu_executor
    print_line = 'print("FIX_APPLIED" if marker else "NO_CHANGE", flush=True)'
    assert CLI_SOURCE.count(print_line) == 1
    if continuous:
        replacement = "import time\nwhile True:\n    os.write(1, b'x' * 8192 + b'\\nFIX_APPLIED\\n')\n    time.sleep(0.001)"
        c.cfg["executor_fix"]["timeout"] = 0.5
    else:
        replacement = "os.write(1, b'x' * " + str(MAX_OUTPUT_BYTES + 17) + " + b'\\n')\n" + print_line
    _script(c, source=CLI_SOURCE.replace(print_line, replacement))
    _capture(c, monkeypatch)
    _start(c, tmp_path, monkeypatch, mode="normal")
    assert c.done.wait(5)
    _finish(c)
    assert c.errors == []
    assert c.result is False
    assert c.counts == {c.idea: 1}
    if continuous:
        assert c.stop_events == ["requested", True]
        assert len(c.confirmed_timeouts) == 1
        output = c.confirmed_timeouts[0]
        assert output.output_complete is False
        assert output.stopped is True
        assert len(output.output.encode("utf-8")) <= MAX_OUTPUT_BYTES
    else:
        assert c.stop_events == []
        assert len(c.completed_outputs) == 1
        output = c.completed_outputs[0]
        assert output.returncode == 0
        assert output.stopped is False
        assert output.output_complete is False
        assert output.stdout_bytes > MAX_OUTPUT_BYTES
        assert len(output.stdout.encode("utf-8")) <= MAX_OUTPUT_BYTES
        assert "FIX_APPLIED" in output.stdout


def test_timeout_sigterm_zero_is_not_an_accepted_fix(
        cpu_executor, tmp_path, monkeypatch):
    c = cpu_executor
    assert CLI_SOURCE.count("            signal.pause()") == 1
    source = CLI_SOURCE.replace("            signal.pause()",
        "            signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))\n            signal.pause()")
    _script(c, source=source)
    _capture(c, monkeypatch)
    _start(c, tmp_path, monkeypatch, mode="timeout")
    assert c.done.wait(5)
    assert not _alive(c.daemon_pidfd)
    _finish(c)
    assert c.errors == []
    assert c.result is False
    assert c.stop_events == ["requested", True]
    assert len(c.confirmed_timeouts) == 1
    error = c.confirmed_timeouts[0]
    assert error.returncode == 0
    assert error.process_tree["worker_returncode"] == 0
    assert error.process_tree["stop_requested"] is True
    assert error.stopped is True
    assert c.counts == {c.idea: 1}


def test_supervisor_loss_is_hold_even_if_retry_configuration_is_disabled(
        cpu_executor, tmp_path, monkeypatch):
    c = cpu_executor
    _capture(c, monkeypatch)
    _start(c, tmp_path, monkeypatch, mode="escaped")
    assert len(c.handles) == 1
    child = c.handles[0]
    fd = next(fd for pid, fd in c.pidfds if pid == child.supervisor_pid)
    assert _alive(fd)
    signal.pidfd_send_signal(fd, signal.SIGKILL)
    _wait_dead(fd)
    assert c.done.wait(5)
    assert c.errors and all(isinstance(exc, BoundedExecutorHOLD) for exc in c.errors)
    assert c.result is None
    assert _alive(c.daemon_pidfd)
    assert c.stop_events == []
    assert c.counts == {}
    captured_before = len(c.pidfds)
    c.cfg["max_fix_attempts"] = 0
    with pytest.raises(BoundedExecutorHOLD):
        failure._try_executor_fix(c.idea, "synthetic failure", c.results, c.cfg, c.counts)
    assert len(c.pidfds) == captured_before
    assert c.stop_events == []
    _finish(c)


def test_ready_policy_change_refuses_go_with_real_blocked_worker_cleanup(
        cpu_executor, tmp_path, monkeypatch):
    c = cpu_executor
    marker = tmp_path / "fixer-user-code-ran"
    _script(c, source="from pathlib import Path\nPath(" + repr(str(marker)) + ").write_bytes(b'ran')\n")

    def changed_after_ready(child):
        assert not marker.exists()
        c.cfg["agent_tool_policy"]["enabled"] = False

    _capture(c, monkeypatch, changed_after_ready)
    with pytest.raises(BoundedExecutorHOLD):
        failure._try_executor_fix(c.idea, "synthetic failure", c.results, c.cfg, c.counts)
    assert len(c.handles) == 1
    closure = c.handles[0].closure_receipt()
    assert closure["stop_requested"] is True
    assert closure["wait_proof"] == "ECHILD_WALL"
    assert c.stop_events == ["requested", True]
    assert not marker.exists()
    assert c.counts == {}
    assert c.completed_outputs == []
    bounded_executor.require_executor_scope_clear(tmp_path)
