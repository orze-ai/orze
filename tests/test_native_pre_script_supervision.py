"""New C2d2 real-CPU protocol mechanisms, not historical API-absence reds.

The original dispatch fixture and its four behavior assertions stay unchanged.
Policy and handoff faults are injected only after an actual blocked READY.
STOP, supervisor loss, SQLite, claims and failure reporting remain real.
"""
import json
from pathlib import Path
import signal

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import launcher, phases, process
from orze.engine.native_pre_script import PreScriptHOLD
from orze.engine.supervised_process import SupervisedProcess, SupervisionUncertain
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_pre_script_tree_completion import (
    cpu_pre_script, _start, _finish, _alive, _wait_dead,
)


def _phase(c):
    c.runner.lake = c.lake
    return phases.OrzePhaseMixin._launch_training(c.runner, [c.idea], True,
        {c.idea: {"title": "Synthetic CPU admission", "priority": "high", "config": {"seed": 13}}})


def _marker(c, tmp_path):
    marker = tmp_path / "pre-script-user-code-ran"
    Path(c.cfg["pre_script"]).write_text(
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_bytes(b'ran')\n")
    c.cfg["pre_args"] = []
    return marker


def _capture(monkeypatch, handles, after_ready=None):
    actual = process.prepare_supervised

    def prepare(*args, **kwargs):
        child = actual(*args, **kwargs)
        assert isinstance(child, SupervisedProcess)
        assert child.poll() is None
        assert child.binding["worker"]["pid"] == child.pid
        handles.append(child)
        if after_ready is not None:
            after_ready(child)
        return child

    monkeypatch.setattr(process, "prepare_supervised", prepare)


def _held(c, state):
    row = current_attempt(c.lake.conn, c.idea, "pre_script")
    assert row["state"] == state
    assert row["terminal"] is None
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
    assert c.runner.failure_counts == {}
    assert c.events == []
    assert not list((c.folder / "_compute_receipts").glob("*/*.json"))
    return row


@pytest.mark.parametrize("policy", ["runtime", "sentinel"])
def test_ready_policy_refusal_never_goes_and_stops_owned_worker(
        cpu_pre_script, tmp_path, monkeypatch, policy):
    c = cpu_pre_script
    marker = _marker(c, tmp_path)
    handles, rejected = [], []
    original = launcher._assert_controller_runtime_attested

    def runtime(cfg):
        if handles and policy == "runtime":
            rejected.append(True)
            raise launcher.LaunchIntegrityError("fixture runtime rejected after real READY")
        return original(cfg)

    def after_ready(child):
        assert not marker.exists()
        if policy == "sentinel":
            (c.results / ".orze_stop_all").write_bytes(b"fixture stop")

    monkeypatch.setattr(launcher, "_assert_controller_runtime_attested", runtime)
    _capture(monkeypatch, handles, after_ready)
    with pytest.raises(PreScriptHOLD):
        _phase(c)
    assert len(handles) == 1
    handles[0].wait(timeout=5)
    closure = handles[0].closure_receipt()
    assert closure["stop_requested"] is True
    assert closure["wait_proof"] == "ECHILD_WALL"
    assert not marker.exists()
    assert len(c.stops) == 1 and c.stops[0]["return"] is True
    row = _held(c, "RUNNING")
    assert row["binding"]["supervision"] == handles[0].binding
    if policy == "runtime":
        assert rejected == [True]
    stopped = c.folder / "_execution_stops" / row["attempt_id"]
    assert (stopped / "requested.json").is_file()
    assert (stopped / "confirmed.json").is_file()


def test_timeout_stop_zero_is_not_success_and_never_calls_fixer(
        cpu_pre_script, tmp_path, monkeypatch):
    c = cpu_pre_script
    script = Path(c.cfg["pre_script"])
    source = script.read_text()
    assert source.count("        signal.pause()") == 1
    script.write_text(source.replace("        signal.pause()",
        "        signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))\n        signal.pause()"))
    handles = []
    _capture(monkeypatch, handles)
    _start(c, tmp_path, mode="timeout")
    assert c.done.wait(5)
    _finish(c)
    assert c.errors == []
    assert len(handles) == 1
    closure = handles[0].closure_receipt()
    assert closure["worker_returncode"] == 0
    assert closure["stop_requested"] is True
    assert not _alive(c.daemon_pidfd)
    row = current_attempt(c.lake.conn, c.idea, "pre_script")
    assert row["state"] == "TERMINAL"
    assert row["terminal"]["return_code"] == 0
    assert row["terminal"]["outcome"] != "completed"
    assert not any(name in ("training_launch", "fixer", "_reset_idea_for_retry") for name, _ in c.events)
    assert c.runner.failure_counts == {c.idea: 1}
    assert c.lake.get_fsm_state(c.idea) == "FAILED"
    claim = json.loads((c.folder / "claim.json").read_bytes())
    assert claim["attempt_id"] == c.claim["attempt_id"]
    assert not list((c.folder / "_compute_receipts").glob("*/start.json"))


def test_actual_supervisor_loss_stays_hold_without_a_second_stop_or_replay(
        cpu_pre_script, tmp_path, monkeypatch):
    c = cpu_pre_script
    handles = []
    _capture(monkeypatch, handles)
    _start(c, tmp_path, mode="escaped")
    assert len(handles) == 1
    supervisor_fd = next(fd for pid, fd in c.pidfds if pid == handles[0].supervisor_pid)
    assert _alive(supervisor_fd)
    signal.pidfd_send_signal(supervisor_fd, signal.SIGKILL)
    _wait_dead(supervisor_fd)
    assert c.done.wait(5)
    assert c.errors and all(isinstance(exc, TerminationUnconfirmed) for exc in c.errors)
    assert _alive(c.daemon_pidfd)
    _held(c, "RUNNING")
    assert c.stops == []
    roots_before = len(c.roots)
    with pytest.raises(PreScriptHOLD):
        process.run_pre_script(c.idea, 4, c.cfg, c.results, lake=c.lake)
    assert len(c.roots) == roots_before
    assert c.stops == []
    _held(c, "RUNNING")
    _finish(c)


def test_created_but_unreturned_ready_handle_does_not_become_not_started(
        cpu_pre_script, tmp_path, monkeypatch):
    c = cpu_pre_script
    marker = _marker(c, tmp_path)
    handles = []

    def handoff_uncertain(child):
        assert _alive(c.worker_pidfd)
        raise SupervisionUncertain("fixture actual READY handoff became uncertain")

    _capture(monkeypatch, handles, handoff_uncertain)
    with pytest.raises(PreScriptHOLD):
        _phase(c)
    assert len(handles) == 1
    assert not marker.exists()
    assert _alive(c.worker_pidfd)
    _held(c, "LAUNCHING")
    assert c.stops == []
    roots_before = len(c.roots)
    with pytest.raises(PreScriptHOLD):
        process.run_pre_script(c.idea, 4, c.cfg, c.results, lake=c.lake)
    assert len(c.roots) == roots_before
    assert c.stops == []
    assert not marker.exists()
