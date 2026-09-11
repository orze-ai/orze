"""C2d3 new real-CPU mechanisms, separate from the four historical cases.

Actual READY workers, streams, STOP, SQLite and failure reporting are used.
Policy/input/claim and handoff faults are explicit narrow injected boundaries.
Only captured own pidfds are signaled; no resolver, GPU or provider runs.
"""
import hashlib
import json
from pathlib import Path
import signal

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import launcher, phases, process
from orze.engine.native_artifact_preflight import ArtifactPreflightHOLD
from orze.engine.supervised_process import SupervisedProcess, SupervisionUncertain
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_artifact_preflight_tree_completion import (
    cpu_preflight, _start_preflight, _receipt,
)
from test_native_pre_script_tree_completion import (
    cpu_pre_script, _finish, _alive, _wait_dead, WORKER_SOURCE,
)
from test_native_posthoc_tree_completion import DAEMON_SOURCE


def _phase(c):
    c.runner.lake = c.lake
    return phases.OrzePhaseMixin._launch_training(c.runner, [c.idea], True,
        {c.idea: {"title": "Synthetic CPU resolver", "priority": "high",
                  "config": {"seed": 13}}})


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
    row = current_attempt(c.lake.conn, c.idea, "artifact_preflight")
    assert row["state"] == state
    assert row["terminal"] is None
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
    assert c.runner.failure_counts == {}
    assert c.events == []
    assert not (c.folder / "artifact_preflight.json").exists()
    assert not list((c.folder / "_compute_receipts").glob("*/*.json"))
    return row


def test_retained_stdout_after_leader_exit_obeys_timeout_and_closes_writer(
        cpu_preflight, tmp_path, monkeypatch):
    c = cpu_preflight
    marker = b"synthetic-retained-stdout\n"
    daemon = DAEMON_SOURCE.replace("channel = socket.socket",
        "os.write(1, " + repr(marker) + ")\nchannel = socket.socket", 1)
    worker = WORKER_SOURCE.replace("for stream_fd in (0, 1, 2):", "for stream_fd in (0,):")
    Path(c.cfg["artifact_preflight"]["script"]).write_text(
        "DAEMON_SOURCE = " + repr(daemon) + "\n" + worker)
    c.cfg["artifact_preflight"]["timeout"] = 0.5
    handles = []
    _capture(monkeypatch, handles)
    _start_preflight(c, tmp_path, mode="escaped")
    assert c.done.wait(5), "retained stdout bypassed the configured resolver deadline"
    assert not _alive(c.daemon_pidfd)
    _finish(c)
    assert c.errors == []
    assert len(handles) == 1
    closure = handles[0].closure_receipt()
    assert closure["worker_returncode"] == 0
    assert closure["stop_requested"] is True
    assert closure["wait_proof"] == "ECHILD_WALL"
    assert len(c.stops) == 1 and c.stops[0]["return"] is True
    receipt = _receipt(c)
    assert receipt["status"] == "timed_out"
    assert receipt["stdout_sha256"] == hashlib.sha256(marker).hexdigest()
    assert not any(name in {"pre_script", "training_launch", "fixer"} for name, _ in c.events)
    assert all(not live for _, live in c.events)
    assert c.runner.failure_counts == {c.idea: 1}


@pytest.mark.parametrize("fault", ["runtime", "input", "claim"])
def test_ready_drift_refuses_go_and_stops_only_the_owned_worker(
        cpu_preflight, tmp_path, monkeypatch, fault):
    c = cpu_preflight
    marker = tmp_path / "resolver-user-code-ran"
    script = Path(c.cfg["artifact_preflight"]["script"])
    script.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_bytes(b'ran')\n")
    c.cfg["artifact_preflight"]["args"] = []
    handles, rejected = [], []
    original = launcher._assert_controller_runtime_attested

    def runtime(cfg):
        if handles and fault == "runtime":
            rejected.append(True)
            raise launcher.LaunchIntegrityError("fixture runtime rejected after actual READY")
        return original(cfg)

    def after_ready(child):
        assert not marker.exists()
        if fault == "input":
            script.write_text(script.read_text() + "# changed after capture\n")
        if fault == "claim":
            path = c.folder / "claim.json"
            claim = json.loads(path.read_bytes())
            claim["gpu"] = 5
            path.write_text(json.dumps(claim) + "\n")

    monkeypatch.setattr(launcher, "_assert_controller_runtime_attested", runtime)
    _capture(monkeypatch, handles, after_ready)
    with pytest.raises(ArtifactPreflightHOLD):
        _phase(c)
    assert len(handles) == 1
    handles[0].wait(timeout=5)
    closure = handles[0].closure_receipt()
    assert closure["stop_requested"] is True
    assert closure["wait_proof"] == "ECHILD_WALL"
    assert not marker.exists()
    assert len(c.stops) == 1 and c.stops[0]["return"] is True
    row = _held(c, "RUNNING" if fault == "runtime" else "LAUNCHING")
    if fault == "runtime":
        assert rejected == [True]
        assert row["binding"]["supervision"] == handles[0].binding
    stopped = c.folder / "_execution_stops" / row["attempt_id"]
    assert (stopped / "requested.json").is_file()
    assert (stopped / "confirmed.json").is_file()


def test_timeout_sigterm_zero_never_becomes_passed(
        cpu_preflight, tmp_path, monkeypatch):
    c = cpu_preflight
    script = Path(c.cfg["artifact_preflight"]["script"])
    source = script.read_text()
    assert source.count("        signal.pause()") == 1
    script.write_text(source.replace("        signal.pause()",
        "        signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))\n        signal.pause()"))
    handles = []
    _capture(monkeypatch, handles)
    _start_preflight(c, tmp_path, mode="timeout")
    assert c.done.wait(5)
    _finish(c)
    assert c.errors == []
    assert len(handles) == 1
    closure = handles[0].closure_receipt()
    assert closure["worker_returncode"] == 0
    assert closure["stop_requested"] is True
    assert not _alive(c.daemon_pidfd)
    row = current_attempt(c.lake.conn, c.idea, "artifact_preflight")
    assert row["state"] == "TERMINAL"
    assert row["terminal"]["return_code"] == 0
    assert row["terminal"]["outcome"] != "completed"
    assert _receipt(c)["status"] == "timed_out"
    assert _receipt(c)["exit_code"] == 0
    assert not any(name in {"pre_script", "training_launch", "fixer", "_reset_idea_for_retry"} for name, _ in c.events)
    assert c.runner.failure_counts == {c.idea: 1}
    assert c.lake.get_fsm_state(c.idea) == "FAILED"


def test_actual_supervisor_loss_holds_without_second_stop_or_replay(
        cpu_preflight, tmp_path, monkeypatch):
    c = cpu_preflight
    handles = []
    _capture(monkeypatch, handles)
    _start_preflight(c, tmp_path, mode="escaped")
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
    with pytest.raises(ArtifactPreflightHOLD):
        process.run_artifact_preflight(c.idea, c.results, c.cfg, lake=c.lake)
    assert len(c.roots) == roots_before
    assert c.stops == []
    _held(c, "RUNNING")
    _finish(c)


def test_unreturned_actual_ready_is_hold_not_not_started_or_replay(
        cpu_preflight, tmp_path, monkeypatch):
    c = cpu_preflight
    marker = tmp_path / "unreturned-resolver-ran"
    Path(c.cfg["artifact_preflight"]["script"]).write_text(
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_bytes(b'ran')\n")
    c.cfg["artifact_preflight"]["args"] = []
    handles = []

    def handoff_uncertain(child):
        assert _alive(c.worker_pidfd)
        raise SupervisionUncertain("fixture actual READY handoff became uncertain")

    _capture(monkeypatch, handles, handoff_uncertain)
    with pytest.raises(ArtifactPreflightHOLD):
        _phase(c)
    assert len(handles) == 1
    assert not marker.exists()
    assert _alive(c.worker_pidfd)
    _held(c, "LAUNCHING")
    assert c.stops == []
    roots_before = len(c.roots)
    with pytest.raises(ArtifactPreflightHOLD):
        process.run_artifact_preflight(c.idea, c.results, c.cfg, lake=c.lake)
    assert len(c.roots) == roots_before
    assert c.stops == []
    assert not marker.exists()
