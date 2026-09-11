"""Public stop/restart boundaries; all process observations and signals are spies.

These tests never enumerate, signal, or launch a host process. The real public
control flow and temporary sentinel/PID files are exercised; synthetic PID
liveness is deliberately not treated as an ownership or shutdown proof.
"""
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import orze.cli as cli
import orze.cli_setup as cli_setup
import orze.lifecycle as lifecycle
from orze.engine.termination_hold import TerminationUnconfirmed


@pytest.fixture
def control(tmp_path, monkeypatch):
    project = tmp_path / "project"
    results = project / "results"
    results.mkdir(parents=True)
    monkeypatch.chdir(project)
    state = SimpleNamespace(
        cfg={"results_dir": str(results),
             "_config_path": str(project / "orze.yaml"),
             "gpu_scheduling": {"allowed_gpus": []}},
        results=results, pid=810001, alive=True, probe_error=None,
        listed=[], cmdlines={}, signals=[], launches=[], commands=[],
    )
    original_read = Path.read_bytes

    def read_bytes(path):
        if str(path).startswith("/proc/"):
            assert str(path) in state.cmdlines, "unexpected process read"
            return state.cmdlines[str(path)]
        return original_read(path)

    def kill(pid, sig):
        assert pid == state.pid, "unregistered synthetic PID"
        if sig == 0:
            if state.probe_error is not None:
                raise state.probe_error("synthetic probe is inconclusive")
            if not state.alive:
                raise ProcessLookupError(pid)
            return
        state.signals.append(("pid", pid, sig))

    def getpgid(pid):
        assert pid == state.pid, "unregistered synthetic PID"
        return pid + 10

    def killpg(pgid, sig):
        assert pgid == state.pid + 10, "unregistered synthetic PGID"
        state.signals.append(("pgid", pgid, sig))

    def run(cmd, **kwargs):
        state.commands.append(list(cmd))
        if cmd[0] == "pgrep":
            stdout = "\n".join(str(pid) for pid in state.listed)
        elif cmd[0] in {"ps", "nvidia-smi"}:
            stdout = ""
        else:
            pytest.fail("unexpected external command: " + repr(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    def popen(cmd, **kwargs):
        state.launches.append(list(cmd))
        return SimpleNamespace(pid=820001, returncode=None, poll=lambda: None)

    os_double = SimpleNamespace(**vars(os))
    os_double.kill = kill
    os_double.getpgid = getpgid
    os_double.killpg = killpg
    os_double.execv = lambda *args: pytest.fail("unexpected foreground exec")
    subprocess_double = SimpleNamespace(**vars(subprocess))
    subprocess_double.run = run
    subprocess_double.Popen = popen
    time_double = SimpleNamespace(**vars(time))
    time_double.sleep = lambda seconds: None
    monkeypatch.setattr(lifecycle, "os", os_double)
    monkeypatch.setattr(cli_setup, "os", os_double)
    monkeypatch.setattr(lifecycle, "subprocess", subprocess_double)
    monkeypatch.setattr(lifecycle, "time", time_double)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return state


def stop_or_hold(call):
    # Existing typed uncertainty is a valid refusal, not successful shutdown.
    try:
        return call()
    except TerminationUnconfirmed:
        return None


def test_missing_config_path_cannot_authorize_foreign_process(control):
    control.cfg.pop("_config_path")
    control.listed = [control.pid]
    control.cmdlines[f"/proc/{control.pid}/cmdline"] = (
        b"/usr/bin/python\0-m\0orze.cli\0-c\0/foreign/project/orze.yaml\0")

    stop_or_hold(lambda: lifecycle.do_stop(control.cfg, timeout=0))

    assert control.signals == [], "empty config path authorized foreign PID"
    assert control.launches == []
    assert (control.results / ".orze_stop_all").read_text() == "kill"


def test_bare_pid_file_is_not_owned_process_authority(control):
    pid_file = control.results / ".orze.pid"
    pid_file.write_text(str(control.pid), encoding="utf-8")

    stop_or_hold(lambda: lifecycle.do_stop(control.cfg, timeout=0))

    assert control.signals == [], "unbound integer PID authorized signaling"
    assert pid_file.read_text() == str(control.pid)
    assert (control.results / ".orze_disabled").exists()


def test_inconclusive_stop_cannot_restart_or_clear_sentinels(control):
    (control.results / ".orze.pid").write_text(str(control.pid))
    control.probe_error = PermissionError
    shutdown = control.results / ".orze_shutdown"
    shutdown.write_text("operator shutdown", encoding="utf-8")

    stop_or_hold(lambda: lifecycle.do_restart(control.cfg, timeout=0))

    assert control.launches == [], "unknown old controller admitted replacement"
    assert control.signals == []
    assert (control.results / ".orze_disabled").exists()
    assert (control.results / ".orze_stop_all").read_text() == "kill"
    assert shutdown.read_text() == "operator shutdown"


def test_stop_without_discovered_process_still_requests_cooperative_stop(control):
    stop_or_hold(lambda: lifecycle.do_stop(control.cfg, timeout=0))

    assert control.signals == []
    assert control.launches == []
    assert (control.results / ".orze_disabled").is_file()
    assert (control.results / ".orze_stop_all").read_text() == "kill"


def test_restart_flag_cannot_continue_after_bare_pid_stop(control, monkeypatch):
    # Actual CLI -> actual cli_setup.stop_running_instance, not a stop stub.
    # Only final orchestrator construction/run is a launch-boundary spy.
    (control.results / ".orze.pid").write_text(str(control.pid))

    class Runner:
        def __init__(self, gpu_ids, cfg, once=False):
            control.launches.append(("controller", tuple(gpu_ids)))

        def run(self):
            control.launches.append(("run",))

    monkeypatch.setattr("sys.argv", [
        "orze", "--restart", "--once", "--no-admin", "--gpus", "0"])
    monkeypatch.setattr("orze.extensions._find_pro_key", lambda: "fixture-only")
    monkeypatch.setattr("orze.extensions.get_extension", lambda name: None)
    monkeypatch.setattr(cli, "load_project_config", lambda path: dict(control.cfg))
    monkeypatch.setattr(
        cli, "detect_all_gpus", lambda: pytest.fail("no GPU inventory"))
    monkeypatch.setattr("orze.engine.orchestrator.Orze", Runner)
    # cli_setup's legacy local import uses this clock; no process wait occurs.
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    stop_or_hold(cli.main)

    assert control.launches == [], "--restart ignored unconfirmed stop"
    assert control.signals == [], "bare PID must not trigger TERM/KILL"
    assert (control.results / ".orze.pid").read_text() == str(control.pid)
