"""Public upgrade variants must not bypass unresolved controller stop.

Only the package/process boundaries are synthetic: real CLI parsing/dispatch
and real temporary PID/request files are used. No package manager, real signal,
host process discovery or replacement controller is ever invoked.
"""
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import orze.cli as cli


@pytest.mark.parametrize("flags", [(), ("--no-reinstall",), ("--no-restart",)],
                         ids=["default", "no-reinstall", "no-restart"])
def test_upgrade_requires_stop_before_any_mutation(tmp_path, monkeypatch, flags):
    results = tmp_path / "results"
    results.mkdir()
    control = tmp_path / ".orze"
    daemon_pid = control / "state" / "daemon.pid"
    daemon_pid.parent.mkdir(parents=True)
    daemon_pid.write_text("850001", encoding="utf-8")
    pid_bytes = daemon_pid.read_bytes()
    cfg = {"results_dir": str(results), "_orze_dir": str(control),
           "_config_path": str(tmp_path / "orze.yaml")}
    package_calls, signals, replacements = [], [], []
    original_spec = importlib.util.find_spec

    def package_spec(name, *args, **kwargs):
        if name == "orze":
            return SimpleNamespace(submodule_search_locations=[
                str(tmp_path / "site-packages" / "orze")])
        if name == "orze_pro":
            return None
        return original_spec(name, *args, **kwargs)

    def package_run(command, *args, **kwargs):
        assert command[:4] == [sys.executable, "-m", "pip", "install"]
        package_calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def kill(pid, sig):
        assert pid == 850001, "unexpected synthetic PID"
        if sig != 0:
            signals.append((pid, sig))
        # Synthetic owner stays alive even after TERM/KILL; never call real OS.

    def popen(command, *args, **kwargs):
        replacements.append(list(command))
        return SimpleNamespace(pid=850002, returncode=None, poll=lambda: None)

    os_double = SimpleNamespace(**vars(os))
    os_double.kill = kill
    os_double.killpg = lambda *args: pytest.fail("unexpected group signal")
    os_double.execv = lambda *args: pytest.fail("unexpected foreground exec")
    monkeypatch.setattr(cli, "os", os_double)
    monkeypatch.setattr("sys.argv", ["orze", "upgrade", *flags])
    monkeypatch.setattr(cli, "load_project_config", lambda path: dict(cfg))
    monkeypatch.setattr("orze.extensions._find_pro_key", lambda: "fixture-only")
    monkeypatch.setattr("orze.extensions.get_extension", lambda name: None)
    monkeypatch.setattr(importlib.util, "find_spec", package_spec)
    # The upgrade branch imports subprocess/time locally; intercept those
    # actual module methods, not an unused cli.subprocess fixture seam.
    monkeypatch.setattr(subprocess, "run", package_run)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.chdir(tmp_path)

    result = cli.main()

    assert package_calls == [], "unconfirmed stop reached package mutation"
    assert signals == [], "unconfirmed stop signaled a bare PID"
    assert daemon_pid.exists(), "unconfirmed stop removed controller identity"
    assert daemon_pid.read_bytes() == pid_bytes
    assert replacements == [], "unconfirmed stop admitted a new controller"
    assert type(result) is int and result != 0
    assert (results / ".orze_disabled").is_file()
    assert (results / ".orze_stop_all").read_bytes() == b"kill"
