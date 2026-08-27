"""Every managed-role exit path reaps exact detached descendants."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orze.engine import lifecycle
from orze.engine.process import (
    RoleProcess,
    process_is_running,
    terminate_role_process,
)
from orze.engine.roles import OUTCOME_OK, check_active_roles


def _spawn_role(tmp_path: Path) -> tuple[subprocess.Popen, int, Path, object]:
    child_pid_path = tmp_path / "child.pid"
    exit_path = tmp_path / "exit"
    script = tmp_path / "role.py"
    script.write_text(
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)'], start_new_session=True)\n"
        "Path(os.environ['CHILD_PID_PATH']).write_text(str(child.pid))\n"
        "while not Path(os.environ['EXIT_PATH']).exists(): time.sleep(0.01)\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["CHILD_PID_PATH"] = str(child_pid_path)
    env["EXIT_PATH"] = str(exit_path)
    log_path = tmp_path / "role.log"
    log_fh = log_path.open("w", encoding="utf-8")
    parent = subprocess.Popen(
        [sys.executable, str(script)],
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + 3
    while not child_pid_path.exists() and time.time() < deadline:
        time.sleep(0.01)
    assert child_pid_path.exists()
    return parent, int(child_pid_path.read_text()), exit_path, log_fh


def _cleanup(parent: subprocess.Popen, child_pid: int) -> None:
    if parent.poll() is None:
        try:
            os.killpg(parent.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            parent.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    if process_is_running(child_pid):
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _role(tmp_path: Path, parent: subprocess.Popen, log_fh) -> RoleProcess:
    lock_dir = tmp_path / "role.lock"
    lock_dir.mkdir()
    return RoleProcess(
        role_name="engineer",
        process=parent,
        start_time=time.time(),
        log_path=tmp_path / "role.log",
        timeout=60,
        lock_dir=lock_dir,
        cycle_num=1,
        _log_fh=log_fh,
        writes_ideas_file=False,
    )


def test_zero_exit_cannot_leave_observed_setsid_child(tmp_path):
    parent, child_pid, exit_path, log_fh = _spawn_role(tmp_path)
    try:
        rp = _role(tmp_path, parent, log_fh)
        # Exercise the recurring poll path rather than relying only on the
        # construction-time snapshot.
        rp._tracked_descendants.clear()
        active = {"engineer": rp}
        assert check_active_roles(
            active, ideas_file=str(tmp_path / "ideas.md")) == []
        assert any(identity["pid"] == child_pid
                   for identity in rp._tracked_descendants)
        exit_path.touch()
        parent.wait(timeout=3)

        finished = check_active_roles(
            active, ideas_file=str(tmp_path / "ideas.md"))

        assert finished == [("engineer", OUTCOME_OK)]
        assert active == {}
        assert not process_is_running(child_pid)
    finally:
        _cleanup(parent, child_pid)


def test_graceful_shutdown_reaps_setsid_role_child(tmp_path, monkeypatch):
    parent, child_pid, _exit_path, log_fh = _spawn_role(tmp_path)
    try:
        rp = _role(tmp_path, parent, log_fh)
        results = tmp_path / "results"
        results.mkdir()
        monkeypatch.setattr(lifecycle, "save_state", lambda *args, **kwargs: None)
        monkeypatch.setattr(lifecycle, "notify", lambda *args, **kwargs: None)

        lifecycle.graceful_shutdown(
            results, {}, {}, {}, {"engineer": rp}, 1, {}, None,
            "host", "instance", kill_all=False,
        )

        assert parent.poll() is not None
        assert not process_is_running(child_pid)
    finally:
        _cleanup(parent, child_pid)


def test_exited_root_never_uses_unproven_recycled_process_group(monkeypatch):
    proc = MagicMock()
    proc.pid = 424242
    rp = RoleProcess(
        role_name="engineer",
        process=proc,
        start_time=time.time(),
        log_path=Path("role.log"),
        timeout=60,
        lock_dir=Path("role.lock"),
        cycle_num=1,
        _root_start_ticks=99,
        _pgid=424242,
    )
    monkeypatch.setattr(
        "orze.engine.process.process_is_running", lambda *args: False)
    calls = []

    terminate_role_process(
        rp, "finished", reaper=lambda *args, **kwargs: calls.append(kwargs))

    assert calls[0]["pgid"] is None
    assert calls[0]["discover_pgid"] is False
