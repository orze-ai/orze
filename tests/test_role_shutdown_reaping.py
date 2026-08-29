"""Every managed-role exit path reaps exact detached descendants."""

from __future__ import annotations

import os
import hashlib
import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orze.engine import lifecycle
from orze.engine.process import (
    RoleProcess,
    capture_process_identity,
    process_is_running,
    reconcile_orphaned_role_receipts,
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


def _receipt(lock_dir: Path, root: dict, *, host: str | None = None,
             nonce_sha256: str = "a" * 64) -> Path:
    lock_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "role_name": lock_dir.name,
        "host": host or socket.gethostname(),
        "controller": {"pid": 99999999, "pgid": 99999999,
                       "start_ticks": 1},
        "root": root,
        "nonce_sha256": nonce_sha256,
        "descendants": [],
        "updated_ns": time.time_ns(),
    }
    path = lock_dir / "role-process.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_crash_recovery_reaps_nonce_bound_root_and_setsid_child(tmp_path):
    orze_dir = tmp_path / ".orze"
    lock_dir = orze_dir / "locks" / "engineer"
    child_path = tmp_path / "crash-child.pid"
    root_path = tmp_path / "crash-root.pid"
    role_script = tmp_path / "crash-role.py"
    role_script.write_text(
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)'], start_new_session=True)\n"
        "Path(os.environ['CHILD_PATH']).write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    controller_script = tmp_path / "crash-controller.py"
    controller_script.write_text(
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "from orze.engine.process import RoleProcess\n"
        "lock = Path(os.environ['LOCK_DIR']); lock.mkdir(parents=True)\n"
        "(lock / 'lock.json').write_text(json.dumps({"
        "'host': os.uname().nodename, 'pid': os.getpid(), 'time': time.time()}))\n"
        "nonce = os.environ['ROLE_NONCE']\n"
        "env = dict(os.environ); env['ORZE_ROLE_PROCESS_NONCE'] = nonce\n"
        "proc = subprocess.Popen([sys.executable, os.environ['ROLE_SCRIPT']], "
        "env=env, start_new_session=True, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL)\n"
        "deadline = time.time() + 3\n"
        "while not Path(os.environ['CHILD_PATH']).exists() and time.time() < deadline: "
        "time.sleep(0.01)\n"
        "Path(os.environ['ROOT_PATH']).write_text(str(proc.pid))\n"
        "RoleProcess(role_name='engineer', process=proc, start_time=time.time(), "
        "log_path=Path(os.devnull), timeout=60, lock_dir=lock, cycle_num=1, "
        "writes_ideas_file=False, process_nonce=nonce)\n"
        "os._exit(0)\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    nonce = hashlib.sha256(str(lock_dir).encode("utf-8")).hexdigest()
    env.update({
        "LOCK_DIR": str(lock_dir),
        "ROLE_SCRIPT": str(role_script),
        "CHILD_PATH": str(child_path),
        "ROOT_PATH": str(root_path),
        "ROLE_NONCE": nonce,
    })
    source_root = str(Path(__file__).parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        [source_root, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    controller = subprocess.run(
        [sys.executable, str(controller_script)], env=env,
        capture_output=True, text=True, timeout=10)
    assert controller.returncode == 0, controller.stderr
    root_pid = int(root_path.read_text())
    child_pid = int(child_path.read_text())
    receipt_path = lock_dir / "role-process.json"
    receipt_text = receipt_path.read_text(encoding="utf-8")
    assert nonce not in receipt_text
    assert receipt_path.stat().st_mode & 0o077 == 0
    try:
        report = reconcile_orphaned_role_receipts(
            orze_dir, socket.gethostname(), timeout=0.2)

        assert report == {"recovered": ["engineer"], "remote": [],
                          "errors": []}
        assert not process_is_running(root_pid)
        assert not process_is_running(child_pid)
        assert not lock_dir.exists()
    finally:
        for pid in (root_pid, child_pid):
            if process_is_running(pid):
                os.kill(pid, signal.SIGKILL)


def test_missing_launch_nonce_attestation_kills_role(tmp_path):
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True)
    try:
        with pytest.raises(
                RuntimeError, match="role_process_receipt_initialization_failed"):
            RoleProcess(
                role_name="engineer", process=proc, start_time=time.time(),
                log_path=tmp_path / "role.log", timeout=60,
                lock_dir=tmp_path / "role.lock", cycle_num=1,
                writes_ideas_file=False, process_nonce="d" * 64,
            )
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def test_nonce_mismatch_never_authorizes_signal(tmp_path):
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        root = capture_process_identity(sleeper.pid)
        _receipt(tmp_path / ".orze" / "locks" / "engineer", root)

        report = reconcile_orphaned_role_receipts(
            tmp_path / ".orze", socket.gethostname(), timeout=0.1)

        assert report["errors"] == ["role_receipt_nonce_mismatch:engineer"]
        assert process_is_running(sleeper.pid, root["start_ticks"])
    finally:
        sleeper.kill()
        sleeper.wait()


def test_nonce_scan_recovers_unobserved_fast_escape(tmp_path):
    nonce = hashlib.sha256(str(tmp_path).encode("utf-8")).hexdigest()
    env = dict(os.environ)
    env["ORZE_ROLE_PROCESS_NONCE"] = nonce
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env=env, start_new_session=True)
    identity = capture_process_identity(sleeper.pid)
    try:
        lock_dir = tmp_path / ".orze" / "locks" / "engineer"
        _receipt(
            lock_dir,
            {"pid": 99999998, "pgid": 99999998, "start_ticks": 1},
            nonce_sha256=hashlib.sha256(nonce.encode("ascii")).hexdigest(),
        )

        report = reconcile_orphaned_role_receipts(
            tmp_path / ".orze", socket.gethostname(), timeout=0.2)

        assert report["recovered"] == ["engineer"]
        assert not process_is_running(sleeper.pid, identity["start_ticks"])
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
        sleeper.wait()


def test_pid_reuse_mismatch_is_not_signaled(tmp_path):
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        root = capture_process_identity(sleeper.pid)
        stale = dict(root, start_ticks=root["start_ticks"] + 1)
        lock_dir = tmp_path / ".orze" / "locks" / "engineer"
        _receipt(lock_dir, stale)

        report = reconcile_orphaned_role_receipts(
            tmp_path / ".orze", socket.gethostname(), timeout=0.1)

        assert report["recovered"] == ["engineer"]
        assert process_is_running(sleeper.pid, root["start_ticks"])
        assert not lock_dir.exists()
    finally:
        sleeper.kill()
        sleeper.wait()


def test_remote_host_receipt_is_left_untouched(tmp_path):
    lock_dir = tmp_path / ".orze" / "locks" / "engineer"
    receipt = _receipt(
        lock_dir, {"pid": 99999998, "pgid": 99999998, "start_ticks": 1},
        host="other-host")

    report = reconcile_orphaned_role_receipts(
        tmp_path / ".orze", socket.gethostname())

    assert report == {"recovered": [], "remote": ["engineer"], "errors": []}
    assert receipt.exists()


def test_startup_fails_closed_on_invalid_local_role_receipt(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    receipt = tmp_path / ".orze" / "locks" / "engineer" / "role-process.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="role_receipt_invalid:engineer"):
        lifecycle.startup_checks(
            results, {"_orze_dir": str(tmp_path / ".orze")},
            socket.gethostname(), "instance")
