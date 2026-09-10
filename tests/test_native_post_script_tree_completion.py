"""C2d1 public source-bound actions with real, owned tiny CPU processes.

The accepted source uses the existing explicit evaluation OS double; its
SQLite, effect and compute receipts are real. Only the target post-script
uses real Popen/supervision. These actions are not scientific observations.
The controller runs on a separate thread with its own real existing-Lake
connection so the test can observe the synchronous public API while blocked.
No process discovery or kill is based on a host scan, environment, or PGID.
"""
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import threading

import pytest

from orze.core.evaluation_retry_state import open_existing_lake
from orze.core.execution_attempts import current_attempt
from orze.engine import evaluator, process
from orze.engine.supervised_process import prepare_supervised as REAL_PREPARE
from test_completion_event_authority import accepted
from test_stale_evaluation_completion import project
from test_native_training_tree_completion import _alive, _wait_dead
from test_native_posthoc_tree_completion import DAEMON_SOURCE


REAL_POPEN = subprocess.Popen
WORKER_SOURCE = r'''
import json, os, socket, sys
endpoint, path, detached, code = sys.argv[1:]
descriptor = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
os.write(descriptor, b"synthetic-post-script-output")
os.fsync(descriptor)
if detached == "no":
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    channel.settimeout(20)
    channel.connect(endpoint)
    channel.sendall((json.dumps({"pid": os.getpid(), "pgid": os.getpgrp(),
        "inode": os.fstat(descriptor).st_ino, "path": path}) + "\n").encode())
    assert channel.recv(1) == b"L"
    channel.sendall(b"R")
    channel.close()
    os.close(descriptor)
    sys.exit(int(code))
read_end, write_end = os.pipe()
os.set_inheritable(write_end, True)
os.set_inheritable(descriptor, True)
middle = os.fork()
if middle:
    os.close(write_end)
    os.close(descriptor)
    assert os.read(read_end, 1) == b"L"
    os.close(read_end)
    os.waitpid(middle, 0)
    os._exit(int(code))
os.close(read_end)
os.setsid()
if os.fork():
    os._exit(0)
os.execve(sys.executable, [sys.executable, "-c", DAEMON_SOURCE,
    endpoint, str(descriptor), str(write_end), path], {})
'''


@pytest.fixture
def cpu_post_script(project, tmp_path, monkeypatch):
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        pytest.skip("requires Linux pidfds for exact own-process cleanup")
    c = project
    c.folder, c.event = accepted(c, "idea-cpu-post-script")
    c.cfg.update(python=sys.executable,
                 train_extra_env={"ORZE_CPU_TREE_TEST": "must-not-survive-exec"})
    c.roots, c.pidfds, c.channels = [], [], []
    c.daemon_pid = c.daemon_pidfd = c.worker_pid = c.worker_pidfd = None
    c.errors, c.publications = [], []
    c.done = threading.Event()
    c.thread = None
    c.channel = None
    c.script = tmp_path / "tiny_post_script.py"
    c.script.write_text("DAEMON_SOURCE = " + repr(DAEMON_SOURCE) + "\n" + WORKER_SOURCE)
    c.output = c.folder / "post-script.bin"
    c.next_marker = c.folder / "next-script-ran"
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0

    def popen(*args, **kwargs):
        child = REAL_POPEN(*args, **kwargs)
        c.roots.append(child)
        fd = os.pidfd_open(child.pid)
        c.pidfds.append((child.pid, fd))
        if c.worker_pid is None:
            c.worker_pid, c.worker_pidfd = child.pid, fd
        return child

    def prepare(*args, **kwargs):
        child = REAL_PREPARE(*args, **kwargs)
        # Capture the actual worker at READY, before GO, not the supervisor.
        fd = os.pidfd_open(child.pid)
        c.pidfds.append((child.pid, fd))
        if len(c.roots) == 1:
            c.worker_pid, c.worker_pidfd = child.pid, fd
        return child

    record_terminal = evaluator.record_compute_terminal
    def observe_terminal(*args, **kwargs):
        c.publications.append(c.daemon_pidfd is not None and _alive(c.daemon_pidfd))
        return record_terminal(*args, **kwargs)

    monkeypatch.setattr(evaluator.subprocess, "Popen", popen)
    monkeypatch.setattr(evaluator, "prepare_supervised", prepare)
    monkeypatch.setattr(evaluator, "_terminate_and_reap", process._terminate_and_reap)
    monkeypatch.setattr(evaluator, "record_compute_terminal", observe_terminal)
    try:
        yield c
    finally:
        if c.channel is not None and c.daemon_pidfd is not None and _alive(c.daemon_pidfd):
            try:
                c.channel.sendall(b"Q")
            except OSError:
                pass
        for channel in c.channels:
            channel.close()
        for _, fd in c.pidfds:
            if _alive(fd):
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            _wait_dead(fd)
        if c.thread is not None:
            c.thread.join(timeout=5)
            assert not c.thread.is_alive(), "owned post-script controller did not finish"
        for child in c.roots:
            child.wait(timeout=5)
        if c.daemon_pid is not None:
            try:
                os.waitpid(c.daemon_pid, 0)
            except ChildProcessError:
                pass  # A real product supervisor may already have reaped it.
        for _, fd in c.pidfds:
            os.close(fd)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def _start(c, tmp_path, *, detached, code=0, second=False):
    endpoint = tmp_path / "post-script.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(endpoint))
    listener.listen(1)
    listener.settimeout(5)
    c.channels.append(listener)
    c.cfg["post_scripts"] = [{"name": "first", "script": str(c.script),
        "args": [str(endpoint), str(c.output), "yes" if detached else "no", str(code)],
        "timeout": 20}]
    if second:
        next_script = tmp_path / "next_post_script.py"
        next_script.write_text("from pathlib import Path\nPath(" + repr(str(c.next_marker))
                               + ").write_bytes(b'next')\n")
        c.cfg["post_scripts"].append({"name": "second", "script": str(next_script), "timeout": 20})

    def controller():
        lake = None
        try:
            lake = open_existing_lake(c.lake.db_path)
            evaluator.run_post_scripts(c.folder.name, 0, c.results, c.cfg,
                                       lake=lake, source_event=c.event)
        except BaseException as exc:
            c.errors.append(exc)
        finally:
            if lake is not None:
                lake.close()
            c.done.set()

    c.thread = threading.Thread(target=controller, daemon=True)
    c.thread.start()
    channel, _ = listener.accept()
    channel.settimeout(5)
    c.channels.append(channel)
    c.channel = channel
    peer_pid, peer_uid, _ = struct.unpack("3i", channel.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    assert peer_uid == os.geteuid()
    peer_fd = os.pidfd_open(peer_pid)
    c.pidfds.append((peer_pid, peer_fd))
    raw = bytearray()
    while not raw.endswith(b"\n"):
        part = channel.recv(1024)
        assert part, "owned worker ended private handshake"
        raw.extend(part)
        assert len(raw) < 4096
    detail = json.loads(raw)
    assert detail["pid"] == peer_pid
    assert Path(detail["path"]) == c.output
    assert detail["inode"] == c.output.stat().st_ino
    if detached:
        c.daemon_pid, c.daemon_pidfd = peer_pid, peer_fd
        assert detail["pgid"] != c.worker_pid
        assert detail["inherited_orze_keys"] == []
        assert _alive(peer_fd)
    else:
        assert peer_pid == c.worker_pid
    channel.sendall(b"L")
    assert channel.recv(1) == b"R"
    _wait_dead(c.worker_pidfd)
    return c


def _finish(c):
    if c.daemon_pidfd is not None:
        c.channel.sendall(b"Q")
        _wait_dead(c.daemon_pidfd)
    assert c.done.wait(5), "post-script did not finish after actual tree closure"
    c.thread.join(timeout=5)
    assert c.errors == []


def test_source_action_does_not_publish_terminal_with_live_writer(cpu_post_script, tmp_path):
    c = _start(cpu_post_script, tmp_path, detached=True)
    # Baseline returns quickly; a correct synchronous action remains blocked.
    c.done.wait(2)
    c.channel.sendall(b"W")
    assert c.channel.recv(1) == b"W"
    assert c.output.read_bytes() == b"late posthoc writer"
    assert _alive(c.daemon_pidfd)
    row = current_attempt(c.lake.conn, c.folder.name, "post_script")
    terminal_exists = (c.folder / "_compute_receipts" / row["attempt_id"] / "terminal.json").exists()
    state_while_live = row["state"]
    _finish(c)
    assert not terminal_exists, "source-bound action published compute while its writer was live"
    assert state_while_live == "RUNNING"
    assert not any(c.publications)
    row = current_attempt(c.lake.conn, c.folder.name, "post_script")
    assert row["state"] == "TERMINAL"
    assert row["terminal"]["outcome"] == "completed"
    assert c.lake.get_fsm_state(c.folder.name) == "FAILED"


def test_next_source_action_waits_for_previous_owned_tree(cpu_post_script, tmp_path):
    c = _start(cpu_post_script, tmp_path, detached=True, second=True)
    c.done.wait(2)
    assert _alive(c.daemon_pidfd)
    ran_while_live = c.next_marker.exists()
    _finish(c)
    assert not ran_while_live, "run_post_scripts launched the next action before owned tree closure"
    assert c.next_marker.read_bytes() == b"next"
    assert not any(c.publications)
    row = current_attempt(c.lake.conn, c.folder.name, "post_script")
    assert row["state"] == "TERMINAL"
    assert row["generation"] == 2
    assert c.lake.get_fsm_state(c.folder.name) == "FAILED"


@pytest.mark.parametrize("code", [0, 7])
def test_normal_source_action_preserves_real_exit_code(cpu_post_script, tmp_path, code):
    c = _start(cpu_post_script, tmp_path, detached=False, code=code)
    _finish(c)
    row = current_attempt(c.lake.conn, c.folder.name, "post_script")
    assert row["state"] == "TERMINAL"
    assert row["terminal"]["return_code"] == code
    assert row["terminal"]["outcome"] == ("completed" if code == 0 else "failed")
    receipt = json.loads((c.folder / "_compute_receipts" / row["attempt_id"] / "terminal.json").read_bytes())
    assert receipt["return_code"] == code
    assert receipt["process_pid"] == c.worker_pid
    assert c.lake.get_fsm_state(c.folder.name) == "FAILED"
