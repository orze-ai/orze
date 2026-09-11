"""Existing fixer bool boundary with a real, temporary CPU stand-in CLI.

cfg.executor_fix.claude_bin selects the shebang program: actual command policy,
_try_executor_fix, _run_bounded_executor and Popen all run. There is no provider,
native Lake, fabricated source reference, GPU or model workload. A private
SO_PEERCRED handshake registers each worker before it may fork. Cleanup uses
only authenticated own pidfds and restores the test process's subreaper state.
"""
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import struct
import sys
import threading
from types import SimpleNamespace

import pytest

from orze.engine import failure, process
from orze.engine.supervised_process import prepare_supervised as REAL_PREPARE
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_posthoc_tree_completion import DAEMON_SOURCE
from test_native_training_tree_completion import _alive, _wait_dead


CLI_SOURCE = r'''
import json, os, signal, socket, sys
endpoint = os.environ["ORZE_FIXTURE_SOCKET"]
path = os.environ["ORZE_FIXTURE_OUTPUT"]
mode = os.environ["ORZE_FIXTURE_MODE"]
code = int(os.environ["ORZE_FIXTURE_RET"])
marker = os.environ["ORZE_FIXTURE_MARKER"] == "1"
descriptor = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
os.write(descriptor, b"synthetic-fixer-change")
os.fsync(descriptor)
channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
channel.settimeout(20)
channel.connect(endpoint)
channel.sendall((json.dumps({"kind": "worker", "pid": os.getpid(),
    "pgid": os.getpgrp(), "inode": os.fstat(descriptor).st_ino,
    "path": path}) + "\n").encode())
# No descendants exist until the parent has captured this actual worker pidfd.
assert channel.recv(1) == b"L"
channel.sendall(b"R")
channel.close()
if mode != "normal":
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
        if mode == "timeout":
            signal.pause()
    else:
        os.close(read_end)
        os.setsid()
        if os.fork():
            os._exit(0)
        for stream_fd in (0, 1, 2):
            try:
                os.close(stream_fd)
            except OSError:
                pass
        os.execve(sys.executable, [sys.executable, "-c", DAEMON_SOURCE,
            endpoint, str(descriptor), str(write_end), path], {})
else:
    os.close(descriptor)
print("FIX_APPLIED" if marker else "NO_CHANGE", flush=True)
sys.exit(code)
'''


@pytest.fixture
def cpu_executor(tmp_path, monkeypatch):
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        pytest.skip("requires Linux pidfds for exact own-process cleanup")
    results = tmp_path / "results"
    folder = results / "idea-cpu-fixer"
    folder.mkdir(parents=True)
    (folder / "train_output.log").write_text("synthetic runtime failure\n")
    inbox = tmp_path / "ideas.md"
    inbox.write_text("# Ideas\n")
    cli = tmp_path / "cpu-fixer-cli"
    cli.write_text("#!" + sys.executable + "\nDAEMON_SOURCE = " + repr(DAEMON_SOURCE) + "\n" + CLI_SOURCE)
    cli.chmod(0o700)
    cfg = {"max_fix_attempts": 1, "ideas_file": str(inbox),
           "train_script": str(tmp_path / "unused_train.py"),
           "sealed_files": [], "_project_root": str(tmp_path),
           "agent_tool_policy": {"enabled": True},
           "executor_fix": {"claude_bin": str(cli), "timeout": 10, "max_turns": 1}}
    c = SimpleNamespace(cfg=cfg, results=results, folder=folder, idea=folder.name,
        output=tmp_path / "project-output.bin", pidfds=[], handles=[], channels=[],
        worker_pid=None, worker_pidfd=None, daemon_pid=None, daemon_pidfd=None,
        daemon_channel=None, counts={}, errors=[], stops=[], result=None,
        done=threading.Event(), thread=None)
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "none")
    actual_reaper = process._terminate_and_reap

    def reap(*args, **kwargs):
        result = actual_reaper(*args, **kwargs)
        c.stops.append({"return": result, "writer_alive_after_return":
            c.daemon_pidfd is not None and _alive(c.daemon_pidfd)})
        return result

    def prepare(*args, **kwargs):
        child = REAL_PREPARE(*args, **kwargs)
        c.handles.append(child)
        for pid in (child.pid, child.supervisor_pid):
            c.pidfds.append((pid, os.pidfd_open(pid)))
        return child

    monkeypatch.setattr(process, "_terminate_and_reap", reap)
    # Forward-compatible transparent real READY capture for the agreed future
    # helper seam. Old execution ignores it. Popen is never replaced.
    monkeypatch.setattr(failure, "prepare_supervised", prepare, raising=False)
    try:
        yield c
    finally:
        if c.daemon_channel is not None and _alive(c.daemon_pidfd):
            try:
                c.daemon_channel.sendall(b"Q")
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
            assert not c.thread.is_alive(), "owned fixer controller did not finish"
        for pid in {pid for pid, _ in c.pidfds}:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        for child in c.handles:
            child._close_descriptors()
        for _, fd in c.pidfds:
            os.close(fd)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def _accept_owned(c, listener):
    channel, _ = listener.accept()
    channel.settimeout(5)
    c.channels.append(channel)
    pid, uid, _ = struct.unpack("3i", channel.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    assert uid == os.geteuid()
    fd = os.pidfd_open(pid)
    c.pidfds.append((pid, fd))
    raw = bytearray()
    while not raw.endswith(b"\n"):
        part = channel.recv(1024)
        assert part, "own CPU CLI ended private handshake"
        raw.extend(part)
        assert len(raw) < 4096
    detail = json.loads(raw)
    assert detail["pid"] == pid
    assert Path(detail["path"]) == c.output
    assert detail["inode"] == c.output.stat().st_ino
    return channel, pid, fd, detail


def _start(c, tmp_path, monkeypatch, *, mode, code=0, marker=True):
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    endpoint = tmp_path / "fixer.sock"
    listener.bind(str(endpoint))
    listener.listen(2)
    listener.settimeout(5)
    c.channels.append(listener)
    for key, value in {"SOCKET": str(endpoint), "OUTPUT": str(c.output),
                       "MODE": mode, "RET": str(code), "MARKER": "1" if marker else "0"}.items():
        monkeypatch.setenv("ORZE_FIXTURE_" + key, value)
    if mode == "timeout":
        c.cfg["executor_fix"]["timeout"] = 1

    def controller():
        try:
            c.result = failure._try_executor_fix(c.idea, "synthetic runtime failure",
                                                c.results, c.cfg, c.counts)
        except BaseException as exc:
            c.errors.append(exc)
        finally:
            c.done.set()

    c.thread = threading.Thread(target=controller, daemon=True)
    c.thread.start()
    channel, c.worker_pid, c.worker_pidfd, detail = _accept_owned(c, listener)
    assert detail["kind"] == "worker"
    channel.sendall(b"L")
    assert channel.recv(1) == b"R"
    if mode != "normal":
        channel, c.daemon_pid, c.daemon_pidfd, detail = _accept_owned(c, listener)
        c.daemon_channel = channel
        assert detail["pgid"] != c.worker_pid
        assert detail["inherited_orze_keys"] == []
        assert _alive(c.daemon_pidfd)
        channel.sendall(b"L")
        assert channel.recv(1) == b"R"
    if mode != "timeout":
        _wait_dead(c.worker_pidfd)
    return c


def _finish(c):
    if c.daemon_pidfd is not None and _alive(c.daemon_pidfd):
        c.daemon_channel.sendall(b"Q")
        _wait_dead(c.daemon_pidfd)
    assert c.done.wait(5), "fixer did not finish after the owned tree closed"
    c.thread.join(timeout=5)


def test_fix_applied_is_not_accepted_while_escaped_writer_remains(
        cpu_executor, tmp_path, monkeypatch):
    c = _start(cpu_executor, tmp_path, monkeypatch, mode="escaped")
    c.done.wait(2)
    c.daemon_channel.sendall(b"W")
    assert c.daemon_channel.recv(1) == b"W"
    assert c.output.read_bytes() == b"late posthoc writer"
    assert _alive(c.daemon_pidfd)
    accepted_while_live = c.done.is_set() and c.result is True
    _finish(c)
    assert not accepted_while_live, "FIX_APPLIED authorized retry while the fixer writer was live"
    assert c.errors == []
    assert c.result is True
    assert c.counts == {c.idea: 1}


def test_fixer_timeout_does_not_downgrade_live_writer_to_ordinary_false(
        cpu_executor, tmp_path, monkeypatch):
    c = _start(cpu_executor, tmp_path, monkeypatch, mode="timeout")
    assert c.done.wait(5), "bounded fixer timeout did not return or hold"
    ordinary_false_while_live = _alive(c.daemon_pidfd) and c.result is False
    _finish(c)
    assert not ordinary_false_while_live, (
        "uncertain fixer cleanup became an ordinary False result", c.stops)
    assert all(isinstance(exc, TerminationUnconfirmed) for exc in c.errors)
    assert c.result is not True


@pytest.mark.parametrize("code,marker,accepted", [(0, True, True), (7, True, False), (0, False, False)])
def test_normal_cpu_cli_controls_keep_exit_and_marker_contract(
        cpu_executor, tmp_path, monkeypatch, code, marker, accepted):
    c = _start(cpu_executor, tmp_path, monkeypatch, mode="normal", code=code, marker=marker)
    _finish(c)
    assert c.errors == []
    assert c.result is accepted
    assert c.counts == {c.idea: 1}
    assert c.output.read_bytes() == b"synthetic-fixer-change"
    log = c.results / "_fix_logs" / (c.idea + "_attempt1.log")
    assert log.is_file()
    assert ("FIX_APPLIED" in log.read_text()) is marker
