"""New worker-only FD ownership mechanisms; not released-behavior red tests."""
import ctypes
import fcntl
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import pytest

from orze.engine import supervised_process as api
from orze.engine import supervisor_worker as worker


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper contract")


def _prepare(tmp_path, script, **kwargs):
    return api.prepare_supervised(
        [sys.executable, "-c", script],
        identity={"attempt_ref": {"task_id": "idea-fds", "phase": "training",
                                  "attempt_id": "fds-one", "generation": 1},
                  "scope": str(tmp_path)},
        cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)


def _until(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("bounded worker handshake did not arrive")


def test_worker_pipe_eof_does_not_wait_for_tree_but_shared_lease_does(tmp_path):
    reader, writer = os.pipe()
    lease_path = tmp_path / "lease"
    lease = os.open(lease_path, os.O_CREAT | os.O_RDWR, 0o600)
    observer = os.open(lease_path, os.O_RDWR)
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    process = None
    worker_pidfd = None
    try:
        process = _prepare(tmp_path, f"""
import os,time
from pathlib import Path
os.write({writer}, b'attestation')
child = os.fork()
os.close({writer})
os.close({lease})
if child:
    os._exit(0)
os.setsid()
Path('ready').write_text(str(os.getpid()))
deadline = time.monotonic()+5
while not Path('release').exists() and time.monotonic()<deadline:
    time.sleep(.01)
os._exit(0)
""", pass_fds=(lease,), worker_only_fds=(writer,))
        assert process.pid == process.binding["worker"]["pid"]
        assert process.pid != process.supervisor_pid
        assert not os.get_inheritable(writer)
        assert not os.get_inheritable(lease)
        assert not (tmp_path / "ready").exists()
        worker_pidfd = os.pidfd_open(process.pid)
        os.close(writer)
        writer = None
        os.close(lease)
        lease = None
        process.start()
        _until(lambda: (tmp_path / "ready").exists())
        assert select.select([worker_pidfd], [], [], 5)[0], "the real leader must exit"
        received = bytearray()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if not select.select([reader], [], [], .02)[0]:
                continue
            chunk = os.read(reader, 4096)
            if not chunk:
                break
            received.extend(chunk)
        else:
            pytest.fail("supervisor retained the worker-only writer after leader exit")
        assert bytes(received) == b"attestation"
        assert process.poll() is None, "the live descendant still prevents tree closure"
        assert process.closure_receipt() is None
        with pytest.raises(BlockingIOError):
            fcntl.flock(observer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (tmp_path / "release").write_text("go")
        assert process.wait(timeout=5) == 0
        receipt = process.closure_receipt()
        assert receipt["reaped_children"] == 2
        assert receipt["stop_requested"] is False
        fcntl.flock(observer, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if process is not None:
            process.stop(timeout=3)
        for fd in (reader, writer, lease, observer, worker_pidfd):
            if fd is not None:
                os.close(fd)


@pytest.mark.parametrize("bad", ["bool", "stdio", "closed_shared", "closed_worker",
                                 "duplicate", "overlap", "container", "limit"])
def test_invalid_descriptor_contract_rejects_before_popen(tmp_path, monkeypatch, bad):
    descriptors = [os.open(os.devnull, os.O_RDONLY) for _ in range(65)]
    closed = os.dup(descriptors[0])
    os.close(closed)
    shared = ()
    only = (descriptors[0],)
    if bad == "bool":
        only = (True,)
    elif bad == "stdio":
        only = (2,)
    elif bad == "closed_shared":
        shared, only = (closed,), ()
    elif bad == "closed_worker":
        only = (closed,)
    elif bad == "duplicate":
        only = (descriptors[0], descriptors[0])
    elif bad == "overlap":
        shared = (descriptors[0],)
    elif bad == "container":
        only = iter(only)
    elif bad == "limit":
        shared, only = tuple(descriptors[:32]), tuple(descriptors[32:])
    monkeypatch.setattr(api.subprocess, "Popen",
                        lambda *a, **kw: pytest.fail("invalid FDs must not create a process"))
    try:
        with pytest.raises(api.SupervisionUnavailable):
            _prepare(tmp_path, "pass", pass_fds=shared, worker_only_fds=only)
    finally:
        for fd in descriptors:
            os.close(fd)


def test_postfork_worker_fd_close_error_drains_before_any_ready(monkeypatch):
    """The fork is stubbed: verifies owned error routing, not OS closure proof."""
    reader, writer = os.pipe()
    lease = os.open(os.devnull, os.O_RDONLY)
    real_close = os.close
    calls = []
    events = []
    closes = []

    class Libc:
        def prctl(self, operation, value, *args):
            if operation == 37:
                ctypes.cast(value, ctypes.POINTER(ctypes.c_int))[0] = 1
            return 0

    class Channel:
        def settimeout(self, value):
            pass

    def fail_close(fd):
        closes.append(fd)
        real_close(fd)
        if fd == writer:
            raise OSError("injected worker-only close uncertainty")

    def drain(pid, state):
        os.fstat(lease)  # Shared lease remains held during uncertain cleanup.
        calls.append(pid)

    monkeypatch.setattr(worker.ctypes, "CDLL", lambda *a, **kw: Libc())
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker.os, "fork", lambda: 123456)
    monkeypatch.setattr(worker.os, "close", fail_close)
    monkeypatch.setattr(worker, "send_frame", lambda channel, frame: events.append(frame))
    monkeypatch.setattr(worker, "_emergency_drain", drain)
    try:
        with pytest.raises(OSError, match="worker-only"):
            worker._run(Channel(), {"worker_only_fds": [writer]})
        assert calls == [123456]
        assert closes.count(writer) == 1, "an uncertain close must not retry a reused fd"
        assert [event["event"] for event in events] == ["ERROR"]
    finally:
        for fd in (reader, writer, lease):
            try:
                real_close(fd)
            except OSError:
                pass
