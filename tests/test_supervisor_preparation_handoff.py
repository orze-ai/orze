"""Frozen draft handoff faults; first two create an actual CPU supervisor."""
import ctypes
import os
import subprocess
import sys

import pytest

from orze.engine import supervised_process as api
from orze.engine import supervisor_worker as worker


@pytest.mark.parametrize("fault", ["child_close", "constructor"])
def test_all_post_popen_failures_carry_uncertain_handle(tmp_path, monkeypatch, fault):
    created = []
    sockets = []
    popen = subprocess.Popen
    socketpair = api.socket.socketpair
    close = api.socket.socket.close
    initialize = api.SupervisedProcess.__init__

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        created.append(child)
        return child

    def pair():
        values = socketpair()
        sockets.extend(values)
        return values

    def close_fail(channel):
        result = close(channel)
        if created and len(sockets) == 2 and channel is sockets[1]:
            raise OSError("injected close uncertainty")
        return result

    def init_fail(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        raise RuntimeError("injected constructor failure")

    monkeypatch.setattr(api.subprocess, "Popen", spawn)
    monkeypatch.setattr(api.socket, "socketpair", pair)
    if fault == "child_close":
        monkeypatch.setattr(api.socket.socket, "close", close_fail)
    else:
        monkeypatch.setattr(api.SupervisedProcess, "__init__", init_fail)
    observed = None
    try:
        try:
            api.prepare_supervised([sys.executable, "-c", "raise AssertionError()"],
                                   identity={"attempt_ref": {"attempt_id": "one"},
                                             "scope": str(tmp_path)},
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except BaseException as exc:
            observed = exc
        assert created, "the fixture must cross real Popen before injecting the fault"
        assert isinstance(observed, api.SupervisionUncertain)
        assert isinstance(observed.process, api.SupervisedProcess)
        assert observed.process.supervisor_pid == created[0].pid
        assert observed.process.returncode is None
    finally:
        for channel in sockets:
            close(channel)
        for child in created:
            child.wait(timeout=3)
        if isinstance(observed, api.SupervisionUncertain):
            observed.process._close_descriptors()


def test_fork_parent_close_error_enters_owned_drain(monkeypatch):
    """OS fork is stubbed here: this checks exception routing, not tree proof."""
    calls = []
    real_close = os.close
    descriptors = []
    original_pipe = os.pipe

    class Libc:
        def prctl(self, operation, value, *args):
            if operation == 37:
                ctypes.cast(value, ctypes.POINTER(ctypes.c_int))[0] = 1
            return 0

    class Channel:
        def settimeout(self, value):
            pass

    def pipe():
        value = original_pipe()
        descriptors.extend(value)
        return value

    def fail_close(fd):
        real_close(fd)
        if fd == descriptors[0]:
            raise OSError("injected post-fork close")

    monkeypatch.setattr(worker.ctypes, "CDLL", lambda *a, **kw: Libc())
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker.os, "pipe", pipe)
    monkeypatch.setattr(worker.os, "fork", lambda: 123456)
    monkeypatch.setattr(worker.os, "close", fail_close)
    monkeypatch.setattr(worker, "send_frame", lambda *args: None)
    monkeypatch.setattr(worker, "_emergency_drain", lambda pid, state: calls.append(pid))
    try:
        with pytest.raises(OSError, match="post-fork"):
            worker._run(Channel(), {})
        assert calls == [123456]
    finally:
        for fd in descriptors:
            try:
                real_close(fd)
            except OSError:
                pass
