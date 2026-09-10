"""Two-phase Linux execution handle with an exact whole-tree closure receipt.

prepare_supervised forks a *blocked* worker under a dedicated subreaper.
The caller must durably bind ``binding`` before start(). A worker's integer
exit status is returned only after TREE_CLOSED and normal supervisor exit.
No crash recovery/adoption or hostile same-UID process sandbox is claimed.
"""
from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import secrets
import select
import signal
import socket
import subprocess
import sys
import time

from orze.engine.supervisor_worker import (
    MAX_FRAME, PROTOCOL, canonical, process_identity, send_frame, take_frame,
)


class SupervisionUnavailable(ValueError):
    """Rejected locally, before creating any supervisor or worker."""


class SupervisionUncertain(RuntimeError):
    """No closure authority; the caller must preserve its attempt HOLD."""

    def __init__(self, reason, *, process=None):
        super().__init__(reason)
        self.process = process


class SupervisedProcess:
    """A Popen-like handle whose pid is the actual worker, not the supervisor."""

    def __init__(self, supervisor, channel, identity, nonce, command_sha256):
        self._supervisor = supervisor
        self._channel = channel
        self._identity = json.loads(canonical(identity))
        self._nonce = nonce
        self._command_sha256 = command_sha256
        self._binding = None
        self._closed = None
        self._buffer = bytearray()
        self._started = False
        self._stop_sent = False
        self._uncertainty = None
        self._eof = False
        self._supervisor_pidfd = None
        self.pid = None
        self.returncode = None
        self.args = ["<supervised-worker>"]  # never leak argv in TimeoutExpired

    @property
    def supervisor_pid(self):
        return self._supervisor.pid

    @property
    def supervisor_pidfd(self):
        return self._supervisor_pidfd

    @property
    def binding(self):
        return copy.deepcopy(self._binding)

    def _fail(self, reason):
        self._uncertainty = self._uncertainty or reason
        raise SupervisionUncertain(self._uncertainty, process=self)

    def _accept(self, message):
        if self._binding is None:
            if set(message) != {"event", "binding"} or message["event"] != "READY":
                self._fail("supervisor_ready_invalid")
            binding = message["binding"]
            if not isinstance(binding, dict) or set(binding) != {
                    "schema", "protocol", "identity", "nonce_sha256", "command_sha256",
                    "worker", "supervisor"}:
                self._fail("supervisor_binding_invalid")
            expected = {"schema": 1, "protocol": PROTOCOL, "identity": self._identity,
                        "nonce_sha256": hashlib.sha256(self._nonce.encode("ascii")).hexdigest(),
                        "command_sha256": self._command_sha256}
            if canonical({key: binding[key] for key in expected}) != canonical(expected):
                self._fail("supervisor_binding_mismatch")
            for key in ("worker", "supervisor"):
                item = binding[key]
                if (not isinstance(item, dict) or set(item) != {"pid", "start_ticks"}
                        or any(type(item[value]) is not int or item[value] <= 0
                               for value in ("pid", "start_ticks"))):
                    self._fail("supervisor_process_identity_invalid")
            if binding["supervisor"]["pid"] != self.supervisor_pid:
                self._fail("supervisor_process_identity_mismatch")
            observed, _ = process_identity(self.supervisor_pid)
            worker, parent = process_identity(binding["worker"]["pid"])
            if (observed != binding["supervisor"] or worker != binding["worker"]
                    or parent != self.supervisor_pid):
                self._fail("supervisor_process_identity_mismatch")
            self._binding = copy.deepcopy(binding)
            self.pid = worker["pid"]
            return
        expected_keys = {"schema", "event", "binding", "worker_returncode", "stop_requested",
                         "forced_cleanup", "reaped_children", "wait_proof"}
        if (self._closed is not None or set(message) != expected_keys
                or type(message["schema"]) is not int or message["schema"] != 1
                or message["event"] != "TREE_CLOSED"
                or canonical(message["binding"]) != canonical(self._binding)
                or type(message["worker_returncode"]) is not int
                or not -64 <= message["worker_returncode"] <= 255
                or type(message["stop_requested"]) is not bool
                or type(message["forced_cleanup"]) is not bool
                or (message["forced_cleanup"] and not message["stop_requested"])
                or type(message["reaped_children"]) is not int
                or message["reaped_children"] < 1
                or message["wait_proof"] != "ECHILD_WALL"
                or (not self._started and not message["stop_requested"])):
            self._fail("supervisor_closure_invalid")
        self._closed = copy.deepcopy(message)

    def _receive(self):
        if self._uncertainty is not None:
            self._fail(self._uncertainty)
        try:
            while not self._eof:
                try:
                    chunk = self._channel.recv(65536)
                except BlockingIOError:
                    break
                if not chunk:
                    self._eof = True
                    break
                self._buffer.extend(chunk)
                if len(self._buffer) > MAX_FRAME + 4:
                    self._fail("supervisor_frame_limit")
                while True:
                    message = take_frame(self._buffer)
                    if message is None:
                        break
                    self._accept(message)
            if self._eof and (self._buffer or self._closed is None):
                self._fail("supervisor_channel_lost")
        except SupervisionUncertain:
            raise
        except Exception:
            self._fail("supervisor_protocol_uncertain")

    def start(self):
        if self._uncertainty or self._binding is None or self._started or self._stop_sent:
            self._fail("supervisor_start_not_authorized")
        self._send("GO")
        self._started = True

    def _send(self, command):
        try:
            self._channel.settimeout(1)
            send_frame(self._channel, {"command": command, "nonce": self._nonce})
            self._channel.setblocking(False)
        except Exception:
            self._fail("supervisor_control_uncertain")

    def poll(self):
        self._receive()
        supervisor_code = self._supervisor.poll()
        if supervisor_code is not None:
            # Drain any final message already written before waitpid observed exit.
            self._receive()
            if supervisor_code != 0 or self._closed is None:
                self._fail("supervisor_exit_unconfirmed")
            self.returncode = self._closed["worker_returncode"]
            self._close_descriptors()
        return self.returncode

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = self.poll()
            if result is not None:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(0.01)

    def closure_receipt(self):
        return copy.deepcopy(self._closed) if self.poll() is not None else None

    def stop(self, timeout=10):
        if self.poll() is not None:
            return True
        if not self._stop_sent:
            self._send("STOP")
            self._stop_sent = True
        try:
            self.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._fail("supervisor_stop_unconfirmed")
        return True

    def _close_descriptors(self):
        self._channel.close()
        if self._supervisor_pidfd is not None:
            os.close(self._supervisor_pidfd)
            self._supervisor_pidfd = None


def prepare_supervised(cmd, *, identity, env=None, cwd=None, stdout=None,
                       stderr=None, pass_fds=(), ready_timeout=10):
    """Create a blocked worker. Any post-Popen failure carries a HOLD handle."""
    try:
        if (sys.platform != "linux" or not hasattr(os, "pidfd_open")
                or not hasattr(signal, "pidfd_send_signal")
                or not hasattr(ctypes.CDLL(None), "prctl")):
            raise ValueError("supervisor_linux_support_required")
        if (not isinstance(cmd, (list, tuple)) or not cmd
                or any(not isinstance(arg, str) or "\0" in arg for arg in cmd)
                or not isinstance(identity, dict)):
            raise ValueError("supervisor_setup_invalid")
        identity = json.loads(canonical(identity))
        if len(canonical(identity)) > 16384:
            raise ValueError("supervisor_identity_limit")
        if type(ready_timeout) not in (int, float) or not 0 < ready_timeout <= 60:
            raise ValueError("supervisor_ready_timeout_invalid")
        environment = dict(os.environ if env is None else env)
        if any(not isinstance(key, str) or not isinstance(value, str)
               or "\0" in key + value or "=" in key for key, value in environment.items()):
            raise ValueError("supervisor_environment_invalid")
        fds = tuple(pass_fds)
        if any(type(fd) is not int or fd < 3 for fd in fds):
            raise ValueError("supervisor_pass_fds_invalid")
        nonce = secrets.token_hex(32)
        config = {"cmd": list(cmd), "identity": identity, "env": environment, "nonce": nonce}
        if len(canonical(config)) > MAX_FRAME:
            raise ValueError("supervisor_setup_limit")
        # Probe only ourselves, without installing a process-wide subreaper.
        probe = os.pidfd_open(os.getpid(), 0)
        try:
            signal.pidfd_send_signal(probe, 0)
        finally:
            os.close(probe)
    except Exception as exc:
        raise SupervisionUnavailable("supervisor_setup_unavailable") from exc
    parent, child = socket.socketpair()
    try:
        supervisor = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).with_name("supervisor_worker.py")),
             str(child.fileno())], cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, pass_fds=(*fds, child.fileno()),
            start_new_session=True,
        )
    except Exception:
        parent.close()
        child.close()
        raise
    child.close()
    handle = SupervisedProcess(supervisor, parent, identity, nonce,
                               hashlib.sha256(canonical(list(cmd))).hexdigest())
    try:
        handle._supervisor_pidfd = os.pidfd_open(supervisor.pid, 0)
        parent.settimeout(ready_timeout)
        send_frame(parent, config)
        parent.setblocking(False)
        deadline = time.monotonic() + ready_timeout
        while handle._binding is None:
            handle._receive()
            if supervisor.poll() is not None:
                handle._fail("supervisor_ready_unconfirmed")
            if time.monotonic() >= deadline:
                handle._fail("supervisor_ready_timeout")
            select.select([parent], [], [], 0.01)
        return handle
    except BaseException as exc:
        # Closing the channel asks the supervisor to drain; it is NOT proof of
        # completion, nor authority to turn a committed intent into NOT_STARTED.
        parent.close()
        if isinstance(exc, SupervisionUncertain):
            raise
        raise SupervisionUncertain("supervisor_prepare_uncertain", process=handle) from exc
