"""Private Linux subreaper executable; no provider/framework imports.

The sole wait owner uses __WALL, including clone children without SIGCHLD.
Natural leader exit does not terminate useful descendants. Only STOP/channel
loss requests cleanup. Signals target pidfds for our direct/adopted children,
never an external process-group or a PID from an unverified caller.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import socket
import struct
import sys
import time

MAX_FRAME = 1048576
PROTOCOL = "orze.linux_subreaper.v1"
WALL = 0x40000000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("supervisor_duplicate_field")
        result[key] = value
    return result


def decode(raw):
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    if not isinstance(value, dict) or canonical(value) != raw:
        raise ValueError("supervisor_noncanonical_frame")
    return value


def send_frame(channel, value):
    raw = canonical(value)
    if not 0 < len(raw) <= MAX_FRAME:
        raise ValueError("supervisor_frame_limit")
    channel.sendall(struct.pack("!I", len(raw)) + raw)


def take_frame(buffer):
    if len(buffer) < 4:
        return None
    size = struct.unpack("!I", buffer[:4])[0]
    if not 0 < size <= MAX_FRAME:
        raise ValueError("supervisor_frame_limit")
    if len(buffer) < size + 4:
        return None
    raw = bytes(buffer[4:size + 4])
    del buffer[:size + 4]
    return decode(raw)


def process_identity(pid):
    # Reading the owned process's identity is diagnostic binding, not a signal.
    raw = Path(f"/proc/{pid}/stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return {"pid": pid, "start_ticks": int(fields[19])}, int(fields[1])


def _signal_children(sig):
    """Only our own children; one wait owner keeps unreaped PIDs unavailable."""
    with open(f"/proc/self/task/{os.getpid()}/children", "rb") as stream:
        raw = stream.read(MAX_FRAME + 1)
    if len(raw) > MAX_FRAME:
        raise RuntimeError("supervisor_child_limit")
    signalled = False
    for text in raw.split():
        pid = int(text)
        try:
            before, parent = process_identity(pid)
            if parent != os.getpid():
                continue
            fd = os.pidfd_open(pid, 0)
            try:
                after, parent = process_identity(pid)
                if before != after or parent != os.getpid():
                    raise RuntimeError("supervisor_child_identity_changed")
                signal.pidfd_send_signal(fd, sig)
                signalled = True
            finally:
                os.close(fd)
        except ProcessLookupError:
            continue
        except FileNotFoundError:
            continue
    return signalled


def _wait_all(worker_pid, state):
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG | WALL)
        except ChildProcessError:
            if state["worker_returncode"] is None:
                raise RuntimeError("supervisor_worker_status_missing")
            return True
        if pid == 0:
            return False
        state["reaped_children"] += 1
        if pid == worker_pid:
            state["worker_returncode"] = os.waitstatus_to_exitcode(status)


def _emergency_drain(worker_pid, state):
    # No false exit/lease release when cleanup itself cannot be confirmed.
    while True:
        try:
            if _wait_all(worker_pid, state):
                return
            _signal_children(signal.SIGKILL)
        except Exception:
            pass
        time.sleep(0.02)


def _run(channel, config):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "supervisor_subreaper_unavailable")
    observed = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(observed), 0, 0, 0) != 0 or observed.value != 1:
        raise RuntimeError("supervisor_subreaper_unverified")
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    read_gate, write_gate = os.pipe()
    worker_pid = os.fork()
    if worker_pid == 0:
        try:
            channel.close()
            os.close(write_gate)
            os.setpgrp()
            allowed = os.read(read_gate, 1)
            os.close(read_gate)
            if allowed != b"G":
                os._exit(125)
            for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"):
                if hasattr(signal, name):
                    signal.signal(getattr(signal, name), signal.SIG_DFL)
            os.execvpe(config["cmd"][0], config["cmd"], config["env"])
        except BaseException:
            os._exit(127)
    os.close(read_gate)
    state = {"worker_returncode": None, "reaped_children": 0}
    try:
        worker, parent = process_identity(worker_pid)
        supervisor, _ = process_identity(os.getpid())
        if parent != os.getpid():
            raise RuntimeError("supervisor_worker_parent_changed")
        binding = {
            "schema": 1, "protocol": PROTOCOL, "identity": config["identity"],
            "nonce_sha256": hashlib.sha256(config["nonce"].encode("ascii")).hexdigest(),
            "command_sha256": hashlib.sha256(canonical(config["cmd"])).hexdigest(),
            "worker": worker, "supervisor": supervisor,
        }
        send_frame(channel, {"event": "READY", "binding": binding})
        channel.setblocking(False)
        buffer = bytearray()
        started = False
        stop_at = None
        forced = False
        connected = True
        while True:
            if _wait_all(worker_pid, state):
                receipt = {"schema": 1, "event": "TREE_CLOSED", "binding": binding,
                           **state, "stop_requested": stop_at is not None,
                           "forced_cleanup": forced, "wait_proof": "ECHILD_WALL"}
                if connected:
                    channel.settimeout(5)
                    send_frame(channel, receipt)
                return
            if stop_at is not None:
                sig = signal.SIGKILL if time.monotonic() - stop_at >= 0.25 else signal.SIGTERM
                signalled = _signal_children(sig)
                forced = forced or (sig == signal.SIGKILL and signalled)
            ready, _, _ = select.select([channel] if connected else [], [], [], 0.01)
            if not ready:
                continue
            chunk = channel.recv(65536)
            if not chunk:
                connected = False
                stop_at = stop_at or time.monotonic()
                if write_gate >= 0:
                    os.close(write_gate)
                    write_gate = -1
                continue
            buffer.extend(chunk)
            if len(buffer) > MAX_FRAME + 4:
                raise ValueError("supervisor_frame_limit")
            while True:
                message = take_frame(buffer)
                if message is None:
                    break
                if (set(message) != {"command", "nonce"}
                        or message["nonce"] != config["nonce"]):
                    raise ValueError("supervisor_control_invalid")
                if message["command"] == "GO" and not started and stop_at is None:
                    started = True
                    os.write(write_gate, b"G")
                    os.close(write_gate)
                    write_gate = -1
                elif message["command"] == "STOP":
                    stop_at = stop_at or time.monotonic()
                    if write_gate >= 0:
                        os.close(write_gate)
                        write_gate = -1
                else:
                    raise ValueError("supervisor_control_invalid")
    except BaseException:
        try:
            channel.settimeout(0.1)
            send_frame(channel, {"event": "ERROR", "reason": "supervisor_uncertain"})
        except Exception:
            pass
        if write_gate >= 0:
            os.close(write_gate)
            write_gate = -1
        _emergency_drain(worker_pid, state)
        raise
    finally:
        if write_gate >= 0:
            os.close(write_gate)


def main():
    channel = socket.socket(fileno=int(sys.argv[1]))
    channel.settimeout(10)
    buffer = bytearray()
    while True:
        message = take_frame(buffer)
        if message is not None:
            if buffer:
                raise ValueError("supervisor_extra_setup_data")
            break
        chunk = channel.recv(65536)
        if not chunk:
            raise ValueError("supervisor_setup_missing")
        buffer.extend(chunk)
        if len(buffer) > MAX_FRAME + 4:
            raise ValueError("supervisor_frame_limit")
    _run(channel, message)
    channel.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Do not print command, environment, nonce, or an exception payload.
        sys.exit(120)
