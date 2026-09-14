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
import math
import os
from pathlib import Path
import re
import select
import signal
import socket
import struct
import sys
import time

MAX_FRAME = 1048576
PROTOCOL = "orze.linux_subreaper.v1"
RUNTIME_LEASE_PROTOCOL = "orze.linux_subreaper.v2"
MAX_NS = 2**63 - 1
WALL = 0x40000000


class RuntimeLeaseExpired(ValueError):
    """Known expiry, not unknown process ownership or closure."""

    def __init__(self, observed_ns):
        self.observed_ns = observed_ns
        super().__init__("cpu_runtime_lease_expired")


def validate_runtime_lease(value):
    """Detached structural validation only: historical readers do no OS I/O."""
    if (type(value) is not dict or set(value) != {
            "schema", "clock", "hostname", "boot_id", "issued_ns", "deadline_ns"}
            or type(value["schema"]) is not int or value["schema"] != 1
            or value["clock"] != "CLOCK_BOOTTIME"
            or type(value["hostname"]) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", value["hostname"]) is None
            or type(value["boot_id"]) is not str
            or re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value["boot_id"]) is None
            or type(value["issued_ns"]) is not int
            or type(value["deadline_ns"]) is not int
            or not 0 <= value["issued_ns"] < value["deadline_ns"] <= MAX_NS):
        raise ValueError("runtime_lease_descriptor_invalid")
    return dict(value)


def _runtime_lease_identity():
    with open("/proc/sys/kernel/random/boot_id", "rb") as stream:
        raw = stream.read(38)
    if len(raw) != 37 or raw[-1:] != b"\n":
        raise ValueError("runtime_lease_boot_invalid")
    return socket.gethostname(), raw[:-1].decode("ascii")


def _runtime_lease_clock():
    observed = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    if type(observed) is not int or not 0 <= observed <= MAX_NS:
        raise ValueError("runtime_lease_clock_invalid")
    return observed


def runtime_lease_now(descriptor):
    """Observe the actual bound boot/clock, including after known expiry."""
    descriptor = validate_runtime_lease(descriptor)
    expected = (descriptor["hostname"], descriptor["boot_id"])
    if _runtime_lease_identity() != expected:
        raise ValueError("runtime_lease_host_boot_changed")
    observed = _runtime_lease_clock()
    if _runtime_lease_identity() != expected or observed < descriptor["issued_ns"]:
        raise ValueError("runtime_lease_clock_changed")
    return observed


def capture_runtime_lease(ttl_seconds):
    """Capture a bounded, nonrenewable deadline; never round above the TTL."""
    if (type(ttl_seconds) not in (int, float) or ttl_seconds <= 0
            or (type(ttl_seconds) is float and not math.isfinite(ttl_seconds))):
        raise ValueError("runtime_lease_ttl_invalid")
    numerator, denominator = (ttl_seconds.as_integer_ratio() if type(ttl_seconds) is float
                              else (ttl_seconds, 1))
    ttl_ns = numerator * 1_000_000_000 // denominator
    if not 0 < ttl_ns <= MAX_NS:
        raise ValueError("runtime_lease_ttl_invalid")
    hostname, boot_id = _runtime_lease_identity()
    issued = _runtime_lease_clock()
    descriptor = validate_runtime_lease({"schema": 1, "clock": "CLOCK_BOOTTIME",
        "hostname": hostname, "boot_id": boot_id, "issued_ns": issued,
        "deadline_ns": issued + ttl_ns})
    runtime_lease_now(descriptor)
    return descriptor


def require_runtime_lease(descriptor):
    descriptor = validate_runtime_lease(descriptor)
    observed = runtime_lease_now(descriptor)
    if observed >= descriptor["deadline_ns"]:
        raise RuntimeLeaseExpired(observed)
    return observed


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
    lease = (validate_runtime_lease(config["runtime_lease"])
             if "runtime_lease" in config else None)
    if lease is not None:
        # Validate before fork; a later deadline crossing is handled by the
        # same owned stop/drain path, not an unowned setup failure.
        runtime_lease_now(lease)
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
    state = {"worker_returncode": None, "reaped_children": 0}
    try:
        os.close(read_gate)
        # Only the blocked worker keeps these copies (e.g. an attestation
        # writer). Do not retain them through descendant drain or retry a
        # failed close: the descriptor number may already have been reused.
        # This runs inside owned cleanup and strictly before READY.
        for fd in config.get("worker_only_fds", []):
            os.close(fd)
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
        if lease is not None:
            binding.update(schema=2, protocol=RUNTIME_LEASE_PROTOCOL,
                           runtime_lease=lease)
        send_frame(channel, {"event": "READY", "binding": binding})
        channel.setblocking(False)
        buffer = bytearray()
        started = False
        stop_at = None
        forced = False
        connected = True
        lease_observed = None
        while True:
            if lease is not None:
                observed = runtime_lease_now(lease)
                if lease_observed is not None and observed < lease_observed:
                    raise ValueError("runtime_lease_clock_reversed")
                lease_observed = observed
                if observed >= lease["deadline_ns"]:
                    stop_at = stop_at if stop_at is not None else time.monotonic()
                    if write_gate >= 0:
                        os.close(write_gate)
                        write_gate = -1
            if _wait_all(worker_pid, state):
                receipt = {"schema": 1, "event": "TREE_CLOSED", "binding": binding,
                           **state, "stop_requested": stop_at is not None,
                           "forced_cleanup": forced, "wait_proof": "ECHILD_WALL"}
                if lease is not None:
                    receipt.update(schema=2,
                        lease_expired=lease_observed >= lease["deadline_ns"],
                        lease_observed_ns=lease_observed)
                if connected:
                    channel.settimeout(5)
                    send_frame(channel, receipt)
                return
            if stop_at is not None:
                sig = signal.SIGKILL if time.monotonic() - stop_at >= 0.25 else signal.SIGTERM
                signalled = _signal_children(sig)
                forced = forced or (sig == signal.SIGKILL and signalled)
            delay = (min(0.01, max(0.0, (lease["deadline_ns"] - lease_observed) / 1e9))
                     if lease is not None and stop_at is None else 0.01)
            ready, _, _ = select.select([channel] if connected else [], [], [], delay)
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
                    if lease is not None:
                        observed = runtime_lease_now(lease)
                        if observed < lease_observed:
                            raise ValueError("runtime_lease_clock_reversed")
                        lease_observed = observed
                        if observed >= lease["deadline_ns"]:
                            stop_at = time.monotonic()
                            os.close(write_gate)
                            write_gate = -1
                            continue
                    started = True
                    os.write(write_gate, b"G")
                    os.close(write_gate)
                    write_gate = -1
                elif message["command"] == "STOP":
                    stop_at = stop_at or time.monotonic()
                    if write_gate >= 0:
                        os.close(write_gate)
                        write_gate = -1
                elif (message["command"] == "GO" and not started and lease is not None
                      and lease_observed >= lease["deadline_ns"]):
                    # Parent may have sampled just before expiry; the actual
                    # gate owner refuses this late GO without losing closure.
                    continue
                else:
                    raise ValueError("supervisor_control_invalid")
    except BaseException:
        try:
            channel.settimeout(0.1)
            send_frame(channel, {"event": "ERROR", "reason": "supervisor_uncertain"})
        except Exception:
            pass
        if write_gate >= 0:
            owned_gate = write_gate
            write_gate = -1
            try:
                os.close(owned_gate)
            except BaseException:
                pass
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
