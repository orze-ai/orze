"""Persistent local lifecycle owner for explicitly registered controllers.

The foreground host creates controllers itself and retains their actual child
handles. Short-lived watchdogs use a credentialled request channel; they never
spawn the successor. Existing host claims are not reclaimed after uncertainty.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
import subprocess
import sys
import time

from orze.core.idea_source_lock import _acquire, idea_source_lock_owned
from orze.engine.controller_control import ControllerHOLD, _path, _token
from orze.engine.controller_handoff import (
    _ChildSet, _capture, _encode, _receive, _send, _timeout,
    _Route, _optional_request, _validate_payload, restart_controller,
    _new_channel, _inherited_channel, _plain,
)
from orze.engine.controller_session import _Observer, _file_witness, _stop_observed
from orze.engine.supervisor_worker import process_identity

PROFILE = {"version": 1, "profile": "local_service_host_v1"}
_INITIAL_FD_ENV = "ORZE_SERVICE_HOST_INITIAL_FD"


def _initial_hello(cfg, process):
    from orze.core.controller_profile import profile_fingerprint
    return {"schema": 1, "event": "SERVICE_INITIAL", "process": process,
            "config_sha256": profile_fingerprint(cfg),
            "config_witness": _plain(_file_witness(Path(cfg["_config_path"]), 65536)[0])}


def prepare_initial_entry(cfg):
    """Require the actual parent to confirm the child's inputs before startup.

    This does not grant adoption or bypass ordinary initial registration. The
    permanent service boot intention already precedes the parent's Popen.
    """
    channel = _inherited_channel(_INITIAL_FD_ENV)
    issuer_fd = None
    try:
        issuer = process_identity(os.getppid())[0]
        issuer_fd = _capture(issuer)
        deadline = time.monotonic() + 60
        hello = _initial_hello(cfg, process_identity(os.getpid())[0])
        _send(channel, hello, deadline)
        packet = _receive(channel, issuer["pid"], deadline)
        if (packet != {"schema": 1, "event": "SERVICE_GO", "hello_sha256": _sha(_encode(hello).encode())}
                or select.select([issuer_fd], [], [], 0)[0]
                or _initial_hello(cfg, process_identity(os.getpid())[0]) != hello):
            raise ControllerHOLD("service_host_initial_admission_changed")
    finally:
        channel.close()
        if issuer_fd is not None:
            os.close(issuer_fd)


def state_directory(service_config):
    path = Path(service_config)
    return path.with_name(path.name + ".host.lock")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path):
    witness, raw = _file_witness(path, 65536)
    value = json.loads(raw)
    if type(value) is not dict:
        raise ControllerHOLD("service_host_document_invalid")
    return witness, raw, value


def _create(path, value):
    raw = _encode(value).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        while raw:
            count = os.write(fd, raw)
            if count <= 0:
                raise ControllerHOLD("service_host_write_unconfirmed")
            raw = raw[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _service(path):
    path, _ = _path(Path(path), directory=False)
    witness, raw, value = _read(path)
    declaration = value.get("service_owner")
    if (type(declaration) is not dict or declaration != PROFILE
            or type(declaration.get("version")) is not int
            or value.get("service_config_file") != str(path)
            or value.get("method") not in {"process", "systemd", "crontab"}):
        raise ControllerHOLD("service_host_declaration_invalid")
    for name in ("workdir", "results_dir", "config_file", "python"):
        if type(value.get(name)) is not str or not Path(value[name]).is_absolute():
            raise ControllerHOLD("service_host_paths_invalid")
    return path, witness, raw, value


def _endpoint(registration):
    nonce = registration.get("nonce")
    if type(nonce) is not str or re.fullmatch("[0-9a-f]{32}", nonce) is None:
        raise ControllerHOLD("service_host_identity_invalid")
    return "\0orze-service-v1-" + str(os.getuid()) + "-" + nonce


def request(service_config, operation, *, request_id=None, source_controller_id=None, timeout=60):
    """Address the exact current host, not a PID or a global service label."""
    deadline = time.monotonic() + _timeout(timeout)
    path, witness, raw, svc = _service(service_config)
    scope, scope_witness = _path(svc["results_dir"], directory=True)
    owner_file = state_directory(path) / "owner.json"
    owner_witness, _, registration = _read(owner_file)
    if (registration.get("schema") != 1 or registration.get("service_sha256") != _sha(raw)
            or registration.get("service_config_file") != str(path)):
        raise ControllerHOLD("service_host_configuration_changed")
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    channel.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
    channel.settimeout(max(.001, deadline - time.monotonic()))
    fd = None
    try:
        channel.connect(_endpoint(registration))
        pid, uid, gid = struct.unpack("3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        process = registration.get("process")
        if ((uid, gid) != (os.getuid(), os.getgid()) or type(process) is not dict
                or process.get("pid") != pid):
            raise ControllerHOLD("service_host_peer_unconfirmed")
        fd = _capture(process)
        if (_file_witness(path, 65536)[0] != witness
                or _file_witness(owner_file, 65536)[0] != owner_witness
                or _path(scope, directory=True) != (scope, scope_witness)):
            raise ControllerHOLD("service_host_route_changed")
        packet = {"schema": 1, "service_sha256": _sha(raw), "nonce": registration["nonce"],
                  "operation": operation, "request_id": request_id,
                  "source_controller_id": source_controller_id, "timeout": timeout}
        _send(channel, packet, deadline)
        reply = _receive(channel, pid, deadline)
        if (reply.get("schema") != 1 or reply.get("nonce") != registration["nonce"]
                or reply.get("request") != packet or reply.get("status") != "ok"):
            raise ControllerHOLD("service_host_operation_unconfirmed")
        if (_file_witness(path, 65536)[0] != witness
                or _file_witness(owner_file, 65536)[0] != owner_witness):
            raise ControllerHOLD("service_host_route_changed")
        return reply["result"]
    except ControllerHOLD:
        raise
    except Exception as exc:
        raise ControllerHOLD("service_host_connection_unconfirmed") from exc
    finally:
        channel.close()
        if fd is not None:
            os.close(fd)


class Host:
    def __init__(self, service_config):
        from orze.core.config import load_project_config, find_dotenv
        from orze.core.controller_profile import handoff_profile
        self.path, self.witness, self.raw, self.svc = _service(service_config)
        if str(Path.cwd()) != self.svc["workdir"]:
            raise ControllerHOLD("service_host_workdir_changed")
        self.scope, self.scope_witness = _path(self.svc["results_dir"], directory=True)
        if self.scope in self.path.parents:
            raise ControllerHOLD("service_host_state_inside_controller_scope")
        self.config_file = Path(self.svc["config_file"])
        self.config_witness = _file_witness(self.config_file, 65536)[0]
        self.dotenv = find_dotenv(self.config_file, cwd=Path.cwd())
        self.dotenv_witness = None if self.dotenv is None else _file_witness(self.dotenv, 65536)[0]
        self.cfg = load_project_config(str(self.config_file))
        if not handoff_profile(self.cfg) or Path(self.cfg["results_dir"]).resolve() != self.scope:
            raise ControllerHOLD("service_host_handoff_profile_required")
        self.environment = dict(os.environ)
        self.children = _ChildSet()
        self.lease = None
        self.listener = None
        self.owner_witness = None
        self.shutdown = False
        self.signal_stop = False
        self.started_at = None
        self.registration = None
        self.pending_handoff = None
        self._check_inputs()

    def _check_inputs(self):
        from orze.core.config import find_dotenv
        from orze.service.runtime_contract import audit_runtime_contract
        if (_file_witness(self.path, 65536)[0] != self.witness
                or _file_witness(self.config_file, 65536)[0] != self.config_witness
                or _path(self.scope, directory=True) != (self.scope, self.scope_witness)
                or str(Path.cwd()) != self.svc["workdir"]
                or dict(os.environ) != self.environment
                or find_dotenv(self.config_file, cwd=Path.cwd()) != self.dotenv
                or self.dotenv is not None and _file_witness(self.dotenv, 65536)[0] != self.dotenv_witness):
            raise ControllerHOLD("service_host_inputs_changed")
        contract = audit_runtime_contract(self.svc)
        if not contract.get("startup_allowed"):
            raise ControllerHOLD("service_host_runtime_rejected")
        if (_file_witness(self.path, 65536)[0] != self.witness
                or _file_witness(self.config_file, 65536)[0] != self.config_witness):
            raise ControllerHOLD("service_host_inputs_changed")

    def _check_owner(self):
        self.children.check()
        if (not idea_source_lock_owned(self.lease)
                or _path(self.scope, directory=True) != (self.scope, self.scope_witness)
                or self.owner_witness is not None and _file_witness(
                    self.lease.lock_dir / "owner.json", 65536)[0] != self.owner_witness):
            raise ControllerHOLD("service_host_owner_changed")

    def _observe(self):
        self._check_owner()
        observer = _Observer(self.cfg)
        if not self.children.owns(observer.identity["process"]):
            observer.close()
            raise ControllerHOLD("service_host_controller_not_owned")
        return observer

    def start(self):
        self._check_inputs()
        self.lease = _acquire(state_directory(self.path))
        if self.lease is None:
            raise ControllerHOLD("service_host_existing_owner")
        self.registration = {"schema": 1, "nonce": self.lease.owner_nonce,
                             "service_config_file": str(self.path), "service_sha256": _sha(self.raw),
                             "process": process_identity(os.getpid())[0]}
        _create(self.lease.lock_dir / "owner.json", self.registration)
        self.owner_witness = _file_witness(self.lease.lock_dir / "owner.json", 65536)[0]
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        self.listener.bind(_endpoint(self.registration))
        self.listener.listen(8)
        self._check_owner()
        self._check_inputs()
        # Retain a durable boot intention before the one permitted initial spawn.
        # A later host cannot erase this namespace and repeat an unknown launch.
        _create(self.lease.lock_dir / "boot.json", {"schema": 1, "service_sha256": _sha(self.raw)})
        parent_channel, child_channel = _new_channel()
        try:
            env = {**self.environment, _INITIAL_FD_ENV: str(child_channel.fileno())}
            child = subprocess.Popen([sys.executable, "-m", "orze.cli", "-c", str(self.config_file)],
                                     cwd=self.svc["workdir"], env=env, close_fds=True,
                                     pass_fds=(child_channel.fileno(),),
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
            process = process_identity(child.pid)[0]
            fd = _capture(process)
            try:
                self.children.retain(child, process, fd)
            finally:
                os.close(fd)
            child_channel.close()
            deadline = time.monotonic() + 60
            hello = _receive(parent_channel, child.pid, deadline)
            self._check_owner()
            self._check_inputs()
            if hello != _initial_hello(self.cfg, process):
                raise ControllerHOLD("service_host_initial_inputs_mismatch")
            _send(parent_channel, {"schema": 1, "event": "SERVICE_GO",
                                  "hello_sha256": _sha(_encode(hello).encode())}, deadline)
        finally:
            parent_channel.close()
            child_channel.close()
        deadline = time.monotonic() + 60
        while True:
            if child.poll() is not None or time.monotonic() >= deadline:
                raise ControllerHOLD("service_host_initial_start_unconfirmed")
            try:
                observer = self._observe()
            except (ControllerHOLD, OSError):
                time.sleep(.02)
                continue
            try:
                if observer.identity["process"] != process:
                    raise ControllerHOLD("service_host_initial_controller_changed")
                self.started_at = time.monotonic()
                _create(self.lease.lock_dir / "ready.json", {"schema": 1, "controller_id": observer.controller_id})
                return
            finally:
                observer.close()

    def status(self):
        observer = self._observe()
        try:
            observer.check()
            return {"host_process": self.registration["process"], "live_children": self.children.live(),
                    "controller": {"controller_id": observer.controller_id,
                                   "process": observer.identity["process"]}}
        finally:
            observer.close()

    def restart(self, request_id, source, timeout):
        _token(request_id, "service_host_request_id_invalid")
        if self.pending_handoff is not None and self.pending_handoff != (request_id, source):
            raise ControllerHOLD("service_host_pending_handoff")
        self._check_inputs()
        current = self.status()["controller"]["controller_id"]
        route = _Route(self.cfg)
        with route.connection() as conn:
            previous = _optional_request(conn, request_id)
        expected = current if previous is None else _validate_payload(route, previous)["source_controller_id"]
        if source != expected:
            raise ControllerHOLD("service_host_source_controller_changed")
        self.pending_handoff = (request_id, source)
        result = restart_controller(self.cfg, request_id, timeout, _children=self.children)
        self._check_owner()
        if self.status()["controller"]["controller_id"] != result.target_controller_id:
            raise ControllerHOLD("service_host_successor_not_owned")
        if previous is None:
            self.started_at = time.monotonic()
        self.pending_handoff = None
        return {"kind": "handoff", **asdict(result)}

    def stop(self, timeout):
        # Stop uses the captured project's actual owner even if author inputs
        # changed; it creates no new work and never reloads a different target.
        observer = self._observe()
        try:
            result = _stop_observed(observer, timeout)
        finally:
            observer.close()
        if self.children.live():
            raise ControllerHOLD("service_host_children_not_closed")
        proof = {"kind": "stopped", **asdict(result)}
        _create(self.lease.lock_dir / "closed.json", proof)
        self.shutdown = True
        return proof

    def dispatch(self, packet):
        if (set(packet) != {"schema", "service_sha256", "nonce", "operation", "request_id", "source_controller_id", "timeout"}
                or type(packet["schema"]) is not int or packet["schema"] != 1
                or packet["service_sha256"] != _sha(self.raw)
                or packet["nonce"] != self.registration["nonce"]):
            raise ControllerHOLD("service_host_request_invalid")
        timeout = _timeout(packet["timeout"])
        operation = packet["operation"]
        if operation != "restart" and (packet["request_id"] is not None or packet["source_controller_id"] is not None):
            raise ControllerHOLD("service_host_request_invalid")
        if operation == "stop":
            return self.stop(timeout)
        self._check_inputs()
        if operation == "status":
            return self.status()
        if operation == "restart":
            return self.restart(packet["request_id"], packet["source_controller_id"], timeout)
        if operation == "watchdog":
            from orze.service.watchdog import _is_heartbeat_stale
            if self.pending_handoff is not None:
                return self.restart(*self.pending_handoff, timeout)
            current = self.status()
            threshold = self.svc.get("stall_threshold", 1800)
            if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
                raise ControllerHOLD("service_host_stall_threshold_invalid")
            if time.monotonic() - self.started_at < threshold:
                return {"kind": "starting", **current}
            stale, age = _is_heartbeat_stale(self.scope, socket.gethostname(), threshold)
            if not stale:
                return {"kind": "observed", **current}
            source = current["controller"]["controller_id"]
            return self.restart("watchdog-" + source, source, timeout)
        raise ControllerHOLD("service_host_operation_invalid")

    def serve(self):
        self.start()
        while not self.shutdown:
            if self.signal_stop:
                self.stop(60)
                break
            if not select.select([self.listener], [], [], .1)[0]:
                self.children.live()
                continue
            channel, _ = self.listener.accept()
            with channel:
                packet = None
                try:
                    pid, uid, gid = struct.unpack("3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    if (uid, gid) != (os.getuid(), os.getgid()):
                        raise ControllerHOLD("service_host_client_unconfirmed")
                    packet = _receive(channel, pid, time.monotonic() + 2)
                    result = self.dispatch(packet)
                    _send(channel, {"schema": 1, "nonce": self.registration["nonce"],
                                    "request": packet, "status": "ok", "result": result})
                except Exception:
                    # Failed delivery does not undo STARTED or repeat a spawn.
                    # The original request ID remains the replay authority.
                    try:
                        _send(channel, {"schema": 1, "nonce": self.registration["nonce"],
                                        "request": packet, "status": "hold"})
                    except Exception:
                        pass

    def close(self):
        if self.listener is not None:
            self.listener.close()
        self.children.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-config", required=True)
    parser.add_argument("--operation", choices=("status", "restart", "stop", "watchdog"))
    parser.add_argument("--request-id")
    parser.add_argument("--source-controller-id")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args(argv)
    host = None
    try:
        if args.operation:
            result = request(args.service_config, args.operation, request_id=args.request_id,
                             source_controller_id=args.source_controller_id, timeout=args.timeout)
            print(json.dumps(result, sort_keys=True))
        else:
            host = Host(args.service_config)
            def stop_signal(signum, frame):
                host.signal_stop = True
            signal.signal(signal.SIGTERM, stop_signal)
            signal.signal(signal.SIGINT, stop_signal)
            host.serve()
        return 0
    except Exception as exc:
        code = str(exc) if isinstance(exc, ControllerHOLD) and re.fullmatch("[a-z0-9_]{1,96}", str(exc)) else "service_host_unconfirmed"
        print("HOLD: " + code, file=sys.stderr)
        return 75
    finally:
        if host is not None:
            host.close()


if __name__ == "__main__":
    raise SystemExit(main())
