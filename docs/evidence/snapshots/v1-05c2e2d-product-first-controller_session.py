"""Single-local-controller stop protocol, never restart/adoption authority.

The live controller alone certifies its strong member and resource inventory.
An observer captures its exact pidfd before requesting stop, then requires both
the bound durable ACK and that same kernel handle's exit. Neither a PID file,
an empty public process map, a return code, nor this result class grants start.
Persistent registration and stop history are intentionally never released.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import select
import socket
import sqlite3
import stat
import threading
import time

from orze.engine.controller_control import (
    ControllerHOLD, _path, _route, _schema as _registration_schema,
    register_controller,
)
from orze.engine.supervisor_worker import canonical, process_identity


_SQL = """CREATE TABLE controller_sessions (
    controller_id TEXT NOT NULL COLLATE BINARY PRIMARY KEY,
    binding_json TEXT NOT NULL,
    request_json TEXT,
    ack_json TEXT
)"""
_SESSIONS = {}  # Retained even when a public Orze attribute is erased.
_LIMIT = 65536


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _encode(value):
    raw = canonical(value)
    if len(raw) > _LIMIT:
        raise ControllerHOLD("controller_session_metadata_limit")
    return raw.decode("utf-8")


def _schema(conn, *, create=False):
    row = conn.execute("SELECT type,name,sql FROM main.sqlite_master "
                       "WHERE name=? COLLATE NOCASE", ("controller_sessions",)).fetchone()
    if row is None and create:
        if not conn.in_transaction:
            raise ControllerHOLD("controller_session_transaction_required")
        conn.execute(_SQL)
        return _schema(conn)
    norm = lambda value: " ".join(str(value).strip().rstrip(";").split())
    if (row is None or tuple(row[:2]) != ("table", "controller_sessions")
            or norm(row[2]) != norm(_SQL)):
        raise ControllerHOLD("controller_session_schema_invalid")
    if conn.execute("SELECT 1 FROM main.sqlite_master WHERE type='trigger' "
                    "AND tbl_name=? COLLATE NOCASE", ("controller_sessions",)).fetchone():
        raise ControllerHOLD("controller_session_trigger_unsupported")


def _row(conn, controller_id):
    _schema(conn)
    row = conn.execute("SELECT controller_id, "
        "CASE WHEN typeof(binding_json)='text' AND length(CAST(binding_json AS BLOB))<=65536 "
        "THEN binding_json ELSE '' END, "
        "CASE WHEN request_json IS NULL THEN NULL WHEN typeof(request_json)='text' "
        "AND length(CAST(request_json AS BLOB))<=65536 THEN request_json ELSE '' END, "
        "CASE WHEN ack_json IS NULL THEN NULL WHEN typeof(ack_json)='text' "
        "AND length(CAST(ack_json AS BLOB))<=65536 THEN ack_json ELSE '' END "
        "FROM main.controller_sessions WHERE controller_id=? COLLATE BINARY",
        (controller_id,)).fetchone()
    if row is None:
        raise ControllerHOLD("controller_session_missing")
    return tuple(row)


def _request(binding, reason="operator_stop"):
    return {"schema": 1, "kind": "controller_stop", "request_id": secrets.token_hex(24),
            "controller_id": binding["controller_id"],
            "binding_sha256": _sha(_encode(binding).encode()), "reason": reason}


def _validate_request(raw, binding):
    try:
        value = json.loads(raw)
        if (type(value) is not dict or set(value) != {
                "schema", "kind", "request_id", "controller_id", "binding_sha256", "reason"}
                or type(value["schema"]) is not int or value["schema"] != 1
                or value["kind"] != "controller_stop"
                or value["controller_id"] != binding["controller_id"]
                or value["binding_sha256"] != _sha(_encode(binding).encode())
                or type(value["request_id"]) is not str or len(value["request_id"]) != 48
                or any(c not in "0123456789abcdef" for c in value["request_id"])
                or value["reason"] not in {"operator_stop", "member_limit", "normal_exit", "signal"}
                or _encode(value) != raw):
            raise ValueError("invalid")
        return value
    except (ValueError, TypeError, KeyError) as exc:
        raise ControllerHOLD("controller_stop_request_invalid") from exc


def _file_witness(path, limit=4096):
    path, identity = _path(path, directory=False)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if before.st_size > limit or (before.st_dev, before.st_ino) != identity:
            raise ControllerHOLD("controller_resource_file_changed")
        raw = os.read(fd, limit + 1)
        after = os.fstat(fd)
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)
        if len(raw) > limit or stamp(before) != stamp(after) or stamp(after) != stamp(path.lstat()):
            raise ControllerHOLD("controller_resource_file_changed")
        return (str(path), stamp(after), _sha(raw)), raw
    finally:
        os.close(fd)


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fresh_inventory(orze, ctx=None):
    """This first profile cannot import an earlier execution ownership era."""
    conn = orze.lake.conn
    if conn.in_transaction:
        raise ControllerHOLD("controller_start_transaction_active")
    if conn.execute("SELECT 1 FROM main.sqlite_master WHERE name='execution_attempts'").fetchone():
        if conn.execute("SELECT 1 FROM main.execution_attempts LIMIT 1").fetchone():
            raise ControllerHOLD("controller_prior_execution_requires_recovery")
    if conn.execute("SELECT 1 FROM main.ideas WHERE status='running' LIMIT 1").fetchone():
        raise ControllerHOLD("controller_prior_running_owner")
    # Queued rows and input documents are supported; old task/owner folders
    # are not silently imported as an empty current membership inventory.
    for number, path in enumerate(orze.results_dir.iterdir()):
        if number >= 4096:
            raise ControllerHOLD("controller_initial_scope_limit")
        info = path.lstat()
        owner_root = (getattr(ctx, "_anchor_lease", None) or ctx._lease).lock_dir if ctx is not None else None
        if ctx is not None and path.absolute() == owner_root:
            ctx._paths()
            continue
        if (path.name.startswith((".orze.pid", ".orze_heartbeat", ".orze_leader", "heartbeat_"))
                or path.name in {".orze_disabled", ".orze_stop_all", ".orze_shutdown"}):
            raise ControllerHOLD("controller_prior_control_requires_recovery")
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ControllerHOLD("controller_prior_scope_requires_recovery")


class ControllerSession:
    def __init__(self, orze):
        from orze.core.controller_profile import controller_profile, profile_fingerprint
        from orze.idea_lake import IdeaLake

        if controller_profile(orze.cfg) is None or type(orze.lake) is not IdeaLake:
            raise ControllerHOLD("controller_profile_real_lake_required")
        self.orze = orze
        self._thread_id = threading.get_ident()
        self._workdir = _path(Path.cwd(), directory=True)
        if str(self._workdir[0]) != orze.cfg.get("_controller_workdir", str(self._workdir[0])):
            raise ControllerHOLD("controller_loaded_workdir_changed")
        self._fingerprint = profile_fingerprint(orze.cfg, orze.gpu_ids)
        if orze.cfg.get("_controller_profile_fingerprint") != self._fingerprint:
            raise ControllerHOLD("controller_loaded_configuration_changed")
        from orze.engine.controller_handoff import _admission_for_configuration
        self._admission = _admission_for_configuration(orze.cfg)
        if self._admission is None:
            _fresh_inventory(orze)
        else:
            self._admission.validate_prior(orze)
        self._request_json = None
        self._ack_json = None
        self._stopping = threading.Event()
        self._thread = None
        self._started = False
        self._finished = False
        self._failed = False
        self._gpu = None
        self._gpu_witness = None
        self._pid_file = None
        self._pid_witness = None
        self._lake = orze.lake
        self._lake_conn = orze.lake.conn
        try:
            if controller_profile(orze.cfg)["version"] == 2:
                self.ctx = register_controller(orze.lake, orze.results_dir, protocol=2,
                    admission=self._admission, session_registrar=self._register_binding)
            else:
                self.ctx = register_controller(orze.lake, orze.results_dir)
                with self.ctx._connection(write=True) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    self._register_binding(conn, self.ctx)
                    conn.commit()
            with self.ctx.guard():
                if self._admission is None:
                    _fresh_inventory(orze, self.ctx)
                else:
                    self._admission.verify_consumed_in_context(self.ctx)
            with self.ctx._connection() as conn:
                if _row(conn, self.ctx.controller_id) != (
                        self.ctx.controller_id, self._binding_json, None, None):
                    raise ControllerHOLD("controller_session_registration_unconfirmed")
        except BaseException as exc:
            if getattr(self, "ctx", None) is not None:
                self.fail(exc)
            raise

    def _register_binding(self, conn, ctx):
        """Only bounded SQL inside the registration writer; no poll or OS work."""
        from orze.core.controller_profile import controller_profile
        self.ctx = ctx
        _SESSIONS[ctx.controller_id] = self
        self._binding = {"schema": 1, "controller_id": ctx.controller_id,
            "identity": ctx.identity, "profile": controller_profile(self.orze.cfg),
            "config_sha256": self._fingerprint, "physical_gpus": sorted(self.orze.gpu_ids),
            "workdir": str(self._workdir[0]), "workdir_device": self._workdir[1][0],
            "workdir_inode": self._workdir[1][1]}
        self._binding_json = _encode(self._binding)
        ctx._runtime_validator = self._validate_runtime
        _schema(conn, create=True)
        conn.execute("INSERT INTO main.controller_sessions VALUES (?,?,NULL,NULL)",
                     (ctx.controller_id, self._binding_json))

    def _validate_runtime(self):
        from orze.core.controller_profile import profile_fingerprint
        if (_SESSIONS.get(self.ctx.controller_id) is not self
                or _path(Path.cwd(), directory=True) != self._workdir
                or (self._started and getattr(self.orze, "_controller_session", None) is not self)
                or self.orze.lake is not self._lake or self._lake.conn is not self._lake_conn
                or profile_fingerprint(self.orze.cfg, self.orze.gpu_ids) != self._fingerprint):
            raise ControllerHOLD("controller_frozen_runtime_changed")

    @contextmanager
    def _connection(self, *, write=False):
        with self.ctx._connection(write=write) as conn:
            if write:
                conn.execute("BEGIN IMMEDIATE")
            self.ctx.poll_control(conn)
            row = _row(conn, self.ctx.controller_id)
            if row[:2] != (self.ctx.controller_id, self._binding_json):
                raise ControllerHOLD("controller_session_binding_changed")
            if self._request_json is not None and row[2] != self._request_json:
                raise ControllerHOLD("controller_stop_request_replaced")
            if row[3] != self._ack_json:
                raise ControllerHOLD("controller_stop_ack_replaced")
            yield conn, row
            if write:
                conn.commit()

    def start(self):
        if self._thread is not None or threading.get_ident() != self._thread_id:
            raise ControllerHOLD("controller_request_pump_reentry")
        self._thread = threading.Thread(target=self._pump, name="orze-controller-stop", daemon=True)
        self._started = True
        self._thread.start()

    def _pump(self):
        while not self._stopping.wait(0.05):
            try:
                # Signal handlers only set memory flags. The pump performs
                # SQLite work later, never reentrantly inside a native writer.
                if (self._request_json is None and not self.orze.running
                        and self.orze._stop_event.is_set()):
                    self.request_local_stop("signal")
                self._poll_request()
            except BaseException as exc:
                self.fail(exc)
                return

    def _poll_request(self):
        with self.ctx.guard():
            with self._connection() as (_, row):
                raw = row[2]
            if raw is None:
                return
            request = _validate_request(raw, self._binding)
            self._request_json = raw  # Pinned before any quiesce write.
            self.ctx.quiesce(request["request_id"])
            self.orze._stop_kill_all = True
            self.orze.running = False
            self.orze._stop_event.set()

    def request_local_stop(self, reason):
        with self.ctx.guard():
            if self._request_json is not None:
                # A pinned stop remains read-only. Native shutdown still owns
                # its result transaction; polling must not compete for its
                # writer reservation or reinterpret ordinary contention as
                # lost execution ownership.
                self._poll_request()
                return
            with self._connection(write=True) as (conn, row):
                if row[2] is None:
                    raw = _encode(_request(self._binding, reason))
                    _validate_request(raw, self._binding)
                    if conn.execute("UPDATE main.controller_sessions SET request_json=? "
                        "WHERE controller_id=? AND request_json IS NULL AND ack_json IS NULL",
                        (raw, self.ctx.controller_id)).rowcount != 1:
                        raise ControllerHOLD("controller_stop_request_write_unconfirmed")
            self._poll_request()

    def bind_gpu_leases(self, leases):
        from orze.core.gpu_lease import GpuLeaseSet, _registered, _registry_lock
        with self.ctx.guard(), _registry_lock:
            if type(leases) is not GpuLeaseSet or self._gpu is not None or leases._closed:
                raise ControllerHOLD("controller_gpu_resource_invalid")
            self._gpu = leases  # Retain before any fallible witness read.
            if sorted(leases.gpus) != sorted(self.orze.gpu_ids):
                raise ControllerHOLD("controller_gpu_resource_scope_changed")
            self._gpu_witness = []
            for lease in leases._leases:
                info = os.fstat(lease.fd)
                if lease._closed or _registered.get(lease.gpu) is not lease:
                    raise ControllerHOLD("controller_gpu_resource_owner_changed")
                self._gpu_witness.append((lease, lease.fd, info.st_dev, info.st_ino))

    def bind_leader(self, handle):
        # Registration is the sole local-profile leadership authority. The
        # legacy age-based leader mechanism is deliberately not joined.
        if handle is not None:
            self.ctx.hold("controller_legacy_leader_unsupported")
            raise ControllerHOLD("controller_legacy_leader_unsupported")

    def bind_pid_file(self, path):
        with self.ctx.guard():
            if self._pid_file is not None:
                raise ControllerHOLD("controller_pid_resource_rebound")
            self._pid_file = Path(path)
            self._pid_witness, raw = _file_witness(path)
            if int(raw.strip()) != os.getpid():
                raise ControllerHOLD("controller_pid_resource_invalid")

    def _close_resources(self):
        from orze.core.gpu_lease import _registered, _registry_lock
        self._validate_runtime()
        if getattr(self.orze, "_leader_handle", None) is not None:
            raise ControllerHOLD("controller_legacy_leader_unsupported")
        if self._lake_conn.in_transaction:
            raise ControllerHOLD("controller_lake_transaction_pending")
        self._lake.close()
        try:
            self._lake_conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            pass
        else:
            raise ControllerHOLD("controller_lake_close_unconfirmed")
        if self._gpu is not None:
            with _registry_lock:
                if (self.orze._gpu_leases is not self._gpu or self._gpu._closed
                        or self._gpu_witness is None
                        or len(self._gpu._leases) != len(self._gpu_witness)
                        or any(lease is not captured[0]
                               for lease, captured in zip(self._gpu._leases, self._gpu_witness))):
                    raise ControllerHOLD("controller_gpu_resource_changed")
                for lease, fd, dev, ino in self._gpu_witness:
                    info = os.fstat(fd)
                    if (lease._closed or lease.fd != fd or _registered.get(lease.gpu) is not lease
                            or (info.st_dev, info.st_ino) != (dev, ino)):
                        raise ControllerHOLD("controller_gpu_resource_changed")
                self._gpu.close()
                for lease, fd, _, _ in self._gpu_witness:
                    if not lease._closed or _registered.get(lease.gpu) is lease:
                        raise ControllerHOLD("controller_gpu_release_unconfirmed")
                    try:
                        os.fstat(fd)
                    except OSError as exc:
                        if exc.errno != errno.EBADF:
                            raise
                    else:
                        raise ControllerHOLD("controller_gpu_close_unconfirmed")
                self.orze._gpu_leases = None
        elif self.orze._gpu_leases is not None:
            raise ControllerHOLD("controller_gpu_resource_missing")
        if self._pid_file is not None:
            witness, _ = _file_witness(self._pid_file)
            if witness != self._pid_witness:
                raise ControllerHOLD("controller_pid_resource_changed")
            self._pid_file.unlink()
            _sync_directory(self._pid_file.parent)
        return {"lake": "closed", "gpu_scope": sorted(self.orze.gpu_ids),
                "gpu_leases": "closed" if self._gpu is not None else "not_acquired",
                "pid_file": "removed" if self._pid_file is not None else "not_created",
                "leadership": "persistent_registration_retained", "request_pump": "joined"}

    def finish(self):
        from orze.engine.controller_members import prove_drained
        if threading.get_ident() != self._thread_id or self._finished or self._failed:
            raise ControllerHOLD("controller_finalizer_not_authorized")
        try:
            self.request_local_stop("normal_exit")
            self._stopping.set()
            if self._thread is not None:
                self._thread.join(timeout=2)
                if self._thread.is_alive():
                    raise ControllerHOLD("controller_request_pump_not_joined")
            with self.ctx.guard():
                proof = prove_drained(self.ctx)
                resources = self._close_resources()
                request = _validate_request(self._request_json, self._binding)
                ack = {"schema": 1, "kind": "controller_drained", "controller_id": self.ctx.controller_id,
                    "request_id": request["request_id"], "request_sha256": _sha(self._request_json.encode()),
                    "binding_sha256": _sha(self._binding_json.encode()), "members": proof, "resources": resources}
                encoded = _encode(ack)
                with self._connection(write=True) as (conn, row):
                    if row[2] != self._request_json or row[3] is not None:
                        raise ControllerHOLD("controller_ack_state_changed")
                    if conn.execute("UPDATE main.controller_sessions SET ack_json=? "
                        "WHERE controller_id=? AND binding_json=? AND request_json=? AND ack_json IS NULL",
                        (encoded, self.ctx.controller_id, self._binding_json, self._request_json)).rowcount != 1:
                        raise ControllerHOLD("controller_ack_write_unconfirmed")
                self._ack_json = encoded
                with self._connection() as (_, row):
                    if row[3] != encoded:
                        raise ControllerHOLD("controller_ack_commit_unconfirmed")
                self._finished = True
                return json.loads(encoded)
        except BaseException as exc:
            self.fail(exc)
            raise

    def fail(self, exc):
        self._failed = True
        self._stopping.set()
        self.orze.running = False
        self.orze._stop_event.set()
        self.ctx.hold("controller_product_shutdown_unconfirmed")


@dataclass(frozen=True)
class CompletedControllerStop:
    """Informational verified-stop result. Not a start/restart capability."""
    controller_id: str
    request_id: str
    ack_sha256: str
    config_sha256: str
    scope: str
    database: str
    observed_process: dict

    def __bool__(self):
        raise TypeError("CompletedControllerStop is not restart authority")


class _Observer:
    def __init__(self, cfg):
        from orze.core.controller_profile import controller_profile, profile_fingerprint
        if controller_profile(cfg) is None:
            raise ControllerHOLD("controller_stop_profile_required")
        self.scope, self.scope_witness = _path(cfg["results_dir"], directory=True)
        self.db, self.db_witness = _path(cfg["idea_lake_db"], directory=False)
        self.fingerprint = profile_fingerprint(cfg)
        if cfg.get("_controller_profile_fingerprint") != self.fingerprint:
            raise ControllerHOLD("controller_loaded_configuration_changed")
        self.pidfd = None
        self._history_only = False
        self._protocol = controller_profile(cfg)["version"]
        workdir, workdir_witness = _path(Path.cwd(), directory=True)
        if str(workdir) != cfg.get("_controller_workdir", str(workdir)):
            raise ControllerHOLD("controller_loaded_workdir_changed")
        with self.connection() as conn:
            from orze.engine.controller_control import current_instance, registration_version
            if registration_version(conn) != self._protocol:
                raise ControllerHOLD("controller_stop_storage_profile_mismatch")
            current = current_instance(conn, self.scope)
            if current[3] not in {"ACTIVE", "QUIESCING"} or current[5] is not None:
                raise ControllerHOLD("controller_stop_registration_unavailable")
            self.controller_id, self.identity_json = current[0], current[2]
            self.identity = json.loads(self.identity_json)
            row = _row(conn, self.controller_id)
        self.binding_json = row[1]
        self.binding = json.loads(self.binding_json)
        expected = {"schema": 1, "controller_id": self.controller_id, "identity": self.identity,
                    "profile": controller_profile(cfg), "config_sha256": self.fingerprint,
                    "physical_gpus": sorted((cfg.get("gpu_scheduling") or {}).get("allowed_gpus") or []),
                    "workdir": str(workdir), "workdir_device": workdir_witness[0],
                    "workdir_inode": workdir_witness[1]}
        if self.binding != expected or _encode(expected) != self.binding_json:
            raise ControllerHOLD("controller_stop_binding_mismatch")
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if (type(self.identity.get("schema")) is not int or self.identity["schema"] != self._protocol
                or self.identity.get("controller_id") != self.controller_id
                or self.identity.get("host") != socket.gethostname() or self.identity.get("boot_id") != boot
                or self.identity.get("scope") != str(self.scope)
                or (self.identity.get("scope_device"), self.identity.get("scope_inode")) != self.scope_witness
                or self.identity.get("database") != str(self.db)
                or (self.identity.get("database_device"), self.identity.get("database_inode")) != self.db_witness):
            raise ControllerHOLD("controller_stop_identity_mismatch")
        self._owner_witness = self.owner_witness()
        process = self.identity.get("process")
        if (type(process) is not dict or set(process) != {"pid", "start_ticks"}
                or any(type(process[k]) is not int or process[k] <= 0 for k in process)):
            raise ControllerHOLD("controller_stop_process_invalid")
        if process_identity(process["pid"])[0] != process:
            raise ControllerHOLD("controller_stop_process_changed")
        try:
            self.pidfd = os.pidfd_open(process["pid"], 0)
            if process_identity(process["pid"])[0] != process or self.exited():
                raise ControllerHOLD("controller_stop_capture_unconfirmed")
            self.check()
        except BaseException:
            self.close()
            raise

    @contextmanager
    def connection(self, *, write=False):
        if (_path(self.scope, directory=True) != (self.scope, self.scope_witness)
                or _path(self.db, directory=False) != (self.db, self.db_witness)):
            raise ControllerHOLD("controller_stop_route_changed")
        conn = sqlite3.connect(self.db.as_uri() + ("?mode=rw" if write else "?mode=ro"),
                               uri=True, timeout=0.25)
        try:
            if _route(conn) != (self.db, self.db_witness):
                raise ControllerHOLD("controller_stop_route_changed")
            if write:
                conn.execute("BEGIN IMMEDIATE")
            else:
                conn.execute("PRAGMA query_only=ON")
            yield conn
            if write:
                conn.commit()
            if (_path(self.scope, directory=True) != (self.scope, self.scope_witness)
                    or _route(conn) != (self.db, self.db_witness)):
                raise ControllerHOLD("controller_stop_route_changed")
        finally:
            conn.close()

    def owner_witness(self):
        lock = self.scope / "_controller_registration.lock"
        if self._protocol == 2:
            generation = self.identity.get("generation")
            if type(generation) is not int or generation < 0:
                raise ControllerHOLD("controller_stop_generation_invalid")
            owner = lock / f"instance-{generation}-{self.controller_id}.lock"
            if (self.identity.get("anchor_directory") != str(lock)
                    or self.identity.get("owner_directory") != str(owner)
                    or _path(lock, directory=True)[1] != (
                        self.identity.get("anchor_device"), self.identity.get("anchor_inode"))):
                raise ControllerHOLD("controller_stop_owner_changed")
            anchor_meta, anchor_raw = _file_witness(lock / "lock.json")
            anchor_marker, anchor_bytes = _file_witness(lock.with_name(lock.name + ".source-lock"))
            if (_sha(anchor_raw) != self.identity.get("anchor_metadata_sha256")
                    or anchor_bytes != b"orze-idea-source-lock-v1\n"):
                raise ControllerHOLD("controller_stop_owner_changed")
            directory = _path(owner, directory=True)
            meta, raw = _file_witness(owner / "lock.json")
            marker, marker_raw = _file_witness(owner.with_name(owner.name + ".source-lock"))
            if (_sha(raw) != self.identity.get("owner_metadata_sha256")
                    or marker_raw != b"orze-idea-source-lock-v1\n"):
                raise ControllerHOLD("controller_stop_owner_changed")
            return (_path(lock, directory=True), anchor_meta, anchor_marker, directory, meta, marker)
        directory = _path(lock, directory=True)
        meta, raw = _file_witness(lock / "lock.json")
        marker, marker_raw = _file_witness(lock.with_name(lock.name + ".source-lock"))
        if (_sha(raw) != self.identity.get("owner_metadata_sha256")
                or marker_raw != b"orze-idea-source-lock-v1\n"):
            raise ControllerHOLD("controller_stop_owner_changed")
        return directory, meta, marker

    def check(self, conn=None):
        if conn is None:
            with self.connection() as actual:
                return self.check(actual)
        _registration_schema(conn)
        if self._protocol == 2 and not self._history_only:
            from orze.engine.controller_control import current_instance
            if current_instance(conn, self.scope)[0] != self.controller_id:
                raise ControllerHOLD("controller_stop_current_head_changed")
        row = conn.execute("SELECT identity_json,phase,request_id,hold_reason "
            "FROM main.controller_instances WHERE controller_id=? COLLATE BINARY",
            (self.controller_id,)).fetchone()
        if (row is None or row[0] != self.identity_json or row[1] not in {"ACTIVE", "QUIESCING"}
                or row[3] is not None or self.owner_witness() != self._owner_witness):
            raise ControllerHOLD("controller_stop_registration_changed")
        session = _row(conn, self.controller_id)
        if session[:2] != (self.controller_id, self.binding_json):
            raise ControllerHOLD("controller_stop_session_changed")
        return tuple(row), session

    def exited(self):
        return bool(select.select([self.pidfd], [], [], 0)[0])

    def verify_drain(self, ack):
        """Re-read durable members and native effects; labels alone cannot ACK."""
        from orze.engine.controller_members import _schema as member_schema, MAX_MEMBERS, _NATIVE, _REPORTS
        from orze.core import execution_attempts as attempts
        from orze.engine.attempt_effect_receipts import _scan

        resources = ack["resources"]
        if (type(resources) is not dict or set(resources) != {
                "lake", "gpu_scope", "gpu_leases", "pid_file", "leadership", "request_pump"}
                or resources["lake"] != "closed" or resources["gpu_scope"] != self.binding["physical_gpus"]
                or resources["gpu_leases"] not in {"closed", "not_acquired"}
                or resources["pid_file"] not in {"removed", "not_created"}
                or resources["leadership"] != "persistent_registration_retained"
                or resources["request_pump"] != "joined"):
            raise ControllerHOLD("controller_stop_resources_invalid")
        proof = ack["members"]
        if (type(proof) is not dict or set(proof) != {
                "schema", "controller_id", "member_count", "members_sha256"}
                or type(proof["schema"]) is not int or proof["schema"] != 1
                or proof["controller_id"] != self.controller_id
                or type(proof["member_count"]) is not int or not 0 <= proof["member_count"] <= MAX_MEMBERS):
            raise ControllerHOLD("controller_stop_member_proof_invalid")
        digest = hashlib.sha256()
        effects = {}
        with self.connection() as conn:
            self.check(conn)
            member_schema(conn)
            rows = conn.execute("SELECT member_id, CASE WHEN typeof(payload_json)='text' "
                "AND length(CAST(payload_json AS BLOB))<=65536 THEN payload_json ELSE '' END "
                "FROM main.controller_members WHERE controller_id=? ORDER BY member_id COLLATE BINARY LIMIT ?",
                (self.controller_id, MAX_MEMBERS + 1)).fetchall()
            if len(rows) != proof["member_count"]:
                raise ControllerHOLD("controller_stop_members_changed")
            for key, raw in rows:
                value = json.loads(raw)
                if (type(value) is not dict or _encode(value) != raw
                        or value.get("member_id") != key or value.get("controller_id") != self.controller_id
                        or value.get("action_state") != "SETTLED" or value.get("hold_reason") is not None
                        or value.get("os_state") not in {"CLOSED", "NO_EXECUTION", "NOT_REQUIRED"}):
                    raise ControllerHOLD("controller_stop_member_unsettled")
                if value["os_state"] == "CLOSED":
                    closure = value.get("closure")
                    if (type(closure) is not dict or closure.get("event") != "TREE_CLOSED"
                            or closure.get("wait_proof") != "ECHILD_WALL"
                            or closure.get("binding") != value.get("ready")):
                        raise ControllerHOLD("controller_stop_member_closure_invalid")
                if value.get("kind") in _NATIVE | _REPORTS:
                    ref = attempts.AttemptRef(**value["identity"]["attempt_ref"])
                    if not attempts._schema(conn):
                        raise ControllerHOLD("controller_stop_attempt_missing")
                    row = attempts._row(conn.execute(attempts._SELECT +
                        " WHERE attempt_id=? COLLATE BINARY", (ref.attempt_id,)).fetchone())
                    if (row is None or {k: row[k] for k in ("task_id", "phase", "attempt_id", "generation")}
                            != value["identity"]["attempt_ref"] or row["state"] not in {"TERMINAL", "NOT_STARTED"}
                            or _sha(_encode(row).encode()) != value.get("terminal_sha256")):
                        raise ControllerHOLD("controller_stop_terminal_changed")
                    if ref.task_id not in effects:
                        effects[ref.task_id] = _scan(self.scope / ref.task_id)
                    if effects[ref.task_id].get(ref.attempt_id) != (row["terminal"].get("effect_receipt_sha256"), True):
                        raise ControllerHOLD("controller_stop_effect_changed")
                digest.update(canonical([key, _sha(raw.encode())]))
        if digest.hexdigest() != proof["members_sha256"]:
            raise ControllerHOLD("controller_stop_members_digest_changed")

    def close(self):
        if self.pidfd is not None:
            os.close(self.pidfd)
            self.pidfd = None


def _stop_observed(observer, timeout):
    """Private shared stop operation; its caller retains the actual pidfd."""
    deadline = time.monotonic() + timeout
    with observer.connection(write=True) as conn:
        registration, row = observer.check(conn)
        if observer.exited():
            raise ControllerHOLD("controller_exited_before_stop_request")
        if row[2] is None:
            request_json = _encode(_request(observer.binding))
            if conn.execute("UPDATE main.controller_sessions SET request_json=? "
                "WHERE controller_id=? AND binding_json=? AND request_json IS NULL AND ack_json IS NULL",
                (request_json, observer.controller_id, observer.binding_json)).rowcount != 1:
                raise ControllerHOLD("controller_stop_request_unconfirmed")
        else:
            request_json = row[2]
        request = _validate_request(request_json, observer.binding)
    while True:
        registration, row = observer.check()
        if row[2] != request_json:
            raise ControllerHOLD("controller_stop_request_replaced")
        if row[3] is not None:
            ack = json.loads(row[3])
            if (type(ack) is not dict or set(ack) != {"schema", "kind", "controller_id", "request_id",
                    "request_sha256", "binding_sha256", "members", "resources"}
                    or type(ack["schema"]) is not int or ack["schema"] != 1
                    or ack["kind"] != "controller_drained" or ack["controller_id"] != observer.controller_id
                    or ack["request_id"] != request["request_id"]
                    or ack["request_sha256"] != _sha(request_json.encode())
                    or ack["binding_sha256"] != _sha(observer.binding_json.encode())
                    or registration[1:3] != ("QUIESCING", request["request_id"])
                    or _encode(ack) != row[3]):
                raise ControllerHOLD("controller_stop_ack_invalid")
            if observer.exited():
                # Re-read after observing kernel exit: a replaced request,
                # registration or ACK is never consumed from a stale read.
                after_registration, after_row = observer.check()
                if after_registration != registration or after_row != row:
                    raise ControllerHOLD("controller_stop_final_readback_changed")
                observer.verify_drain(ack)
                if observer.check() != (after_registration, after_row):
                    raise ControllerHOLD("controller_stop_final_readback_changed")
                return CompletedControllerStop(observer.controller_id, request["request_id"],
                    _sha(row[3].encode()), observer.fingerprint, str(observer.scope), str(observer.db),
                    dict(observer.identity["process"]))
        elif observer.exited():
            raise ControllerHOLD("controller_exited_without_ack")
        if time.monotonic() >= deadline:
            raise ControllerHOLD("controller_stop_timeout_unconfirmed")
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))


def stop_controller(cfg, timeout=60):
    """Observe an already registered live controller; never signal raw PIDs."""
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ControllerHOLD("controller_stop_timeout_invalid")
    observer = None
    try:
        observer = _Observer(cfg)
        return _stop_observed(observer, timeout)
    except ControllerHOLD:
        raise
    except Exception as exc:
        raise ControllerHOLD("controller_stop_observation_unconfirmed") from exc
    finally:
        if observer is not None:
            observer.close()
