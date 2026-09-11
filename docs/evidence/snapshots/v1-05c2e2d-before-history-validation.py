"""One-shot local controller handoff; no PID-label or retry-on-unknown grant.

The issuer owns a live captured observer until the old instance has drained
and exited. A successor starts only through a create-only durable launch
intent, kernel-credentialled channel, and current-head transaction. Neither
an informational result nor a disconnected bootstrap channel is authority.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import select
import socket
import sqlite3
import struct
import subprocess
import sys
import threading
import time

from orze.engine.controller_control import ControllerHOLD, _path, _route, _token
from orze.engine.supervisor_worker import canonical, process_identity


_FD_ENV = "ORZE_CONTROLLER_HANDOFF_FD"
_LIMIT = 65536
_ADMISSION = None
_ISSUERS = {}
_SQL = """CREATE TABLE controller_handoffs (
    grant_id TEXT NOT NULL COLLATE BINARY PRIMARY KEY,
    request_id TEXT NOT NULL COLLATE BINARY UNIQUE,
    scope TEXT NOT NULL COLLATE BINARY,
    source_controller_id TEXT NOT NULL COLLATE BINARY UNIQUE,
    target_controller_id TEXT NOT NULL COLLATE BINARY UNIQUE,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('RESERVED','SPAWNING','ISSUED','CONSUMED','PREPARED','STARTED','HOLD')),
    ready_json TEXT,
    started_json TEXT,
    hold_reason TEXT,
    CHECK ((state = 'HOLD') = (hold_reason IS NOT NULL))
)"""


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _encode(value):
    raw = canonical(value)
    if len(raw) > _LIMIT:
        raise ControllerHOLD("controller_handoff_metadata_limit")
    return raw.decode("utf-8")


def _decode(raw):
    try:
        value = json.loads(raw)
        if type(value) is not dict or _encode(value) != raw:
            raise ValueError("noncanonical")
        return value
    except (TypeError, ValueError) as exc:
        raise ControllerHOLD("controller_handoff_metadata_invalid") from exc


def _schema(conn, *, create=False):
    row = conn.execute("SELECT type,name,sql FROM main.sqlite_master "
                       "WHERE name=? COLLATE NOCASE", ("controller_handoffs",)).fetchone()
    if row is None and create:
        if not conn.in_transaction:
            raise ControllerHOLD("controller_handoff_transaction_required")
        conn.execute(_SQL)
        return _schema(conn)
    norm = lambda value: " ".join(str(value).strip().rstrip(";").split())
    if (row is None or tuple(row[:2]) != ("table", "controller_handoffs")
            or norm(row[2]) != norm(_SQL)):
        raise ControllerHOLD("controller_handoff_schema_invalid")
    if conn.execute("SELECT 1 FROM main.sqlite_master WHERE type='trigger' "
                    "AND tbl_name=? COLLATE NOCASE", ("controller_handoffs",)).fetchone():
        raise ControllerHOLD("controller_handoff_trigger_unsupported")


def _row(conn, grant_id=None, *, request_id=None):
    _schema(conn)
    column, key = ("grant_id", grant_id) if grant_id is not None else ("request_id", request_id)
    row = conn.execute("SELECT grant_id,request_id,scope,source_controller_id,target_controller_id,"
        "CASE WHEN typeof(payload_json)='text' AND length(CAST(payload_json AS BLOB))<=65536 "
        "THEN payload_json ELSE '' END,state,"
        "CASE WHEN ready_json IS NULL THEN NULL WHEN typeof(ready_json)='text' "
        "AND length(CAST(ready_json AS BLOB))<=65536 THEN ready_json ELSE '' END,"
        "CASE WHEN started_json IS NULL THEN NULL WHEN typeof(started_json)='text' "
        "AND length(CAST(started_json AS BLOB))<=65536 THEN started_json ELSE '' END,hold_reason "
        "FROM main.controller_handoffs WHERE " + column + "=? COLLATE BINARY", (key,)).fetchone()
    if row is None:
        return None
    if (row[6] not in {"RESERVED", "SPAWNING", "ISSUED", "CONSUMED", "PREPARED", "STARTED", "HOLD"}
            or (row[6] == "HOLD") != (row[9] is not None)):
        raise ControllerHOLD("controller_handoff_state_invalid")
    return tuple(row)


def _timeout(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 3600:
        raise ControllerHOLD("controller_handoff_timeout_invalid")
    return float(value)


def _profile(cfg):
    from orze.core.controller_profile import controller_profile, profile_fingerprint
    if controller_profile(cfg) != {"version": 2, "profile": "local_handoff_v1"}:
        raise ControllerHOLD("controller_handoff_profile_required")
    fingerprint = profile_fingerprint(cfg)
    if cfg.get("_controller_profile_fingerprint") != fingerprint:
        raise ControllerHOLD("controller_loaded_configuration_changed")
    if str(Path.cwd()) != cfg.get("_controller_workdir"):
        raise ControllerHOLD("controller_loaded_workdir_changed")
    return fingerprint


def _send(channel, value, deadline=None):
    raw = _encode(value).encode()
    remaining = 1.0 if deadline is None else deadline - time.monotonic()
    if remaining <= 0:
        raise ControllerHOLD("controller_handoff_channel_timeout")
    previous = channel.gettimeout()
    try:
        channel.settimeout(remaining)
        if channel.send(raw) != len(raw):
            raise ControllerHOLD("controller_handoff_packet_short_write")
    finally:
        channel.settimeout(previous)


def _receive(channel, peer_pid, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not select.select([channel], [], [], remaining)[0]:
        raise ControllerHOLD("controller_handoff_channel_timeout")
    raw, ancillary, flags, _ = channel.recvmsg(_LIMIT + 1, socket.CMSG_SPACE(struct.calcsize("3i")))
    if (not raw or len(raw) > _LIMIT or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
            or len(ancillary) != 1):
        raise ControllerHOLD("controller_handoff_packet_unconfirmed")
    level, kind, credentials = ancillary[0]
    if (level != socket.SOL_SOCKET or kind != socket.SCM_CREDENTIALS
            or len(credentials) != struct.calcsize("3i")
            or struct.unpack("3i", credentials) != (peer_pid, os.getuid(), os.getgid())):
        raise ControllerHOLD("controller_handoff_peer_unconfirmed")
    try:
        return _decode(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ControllerHOLD("controller_handoff_packet_invalid") from exc


def _new_channel():
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    for channel in (parent, child):
        channel.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        channel.set_inheritable(False)
    return parent, child


def _inherited_channel():
    raw = os.environ.get(_FD_ENV)
    if raw is None or not raw.isascii() or not raw.isdecimal() or not 3 <= int(raw) <= 1048576:
        raise ControllerHOLD("controller_handoff_fd_invalid")
    channel = socket.socket(fileno=int(raw))
    if channel.family != socket.AF_UNIX or channel.type != socket.SOCK_SEQPACKET:
        channel.detach()  # Invalid input is not authority to close an arbitrary FD.
        raise ControllerHOLD("controller_handoff_channel_invalid")
    channel.set_inheritable(False)
    channel.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
    os.environ.pop(_FD_ENV, None)  # Never leak bootstrap admission into workers.
    return channel


def _validated_admission(admission, lake, scope):
    # The actual class and global owner are populated only by the sealed
    # successor entry. Arbitrary mappings, copied objects and fork copies are
    # never a backdoor through the permanent namespace.
    if (admission is None or admission is not _ADMISSION
            or type(admission) is not _Admission or admission._pid != os.getpid()):
        raise ControllerHOLD("controller_handoff_admission_invalid")
    admission.validate(lake, scope)
    return admission


def _admission_for_configuration(cfg):
    if _ADMISSION is None:
        return None
    if type(_ADMISSION) is not _Admission or _ADMISSION._pid != os.getpid():
        raise ControllerHOLD("controller_handoff_admission_invalid")
    if _profile(cfg) != _ADMISSION._fingerprint:
        raise ControllerHOLD("controller_handoff_configuration_changed")
    return _ADMISSION


@dataclass(frozen=True)
class CompletedControllerHandoff:
    """A verified operation record, never another launch capability."""
    request_id: str
    grant_id: str
    source_controller_id: str
    target_controller_id: str
    generation: int
    started_sha256: str

    def __bool__(self):
        raise TypeError("CompletedControllerHandoff is not launch authority")


class _Route:
    def __init__(self, cfg):
        from orze.engine.controller_session import _file_witness
        self.cfg = cfg
        self.fingerprint = _profile(cfg)
        self.scope, self.scope_witness = _path(cfg["results_dir"], directory=True)
        self.db, self.db_witness = _path(cfg["idea_lake_db"], directory=False)
        self.workdir = _path(Path.cwd(), directory=True)
        self.config_file, _ = _path(cfg["_config_path"], directory=False)
        self.config_witness, _ = _file_witness(self.config_file, _LIMIT)

    def check(self):
        from orze.engine.controller_session import _file_witness
        if (_profile(self.cfg) != self.fingerprint
                or _path(Path.cwd(), directory=True) != self.workdir
                or _path(self.scope, directory=True) != (self.scope, self.scope_witness)
                or _path(self.db, directory=False) != (self.db, self.db_witness)
                or _file_witness(self.config_file, _LIMIT)[0] != self.config_witness):
            raise ControllerHOLD("controller_handoff_route_changed")

    @contextmanager
    def connection(self, *, write=False):
        self.check()
        conn = sqlite3.connect(self.db.as_uri() + ("?mode=rw" if write else "?mode=ro"),
                               uri=True, timeout=0.25)
        try:
            if _route(conn) != (self.db, self.db_witness):
                raise ControllerHOLD("controller_handoff_database_changed")
            if write:
                conn.execute("BEGIN IMMEDIATE")
            else:
                conn.execute("PRAGMA query_only=ON")
            yield conn
            if write:
                conn.commit()
            self.check()
            if _route(conn) != (self.db, self.db_witness):
                raise ControllerHOLD("controller_handoff_database_changed")
        finally:
            conn.close()


def _history_observer(route, identity_json, binding_json):
    """A read-only historical verifier, intentionally without a kernel handle.

    This cannot issue a stop or grant: public operations first construct their
    own live _Observer. It only checks sealed history of an already qualified
    source or the ancestry that source was itself admitted from.
    """
    from orze.engine.controller_session import _Observer
    observer = object.__new__(_Observer)
    observer.scope, observer.scope_witness = route.scope, route.scope_witness
    observer.db, observer.db_witness = route.db, route.db_witness
    observer.fingerprint = route.fingerprint
    observer._protocol, observer._history_only, observer.pidfd = 2, True, None
    observer.identity_json, observer.binding_json = identity_json, binding_json
    observer.identity, observer.binding = _decode(identity_json), _decode(binding_json)
    observer.controller_id = observer.identity.get("controller_id")
    if (observer.identity.get("schema") != 2
            or observer.binding.get("identity") != observer.identity
            or observer.binding.get("controller_id") != observer.controller_id
            or observer.binding.get("profile") != {"version": 2, "profile": "local_handoff_v1"}
            or observer.binding.get("config_sha256") != route.fingerprint
            or observer.binding.get("physical_gpus") != sorted(route.cfg["gpu_scheduling"]["allowed_gpus"])
            or observer.identity.get("scope") != str(route.scope)
            or (observer.identity.get("scope_device"), observer.identity.get("scope_inode")) != route.scope_witness
            or observer.identity.get("database") != str(route.db)
            or (observer.identity.get("database_device"), observer.identity.get("database_inode")) != route.db_witness
            or observer.binding.get("workdir") != str(route.workdir[0])
            or (observer.binding.get("workdir_device"), observer.binding.get("workdir_inode")) != route.workdir[1]):
        raise ControllerHOLD("controller_handoff_history_binding_changed")
    observer._owner_witness = observer.owner_witness()
    return observer


def _validate_history(route, source_id, generation, *, expected=None, closed_inventory=True):
    """Closed history is preserved, never reset/requeued as part of handoff."""
    from orze.engine.controller_control import registration_version
    from orze.engine.controller_session import _row as session_row, _validate_request
    from orze.engine.controller_members import _NATIVE, _REPORTS, MAX_MEMBERS
    if type(generation) is not int or not 0 <= generation < 64:
        raise ControllerHOLD("controller_handoff_generation_limit")
    with route.connection() as conn:
        if registration_version(conn) != 2:
            raise ControllerHOLD("controller_handoff_storage_required")
        rows = conn.execute("SELECT controller_id,generation,predecessor,identity_json,phase,request_id,hold_reason "
            "FROM main.controller_instances WHERE scope=? AND generation<=? ORDER BY generation LIMIT 65",
            (str(route.scope), generation)).fetchall()
        if len(rows) != generation + 1 or rows[-1][0] != source_id:
            raise ControllerHOLD("controller_handoff_history_incomplete")
        if closed_inventory and conn.execute("SELECT 1 FROM main.ideas WHERE status='running' LIMIT 1").fetchone():
            raise ControllerHOLD("controller_handoff_running_idea")
        if closed_inventory and conn.execute("SELECT 1 FROM main.idea_state WHERE current_state IN ('CLAIMED','IN_PROGRESS') LIMIT 1").fetchone():
            raise ControllerHOLD("controller_handoff_pending_claim")
        if closed_inventory and conn.execute("SELECT 1 FROM main.idea_stage_state WHERE current_state='IN_PROGRESS' LIMIT 1").fetchone():
            raise ControllerHOLD("controller_handoff_pending_stage")
        snapshots = [(tuple(row), session_row(conn, row[0])) for row in rows]
    digest = hashlib.sha256()
    previous = None
    refs = set()
    member_count = 0
    for index, (registration, session) in enumerate(snapshots):
        controller_id, actual_generation, predecessor, identity_json, phase, request_id, held = registration
        if (actual_generation != index or predecessor != (previous[0][0] if previous else None)
                or phase != "QUIESCING" or held is not None or session[2] is None or session[3] is None):
            raise ControllerHOLD("controller_handoff_history_unsettled")
        observer = _history_observer(route, identity_json, session[1])
        request = _validate_request(session[2], observer.binding)
        ack = _decode(session[3])
        if (request_id != request["request_id"] or ack.get("schema") != 1
                or ack.get("kind") != "controller_drained" or ack.get("controller_id") != controller_id
                or ack.get("request_id") != request_id or ack.get("request_sha256") != _sha(session[2].encode())
                or ack.get("binding_sha256") != _sha(session[1].encode())):
            raise ControllerHOLD("controller_handoff_history_ack_changed")
        observer.verify_drain(ack)
        current, actual_session = observer.check()
        if current != (identity_json, phase, request_id, held) or actual_session != session:
            raise ControllerHOLD("controller_handoff_history_changed")
        with route.connection() as conn:
            members = conn.execute("SELECT payload_json FROM main.controller_members "
                "WHERE controller_id=? LIMIT ?", (controller_id, MAX_MEMBERS + 1)).fetchall()
            member_count += len(members)
            if member_count > MAX_MEMBERS:
                raise ControllerHOLD("controller_handoff_history_member_limit")
            for member in members:
                value = _decode(member[0])
                if value.get("kind") in _NATIVE | _REPORTS:
                    ref = value["identity"]["attempt_ref"]
                    refs.add(tuple(ref[key] for key in ("task_id", "phase", "attempt_id", "generation")))
            if previous is not None:
                _schema(conn)
                grant = conn.execute("SELECT grant_id FROM main.controller_handoffs WHERE target_controller_id=?",
                                     (controller_id,)).fetchone()
                prior = _row(conn, grant[0]) if grant is not None else None
                if prior is None or prior[6] != "STARTED" or prior[8] is None:
                    raise ControllerHOLD("controller_handoff_ancestry_unconfirmed")
                payload = _decode(prior[5])
                if (payload.get("source_controller_id") != predecessor
                        or payload.get("source_identity_json") != previous[0][3]
                        or payload.get("source_binding_json") != previous[1][1]
                        or payload.get("source_request_json") != previous[1][2]
                        or payload.get("source_ack_json") != previous[1][3]):
                    raise ControllerHOLD("controller_handoff_ancestry_changed")
                started = _decode(prior[8])
                if (started.get("binding_sha256") != _sha(session[1].encode())
                        or started.get("controller_id") != controller_id):
                    raise ControllerHOLD("controller_handoff_started_binding_changed")
        digest.update(canonical([controller_id, _sha(identity_json.encode()),
                                 _sha(session[1].encode()), _sha(session[2].encode()), _sha(session[3].encode())]))
        previous = registration, session
    with route.connection() as conn:
        if closed_inventory and conn.execute("SELECT 1 FROM main.sqlite_master WHERE name='execution_attempts'").fetchone():
            attempts = conn.execute("SELECT task_id,phase,attempt_id,generation,state FROM main.execution_attempts LIMIT ?",
                                    (MAX_MEMBERS + 1,)).fetchall()
            if (len(attempts) > MAX_MEMBERS or any(row[4] not in {"TERMINAL", "NOT_STARTED"} for row in attempts)
                    or {tuple(row[:4]) for row in attempts} != refs):
                raise ControllerHOLD("controller_handoff_unregistered_attempt")
        elif closed_inventory and refs:
            raise ControllerHOLD("controller_handoff_attempt_history_missing")
    value = digest.hexdigest()
    if expected is not None and value != expected:
        raise ControllerHOLD("controller_handoff_history_digest_changed")
    return value


def _process(value):
    if (type(value) is not dict or set(value) != {"pid", "start_ticks"}
            or any(type(value[key]) is not int or value[key] <= 0 for key in value)):
        raise ControllerHOLD("controller_handoff_process_invalid")
    return value


def _runtime():
    from orze import cli
    from orze.engine.controller_session import _file_witness
    executable, identity = _path(Path(sys.executable).resolve(strict=True), directory=False)
    source, _ = _file_witness(Path(cli.__file__).resolve(strict=True), 1048576)
    return _plain({"executable": sys.executable, "binary": (executable, identity), "cli": source})


class _Admission:
    """One process-local sealed channel; never reconstructed from disk alone."""
    def __init__(self, route, payload_json, ready_json, channel, issuer_pidfd, token, deadline):
        from orze.core.idea_source_lock import SourceLockLease, idea_source_lock_owned
        from orze.engine.controller_session import _file_witness
        self._pid = os.getpid()
        self._thread_id = threading.get_ident()
        self._route = route
        self._fingerprint = route.fingerprint
        self._payload_json, self._ready_json = payload_json, ready_json
        self._payload, self._ready = _decode(payload_json), _decode(ready_json)
        self._channel, self._issuer_pidfd = channel, issuer_pidfd
        self._token, self._deadline = token, deadline
        self._ctx, self._committed = None, False
        value = self._payload
        self.target_id = value["target_controller_id"]
        self.source_controller_id = value["source_controller_id"]
        self.generation = value["target_generation"]
        self.grant_id = value["grant_id"]
        self.target_process = _process(self._ready["process"])
        source = _decode(value["source_identity_json"])
        anchor = route.scope / "_controller_registration.lock"
        _, raw = _file_witness(anchor / "lock.json")
        meta = json.loads(raw)
        self.anchor_lease = SourceLockLease(anchor, meta["owner_nonce"], source["anchor_device"],
                                          source["anchor_inode"], source["anchor_metadata_sha256"])
        if not idea_source_lock_owned(self.anchor_lease):
            raise ControllerHOLD("controller_handoff_anchor_changed")
        self._check_route()

    def _check_route(self):
        self._route.check()
        value = self._payload
        if (self._pid != os.getpid() or threading.get_ident() != self._thread_id
                or self._fingerprint != value["config_sha256"]
                or _plain(self._route.config_witness) != value["config_witness"]
                or _plain(self._route.workdir) != value["workdir"]
                or _plain((self._route.scope, self._route.scope_witness)) != value["scope"]
                or _plain((self._route.db, self._route.db_witness)) != value["database"]
                or _runtime() != value["runtime"]
                or _sha(self._token.encode()) != value["token_sha256"]
                or process_identity(os.getpid())[0] != self.target_process):
            raise ControllerHOLD("controller_handoff_admission_changed")
        if not self._committed and select.select([self._issuer_pidfd], [], [], 0)[0]:
            raise ControllerHOLD("controller_handoff_issuer_lost")

    def _check_row(self, conn, states):
        row = _row(conn, self.grant_id)
        if (row is None or row[0:5] != (self.grant_id, self._payload["request_id"], str(self._route.scope),
                self.source_controller_id, self.target_id) or row[5] != self._payload_json
                or row[6] not in states or row[7] != self._ready_json or row[9] is not None):
            raise ControllerHOLD("controller_handoff_grant_changed")
        return row

    def validate(self, lake, scope):
        from orze.idea_lake import IdeaLake
        from orze.engine.controller_control import current_instance
        self._check_route()
        if (type(lake) is not IdeaLake or lake.conn.in_transaction
                or _route(lake.conn) != (self._route.db, self._route.db_witness)
                or _path(scope, directory=True) != (self._route.scope, self._route.scope_witness)):
            raise ControllerHOLD("controller_handoff_lake_changed")
        with self._route.connection() as conn:
            self._check_row(conn, {"ISSUED"})
            head = conn.execute("SELECT current_id,generation,pending_grant FROM main.controller_scope_heads "
                                "WHERE scope=?", (str(self._route.scope),)).fetchone()
            current = current_instance(conn, self._route.scope)
            if (tuple(head or ()) != (self.source_controller_id, self.generation - 1, self.grant_id)
                    or current[0] != self.source_controller_id or current[3] != "QUIESCING"):
                raise ControllerHOLD("controller_handoff_pending_head_changed")

    def validate_prior(self, orze):
        self.validate(orze.lake, orze.results_dir)
        _validate_history(self._route, self.source_controller_id, self.generation - 1,
                          expected=self._payload["history_sha256"])

    def consume(self, conn, ctx):
        self._check_route()
        self._check_row(conn, {"ISSUED"})
        head = conn.execute("SELECT current_id,generation,pending_grant FROM main.controller_scope_heads "
                            "WHERE scope=?", (str(self._route.scope),)).fetchone()
        if (not conn.in_transaction or tuple(head or ()) != (self.target_id, self.generation, None)
                or ctx.controller_id != self.target_id or ctx.generation != self.generation
                or ctx.predecessor != self.source_controller_id or ctx.identity["process"] != self.target_process):
            raise ControllerHOLD("controller_handoff_consume_binding_changed")
        count = conn.execute("UPDATE main.controller_handoffs SET state='CONSUMED' WHERE grant_id=? "
            "AND state='ISSUED' AND payload_json=? AND ready_json=? AND started_json IS NULL AND hold_reason IS NULL",
            (self.grant_id, self._payload_json, self._ready_json)).rowcount
        if count != 1:
            raise ControllerHOLD("controller_handoff_consume_unconfirmed")
        self._ctx = ctx

    def verify_consumed(self, conn, ctx):
        self._check_route()
        self._check_row(conn, {"CONSUMED"})
        if self._ctx is not ctx or ctx.controller_id != self.target_id:
            raise ControllerHOLD("controller_handoff_context_changed")
        head = conn.execute("SELECT current_id,generation,pending_grant FROM main.controller_scope_heads "
                            "WHERE scope=?", (str(self._route.scope),)).fetchone()
        if tuple(head or ()) != (self.target_id, self.generation, None):
            raise ControllerHOLD("controller_handoff_head_readback_changed")

    def verify_consumed_in_context(self, ctx):
        with self._route.connection() as conn:
            self.verify_consumed(conn, ctx)

    def mark_started(self, session):
        """Resource-ready is not RUN permission: wait for the sealed COMMIT."""
        from orze.core.gpu_lease import _registered, _registry_lock
        from orze.engine.controller_session import _file_witness
        self._check_route()
        if self._ctx is not session.ctx or session._admission is not self:
            raise ControllerHOLD("controller_handoff_session_changed")
        with session.ctx.guard(), _registry_lock:
            session.ctx.check_admission()
            session._validate_runtime()
            owned, witnesses = session._gpu, session._gpu_witness
            if (owned is None or owned is not session.orze._gpu_leases or owned._closed
                    or witnesses is None or len(owned._leases) != len(witnesses)
                    or any(lease is not witness[0] for lease, witness in zip(owned._leases, witnesses))
                    or session._pid_file is None or _file_witness(session._pid_file)[0] != session._pid_witness
                    or session._lake_conn.in_transaction):
                raise ControllerHOLD("controller_handoff_resources_unconfirmed")
            session._lake_conn.execute("SELECT 1")
            for lease, fd, dev, ino in witnesses:
                info = os.fstat(fd)
                if (lease._closed or lease.fd != fd or _registered.get(lease.gpu) is not lease
                        or (info.st_dev, info.st_ino) != (dev, ino)):
                    raise ControllerHOLD("controller_handoff_lease_changed")
            started = {"schema": 1, "event": "STARTED", "grant_id": self.grant_id,
                "controller_id": self.target_id, "generation": self.generation,
                "process": self.target_process, "binding_sha256": _sha(session._binding_json.encode()),
                "gpu_scope": sorted(session.orze.gpu_ids), "lease_count": len(witnesses),
                "pid_file_sha256": session._pid_witness[2], "stage": "leased_before_first_probe"}
            encoded = _encode(started)
            with self._route.connection(write=True) as conn:
                self._check_row(conn, {"CONSUMED"})
                if conn.execute("UPDATE main.controller_handoffs SET state='PREPARED',started_json=? "
                    "WHERE grant_id=? AND state='CONSUMED' AND started_json IS NULL",
                    (encoded, self.grant_id)).rowcount != 1:
                    raise ControllerHOLD("controller_handoff_prepare_unconfirmed")
            with self._route.connection() as conn:
                if self._check_row(conn, {"PREPARED"})[8] != encoded:
                    raise ControllerHOLD("controller_handoff_prepare_readback_changed")
        _send(self._channel, {"schema": 1, "event": "PREPARED", "grant_id": self.grant_id,
                             "started_sha256": _sha(encoded.encode())}, self._deadline)
        packet = _receive(self._channel, self._payload["issuer"]["pid"], self._deadline)
        if packet != {"schema": 1, "event": "COMMIT", "grant_id": self.grant_id,
                      "started_sha256": _sha(encoded.encode())}:
            raise ControllerHOLD("controller_handoff_commit_packet_invalid")
        with session.ctx.guard():
            session.ctx.check_admission()
            self._check_route()
            with self._route.connection(write=True) as conn:
                if self._check_row(conn, {"PREPARED"})[8] != encoded:
                    raise ControllerHOLD("controller_handoff_start_binding_changed")
                if conn.execute("UPDATE main.controller_handoffs SET state='STARTED' "
                    "WHERE grant_id=? AND state='PREPARED' AND started_json=?",
                    (self.grant_id, encoded)).rowcount != 1:
                    raise ControllerHOLD("controller_handoff_start_unconfirmed")
            with self._route.connection() as conn:
                if self._check_row(conn, {"STARTED"})[8] != encoded:
                    raise ControllerHOLD("controller_handoff_start_readback_changed")
            self._committed = True
        # A lost final reply is unknown to the issuer, not a second launch.
        # The successor already owns the head and committed execution gate.
        try:
            _send(self._channel, {"schema": 1, "event": "STARTED", "grant_id": self.grant_id,
                                 "started_sha256": _sha(encoded.encode())}, self._deadline)
        except (OSError, ControllerHOLD):
            pass
        self._channel.close()
        os.close(self._issuer_pidfd)
        self._issuer_pidfd = None


def require_pending_start(scope):
    admission = _ADMISSION
    if admission is None or type(admission) is not _Admission or admission._pid != os.getpid():
        raise ControllerHOLD("controller_handoff_admission_required")
    admission._check_route()
    if _path(scope, directory=True) != (admission._route.scope, admission._route.scope_witness):
        raise ControllerHOLD("controller_handoff_start_scope_changed")
    with admission._route.connection() as conn:
        admission._check_row(conn, {"ISSUED"})
