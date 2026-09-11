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
_ISSUER_LOCK = threading.Lock()
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
    import re
    from orze.engine.controller_session import _Observer
    if type(identity_json) is not str or len(identity_json.encode("utf-8")) > 16384:
        raise ControllerHOLD("controller_handoff_history_identity_invalid")
    observer = object.__new__(_Observer)
    observer.scope, observer.scope_witness = route.scope, route.scope_witness
    observer.db, observer.db_witness = route.db, route.db_witness
    observer.fingerprint = route.fingerprint
    observer._protocol, observer._history_only, observer.pidfd = 2, True, None
    observer.identity_json, observer.binding_json = identity_json, binding_json
    observer.identity, observer.binding = _decode(identity_json), _decode(binding_json)
    observer.controller_id = observer.identity.get("controller_id")
    identity = observer.identity
    integer_fields = ("scope_device", "scope_inode", "database_device", "database_inode",
                      "anchor_device", "anchor_inode", "generation")
    if (set(identity) != {"schema", "controller_id", "scope", "scope_device", "scope_inode",
            "database", "database_device", "database_inode", "host", "boot_id", "process",
            "owner_metadata_sha256", "generation", "predecessor", "owner_directory",
            "anchor_directory", "anchor_device", "anchor_inode", "anchor_metadata_sha256"}
            or type(identity["schema"]) is not int or identity["schema"] != 2
            or type(observer.controller_id) is not str or not re.fullmatch(r"[0-9a-f]{48}", observer.controller_id)
            or any(type(identity[key]) is not int or identity[key] < 0 for key in integer_fields)
            or not 0 <= identity["generation"] < 64
            or (identity["generation"] == 0) != (identity["predecessor"] is None)
            or (identity["predecessor"] is not None and (type(identity["predecessor"]) is not str
                or not re.fullmatch(r"[0-9a-f]{48}", identity["predecessor"])
                or identity["predecessor"] == observer.controller_id))
            or type(identity["host"]) is not str or not 0 < len(identity["host"].encode("utf-8")) <= 1024
            or type(identity["boot_id"]) is not str
            or not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", identity["boot_id"])
            or any(type(identity[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", identity[key])
                   for key in ("owner_metadata_sha256", "anchor_metadata_sha256"))
            or identity["scope"] != str(route.scope)
            or (identity["scope_device"], identity["scope_inode"]) != route.scope_witness
            or identity["database"] != str(route.db)
            or (identity["database_device"], identity["database_inode"]) != route.db_witness):
        raise ControllerHOLD("controller_handoff_history_identity_invalid")
    _process(identity["process"])  # Validate metadata only; never probe a historical PID.
    expected_binding = {"schema": 1, "controller_id": observer.controller_id, "identity": identity,
        "profile": {"version": 2, "profile": "local_handoff_v1"}, "config_sha256": route.fingerprint,
        "physical_gpus": sorted(route.cfg["gpu_scheduling"]["allowed_gpus"]),
        "workdir": str(route.workdir[0]), "workdir_device": route.workdir[1][0],
        "workdir_inode": route.workdir[1][1]}
    if _encode(expected_binding) != binding_json:
        raise ControllerHOLD("controller_handoff_history_binding_changed")
    observer._owner_witness = observer.owner_witness()
    return observer


def _closed_history_catalog(conn):
    """Read existing FSM projections; never normalize or restore unknown rows."""
    from orze.idea_lake import (
        STATE_TO_STATUS, STATUS_TO_STATE, VALID_STAGE_TRANSITIONS, PIPELINE_STAGES,
    )
    def unknown(table, column, allowed):
        placeholders = ",".join("?" for _ in allowed)
        return conn.execute("SELECT 1 FROM main." + table + " WHERE typeof(" + column + ")!='text' "
            "OR " + column + " NOT IN (" + placeholders + ") LIMIT 1", tuple(allowed)).fetchone()
    if (unknown("ideas", "status", STATUS_TO_STATE)
            or unknown("idea_state", "current_state", STATE_TO_STATUS)
            or unknown("idea_stage_state", "stage", PIPELINE_STAGES)
            or unknown("idea_stage_state", "current_state", VALID_STAGE_TRANSITIONS)):
        raise ControllerHOLD("controller_handoff_unknown_lifecycle")
    if conn.execute("SELECT 1 FROM main.idea_state WHERE current_state IN ('CLAIMED','IN_PROGRESS') LIMIT 1").fetchone():
        raise ControllerHOLD("controller_handoff_pending_claim")
    if conn.execute("SELECT 1 FROM main.idea_stage_state WHERE current_state='IN_PROGRESS' LIMIT 1").fetchone():
        raise ControllerHOLD("controller_handoff_pending_stage")
    # The legacy aliases are authoritative only when they agree with the FSM.
    projection = "CASE i.status " + " ".join("WHEN ? THEN ?" for _ in STATUS_TO_STATE) + " END"
    parameters = tuple(value for pair in STATUS_TO_STATE.items() for value in pair)
    if (conn.execute("SELECT 1 FROM main.ideas i LEFT JOIN main.idea_state s "
            "ON i.idea_id=s.idea_id COLLATE BINARY WHERE s.idea_id IS NULL "
            "OR s.current_state != " + projection + " LIMIT 1", parameters).fetchone()
            or conn.execute("SELECT 1 FROM main.idea_state s LEFT JOIN main.ideas i "
                "ON i.idea_id=s.idea_id COLLATE BINARY WHERE i.idea_id IS NULL LIMIT 1").fetchone()
            or conn.execute("SELECT 1 FROM main.idea_stage_state s LEFT JOIN main.ideas i "
                "ON i.idea_id=s.idea_id COLLATE BINARY WHERE i.idea_id IS NULL LIMIT 1").fetchone()):
        raise ControllerHOLD("controller_handoff_lifecycle_projection_changed")


def _validate_history(route, source_id, generation, *, expected=None, closed_inventory=True):
    """Closed history is preserved, never reset/requeued as part of handoff."""
    from orze.engine.controller_control import registration_version
    from orze.engine.controller_session import _row as session_row, _validate_request
    from orze.engine.controller_members import _NATIVE, _REPORTS, MAX_MEMBERS, _schema as member_schema
    from orze.core.execution_attempts import AttemptRef
    if type(generation) is not int or not 0 <= generation < 64:
        raise ControllerHOLD("controller_handoff_generation_limit")
    with route.connection() as conn:
        if registration_version(conn) != 2:
            raise ControllerHOLD("controller_handoff_storage_required")
        rows = conn.execute("SELECT controller_id,generation,predecessor,"
            "CASE WHEN typeof(identity_json)='text' AND length(CAST(identity_json AS BLOB))<=16384 "
            "THEN identity_json ELSE '' END,phase,request_id,hold_reason "
            "FROM main.controller_instances WHERE scope=? AND generation<=? ORDER BY generation LIMIT 65",
            (str(route.scope), generation)).fetchall()
        if len(rows) != generation + 1 or rows[-1][0] != source_id:
            raise ControllerHOLD("controller_handoff_history_incomplete")
        if closed_inventory:
            _closed_history_catalog(conn)
        member_schema(conn)
        placeholders = ",".join("?" for _ in rows)
        count = conn.execute("SELECT COUNT(*) FROM (SELECT 1 FROM main.controller_members "
            "WHERE controller_id IN (" + placeholders + ") LIMIT ?)",
            (*[row[0] for row in rows], MAX_MEMBERS + 1)).fetchone()[0]
        if count > MAX_MEMBERS:
            raise ControllerHOLD("controller_handoff_history_member_limit")
        snapshots = [(tuple(row), session_row(conn, row[0])) for row in rows]
    digest = hashlib.sha256()
    previous = None
    refs = set()
    member_count = 0
    for index, (registration, session) in enumerate(snapshots):
        controller_id, actual_generation, predecessor, identity_json, phase, request_id, held = registration
        if (type(actual_generation) is not int or actual_generation != index
                or predecessor != (previous[0][0] if previous else None)
                or phase != "QUIESCING" or held is not None or session[2] is None or session[3] is None):
            raise ControllerHOLD("controller_handoff_history_unsettled")
        observer = _history_observer(route, identity_json, session[1])
        if (observer.controller_id != controller_id or observer.identity["generation"] != actual_generation
                or observer.identity["predecessor"] != predecessor):
            raise ControllerHOLD("controller_handoff_history_identity_changed")
        request = _validate_request(session[2], observer.binding)
        ack = _decode(session[3])
        if (set(ack) != {"schema", "kind", "controller_id", "request_id", "request_sha256",
                        "binding_sha256", "members", "resources"}
                or type(ack["schema"]) is not int or ack["schema"] != 1
                or request_id != request["request_id"]
                or ack.get("kind") != "controller_drained" or ack.get("controller_id") != controller_id
                or ack.get("request_id") != request_id or ack.get("request_sha256") != _sha(session[2].encode())
                or ack.get("binding_sha256") != _sha(session[1].encode())):
            raise ControllerHOLD("controller_handoff_history_ack_changed")
        if (type(ack["resources"]) is not dict
                or canonical(ack["resources"].get("gpu_scope")) != canonical(observer.binding["physical_gpus"])):
            raise ControllerHOLD("controller_handoff_history_resources_changed")
        observer.verify_drain(ack)
        current, actual_session = observer.check()
        if current != (identity_json, phase, request_id, held) or actual_session != session:
            raise ControllerHOLD("controller_handoff_history_changed")
        with route.connection() as conn:
            members = conn.execute("SELECT member_id,CASE WHEN typeof(payload_json)='text' "
                "AND length(CAST(payload_json AS BLOB))<=65536 THEN payload_json ELSE '' END "
                "FROM main.controller_members WHERE controller_id=? ORDER BY member_id COLLATE BINARY LIMIT ?",
                (controller_id, MAX_MEMBERS - member_count + 1)).fetchall()
            member_count += len(members)
            if member_count > MAX_MEMBERS:
                raise ControllerHOLD("controller_handoff_history_member_limit")
            member_digest = hashlib.sha256()
            for key, raw in members:
                value = _decode(raw)
                member_digest.update(canonical([key, _sha(raw.encode())]))
                if value.get("kind") in _NATIVE | _REPORTS:
                    ref = AttemptRef(**value["identity"]["attempt_ref"])
                    refs.add((ref.task_id, ref.phase, ref.attempt_id, ref.generation))
            if (len(members) != ack["members"]["member_count"]
                    or member_digest.hexdigest() != ack["members"]["members_sha256"]):
                raise ControllerHOLD("controller_handoff_history_members_changed")
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


def _capture(process):
    process = _process(process)
    if process_identity(process["pid"])[0] != process:
        raise ControllerHOLD("controller_handoff_process_changed")
    fd = os.pidfd_open(process["pid"], 0)
    try:
        if process_identity(process["pid"])[0] != process or select.select([fd], [], [], 0)[0]:
            raise ControllerHOLD("controller_handoff_capture_unconfirmed")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _optional_request(conn, request_id):
    if conn.execute("SELECT 1 FROM main.sqlite_master WHERE name=? COLLATE NOCASE",
                    ("controller_handoffs",)).fetchone() is None:
        return None
    return _row(conn, request_id=request_id)


def _validate_payload(route, row):
    from orze.engine.controller_control import _ID
    value = _decode(row[5])
    keys = {"schema", "grant_id", "request_id", "source_controller_id", "target_controller_id",
        "source_generation", "target_generation", "source_identity_json", "source_binding_json",
        "source_request_json", "source_ack_json", "history_sha256", "config_sha256", "config_witness",
        "workdir", "scope", "database", "runtime", "token_sha256", "issuer"}
    if (set(value) != keys or type(value["schema"]) is not int or value["schema"] != 1
            or row[:5] != (value["grant_id"], value["request_id"], str(route.scope),
                value["source_controller_id"], value["target_controller_id"])
            or type(value["source_generation"]) is not int or not 0 <= value["source_generation"] < 64
            or type(value["target_generation"]) is not int
            or value["target_generation"] != value["source_generation"] + 1
            or value["config_sha256"] != route.fingerprint
            or value["config_witness"] != _plain(route.config_witness)
            or value["workdir"] != _plain(route.workdir)
            or value["scope"] != _plain((route.scope, route.scope_witness))
            or value["database"] != _plain((route.db, route.db_witness))
            or value["runtime"] != _runtime()):
        raise ControllerHOLD("controller_handoff_payload_invalid")
    for key in ("grant_id", "source_controller_id", "target_controller_id"):
        if type(value[key]) is not str or not _ID.fullmatch(value[key]):
            raise ControllerHOLD("controller_handoff_identifier_invalid")
    _token(value["request_id"], "controller_handoff_request_id_invalid")
    _process(value["issuer"])
    for key in ("history_sha256", "token_sha256"):
        if (type(value[key]) is not str or len(value[key]) != 64
                or any(c not in "0123456789abcdef" for c in value[key])):
            raise ControllerHOLD("controller_handoff_digest_invalid")
    return value


def _validate_hello(route, hello, process):
    expected = {"schema": 1, "event": "HELLO", "process": _process(process),
                "nonce": hello.get("nonce"), "config_sha256": route.fingerprint, "runtime": _runtime()}
    nonce = hello.get("nonce")
    if (type(nonce) is not str or len(nonce) != 48 or any(c not in "0123456789abcdef" for c in nonce)
            or _encode(hello) != _encode(expected)):
        raise ControllerHOLD("controller_handoff_hello_invalid")


def prepare_successor_entry(cfg, args):
    """Original CLI entry accepts only a live issuer's sealed, one-shot GO."""
    from orze.core.controller_profile import validate_successor_cli
    global _ADMISSION
    validate_successor_cli(args)
    if _ADMISSION is not None:
        raise ControllerHOLD("controller_handoff_entry_already_consumed")
    route = _Route(cfg)
    channel, issuer_fd = None, None
    try:
        issuer = process_identity(os.getppid())[0]
        issuer_fd = _capture(issuer)
        channel = _inherited_channel()
        deadline = time.monotonic() + 60
        hello = {"schema": 1, "event": "HELLO", "process": process_identity(os.getpid())[0],
                 "nonce": secrets.token_hex(24), "config_sha256": route.fingerprint, "runtime": _runtime()}
        _send(channel, hello, deadline)
        packet = _receive(channel, issuer["pid"], deadline)
        if (set(packet) != {"schema", "event", "grant_id", "token"}
                or type(packet["schema"]) is not int or packet["schema"] != 1 or packet["event"] != "GO"
                or type(packet["token"]) is not str or len(packet["token"]) != 64
                or any(c not in "0123456789abcdef" for c in packet["token"])):
            raise ControllerHOLD("controller_handoff_go_invalid")
        with route.connection() as conn:
            row = _row(conn, packet["grant_id"])
            if row is None or row[6] != "ISSUED" or row[7] != _encode(hello) or row[8:] != (None, None):
                raise ControllerHOLD("controller_handoff_entry_grant_unconfirmed")
            payload = _validate_payload(route, row)
            if (_encode(payload["issuer"]) != _encode(issuer)
                    or _sha(packet["token"].encode()) != payload["token_sha256"]):
                raise ControllerHOLD("controller_handoff_issuer_binding_changed")
            head = conn.execute("SELECT current_id,generation,pending_grant FROM main.controller_scope_heads "
                                "WHERE scope=?", (str(route.scope),)).fetchone()
            if tuple(head or ()) != (payload["source_controller_id"], payload["source_generation"], row[0]):
                raise ControllerHOLD("controller_handoff_entry_head_changed")
        _validate_history(route, payload["source_controller_id"], payload["source_generation"],
                          expected=payload["history_sha256"])
        admission = _Admission(route, row[5], row[7], channel, issuer_fd, packet["token"], deadline)
        _ADMISSION = admission
        channel, issuer_fd = None, None  # Strong admission owns both from here.
    except ControllerHOLD:
        raise
    except Exception as exc:
        raise ControllerHOLD("controller_handoff_entry_unconfirmed") from exc
    finally:
        if channel is not None:
            channel.close()
        if issuer_fd is not None:
            os.close(issuer_fd)


def _verify_started(route, row, *, state, expected_process=None):
    """Fresh current registration + resource-start record, not an ACK label."""
    from orze.engine.controller_session import _Observer
    payload = _validate_payload(route, row)
    if row[6] != state or row[7] is None or row[8] is None or row[9] is not None:
        raise ControllerHOLD("controller_handoff_start_unconfirmed")
    hello, started = _decode(row[7]), _decode(row[8])
    process = _process(hello.get("process"))
    if expected_process is not None and _encode(process) != _encode(expected_process):
        raise ControllerHOLD("controller_handoff_successor_changed")
    _validate_hello(route, hello, process)
    observer = _Observer(route.cfg)
    try:
        registration, session = observer.check()
        expected = {"schema": 1, "event": "STARTED", "grant_id": row[0],
            "controller_id": payload["target_controller_id"], "generation": payload["target_generation"],
            "process": process, "binding_sha256": _sha(session[1].encode()),
            "gpu_scope": sorted(route.cfg["gpu_scheduling"]["allowed_gpus"]),
            "lease_count": len(route.cfg["gpu_scheduling"]["allowed_gpus"]),
            "pid_file_sha256": started.get("pid_file_sha256"), "stage": "leased_before_first_probe"}
        pid_hash = started.get("pid_file_sha256")
        if (observer.controller_id != payload["target_controller_id"]
                or _encode(observer.identity["process"]) != _encode(process)
                or observer.identity.get("generation") != payload["target_generation"]
                or observer.identity.get("predecessor") != payload["source_controller_id"]
                or registration[1:] != ("ACTIVE", None, None) or session[2:] != (None, None)
                or _encode(started) != _encode(expected) or type(pid_hash) is not str
                or len(pid_hash) != 64 or any(c not in "0123456789abcdef" for c in pid_hash)):
            raise ControllerHOLD("controller_handoff_start_binding_changed")
        with route.connection() as conn:
            if _row(conn, row[0]) != row:
                raise ControllerHOLD("controller_handoff_start_readback_changed")
            head = conn.execute("SELECT current_id,generation,pending_grant FROM main.controller_scope_heads "
                                "WHERE scope=?", (str(route.scope),)).fetchone()
            if tuple(head or ()) != (payload["target_controller_id"], payload["target_generation"], None):
                raise ControllerHOLD("controller_handoff_started_head_changed")
        if observer.exited():
            raise ControllerHOLD("controller_handoff_successor_exited")
        return CompletedControllerHandoff(payload["request_id"], row[0], payload["source_controller_id"],
            payload["target_controller_id"], payload["target_generation"], _sha(row[8].encode()))
    finally:
        observer.close()


class _Coordinator:
    def __init__(self, cfg, request_id, timeout):
        self.route = _Route(cfg)
        self.request_id = _token(request_id, "controller_handoff_request_id_invalid")
        self.deadline = time.monotonic() + _timeout(timeout)
        self.observer = None
        self.channel = self.child_channel = None
        self.child = self.child_fd = None
        self.grant_id = self.payload_json = self.ready_json = None
        self._pid, self._thread = os.getpid(), threading.get_ident()

    def check(self):
        if self._pid != os.getpid() or self._thread != threading.get_ident():
            raise ControllerHOLD("controller_handoff_coordinator_changed")
        self.route.check()
        if self.observer is not None and not self.observer.exited():
            raise ControllerHOLD("controller_handoff_old_process_alive")
        if self.child_fd is not None and select.select([self.child_fd], [], [], 0)[0]:
            raise ControllerHOLD("controller_handoff_successor_exited")
        if time.monotonic() >= self.deadline:
            raise ControllerHOLD("controller_handoff_timeout_unconfirmed")

    def read(self, conn, states):
        row = _row(conn, self.grant_id)
        if (row is None or row[5] != self.payload_json or row[6] not in states
                or row[7] != self.ready_json or row[9] is not None):
            raise ControllerHOLD("controller_handoff_operation_changed")
        return row

    def transition(self, before, after, ready_json=None):
        self.check()
        with self.route.connection(write=True) as conn:
            self.read(conn, {before})
            if conn.execute("UPDATE main.controller_handoffs SET state=?,ready_json=? "
                "WHERE grant_id=? AND state=? AND payload_json=? AND ready_json IS ? AND started_json IS NULL",
                (after, ready_json, self.grant_id, before, self.payload_json, self.ready_json)).rowcount != 1:
                raise ControllerHOLD("controller_handoff_transition_unconfirmed")
        self.ready_json = ready_json
        with self.route.connection() as conn:
            self.read(conn, {after})

    def run(self):
        from orze.engine.controller_session import _Observer, _stop_observed
        with self.route.connection() as conn:
            previous = _optional_request(conn, self.request_id)
        if previous is not None:
            payload = _validate_payload(self.route, previous)
            if previous[6] != "STARTED":
                raise ControllerHOLD("controller_handoff_request_already_reserved")
            _validate_history(self.route, payload["source_controller_id"], payload["source_generation"],
                              expected=payload["history_sha256"], closed_inventory=False)
            return _verify_started(self.route, previous, state="STARTED")
        self.observer = _Observer(self.route.cfg)
        _stop_observed(self.observer, _timeout(self.deadline - time.monotonic()))
        self.check()
        source_registration, source_session = self.observer.check()
        generation = self.observer.identity["generation"]
        history = _validate_history(self.route, self.observer.controller_id, generation)
        token = secrets.token_hex(32)
        self.grant_id = secrets.token_hex(24)
        payload = {"schema": 1, "grant_id": self.grant_id, "request_id": self.request_id,
            "source_controller_id": self.observer.controller_id, "target_controller_id": secrets.token_hex(24),
            "source_generation": generation, "target_generation": generation + 1,
            "source_identity_json": self.observer.identity_json, "source_binding_json": source_session[1],
            "source_request_json": source_session[2], "source_ack_json": source_session[3],
            "history_sha256": history, "config_sha256": self.route.fingerprint,
            "config_witness": _plain(self.route.config_witness), "workdir": _plain(self.route.workdir),
            "scope": _plain((self.route.scope, self.route.scope_witness)),
            "database": _plain((self.route.db, self.route.db_witness)), "runtime": _runtime(),
            "token_sha256": _sha(token.encode()), "issuer": process_identity(os.getpid())[0]}
        self.payload_json = _encode(payload)
        self.check()
        with self.route.connection(write=True) as conn:
            _schema(conn, create=True)
            if self.observer.check(conn) != (source_registration, source_session):
                raise ControllerHOLD("controller_handoff_source_changed")
            conn.execute("INSERT INTO main.controller_handoffs VALUES (?,?,?,?,?,?,'RESERVED',NULL,NULL,NULL)",
                (self.grant_id, self.request_id, str(self.route.scope), self.observer.controller_id,
                 payload["target_controller_id"], self.payload_json))
            if conn.execute("UPDATE main.controller_scope_heads SET pending_grant=? "
                "WHERE scope=? AND current_id=? AND generation=? AND pending_grant IS NULL",
                (self.grant_id, str(self.route.scope), self.observer.controller_id, generation)).rowcount != 1:
                raise ControllerHOLD("controller_handoff_reserve_unconfirmed")
        with self.route.connection() as conn:
            self.read(conn, {"RESERVED"})
            if tuple(conn.execute("SELECT current_id,generation,pending_grant FROM main.controller_scope_heads "
                "WHERE scope=?", (str(self.route.scope),)).fetchone() or ()) != (
                    self.observer.controller_id, generation, self.grant_id):
                raise ControllerHOLD("controller_handoff_reserve_readback_changed")
        self.transition("RESERVED", "SPAWNING")  # Commit + readback BEFORE Popen, never reset on unknown.
        self.channel, self.child_channel = _new_channel()
        env = dict(os.environ)
        env[_FD_ENV] = str(self.child_channel.fileno())
        self.check()
        self.child = subprocess.Popen([sys.executable, "-m", "orze.cli", "-c", str(self.route.config_file)],
            cwd=str(self.route.workdir[0]), env=env, close_fds=True, pass_fds=(self.child_channel.fileno(),),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        process = process_identity(self.child.pid)[0]
        self.child_fd = _capture(process)
        self.child_channel.close()
        self.child_channel = None
        hello = _receive(self.channel, self.child.pid, self.deadline)
        self.check()
        if process_identity(self.child.pid)[0] != process:
            raise ControllerHOLD("controller_handoff_child_birth_changed")
        _validate_hello(self.route, hello, process)
        self.transition("SPAWNING", "ISSUED", _encode(hello))
        _send(self.channel, {"schema": 1, "event": "GO", "grant_id": self.grant_id, "token": token}, self.deadline)
        prepared = _receive(self.channel, self.child.pid, self.deadline)
        self.check()
        with self.route.connection() as conn:
            row = self.read(conn, {"PREPARED"})
        result = _verify_started(self.route, row, state="PREPARED", expected_process=process)
        expected = {"schema": 1, "event": "PREPARED", "grant_id": self.grant_id,
                    "started_sha256": result.started_sha256}
        if _encode(prepared) != _encode(expected):
            raise ControllerHOLD("controller_handoff_prepared_packet_invalid")
        self.check()
        _send(self.channel, {**expected, "event": "COMMIT"}, self.deadline)
        final = _receive(self.channel, self.child.pid, self.deadline)
        if _encode(final) != _encode({**expected, "event": "STARTED"}):
            raise ControllerHOLD("controller_handoff_started_packet_invalid")
        self.check()
        with self.route.connection() as conn:
            row = self.read(conn, {"STARTED"})
        return _verify_started(self.route, row, state="STARTED", expected_process=process)

    def close(self):
        # No raw PID signal and no fallback launch. Closing an uncommitted
        # channel denies COMMIT; an already STARTED controller is independent.
        for channel in (self.channel, self.child_channel):
            if channel is not None:
                channel.close()
        if self.child_fd is not None:
            os.close(self.child_fd)
        if self.observer is not None:
            self.observer.close()


def restart_controller(cfg, request_id, timeout=60):
    """One durable same-config/scope successor; replay cannot spawn again."""
    from orze.engine.controller_control import current_controller
    if current_controller() is not None:
        raise ControllerHOLD("controller_handoff_external_issuer_required")
    coordinator = _Coordinator(cfg, request_id, timeout)
    key = secrets.token_hex(24)
    with _ISSUER_LOCK:
        if len(_ISSUERS) >= 128:
            raise ControllerHOLD("controller_handoff_issuer_limit")
        _ISSUERS[key] = coordinator
    try:
        return coordinator.run()
    except ControllerHOLD:
        raise
    except Exception as exc:
        raise ControllerHOLD("controller_handoff_operation_unconfirmed") from exc
    finally:
        coordinator.close()
        with _ISSUER_LOCK:
            _ISSUERS.pop(key, None)
