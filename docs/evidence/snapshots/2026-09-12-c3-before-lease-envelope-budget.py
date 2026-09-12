"""Persistent CPU action envelopes in the supplied IdeaLake, not CPU billing.

All public APIs own short transactions and reject an already-open caller
transaction. A permit is metadata whose exact durable row must be rechecked;
it is neither process authority nor permission to replay an unknown launch.
Slots remain occupied until confirmed native settlement. Reserved wall time
is charged forever, including unused time; no age/PID-based refund exists.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import asdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import time

from orze.core.execution_attempts import AttemptRef, require_current, _json as _attempt_json

_LIMIT = 16384
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_HEX = re.compile(r"[0-9a-f]{48}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_HELD = set()
_SQL = {
    "cpu_action_scopes": """CREATE TABLE cpu_action_scopes (
        scope TEXT NOT NULL COLLATE BINARY PRIMARY KEY,
        binding_json TEXT NOT NULL,
        stop_json TEXT
    )""",
    "cpu_action_reservations": """CREATE TABLE cpu_action_reservations (
        reservation_id TEXT NOT NULL COLLATE BINARY PRIMARY KEY,
        scope TEXT NOT NULL COLLATE BINARY,
        task_id TEXT NOT NULL COLLATE BINARY,
        slot INTEGER NOT NULL CHECK (slot >= 0),
        permit_json TEXT NOT NULL,
        ref_json TEXT,
        state TEXT NOT NULL CHECK (state IN ('RESERVED','BOUND','SETTLED')),
        terminal_sha256 TEXT
    )""",
    "cpu_action_decisions": """CREATE TABLE cpu_action_decisions (
        decision_id TEXT NOT NULL COLLATE BINARY PRIMARY KEY,
        scope TEXT NOT NULL COLLATE BINARY,
        record_json TEXT NOT NULL
    )""",
    "cpu_action_recovery": """CREATE TABLE cpu_action_recovery (
        scope TEXT NOT NULL COLLATE BINARY PRIMARY KEY,
        binding_json TEXT NOT NULL,
        nonce TEXT NOT NULL COLLATE BINARY,
        state TEXT NOT NULL CHECK (state IN ('IN_PROGRESS','COMPLETE')),
        summary_json TEXT
    )""",
}
_INDEX = {
    "cpu_action_live_slot": "CREATE UNIQUE INDEX cpu_action_live_slot ON cpu_action_reservations(scope,slot) WHERE state != 'SETTLED'",
    "cpu_action_live_task": "CREATE UNIQUE INDEX cpu_action_live_task ON cpu_action_reservations(scope,task_id) WHERE state != 'SETTLED'",
    "cpu_action_bound_attempt": "CREATE UNIQUE INDEX cpu_action_bound_attempt ON cpu_action_reservations(ref_json) WHERE ref_json IS NOT NULL",
}


class CpuBudgetHOLD(RuntimeError):
    """No launch/settlement authority; never an ordinary resource Wait."""


def _fail(reason):
    raise CpuBudgetHOLD("cpu_budget_" + reason)


def _json(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
        if len(raw.encode("utf-8")) > _LIMIT:
            _fail("metadata_limit")
        return raw
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise CpuBudgetHOLD("cpu_budget_metadata_invalid") from exc


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail("duplicate_key")
            result[key] = value
        return result
    try:
        if type(raw) is not str:
            _fail("stored_metadata_invalid")
        value = json.loads(raw, object_pairs_hook=pairs)
        if _json(value) != raw:
            _fail("stored_metadata_invalid")
        return value
    except (ValueError, TypeError, RecursionError) as exc:
        raise CpuBudgetHOLD("cpu_budget_stored_metadata_invalid") from exc


def _number(value):
    try:
        if type(value) not in (int, float) or value <= 0 or not math.isfinite(value):
            _fail("wall_limit_invalid")
    except (OverflowError, TypeError, ValueError):
        _fail("wall_limit_invalid")
    return value


def _ns(value, *, reservation=False):
    _number(value)
    with localcontext() as ctx:
        ctx.prec = 1000
        return int((Decimal(str(value)) * Decimal(1000000000)).to_integral_value(
            rounding=ROUND_CEILING if reservation else ROUND_FLOOR))


def _seconds(value):
    with localcontext() as ctx:
        ctx.prec = 1000
        return float(Decimal(value) / Decimal(1000000000))


def _token(value):
    if type(value) is not str or value in (".", "..") or not _TOKEN.fullmatch(value):
        _fail("task_id_invalid")


def _path(path, *, directory):
    p = Path(path).absolute()
    if ".." in p.parts or len(str(p).encode()) > 4096:
        _fail("path_invalid")
    for parent in reversed(p.parents):
        if not stat.S_ISDIR(parent.lstat().st_mode):
            _fail("path_redirected")
    info = p.lstat()
    if (directory and not stat.S_ISDIR(info.st_mode)) or (
            not directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1)):
        _fail("path_redirected")
    return str(p), [info.st_dev, info.st_ino]


def _route(lake):
    if lake.conn.in_transaction:
        _fail("caller_transaction_active")
    rows = [row[2] for row in lake.conn.execute("PRAGMA database_list") if row[1] == "main"]
    if len(rows) != 1 or not rows[0]:
        _fail("persistent_main_required")
    actual = _path(rows[0], directory=False)
    if actual != _path(lake.db_path, directory=False):
        _fail("database_route_changed")
    return actual


def _declaration(value):
    if (type(value) is not dict or set(value) != {
            "version", "resource", "slots", "wall_budget_seconds"}
            or type(value["version"]) is not int or value["version"] != 1
            or value["resource"] != "cpu" or type(value["slots"]) is not int
            or not 0 < value["slots"] <= 64):
        _fail("declaration_invalid")
    _number(value["wall_budget_seconds"])
    return _decode(_json(value))


def _scope(value):
    if type(value) is not dict or set(value) != {
            "schema", "results_dir", "database", "database_identity",
            "directory_identity", "declaration", "policy_sha256"}:
        _fail("scope_invalid")
    if type(value["schema"]) is not int or value["schema"] != 1:
        _fail("scope_invalid")
    _declaration(value["declaration"])
    original = {k: v for k, v in value.items() if k != "policy_sha256"}
    if hashlib.sha256(_json(original).encode()).hexdigest() != value["policy_sha256"]:
        _fail("scope_invalid")
    return _decode(_json(value))


def _check_route(lake, scope):
    scope = _scope(scope)
    if (_route(lake) != (scope["database"], scope["database_identity"])
            or _path(scope["results_dir"], directory=True) !=
            (scope["results_dir"], scope["directory_identity"])):
        _fail("scope_route_changed")
    return scope


def _schema(conn, *, create=False):
    normalize = lambda s: " ".join(str(s).strip().rstrip(";").split())
    for name, sql in {**_SQL, **_INDEX}.items():
        row = conn.execute("SELECT type,name,sql FROM main.sqlite_master WHERE name=? COLLATE NOCASE",
                           (name,)).fetchone()
        if row is None and create:
            conn.execute(sql)
            row = conn.execute("SELECT type,name,sql FROM main.sqlite_master WHERE name=?",
                               (name,)).fetchone()
        if (row is None or row[0] != ("table" if name in _SQL else "index")
                or row[1] != name or normalize(row[2]) != normalize(sql)):
            _fail("schema_invalid")
    for name in _SQL:
        if conn.execute("SELECT 1 FROM main.sqlite_master WHERE type='trigger' AND tbl_name=? COLLATE NOCASE",
                        (name,)).fetchone():
            _fail("schema_trigger_unsupported")


def _scope_row(conn, scope):
    row = conn.execute("SELECT CASE WHEN length(CAST(binding_json AS BLOB))<=16384 THEN binding_json END,"
                       "CASE WHEN stop_json IS NULL OR length(CAST(stop_json AS BLOB))<=16384 THEN stop_json ELSE '' END "
                       "FROM main.cpu_action_scopes WHERE scope=?", (scope["results_dir"],)).fetchone()
    if row is None or row[0] != _json(scope):
        _fail("scope_binding_changed")
    stopped = None if row[1] is None else _decode(row[1])
    if stopped is not None and (type(stopped) is not dict or set(stopped) != {"kind", "reason", "wakeup"}
            or stopped.get("kind") != "Stop" or stopped.get("wakeup") is not None
            or type(stopped.get("reason")) is not str or not stopped["reason"].strip()):
        _fail("stop_invalid")
    return stopped


def _recovery_summary(value, scope):
    fields = {"schema", "examined", "settled", "already_settled", "retained"}
    if (type(value) is not dict or set(value) != fields
            or type(value["schema"]) is not int or value["schema"] != 1
            or type(value["examined"]) is not int
            or not 0 <= value["examined"] <= scope["declaration"]["slots"]):
        _fail("recovery_summary_invalid")
    identities = []
    for key in ("settled", "already_settled", "retained"):
        if type(value[key]) is not list or len(value[key]) > value["examined"]:
            _fail("recovery_summary_invalid")
        for item in value[key]:
            if key == "retained":
                if (type(item) is not dict or set(item) != {"reservation_id", "reason"}
                        or item["reason"] not in {"scope_stopped", "reserved", "launching",
                                                 "running", "in_doubt", "not_started"}):
                    _fail("recovery_summary_invalid")
                item = item["reservation_id"]
            if type(item) is not str or not _HEX.fullmatch(item):
                _fail("recovery_summary_invalid")
            identities.append(item)
    if len(identities) != value["examined"] or len(set(identities)) != len(identities):
        _fail("recovery_summary_invalid")
    return _decode(_json(value))


def _recovery_row(conn, scope):
    row = conn.execute(
        "SELECT CASE WHEN length(CAST(binding_json AS BLOB))<=16384 THEN binding_json END,"
        "nonce,state,CASE WHEN summary_json IS NULL OR length(CAST(summary_json AS BLOB))<=16384 "
        "THEN summary_json ELSE '' END FROM main.cpu_action_recovery WHERE scope=?",
        (scope["results_dir"],)).fetchone()
    if row is None:
        return None
    row = tuple(row)
    if (row[0] != _json(scope) or type(row[1]) is not str or not _HEX.fullmatch(row[1])
            or row[2] not in {"IN_PROGRESS", "COMPLETE"}
            or (row[2] == "IN_PROGRESS") != (row[3] is None)):
        _fail("recovery_barrier_invalid")
    if row[3] is not None:
        _recovery_summary(_decode(row[3]), scope)
    return row


def _recovery_available(conn, scope):
    row = _recovery_row(conn, scope)
    if row is not None and row[2] == "IN_PROGRESS":
        _fail("recovery_in_progress")
    # The row precedes all settlement, but a partial lock acquisition may not
    # yet have written that row. Absence alone is never proof of owner death.
    gate = Path(scope["results_dir"]) / "_cpu_terminal_recovery.lock"
    try:
        gate.lstat()
    except FileNotFoundError:
        return row
    except OSError as exc:
        raise CpuBudgetHOLD("cpu_budget_recovery_owner_unverifiable") from exc
    _fail("recovery_owner_unavailable")


def _recovery_owned(conn, scope, nonce, lease):
    from orze.core.idea_source_lock import idea_source_lock_owned
    expected = (_json(scope), nonce, "IN_PROGRESS", None)
    if (_recovery_row(conn, scope) != expected
            or lease.lock_dir != Path(scope["results_dir"]) / "_cpu_terminal_recovery.lock"
            or not idea_source_lock_owned(lease)):
        _fail("recovery_ownership_changed")
    stopped = _scope_row(conn, scope)
    if stopped is not None or _key(scope) in _HELD:
        _fail("recovery_stopped")
    return expected


def _permit(value):
    if type(value) is not dict or set(value) != {
            "schema", "budget_scope", "reservation_id", "task_id", "slot",
            "wall_limit_seconds", "reserved_nanoseconds"}:
        _fail("permit_invalid")
    scope = _scope(value["budget_scope"])
    _token(value["task_id"])
    if (type(value["schema"]) is not int or value["schema"] != 1
            or type(value["reservation_id"]) is not str or not _HEX.fullmatch(value["reservation_id"])
            or type(value["slot"]) is not int or not 0 <= value["slot"] < scope["declaration"]["slots"]
            or type(value["reserved_nanoseconds"]) is not str
            or value["reserved_nanoseconds"] != str(_ns(value["wall_limit_seconds"], reservation=True))):
        _fail("permit_invalid")
    return _decode(_json(value))


def _ref(ref):
    if type(ref) is not AttemptRef:
        _fail("reference_invalid")
    return _json(asdict(ref))


def _bound_current(conn, permit, ref, *, states):
    row = require_current(conn, ref, states=states)
    if row["binding"].get("reservation_id") != permit["reservation_id"]:
        _fail("attempt_permit_changed")
    return row


def _reservation(conn, permit):
    rows = conn.execute("SELECT scope,task_id,slot,"
        "CASE WHEN length(CAST(permit_json AS BLOB))<=16384 THEN permit_json END,"
        "CASE WHEN ref_json IS NULL OR length(CAST(ref_json AS BLOB))<=16384 THEN ref_json ELSE '' END,"
        "state,terminal_sha256 FROM main.cpu_action_reservations WHERE reservation_id=?",
        (permit["reservation_id"],)).fetchall()
    if len(rows) != 1 or tuple(rows[0][:4]) != (
            permit["budget_scope"]["results_dir"], permit["task_id"], permit["slot"], _json(permit)):
        _fail("permit_changed")
    row = tuple(rows[0])
    if row[5] not in {"RESERVED", "BOUND", "SETTLED"} or (
            (row[5] == "RESERVED") != (row[4] is None)) or (
            (row[5] == "SETTLED") != (row[6] is not None)):
        _fail("reservation_state_invalid")
    if row[4] is not None:
        parsed = _decode(row[4])
        if _ref(AttemptRef(**parsed)) != row[4] or parsed["task_id"] != permit["task_id"]:
            _fail("reference_invalid")
    return row


def _totals(conn, scope):
    charged, active = 0, {}
    rows = conn.execute("SELECT CASE WHEN length(CAST(permit_json AS BLOB))<=16384 THEN permit_json END "
                        "FROM main.cpu_action_reservations WHERE scope=? ORDER BY reservation_id",
                        (scope["results_dir"],))
    for raw, in rows:
        permit = _permit(_decode(raw))
        if _json(permit["budget_scope"]) != _json(scope):
            _fail("reservation_scope_changed")
        row = _reservation(conn, permit)
        charged += int(permit["reserved_nanoseconds"])
        if row[5] != "SETTLED":
            if permit["slot"] in active:
                _fail("slot_conflict")
            active[permit["slot"]] = permit["task_id"]
    if charged > _ns(scope["declaration"]["wall_budget_seconds"]):
        _fail("charged_budget_invalid")
    return charged, active


@contextmanager
def _read(lake, scope):
    _check_route(lake, scope)
    with closing(sqlite3.connect(Path(scope["database"]).as_uri() + "?mode=ro", uri=True, timeout=1)) as conn:
        conn.execute("BEGIN")
        try:
            _schema(conn)
            _scope_row(conn, scope)
            yield conn
            _check_route(lake, scope)
        finally:
            conn.rollback()


def _key(scope):
    return scope["database"], tuple(scope["database_identity"]), scope["results_dir"]


def _latch(scope):
    _HELD.add(_key(scope))
    # Best effort durable stop after an uncertain commit, never a second debit.
    try:
        if _path(scope["database"], directory=False) != (scope["database"], scope["database_identity"]):
            return
        conn = sqlite3.connect(Path(scope["database"]).as_uri() + "?mode=rw", uri=True, timeout=0.25)
        try:
            conn.execute("BEGIN IMMEDIATE")
            _schema(conn)
            conn.execute("UPDATE main.cpu_action_scopes SET stop_json=COALESCE(stop_json,?) "
                         "WHERE scope=? AND binding_json=?",
                         (_json({"kind": "Stop", "reason": "budget_storage_unconfirmed", "wakeup": None}),
                          scope["results_dir"], _json(scope)))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def _write(lake, scope, callback, *, create=False, watch=None, completion=False):
    _check_route(lake, scope)
    if _key(scope) in _HELD:
        _fail("storage_unconfirmed")
    conn = lake.conn
    attempted = False
    try:
        conn.execute("BEGIN IMMEDIATE")
        _schema(conn, create=create)
        result = callback(conn)
        if not conn.in_transaction:
            attempted = True
            _fail("transaction_ended_early")
        _scope_row(conn, scope)
        expected = watch(conn, result) if watch is not None else None
        attempted = True
        conn.commit()
        if conn.in_transaction:
            _fail("commit_unconfirmed")
        with _read(lake, scope) as check:
            _totals(check, scope)
            # A legitimate peer may reserve a different slot, append a policy
            # decision, or stop this scope after our commit. Only our captured
            # rows must remain exact; require_permit gates GO on current Stop.
            if watch is not None and watch(check, result) != expected:
                _fail("committed_rows_changed")
        return result
    except BaseException as exc:
        if conn.in_transaction:
            try:
                conn.rollback()
            except BaseException:
                attempted = True
        if attempted:
            if completion:
                # Only the final recovery ACK uses this path, after every
                # physical guard exit was confirmed. A committed COMPLETE is
                # safe to recognize on restart even if its response was lost.
                _HELD.add(_key(scope))
            else:
                _latch(scope)
        if isinstance(exc, CpuBudgetHOLD):
            raise
        raise CpuBudgetHOLD("cpu_budget_transaction_unconfirmed") from exc


def initialize(lake, results_dir, declaration):
    """Pin one policy per canonical results directory; never reset limits."""
    database, db_identity = _route(lake)
    scope_path, directory_identity = _path(results_dir, directory=True)
    scope = {"schema": 1, "results_dir": scope_path, "database": database,
             "database_identity": db_identity, "directory_identity": directory_identity,
             "declaration": _declaration(declaration)}
    scope["policy_sha256"] = hashlib.sha256(_json(scope).encode()).hexdigest()
    def action(conn):
        prior = conn.execute("SELECT binding_json FROM main.cpu_action_scopes WHERE scope=?",
                             (scope_path,)).fetchone()
        if prior is None:
            conn.execute("INSERT INTO main.cpu_action_scopes VALUES (?,?,NULL)", (scope_path, _json(scope)))
        _scope_row(conn, scope)
        _totals(conn, scope)
        return _decode(_json(scope))
    return _write(lake, scope, action, create=True)


def snapshot(lake, scope):
    scope = _scope(scope)
    with _read(lake, scope) as conn:
        stopped = _scope_row(conn, scope)
        charged, active = _totals(conn, scope)
    return {"schema": 1, "scope": scope["results_dir"],
            "reserved_wall_seconds": _seconds(charged),
            "remaining_wall_seconds": _seconds(_ns(scope["declaration"]["wall_budget_seconds"]) - charged),
            "free_slots": scope["declaration"]["slots"] - len(active),
            "active_reservations": len(active),
            "stopped": stopped is not None or _key(scope) in _HELD, "stop": stopped}


def reserve(lake, scope, task_id, timeout_seconds):
    """Return a durable permit or None without an attempt/claim/debit on Wait."""
    scope = _scope(scope)
    _token(task_id)
    amount = _ns(timeout_seconds, reservation=True)
    def action(conn):
        if _scope_row(conn, scope) is not None:
            _fail("stopped")
        _recovery_available(conn, scope)
        charged, active = _totals(conn, scope)
        if (task_id in active.values() or len(active) == scope["declaration"]["slots"]
                or charged + amount > _ns(scope["declaration"]["wall_budget_seconds"])):
            return None
        slot = next(index for index in range(scope["declaration"]["slots"]) if index not in active)
        permit = {"schema": 1, "budget_scope": scope, "reservation_id": secrets.token_hex(24),
                  "task_id": task_id, "slot": slot, "wall_limit_seconds": timeout_seconds,
                  "reserved_nanoseconds": str(amount)}
        conn.execute("INSERT INTO main.cpu_action_reservations VALUES (?,?,?,?,?,NULL,'RESERVED',NULL)",
                     (permit["reservation_id"], scope["results_dir"], task_id, slot, _json(permit)))
        _reservation(conn, permit)
        _totals(conn, scope)
        return _decode(_json(permit))
    return _write(lake, scope, action,
                  watch=lambda conn, result: None if result is None else _reservation(conn, result))


def require_permit(lake, permit, ref=None):
    """Read exact authorization; Stop or any changed/unsettled identity refuses."""
    permit = _permit(permit)
    scope = permit["budget_scope"]
    if _key(scope) in _HELD:
        _fail("storage_unconfirmed")
    with _read(lake, scope) as conn:
        if _scope_row(conn, scope) is not None:
            _fail("stopped")
        _recovery_available(conn, scope)
        _totals(conn, scope)
        row = _reservation(conn, permit)
        if row[5] == "SETTLED" or (ref is not None and (
                row[4] != _ref(ref) or row[5] != "BOUND")):
            _fail("permit_not_active")
        if ref is not None:
            _bound_current(conn, permit, ref, states=("LAUNCHING", "RUNNING"))
    return _decode(_json(permit))


def bind(lake, permit, ref):
    permit = _permit(permit)
    scope, encoded_ref = permit["budget_scope"], _ref(ref)
    if ref.task_id != permit["task_id"]:
        _fail("reference_task_mismatch")
    def action(conn):
        if _scope_row(conn, scope) is not None:
            _fail("stopped")
        _recovery_available(conn, scope)
        row = _reservation(conn, permit)
        _bound_current(conn, permit, ref, states=("LAUNCHING", "RUNNING"))
        if row[5] == "BOUND" and row[4] == encoded_ref:
            return _decode(_json(permit))
        if row[5] != "RESERVED":
            _fail("permit_already_bound")
        conn.execute("UPDATE main.cpu_action_reservations SET ref_json=?,state='BOUND' "
                     "WHERE reservation_id=? AND state='RESERVED'", (encoded_ref, permit["reservation_id"]))
        if _reservation(conn, permit)[4:6] != (encoded_ref, "BOUND"):
            _fail("bind_unconfirmed")
        return _decode(_json(permit))
    return _write(lake, scope, action, watch=lambda conn, result: (
        _reservation(conn, permit), _attempt_json(_bound_current(
            conn, permit, ref, states=("LAUNCHING", "RUNNING")))))


def _terminal_runtime_lease(bound, terminal):
    """Validate historical v2 provenance, never current execution permission.

    A confirmed sample is compared only with its captured descriptor. Reading
    today's clock/boot here would make historical F settlement time-dependent.
    The caller still checks the actual full Ref, effect and closure authority.
    """
    ready, closure = bound.get("supervision"), terminal.get("process_tree")
    versioned = ("runtime_lease" in bound or "runtime_lease" in terminal
                 or bound.get("process_supervision_protocol") == "orze.linux_subreaper.v2"
                 or (type(ready) is dict and ("runtime_lease" in ready
                     or ready.get("schema") == 2 or ready.get("protocol") == "orze.linux_subreaper.v2"))
                 or (type(closure) is dict and (closure.get("schema") == 2
                     or "lease_expired" in closure or "lease_observed_ns" in closure)))
    if not versioned:
        return None
    terminal_fields = {"outcome", "reason_code", "return_code", "process_tree", "artifact_ids",
                       "observation_ids", "lifecycle_phase", "elapsed_wall_seconds", "lifecycle",
                       "effect_receipt_sha256", "runtime_lease"}
    elapsed = terminal.get("elapsed_wall_seconds")
    if (set(terminal) != terminal_fields or terminal["lifecycle_phase"] != "action"
            or type(elapsed) not in (int, float) or not 0 <= elapsed <= 2**63 - 1
            or not math.isfinite(elapsed)):
        _fail("runtime_lease_terminal_invalid")
    from orze.engine.supervisor_worker import validate_runtime_lease
    try:
        lease = validate_runtime_lease(bound.get("runtime_lease"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise CpuBudgetHOLD("cpu_budget_runtime_lease_invalid") from exc
    if lease["deadline_ns"] - lease["issued_ns"] > _ns(bound.get("timeout_seconds")):
        _fail("runtime_lease_exceeds_envelope")
    fields = {"schema", "protocol", "identity", "nonce_sha256", "command_sha256",
              "worker", "supervisor", "runtime_lease"}
    if (bound.get("process_supervision_protocol") != "orze.linux_subreaper.v2"
            or type(ready) is not dict or set(ready) != fields
            or type(ready["schema"]) is not int or ready["schema"] != 2
            or ready["protocol"] != "orze.linux_subreaper.v2"
            or _attempt_json(ready["runtime_lease"]) != _attempt_json(lease)
            or ready["command_sha256"] != bound.get("command_sha256")
            or any(type(ready[key]) is not str or not _SHA.fullmatch(ready[key])
                   for key in ("command_sha256", "nonce_sha256"))):
        _fail("runtime_lease_ready_invalid")
    for key in ("worker", "supervisor"):
        member = ready[key]
        if (type(member) is not dict or set(member) != {"pid", "start_ticks"}
                or type(member["pid"]) is not int or member["pid"] <= 0
                or type(member["start_ticks"]) is not int or member["start_ticks"] < 0):
            _fail("runtime_lease_identity_invalid")
    if (type(bound.get("process_pid")) is not int or bound["process_pid"] != ready["worker"]["pid"]
            or ready["worker"]["pid"] == ready["supervisor"]["pid"]):
        _fail("runtime_lease_identity_invalid")
    closure_fields = {"schema", "event", "binding", "worker_returncode", "stop_requested",
                      "forced_cleanup", "reaped_children", "wait_proof",
                      "lease_expired", "lease_observed_ns"}
    if (type(closure) is not dict or set(closure) != closure_fields
            or type(closure["schema"]) is not int or closure["schema"] != 2
            or _attempt_json(closure["binding"]) != _attempt_json(ready)
            or type(closure["lease_expired"]) is not bool
            or type(closure["lease_observed_ns"]) is not int
            or not lease["issued_ns"] <= closure["lease_observed_ns"] <= 2**63 - 1
            or closure["lease_expired"] != (closure["lease_observed_ns"] >= lease["deadline_ns"])
            or (closure["lease_expired"] and closure["stop_requested"] is not True)):
        _fail("runtime_lease_closure_invalid")
    sample = terminal.get("runtime_lease")
    if (type(sample) is not dict or set(sample) != {"schema", "status", "observed_ns"}
            or type(sample["schema"]) is not int or sample["schema"] != 1
            or type(sample["status"]) is not str or sample["status"] not in {"authorized", "expired"}
            or type(sample["observed_ns"]) is not int
            or not lease["issued_ns"] <= sample["observed_ns"] <= 2**63 - 1):
        _fail("runtime_lease_terminal_invalid")
    expired = sample["observed_ns"] >= lease["deadline_ns"]
    if (sample["status"] != ("expired" if expired else "authorized")
            or sample["observed_ns"] < closure["lease_observed_ns"]
            or (closure["lease_expired"] and not expired)):
        _fail("runtime_lease_terminal_invalid")
    if expired and (terminal.get("outcome") != "interrupted"
            or terminal.get("reason_code") != "cpu_runtime_lease_expired"
            or type(terminal.get("artifact_ids")) is not list or terminal["artifact_ids"]
            or type(terminal.get("observation_ids")) is not list or terminal["observation_ids"]):
        _fail("runtime_lease_expiry_invalid")
    if not expired:
        outcome = ("interrupted" if closure["stop_requested"] or closure["forced_cleanup"]
                   else "completed" if closure["worker_returncode"] == 0 else "failed")
        if (terminal.get("outcome") != outcome
                or terminal.get("reason_code") != "cpu_action_" + outcome):
            _fail("runtime_lease_terminal_invalid")
    return sample["status"]


def _terminal(conn, permit, ref, evidence):
    from orze.engine.attempt_effect_receipts import _scan, _read as effect_read, _decode as effect_decode, _ref_fields
    scope = permit["budget_scope"]
    row = _bound_current(conn, permit, ref, states=("TERMINAL", "NOT_STARTED"))
    terminal = row["terminal"]
    if row["state"] == "NOT_STARTED" and ("runtime_lease" in row["binding"]
            or "runtime_lease" in terminal
            or row["binding"].get("process_supervision_protocol") == "orze.linux_subreaper.v2"):
        _fail("runtime_lease_not_started_unconfirmed")
    if _attempt_json(terminal) != _attempt_json(evidence):
        _fail("terminal_evidence_changed")
    folder = Path(scope["results_dir"]) / ref.task_id
    digest = terminal.get("effect_receipt_sha256")
    if _scan(folder).get(ref.attempt_id) != (digest, True):
        _fail("effect_unconfirmed")
    raw = effect_read(folder / "_execution_effects" / ref.attempt_id / "prepared.json")
    prepared = effect_decode(raw)
    identity = _ref_fields(ref, folder)
    if (hashlib.sha256(raw).hexdigest() != digest
            or _json({k: prepared.get(k) for k in identity}) != _json(identity)):
        _fail("effect_reference_changed")
    if row["state"] == "TERMINAL":
        lease_status = _terminal_runtime_lease(row["binding"], terminal)
        closure = terminal.get("process_tree")
        fields = {"schema", "event", "binding", "worker_returncode", "stop_requested",
                  "forced_cleanup", "reaped_children", "wait_proof"}
        if lease_status is not None:
            fields |= {"lease_expired", "lease_observed_ns"}
        if (type(closure) is not dict or set(closure) != fields
                or type(closure["schema"]) is not int or closure["schema"] != (2 if lease_status else 1)
                or closure["event"] != "TREE_CLOSED" or closure["wait_proof"] != "ECHILD_WALL"
                or type(closure["worker_returncode"]) is not int
                or type(closure["stop_requested"]) is not bool
                or type(closure["forced_cleanup"]) is not bool
                or type(closure["reaped_children"]) is not int or closure["reaped_children"] < 1
                or _json(closure["binding"]) != _json(row["binding"].get("supervision"))
                or _json(closure["binding"].get("identity")) !=
                _json({"attempt_ref": asdict(ref), "scope": str(folder)})):
            _fail("tree_closure_unconfirmed")
        codes = [terminal[k] for k in ("return_code", "exit_code") if k in terminal]
        if not codes or any(type(v) is not int or v != closure["worker_returncode"] for v in codes):
            _fail("return_code_unconfirmed")
        if lease_status is not None:
            planned = {key: value for key, value in terminal.items()
                       if key not in {"lifecycle", "effect_receipt_sha256"}}
            if _attempt_json(prepared.get("plan")) != _attempt_json(
                    {"operation": "cpu_action_terminal", **planned}):
                _fail("runtime_lease_effect_plan_changed")
    return hashlib.sha256(_attempt_json(terminal).encode()).hexdigest()


def _recovery_terminal(conn, scope, permit, ref, evidence):
    """Read native metadata only; no old handle, Domain callback or OS query.

    This is stricter than public explicit settlement. Its snapshot is checked
    before and after the short writer, using the supplied actual connection.
    Content is not republished or scientifically reinterpreted by recovery.
    """
    from types import SimpleNamespace
    import yaml
    from orze.core.cpu_action_contract import validate_action, action_fingerprint
    from orze.core.artifact_contract import validate_artifact_publication_binding
    from orze.core.research_artifacts import artifacts_for_attempt
    from orze.core.research_observations import observations_for_attempt
    from orze.engine.execution_authority import lifecycle_fence
    from orze.engine.execution_catalog import declared_catalog
    from orze.engine import claim_authority
    from orze.engine.attempt_effect_receipts import _read as effect_read, _decode as effect_decode
    from orze.engine.process_supervision import PROTOCOL
    from orze.engine.supervisor_worker import canonical

    if ref.phase != "action":
        _fail("recovery_phase_invalid")
    digest = _terminal(conn, permit, ref, evidence)
    row = require_current(conn, ref, states=("TERMINAL",))
    bound, terminal = row["binding"], row["terminal"]
    lease_status = _terminal_runtime_lease(bound, terminal)
    folder = Path(scope["results_dir"]) / ref.task_id
    expected = {"origin": "native_cpu_action", "kind": "native_cpu_action", "resource": "cpu",
        "attempt_id": ref.attempt_id, "attempt_ref": asdict(ref), "scope": str(folder),
        "work_dir": str(folder / "_action_attempts" / ref.attempt_id / "work"),
        "timeout_seconds": permit["wall_limit_seconds"], "reservation_id": permit["reservation_id"],
        "process_supervision_protocol": "orze.linux_subreaper.v2" if lease_status else PROTOCOL,
        "lifecycle_phase": "action"}
    if _attempt_json({key: bound.get(key) for key in expected}) != _attempt_json(expected):
        _fail("recovery_native_binding_changed")
    for key in ("action_sha256", "command_sha256", "inputs_sha256"):
        if type(bound.get(key)) is not str or not _SHA.fullmatch(bound[key]):
            _fail("recovery_native_digest_invalid")
    ready = bound.get("supervision")
    ready_fields = {"schema", "protocol", "identity", "nonce_sha256", "command_sha256", "worker", "supervisor"}
    if lease_status is not None:
        ready_fields.add("runtime_lease")
    if (type(ready) is not dict or set(ready) != ready_fields
            or type(ready["schema"]) is not int or ready["schema"] != (2 if lease_status else 1)
            or ready["protocol"] != expected["process_supervision_protocol"]
            or ready["command_sha256"] != bound["command_sha256"]
            or type(ready["nonce_sha256"]) is not str or not _SHA.fullmatch(ready["nonce_sha256"])):
        _fail("recovery_ready_invalid")
    for key in ("worker", "supervisor"):
        member = ready[key]
        if (type(member) is not dict or set(member) != {"pid", "start_ticks"}
                or type(member["pid"]) is not int or member["pid"] <= 0
                or type(member["start_ticks"]) is not int or member["start_ticks"] < 0):
            _fail("recovery_ready_identity_invalid")
    if (type(bound.get("process_pid")) is not int or bound["process_pid"] != ready["worker"]["pid"]
            or ready["worker"]["pid"] == ready["supervisor"]["pid"]):
        _fail("recovery_process_binding_changed")

    source = bound.get("source")
    source_fields = {"claim_attempt_id", "claim_sha256", "config_sha256", "database"}
    if (type(source) is not dict or set(source) not in (source_fields, source_fields | {"replication_request"})
            or source["database"] != scope["database"] or declared_catalog(folder) != scope["database"]):
        _fail("recovery_source_invalid")
    claim, claim_sha = claim_authority.read_claim_snapshot(
        folder / "claim.json", limit=8192, required=True)
    if (claim.get("attempt_id") != source["claim_attempt_id"] or claim_sha != source["claim_sha256"]
            or claim.get("resource") != "cpu" or "gpu" not in claim or claim["gpu"] is not None
            or claim.get("lifecycle_db") != scope["database"]):
        _fail("recovery_claim_changed")
    _token(source["claim_attempt_id"])
    task = conn.execute("SELECT kind,CASE WHEN typeof(config)='text' AND "
        "length(CAST(config AS BLOB))<=65536 THEN config END FROM main.ideas "
        "WHERE idea_id=? COLLATE BINARY LIMIT 2", (ref.task_id,)).fetchall()
    if (len(task) != 1 or task[0][0] != "native_cpu_action" or type(task[0][1]) is not str
            or hashlib.sha256(task[0][1].encode()).hexdigest() != source["config_sha256"]):
        _fail("recovery_task_changed")
    from orze.core.replication_requests import request_for_task
    replication = request_for_task(conn, ref.task_id)
    if (replication is None) != ("replication_request" not in source):
        _fail("recovery_replication_changed")
    if replication is not None:
        if (replication["schema"] != 2 or replication.get("adapter") != "native_cpu_action"
                or replication["scope"] != scope["results_dir"] or replication["database"] != scope["database"]
                or replication["action_sha256"] != bound["action_sha256"]
                or _attempt_json(replication) != _attempt_json(source["replication_request"])):
            _fail("recovery_replication_changed")
    configured = yaml.safe_load(task[0][1])
    if type(configured) is not dict or configured.get("kind", "native_cpu_action") != "native_cpu_action":
        _fail("recovery_task_invalid")
    publication = validate_artifact_publication_binding(bound.get("artifact_publication"))
    if publication["scope"] != scope["results_dir"] or publication["spec_fingerprint"] != bound["action_sha256"]:
        _fail("recovery_artifact_binding_changed")
    if "domain_run" not in bound:
        action = validate_action(configured.get("action"))
        if (action_fingerprint(action) != bound["action_sha256"]
                or hashlib.sha256(canonical(action["command"])).hexdigest() != bound["command_sha256"]
                or hashlib.sha256(_attempt_json(action["inputs"]).encode()).hexdigest() != bound["inputs_sha256"]
                or _attempt_json({"timeout": action["timeout_seconds"], "outputs": action["outputs"]}) !=
                   _attempt_json({"timeout": bound["timeout_seconds"], "outputs": publication["contract"]["outputs"]})
                or "observation_publication" in bound):
            _fail("recovery_action_changed")
    else:
        # A prepared Domain cannot be recreated from metadata. Validate its
        # pinned request/route relationship, never call prepare or interpret.
        from orze.core.research_interfaces import parse_domain_task
        request = parse_domain_task(task[0][1])
        domain = bound["domain_run"]
        if (type(domain) is not dict or set(domain) != {"schema", "domain_id", "domain_kind",
                "domain_config_sha256", "request_sha256", "action_sha256", "source_snapshot", "observation"}
                or type(domain["schema"]) is not int or domain["schema"] != 1
                or domain["request_sha256"] != source["config_sha256"]
                or domain["action_sha256"] != bound["action_sha256"]
                or type(domain["domain_config_sha256"]) is not str or not _SHA.fullmatch(domain["domain_config_sha256"])
                or _attempt_json({"timeout": request["timeout_seconds"], "outputs": request["outputs"]}) !=
                   _attempt_json({"timeout": bound["timeout_seconds"], "outputs": publication["contract"]["outputs"]})):
            _fail("recovery_domain_changed")
        _token(domain["domain_id"])
        _token(domain["domain_kind"])
        sources = domain["source_snapshot"]
        identities = {"schema": 1, "scope": scope["results_dir"], "database": scope["database"],
                      "database_identity": scope["database_identity"], "scope_identity": scope["directory_identity"]}
        if (type(sources) is not dict or set(sources) != set(identities) | {"inputs"}
                or _attempt_json({key: sources.get(key) for key in identities}) != _attempt_json(identities)
                or type(sources["inputs"]) is not list or len(sources["inputs"]) > 32):
            _fail("recovery_domain_source_changed")
        from orze.core.research_artifacts import _record as encode_artifact_record
        records = []
        for item in sources["inputs"]:
            if (type(item) is not dict or set(item) != {"artifact", "source_sha256", "effect_sha256"}
                    or any(type(item[key]) is not str or not _SHA.fullmatch(item[key])
                           for key in ("source_sha256", "effect_sha256"))):
                _fail("recovery_domain_source_invalid")
            record = json.loads(encode_artifact_record(item["artifact"]))
            if record["scope"] != scope["results_dir"]:
                _fail("recovery_domain_source_changed")
            records.append(record)
        if [item["artifact_id"] for item in records] != request["input_artifact_ids"]:
            _fail("recovery_domain_inputs_changed")
        declaration = domain["observation"]
        if declaration is None:
            if "observation_publication" in bound:
                _fail("recovery_observation_binding_changed")
        else:
            from orze.core.cpu_observation_contract import cpu_observation_binding
            if type(declaration) is not dict or set(declaration) != {
                    "adapter_id", "spec_fingerprint", "protocol_fingerprint", "result_output"}:
                _fail("recovery_observation_declaration_invalid")
            output = publication["contract"]["outputs"].get(declaration["result_output"])
            expected_observation = cpu_observation_binding(adapter_id=declaration["adapter_id"],
                spec_fingerprint=declaration["spec_fingerprint"], protocol_fingerprint=declaration["protocol_fingerprint"],
                scope=scope["results_dir"], input_artifacts=records)
            if (output is None or output["max_bytes"] > 1048576
                    or _attempt_json(bound.get("observation_publication")) != _attempt_json(expected_observation)):
                _fail("recovery_observation_binding_changed")

    closure = terminal["process_tree"]
    outcome = ("interrupted" if closure["stop_requested"] or closure["forced_cleanup"]
               else "completed" if closure["worker_returncode"] == 0 else "failed")
    if lease_status == "expired":
        outcome = "interrupted"
    fields = {"outcome", "reason_code", "return_code", "process_tree", "artifact_ids", "observation_ids",
              "lifecycle_phase", "elapsed_wall_seconds", "lifecycle", "effect_receipt_sha256"}
    if lease_status is not None:
        fields.add("runtime_lease")
    reason = "cpu_runtime_lease_expired" if lease_status == "expired" else "cpu_action_" + outcome
    elapsed = terminal.get("elapsed_wall_seconds")
    if (set(terminal) != fields or terminal["outcome"] != outcome
            or terminal["reason_code"] != reason or terminal["lifecycle_phase"] != "action"
            or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
        _fail("recovery_terminal_invalid")
    lifecycle = lifecycle_fence(SimpleNamespace(conn=conn), ref.task_id, "action")
    state = "COMPLETE" if outcome == "completed" else "FAILED"
    if (lifecycle["global_state"] != state or lifecycle["phase_state"] != state
            or _attempt_json(bound.get("lifecycle")) != _attempt_json(lifecycle)
            or _attempt_json(terminal["lifecycle"]) != _attempt_json(lifecycle)):
        _fail("recovery_lifecycle_changed")
    kinds = conn.execute("SELECT s.sop_type,t.sop_type FROM main.idea_state s "
        "JOIN main.idea_transitions t ON t.id=? WHERE s.idea_id=? COLLATE BINARY",
        (lifecycle["global_transition_id"], ref.task_id)).fetchall()
    if len(kinds) != 1 or tuple(kinds[0]) != ("action", "action"):
        _fail("recovery_lifecycle_kind_changed")
    prepared_raw = effect_read(folder / "_execution_effects" / ref.attempt_id / "prepared.json")
    if hashlib.sha256(prepared_raw).hexdigest() != terminal["effect_receipt_sha256"]:
        _fail("recovery_effect_changed")
    prepared = effect_decode(prepared_raw)
    planned = {key: value for key, value in terminal.items() if key not in {"lifecycle", "effect_receipt_sha256"}}
    if _attempt_json(prepared.get("plan")) != _attempt_json({"operation": "cpu_action_terminal", **planned}):
        _fail("recovery_effect_plan_changed")
    artifacts = artifacts_for_attempt(conn, ref)
    observations = observations_for_attempt(conn, ref)
    for item in artifacts:
        output = publication["contract"]["outputs"].get(item["logical_name"])
        if (item["scope"] != scope["results_dir"] or item["spec_fingerprint"] != bound["action_sha256"]
                or item["path"] != str(Path(publication["root"]) / item["artifact_id"] / "content")
                or output is None or item["size_bytes"] > output["max_bytes"]):
            _fail("recovery_artifact_record_changed")
    if outcome == "completed" and {item["logical_name"] for item in artifacts} != set(publication["contract"]["outputs"]):
        _fail("recovery_artifact_set_changed")
    for key, values, identity in (("artifact_ids", artifacts, "artifact_id"),
                                  ("observation_ids", observations, "observation_id")):
        ids = terminal[key]
        if (type(ids) is not list or len(ids) > 32 or any(type(item) is not str for item in ids)
                or len(ids) != len(set(ids)) or set(ids) != {item[identity] for item in values}
                or (outcome != "completed" and ids)):
            _fail("recovery_published_set_changed")
    # Comparing all captured small rows is a fence, not new artifact content
    # hashing or a claim that archived domain inputs can become live handles.
    return (digest, _attempt_json(bound), _attempt_json(terminal),
            tuple(_attempt_json(item) for item in artifacts), tuple(_attempt_json(item) for item in observations))


def _settle(lake, permit, ref, terminal_evidence, *, recovery=None):
    from orze.engine.attempt_effect_lock import attempt_effect_lock
    permit = _permit(permit)
    scope, encoded_ref = permit["budget_scope"], _ref(ref)
    if ref.task_id != permit["task_id"]:
        _fail("reference_task_mismatch")
    _check_route(lake, scope)
    folder = Path(scope["results_dir"]) / ref.task_id
    with attempt_effect_lock(folder):
        def fence(conn):
            if recovery is None:
                return _terminal(conn, permit, ref, terminal_evidence)
            nonce, lease, expected = recovery
            _recovery_owned(conn, scope, nonce, lease)
            actual = _recovery_terminal(conn, scope, permit, ref, terminal_evidence)
            if actual != expected:
                _fail("recovery_terminal_changed")
            return actual

        def action(conn):
            _scope_row(conn, scope)  # Settlement remains allowed after Stop.
            row = _reservation(conn, permit)
            if row[4] != encoded_ref:
                _fail("settlement_reference_changed")
            digest = _terminal(conn, permit, ref, terminal_evidence)
            fence(conn)
            if row[5] == "SETTLED":
                if row[6] != digest:
                    _fail("settlement_changed")
                return "duplicate"
            if row[5] != "BOUND":
                _fail("settlement_unbound")
            conn.execute("UPDATE main.cpu_action_reservations SET state='SETTLED',terminal_sha256=? "
                         "WHERE reservation_id=? AND state='BOUND'", (digest, permit["reservation_id"]))
            if _reservation(conn, permit)[5:] != ("SETTLED", digest):
                _fail("settlement_unconfirmed")
            return "settled"
        return _write(lake, scope, action, watch=lambda conn, result: (
            _reservation(conn, permit), fence(conn)))


def settle(lake, permit, ref, terminal_evidence):
    """Release only this slot after durable native proof; never refund wall."""
    return _settle(lake, permit, ref, terminal_evidence)


def reconcile_confirmed_terminals(lake, scope):
    """Resume only confirmed CPU settlement, never execution or publication.

    Inspect at most slots live rows; existing _totals/history validation is
    not globally bounded. Each settlement commits independently. An unknown
    guard exit leaves IN_PROGRESS even when the slot row is already SETTLED.
    """
    from orze.core.idea_source_lock import SourceLockInDoubt, idea_source_lock, idea_source_lock_owned

    scope = _check_route(lake, scope)
    if _key(scope) in _HELD:
        _fail("storage_unconfirmed")
    summary = {"schema": 1, "examined": 0, "settled": [], "already_settled": [], "retained": []}
    try:
        candidates = []
        with _read(lake, scope) as conn:
            _recovery_available(conn, scope)  # Required even with zero BOUND.
            stopped = _scope_row(conn, scope)
            if stopped is not None and stopped["reason"] == "budget_storage_unconfirmed":
                _fail("storage_unconfirmed")
            rows = conn.execute("SELECT CASE WHEN length(CAST(permit_json AS BLOB))<=16384 "
                "THEN permit_json END FROM main.cpu_action_reservations WHERE scope=? AND state!='SETTLED' "
                "ORDER BY reservation_id LIMIT ?", (scope["results_dir"], scope["declaration"]["slots"] + 1)).fetchall()
            if len(rows) > scope["declaration"]["slots"]:
                _fail("recovery_active_limit")
            for raw, in rows:
                permit = _permit(_decode(raw))
                if _json(permit["budget_scope"]) != _json(scope):
                    _fail("recovery_scope_changed")
                saved = _reservation(conn, permit)
                summary["examined"] += 1
                if stopped is not None:
                    reason = "scope_stopped"
                elif saved[5] == "RESERVED":
                    reason = "reserved"
                else:
                    ref = AttemptRef(**_decode(saved[4]))
                    current = _bound_current(conn, permit, ref,
                        states=("LAUNCHING", "RUNNING", "IN_DOUBT", "NOT_STARTED", "TERMINAL"))
                    if current["state"] == "TERMINAL":
                        evidence = current["terminal"]
                        expected = _recovery_terminal(conn, scope, permit, ref, evidence)
                        candidates.append((permit, ref, evidence, expected))
                        continue
                    reason = current["state"].lower()
                summary["retained"].append({"reservation_id": permit["reservation_id"], "reason": reason})
        if not candidates:
            return _recovery_summary(summary, scope)

        gate = Path(scope["results_dir"]) / "_cpu_terminal_recovery.lock"
        nonce = secrets.token_hex(24)
        with idea_source_lock(gate) as lease:
            if lease is None:
                _fail("recovery_owner_unavailable")  # Never poison a competing owner.
            try:
                def begin(conn):
                    prior = _recovery_row(conn, scope)
                    if (prior is not None and prior[2] != "COMPLETE") or _scope_row(conn, scope) is not None:
                        _fail("recovery_unavailable")
                    if not idea_source_lock_owned(lease):
                        _fail("recovery_ownership_changed")
                    conn.execute("INSERT INTO main.cpu_action_recovery VALUES (?,?,?,'IN_PROGRESS',NULL) "
                        "ON CONFLICT(scope) DO UPDATE SET binding_json=excluded.binding_json,nonce=excluded.nonce,"
                        "state='IN_PROGRESS',summary_json=NULL", (scope["results_dir"], _json(scope), nonce))
                    return None
                _write(lake, scope, begin, watch=lambda conn, result: _recovery_owned(conn, scope, nonce, lease))
                for permit, ref, evidence, expected in candidates:
                    result = _settle(lake, permit, ref, evidence, recovery=(nonce, lease, expected))
                    if result not in {"settled", "duplicate"}:
                        _fail("recovery_settlement_unconfirmed")
                    key = "settled" if result == "settled" else "already_settled"
                    summary[key].append(permit["reservation_id"])
                summary = _recovery_summary(summary, scope)
                with _read(lake, scope) as conn:
                    _recovery_owned(conn, scope, nonce, lease)
            except BaseException as exc:
                # Conversion must occur inside the source-lock context so any
                # uncertain body/inner guard retains its original directory.
                raise SourceLockInDoubt("cpu_terminal_recovery_unconfirmed") from exc
        # SourceLock itself may have removed its directory before a failing
        # fsync. IN_PROGRESS, committed above, outlives that failure window.
        expected = (_json(scope), nonce, "COMPLETE", _json(summary))
        def complete(conn):
            if _recovery_row(conn, scope) != (_json(scope), nonce, "IN_PROGRESS", None):
                _fail("recovery_completion_changed")
            if _scope_row(conn, scope) is not None:
                _fail("recovery_stopped")
            try:
                gate.lstat()
            except FileNotFoundError:
                pass
            else:
                _fail("recovery_release_unconfirmed")
            changed = conn.execute("UPDATE main.cpu_action_recovery SET state='COMPLETE',summary_json=? "
                "WHERE scope=? AND nonce=? AND state='IN_PROGRESS' AND summary_json IS NULL",
                (_json(summary), scope["results_dir"], nonce)).rowcount
            if changed != 1 or _recovery_row(conn, scope) != expected:
                _fail("recovery_completion_unconfirmed")
            return None
        _write(lake, scope, complete, watch=lambda conn, result: _recovery_row(conn, scope), completion=True)
        return _recovery_summary(summary, scope)
    except CpuBudgetHOLD:
        raise
    except BaseException as exc:
        raise CpuBudgetHOLD("cpu_budget_recovery_unconfirmed") from exc


def record_decision(lake, scope, decision):
    scope = _scope(scope)
    if (type(decision) is not dict or set(decision) != {"kind", "reason", "wakeup"}
            or decision["kind"] not in {"Wait", "Stop"}
            or type(decision["reason"]) is not str or not decision["reason"].strip()
            or len(decision["reason"].encode()) > 128
            or (decision["kind"] == "Stop" and decision["wakeup"] is not None)
            or (decision["kind"] == "Wait" and decision["wakeup"] is None)):
        _fail("decision_invalid")
    if decision["wakeup"] is not None:
        _number(decision["wakeup"])
    decision = _decode(_json(decision))
    def verify(conn, record):
        row = conn.execute("SELECT scope,record_json FROM main.cpu_action_decisions WHERE decision_id=?",
                           (record["decision_id"],)).fetchone()
        if row is None or tuple(row) != (scope["results_dir"], _json(record)):
            _fail("decision_unconfirmed")
        if decision["kind"] == "Stop" and _scope_row(conn, scope) is None:
            _fail("stop_unconfirmed")
        return tuple(row)
    def action(conn):
        _scope_row(conn, scope)
        record = {**decision, "decision_id": secrets.token_hex(24), "recorded_at": time.time()}
        conn.execute("INSERT INTO main.cpu_action_decisions VALUES (?,?,?)",
                     (record["decision_id"], scope["results_dir"], _json(record)))
        if decision["kind"] == "Stop":
            conn.execute("UPDATE main.cpu_action_scopes SET stop_json=COALESCE(stop_json,?) WHERE scope=?",
                         (_json(decision), scope["results_dir"]))
        verify(conn, record)
        return record
    return _write(lake, scope, action, watch=verify)
