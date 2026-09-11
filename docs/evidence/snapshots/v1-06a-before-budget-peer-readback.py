"""Persistent CPU action envelopes in the supplied IdeaLake, not CPU billing.

All public APIs own short transactions and reject an already-open caller
transaction. A permit is metadata whose exact durable row must be rechecked;
it is neither process authority nor permission to replay an unknown launch.
Slots remain occupied until confirmed native settlement. Reserved wall time
is charged forever, including unused time; no age/PID-based refund exists.
"""
from __future__ import annotations

from contextlib import contextmanager
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

from orze.core.execution_attempts import AttemptRef, require_current

_LIMIT = 16384
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_HEX = re.compile(r"[0-9a-f]{48}\Z")
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
    if stopped is not None and stopped.get("kind") != "Stop":
        _fail("stop_invalid")
    return stopped


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


def _fingerprint(conn, scope):
    digest = hashlib.sha256()
    for table, order in (("cpu_action_scopes", "scope"),
                         ("cpu_action_reservations", "reservation_id"),
                         ("cpu_action_decisions", "decision_id")):
        for row in conn.execute(f"SELECT * FROM main.{table} WHERE scope=? ORDER BY {order}",
                                (scope["results_dir"],)):
            digest.update(_json(list(row)).encode())
            digest.update(b"\n")
    return digest.digest()


@contextmanager
def _read(lake, scope):
    _check_route(lake, scope)
    with sqlite3.connect(Path(scope["database"]).as_uri() + "?mode=ro", uri=True, timeout=1) as conn:
        conn.execute("BEGIN")
        try:
            _schema(conn)
            _scope_row(conn, scope)
            yield conn
            _check_route(lake, scope)
        finally:
            conn.rollback()
    conn.close()


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


def _write(lake, scope, callback, *, create=False):
    _check_route(lake, scope)
    if _key(scope) in _HELD:
        _fail("storage_unconfirmed")
    conn = lake.conn
    attempted = False
    try:
        conn.execute("BEGIN IMMEDIATE")
        _schema(conn, create=create)
        result = callback(conn)
        expected = _fingerprint(conn, scope)
        attempted = True
        conn.commit()
        if conn.in_transaction:
            _fail("commit_unconfirmed")
        with _read(lake, scope) as check:
            if _fingerprint(check, scope) != expected:
                _fail("committed_rows_changed")
        return result
    except BaseException as exc:
        if conn.in_transaction:
            try:
                conn.rollback()
            except BaseException:
                attempted = True
        if attempted:
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
    return _write(lake, scope, action)


def require_permit(lake, permit, ref=None):
    """Read exact authorization; Stop or any changed/unsettled identity refuses."""
    permit = _permit(permit)
    scope = permit["budget_scope"]
    if _key(scope) in _HELD:
        _fail("storage_unconfirmed")
    with _read(lake, scope) as conn:
        if _scope_row(conn, scope) is not None:
            _fail("stopped")
        _totals(conn, scope)
        row = _reservation(conn, permit)
        if row[5] == "SETTLED" or (ref is not None and (
                row[4] != _ref(ref) or row[5] != "BOUND")):
            _fail("permit_not_active")
        if ref is not None:
            require_current(conn, ref, states=("LAUNCHING", "RUNNING"))
    return _decode(_json(permit))


def bind(lake, permit, ref):
    permit = _permit(permit)
    scope, encoded_ref = permit["budget_scope"], _ref(ref)
    if ref.task_id != permit["task_id"]:
        _fail("reference_task_mismatch")
    def action(conn):
        if _scope_row(conn, scope) is not None:
            _fail("stopped")
        row = _reservation(conn, permit)
        require_current(conn, ref, states=("LAUNCHING", "RUNNING"))
        if row[5] == "BOUND" and row[4] == encoded_ref:
            return _decode(_json(permit))
        if row[5] != "RESERVED":
            _fail("permit_already_bound")
        conn.execute("UPDATE main.cpu_action_reservations SET ref_json=?,state='BOUND' "
                     "WHERE reservation_id=? AND state='RESERVED'", (encoded_ref, permit["reservation_id"]))
        if _reservation(conn, permit)[4:6] != (encoded_ref, "BOUND"):
            _fail("bind_unconfirmed")
        return _decode(_json(permit))
    return _write(lake, scope, action)


def _terminal(conn, permit, ref, evidence):
    from orze.engine.attempt_effect_receipts import _scan, _read as effect_read, _decode as effect_decode, _ref_fields
    scope = permit["budget_scope"]
    row = require_current(conn, ref, states=("TERMINAL", "NOT_STARTED"))
    terminal = row["terminal"]
    if _json(terminal) != _json(evidence):
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
        closure = terminal.get("process_tree")
        fields = {"schema", "event", "binding", "worker_returncode", "stop_requested",
                  "forced_cleanup", "reaped_children", "wait_proof"}
        if (type(closure) is not dict or set(closure) != fields
                or type(closure["schema"]) is not int or closure["schema"] != 1
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
    return hashlib.sha256(_json(terminal).encode()).hexdigest()


def settle(lake, permit, ref, terminal_evidence):
    """Release only this slot after durable native proof; never refund wall."""
    from orze.engine.attempt_effect_lock import attempt_effect_lock
    permit = _permit(permit)
    scope, encoded_ref = permit["budget_scope"], _ref(ref)
    _check_route(lake, scope)
    folder = Path(scope["results_dir"]) / ref.task_id
    with attempt_effect_lock(folder):
        def action(conn):
            _scope_row(conn, scope)  # Settlement remains allowed after Stop.
            row = _reservation(conn, permit)
            if row[4] != encoded_ref:
                _fail("settlement_reference_changed")
            digest = _terminal(conn, permit, ref, terminal_evidence)
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
        return _write(lake, scope, action)


def record_decision(lake, scope, decision):
    scope = _scope(scope)
    if (type(decision) is not dict or set(decision) != {"kind", "reason", "wakeup"}
            or decision["kind"] not in {"Wait", "Stop"}
            or type(decision["reason"]) is not str or not decision["reason"].strip()
            or len(decision["reason"].encode()) > 128
            or (decision["kind"] == "Stop" and decision["wakeup"] is not None)):
        _fail("decision_invalid")
    if decision["wakeup"] is not None:
        _number(decision["wakeup"])
    decision = _decode(_json(decision))
    def action(conn):
        _scope_row(conn, scope)
        record = {**decision, "decision_id": secrets.token_hex(24), "recorded_at": time.time()}
        conn.execute("INSERT INTO main.cpu_action_decisions VALUES (?,?,?)",
                     (record["decision_id"], scope["results_dir"], _json(record)))
        if decision["kind"] == "Stop":
            conn.execute("UPDATE main.cpu_action_scopes SET stop_json=COALESCE(stop_json,?) WHERE scope=?",
                         (_json(decision), scope["results_dir"]))
        return record
    return _write(lake, scope, action)
