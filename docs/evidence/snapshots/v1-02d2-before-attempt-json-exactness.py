"""Generic execution-attempt authority inside a caller's SQLite transaction.

Writers require ``conn.in_transaction`` but never BEGIN, COMMIT, or ROLLBACK.
The adapter must acquire its per-task file guard and BEGIN IMMEDIATE first;
Python's SQLite API cannot establish the transaction's locking mode here.
Creating an attempt records launch intent, not process creation. RUNNING means
the framework observed execution creation, not that the process is still alive
at this instant. No CPU/GPU, exit-code, evaluation, or scientific semantics are
inferred. Ambiguous execution stays IN_DOUBT until an external resolution exists.

Every failed write raises so the caller can roll back its *whole* transaction,
including lifecycle changes. This module does not undo trigger side effects or
other caller writes on its own. The table belongs to this versioned contract;
an existing incompatible schema is rejected, never migrated implicitly.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass


MAX_JSON_BYTES = 65536
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 2048
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_STATES = {"LAUNCHING", "RUNNING", "TERMINAL", "NOT_STARTED", "IN_DOUBT"}
_CLOSED = {"TERMINAL", "NOT_STARTED"}
_SQL = """CREATE TABLE execution_attempts (
    attempt_id TEXT NOT NULL COLLATE BINARY PRIMARY KEY,
    task_id TEXT NOT NULL COLLATE BINARY,
    phase TEXT NOT NULL COLLATE BINARY,
    generation INTEGER NOT NULL CHECK (generation > 0),
    state TEXT NOT NULL CHECK (state IN ('LAUNCHING', 'RUNNING', 'TERMINAL', 'NOT_STARTED', 'IN_DOUBT')),
    binding_json TEXT NOT NULL,
    terminal_json TEXT,
    hold_reason TEXT,
    UNIQUE (task_id, phase, generation)
)"""
_COLUMNS = [
    ("attempt_id", "TEXT", 1, 1), ("task_id", "TEXT", 1, 0),
    ("phase", "TEXT", 1, 0), ("generation", "INTEGER", 1, 0),
    ("state", "TEXT", 1, 0), ("binding_json", "TEXT", 1, 0),
    ("terminal_json", "TEXT", 0, 0), ("hold_reason", "TEXT", 0, 0),
]
_SELECT = """SELECT attempt_id, task_id, phase, generation, state,
    CASE WHEN typeof(binding_json) = 'text'
         AND length(CAST(binding_json AS BLOB)) <= 65536
         THEN binding_json ELSE NULL END AS binding_json,
    CASE WHEN terminal_json IS NULL THEN NULL
         WHEN typeof(terminal_json) = 'text'
         AND length(CAST(terminal_json AS BLOB)) <= 65536
         THEN terminal_json ELSE '' END AS terminal_json,
    CASE WHEN hold_reason IS NULL THEN NULL
         WHEN typeof(hold_reason) = 'text'
         AND length(CAST(hold_reason AS BLOB)) <= 1024
         THEN hold_reason ELSE '' END AS hold_reason
    FROM main.execution_attempts"""


class AttemptAuthorityError(RuntimeError):
    """The attempt contract is unavailable, invalid, or refuses this effect."""


class StaleAttempt(AttemptAuthorityError):
    """The exact attempt is no longer the current owner of its task/phase."""


def _token(value, label):
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise AttemptAuthorityError(f"attempt_{label}_invalid")
    return value


@dataclass(frozen=True)
class AttemptRef:
    task_id: str
    phase: str
    attempt_id: str
    generation: int

    def __post_init__(self):
        _token(self.task_id, "task_id")
        _token(self.phase, "phase")
        _token(self.attempt_id, "id")
        if type(self.generation) is not int or not 0 < self.generation <= 2**63 - 1:
            raise AttemptAuthorityError("attempt_generation_invalid")


def _json(value):
    if type(value) is not dict:
        raise AttemptAuthorityError("attempt_json_requires_object")
    nodes = 0
    string_bytes = 0
    active = set()

    def visit(item, depth):
        nonlocal nodes, string_bytes
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise AttemptAuthorityError("attempt_json_complexity_limit")
        kind = type(item)
        if kind in (dict, list):
            if id(item) in active:
                raise AttemptAuthorityError("attempt_json_recursive")
            active.add(id(item))
            if kind is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise AttemptAuthorityError("attempt_json_key_invalid")
                    visit(key, depth + 1)
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
            active.remove(id(item))
        elif kind not in (str, int, float, bool, type(None)):
            raise AttemptAuthorityError("attempt_json_value_invalid")
        elif kind is str:
            try:
                if len(item) > MAX_JSON_BYTES:
                    raise AttemptAuthorityError("attempt_json_byte_limit")
                string_bytes += len(item.encode("utf-8"))
            except UnicodeError as exc:
                raise AttemptAuthorityError("attempt_json_encoding_invalid") from exc
            if string_bytes > MAX_JSON_BYTES:
                raise AttemptAuthorityError("attempt_json_byte_limit")
        elif kind is float and not math.isfinite(item):
            raise AttemptAuthorityError("attempt_json_nonfinite")

    visit(value, 0)
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
            raise AttemptAuthorityError("attempt_json_byte_limit")
    except (ValueError, UnicodeError, OverflowError) as exc:
        raise AttemptAuthorityError("attempt_json_encoding_invalid") from exc
    return encoded


def _decode(raw):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AttemptAuthorityError("attempt_json_duplicate_key")
            result[key] = value
        return result

    try:
        if not isinstance(raw, str):
            raise ValueError("missing or oversized JSON")
        value = json.loads(raw, object_pairs_hook=object_pairs)
        if _json(value) != raw:
            raise ValueError("noncanonical JSON")
        return value
    except (ValueError, TypeError, RecursionError) as exc:
        raise AttemptAuthorityError("attempt_stored_json_invalid") from exc


def _write_transaction(conn):
    if not conn.in_transaction:
        raise AttemptAuthorityError("attempt_caller_transaction_required")


def _schema(conn):
    row = conn.execute(
        "SELECT type, sql, name FROM main.sqlite_master WHERE name = ? COLLATE NOCASE",
        ("execution_attempts",),
    ).fetchone()
    if row is None:
        return False
    normalize = lambda sql: " ".join(str(sql).strip().rstrip(";").split())
    if (row[2] != "execution_attempts" or row[0] != "table"
            or normalize(row[1]) != normalize(_SQL)):
        raise AttemptAuthorityError("attempt_schema_incompatible")
    columns = conn.execute("PRAGMA main.table_xinfo(execution_attempts)").fetchall()
    actual = [(r[1], r[2], r[3], r[5]) for r in columns]
    if (actual != _COLUMNS or any(r[4] is not None or r[6] != 0 for r in columns)):
        raise AttemptAuthorityError("attempt_schema_columns_invalid")
    indexes = conn.execute("PRAGMA main.index_list(execution_attempts)").fetchall()
    identities = []
    for index in indexes:
        if not index[2]:
            continue
        # Names are SQLite metadata, never interpolated as SQL identifiers.
        keys = conn.execute(
            "SELECT name, coll, desc FROM pragma_index_xinfo(?, 'main') "
            "WHERE key = 1 ORDER BY seqno", (index[1],),
        ).fetchall()
        if index[4] or any(r[1] != "BINARY" or r[2] != 0 for r in keys):
            raise AttemptAuthorityError("attempt_schema_identity_invalid")
        identities.append((index[3], tuple(r[0] for r in keys)))
    if sorted(identities) != sorted([
            ("pk", ("attempt_id",)), ("u", ("task_id", "phase", "generation"))]):
        raise AttemptAuthorityError("attempt_schema_identity_invalid")
    return True


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create only in an explicit caller transaction; reject incompatible tables."""
    _write_transaction(conn)
    if not _schema(conn):
        conn.execute(_SQL)
    if not _schema(conn):
        raise AttemptAuthorityError("attempt_schema_creation_failed")


def _row(raw):
    if raw is None:
        return None
    attempt, task, phase, generation, state, binding, terminal, reason = raw
    AttemptRef(task, phase, attempt, generation)
    if state not in _STATES:
        raise AttemptAuthorityError("attempt_state_invalid")
    decoded_binding = _decode(binding)
    decoded_terminal = None if terminal is None else _decode(terminal)
    if (state in _CLOSED) != (decoded_terminal is not None):
        raise AttemptAuthorityError("attempt_terminal_state_conflict")
    if ((state == "IN_DOUBT") != (reason is not None)
            or (reason is not None and (not isinstance(reason, str) or not reason))):
        raise AttemptAuthorityError("attempt_hold_state_conflict")
    return {"attempt_id": attempt, "task_id": task, "phase": phase,
            "generation": generation, "state": state,
            "binding": decoded_binding, "terminal": decoded_terminal,
            "hold_reason": reason}


def current_attempt(conn, task_id: str, phase: str):
    """Read an independent value snapshot; a missing table remains missing."""
    _token(task_id, "task_id")
    _token(phase, "phase")
    if not _schema(conn):
        return None
    row = _row(conn.execute(
        _SELECT + " WHERE task_id = ? COLLATE BINARY AND phase = ? COLLATE BINARY"
        " ORDER BY generation DESC LIMIT 1", (task_id, phase),
    ).fetchone())
    if row is not None:
        count, invalid_prior = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN generation < ? AND state NOT IN "
            "('TERMINAL','NOT_STARTED') THEN 1 ELSE 0 END) "
            "FROM main.execution_attempts WHERE task_id = ? COLLATE BINARY "
            "AND phase = ? COLLATE BINARY", (row["generation"], task_id, phase),
        ).fetchone()
        if count != row["generation"] or invalid_prior:
            raise AttemptAuthorityError("attempt_history_inconsistent")
    return row


def require_current(conn, ref: AttemptRef, states=("LAUNCHING", "RUNNING")):
    if not isinstance(ref, AttemptRef):
        raise AttemptAuthorityError("attempt_reference_invalid")
    if (not isinstance(states, (tuple, list)) or not states
            or any(state not in _STATES for state in states)):
        raise AttemptAuthorityError("attempt_expected_states_invalid")
    row = current_attempt(conn, ref.task_id, ref.phase)
    if (row is None or row["attempt_id"] != ref.attempt_id
            or row["generation"] != ref.generation):
        raise StaleAttempt("attempt_not_current")
    if row["state"] not in states:
        raise AttemptAuthorityError("attempt_state_not_authorized")
    return row


def create_attempt(conn, task_id: str, phase: str, attempt_id: str, binding: dict):
    _write_transaction(conn)
    _token(task_id, "task_id")
    _token(phase, "phase")
    _token(attempt_id, "id")
    encoded = _json(binding)
    ensure_schema(conn)
    if conn.execute(
            "SELECT 1 FROM main.execution_attempts WHERE attempt_id = ? COLLATE BINARY",
            (attempt_id,)).fetchone():
        raise AttemptAuthorityError("attempt_id_already_used")
    previous = current_attempt(conn, task_id, phase)
    if previous is not None and previous["state"] not in _CLOSED:
        raise AttemptAuthorityError("attempt_previous_not_closed")
    generation = 1 if previous is None else previous["generation"] + 1
    ref = AttemptRef(task_id, phase, attempt_id, generation)
    changed = conn.execute(
        "INSERT INTO main.execution_attempts "
        "(attempt_id,task_id,phase,generation,state,binding_json) VALUES (?,?,?,?,?,?)",
        (attempt_id, task_id, phase, generation, "LAUNCHING", encoded),
    ).rowcount
    expected = {"attempt_id": attempt_id, "task_id": task_id, "phase": phase,
                "generation": generation, "state": "LAUNCHING",
                "binding": _decode(encoded), "terminal": None, "hold_reason": None}
    if changed != 1 or require_current(conn, ref) != expected:
        raise AttemptAuthorityError("attempt_insert_not_confirmed")
    return ref


def _update(conn, ref, before, *, state, binding=None, terminal=None, reason=None):
    expected = dict(before)
    expected.update(state=state, terminal=terminal, hold_reason=reason)
    if binding is not None:
        expected["binding"] = binding
    binding_json = _json(expected["binding"])
    terminal_json = None if terminal is None else _json(terminal)
    changed = conn.execute(
        "UPDATE main.execution_attempts SET state=?,binding_json=?,terminal_json=?,hold_reason=? "
        "WHERE attempt_id=? COLLATE BINARY AND task_id=? COLLATE BINARY "
        "AND phase=? COLLATE BINARY AND generation=? AND state=?",
        (state, binding_json, terminal_json, reason, ref.attempt_id,
         ref.task_id, ref.phase, ref.generation, before["state"]),
    ).rowcount
    if changed != 1 or require_current(conn, ref, states=(state,)) != expected:
        raise AttemptAuthorityError("attempt_update_not_confirmed")


def mark_running(conn, ref: AttemptRef, binding=None) -> None:
    """Record observed execution creation, even if it has since stopped."""
    _write_transaction(conn)
    before = require_current(conn, ref, states=("LAUNCHING",))
    if binding is not None:
        binding = _decode(_json(binding))
    _update(conn, ref, before, state="RUNNING", binding=binding)


def finish_attempt(conn, ref: AttemptRef, terminal: dict, not_started=False):
    _write_transaction(conn)
    if type(not_started) is not bool:
        raise AttemptAuthorityError("attempt_not_started_flag_invalid")
    terminal = _decode(_json(terminal))
    try:
        before = require_current(conn, ref, states=tuple(_STATES))
    except StaleAttempt:
        return "stale"
    target = "NOT_STARTED" if not_started else "TERMINAL"
    if before["state"] in _CLOSED:
        if before["state"] == target and before["terminal"] == terminal:
            return "duplicate"
        raise AttemptAuthorityError("attempt_terminal_conflict")
    source = "LAUNCHING" if not_started else "RUNNING"
    if before["state"] != source:
        raise AttemptAuthorityError("attempt_finish_not_authorized")
    _update(conn, ref, before, state=target, terminal=terminal)
    return "committed"


def hold_attempt(conn, ref: AttemptRef, reason: str) -> None:
    _write_transaction(conn)
    try:
        if (not isinstance(reason, str) or not reason.strip()
                or len(reason.encode("utf-8")) > 1024):
            raise ValueError("invalid reason")
    except (ValueError, UnicodeError) as exc:
        raise AttemptAuthorityError("attempt_hold_reason_invalid") from exc
    before = require_current(conn, ref, states=("LAUNCHING", "RUNNING", "IN_DOUBT"))
    if before["state"] == "IN_DOUBT":
        if before["hold_reason"] != reason:
            raise AttemptAuthorityError("attempt_hold_reason_conflict")
        return
    _update(conn, ref, before, state="IN_DOUBT", reason=reason)
