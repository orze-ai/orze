"""Storage primitives for fenced trigger delivery (no subprocess operations).

CALLING SPEC: connect(path, write=False, create=False) returns a short-lived
connection; schema_status/ensure_schema inspect or initialize only this module's
tables. All mutations belong to the caller's BEGIN IMMEDIATE transaction.
Legacy consumption rows are never inferred to be runnable deliveries.
"""
from __future__ import annotations

import math
import sqlite3
import stat
import time
from pathlib import Path

from orze.core.sqlite_policy import (
    apply_shared_database_policy, inspect_shared_database_policy,
)


class TriggerDeliveryError(ValueError):
    """Closed delivery-contract error; never includes a trigger payload."""


STATES = ("PENDING", "LEASED", "LAUNCHING", "STARTED", "TERMINAL", "IN_DOUBT")
_DDL = (
    """CREATE TABLE trigger_delivery_schema (version INTEGER PRIMARY KEY)""",
    """CREATE TABLE trigger_deliveries (
        delivery_id TEXT PRIMARY KEY, legacy_consumption_id INTEGER NOT NULL UNIQUE,
        scope TEXT NOT NULL, role_name TEXT NOT NULL, source_key TEXT NOT NULL,
        file_path TEXT NOT NULL, fingerprint TEXT NOT NULL,
        payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN
            ('PENDING','LEASED','LAUNCHING','STARTED','TERMINAL','IN_DOUBT')),
        generation INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0),
        owner TEXT, lease_until REAL, attempt_id TEXT, process_pid INTEGER,
        created_at REAL NOT NULL, updated_at REAL NOT NULL,
        UNIQUE(scope, role_name, source_key))""",
    """CREATE TABLE trigger_delivery_attempts (
        attempt_id TEXT PRIMARY KEY, delivery_id TEXT NOT NULL,
        generation INTEGER NOT NULL, owner TEXT NOT NULL,
        nonce_sha256 TEXT NOT NULL, command_sha256 TEXT NOT NULL,
        created_at REAL NOT NULL, UNIQUE(delivery_id, generation))""",
    """CREATE TABLE trigger_delivery_transitions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, delivery_id TEXT NOT NULL,
        generation INTEGER NOT NULL, attempt_id TEXT, owner TEXT,
        from_state TEXT, to_state TEXT NOT NULL, reason TEXT NOT NULL,
        outcome TEXT, exit_code INTEGER, cleanup_verified INTEGER,
        created_at REAL NOT NULL)""",
    """CREATE INDEX trigger_delivery_queue
        ON trigger_deliveries(scope, role_name, state, created_at, delivery_id)""",
    """CREATE INDEX trigger_delivery_history
        ON trigger_delivery_transitions(delivery_id, id)""",
)
_LEGACY_DDL = """CREATE TABLE trigger_consumptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, role_name TEXT NOT NULL,
    file_path TEXT NOT NULL, fingerprint TEXT NOT NULL, payload TEXT,
    consumed_at TEXT NOT NULL, consumed_by_host TEXT, consumed_by_pid INTEGER,
    UNIQUE(role_name, fingerprint))"""
_COLUMNS = {
    "trigger_delivery_schema": {"version"},
    "trigger_deliveries": {
        "delivery_id", "legacy_consumption_id", "scope", "role_name", "source_key",
        "file_path", "fingerprint", "payload", "payload_sha256", "state",
        "generation", "owner", "lease_until", "attempt_id", "process_pid",
        "created_at", "updated_at",
    },
    "trigger_delivery_attempts": {
        "attempt_id", "delivery_id", "generation", "owner", "nonce_sha256",
        "command_sha256", "created_at",
    },
    "trigger_delivery_transitions": {
        "id", "delivery_id", "generation", "attempt_id", "owner", "from_state",
        "to_state", "reason", "outcome", "exit_code", "cleanup_verified", "created_at",
    },
    "trigger_consumptions": {
        "id", "role_name", "file_path", "fingerprint", "payload", "consumed_at",
        "consumed_by_host", "consumed_by_pid",
    },
}
_PK = {
    "trigger_delivery_schema": ("version",),
    "trigger_deliveries": ("delivery_id",),
    "trigger_delivery_attempts": ("attempt_id",),
    "trigger_delivery_transitions": ("id",),
    "trigger_consumptions": ("id",),
}


def timestamp(now=None):
    value = time.time() if now is None else now
    if (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
        raise TriggerDeliveryError("trigger_time_invalid")
    return float(value)


def bounded_text(value, field, limit=4096):
    if (not isinstance(value, str) or not value or "\x00" in value
            or len(value.encode("utf-8")) > limit):
        raise TriggerDeliveryError("trigger_" + field + "_invalid")
    return value


def digest(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise TriggerDeliveryError("trigger_digest_invalid")
    return value


def connect(db_path, *, write=False, create=False):
    path = Path(db_path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise TriggerDeliveryError("trigger_database_redirected")
    if path.exists():
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise TriggerDeliveryError("trigger_database_not_regular")
    elif not create:
        raise TriggerDeliveryError("trigger_database_missing")
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            if current.is_symlink():
                raise TriggerDeliveryError("trigger_database_redirected")
    mode = "rwc" if create else "rw" if write else "ro"
    conn = sqlite3.connect(path.as_uri() + "?mode=" + mode, uri=True, timeout=5)
    try:
        # Only a newly created database may have its journal policy initialized.
        # Existing DBs are never silently migrated from WAL here.
        if create and conn.execute("PRAGMA page_count").fetchone()[0] == 0:
            apply_shared_database_policy(conn)
        elif not inspect_shared_database_policy(conn)["compliant"]:
            raise TriggerDeliveryError("trigger_database_policy_invalid")
        if not write:
            conn.execute("PRAGMA query_only=ON")
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        conn.close()
        raise


def _table_valid(conn, name):
    entry = conn.execute("SELECT type FROM sqlite_master WHERE name=?", (name,)).fetchone()
    if entry is None:
        return False
    columns = conn.execute("PRAGMA table_info(" + name + ")").fetchall()
    pk = tuple(row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5])
    if entry[0] != "table" or not _COLUMNS[name].issubset({row[1] for row in columns}) or pk != _PK[name]:
        raise TriggerDeliveryError("trigger_database_schema_invalid")
    return True


def _unique(conn, table, wanted):
    for index in conn.execute("PRAGMA index_list(" + table + ")"):
        if index[2] and not index[4]:
            # Index names are database data; quote rather than interpolate raw.
            name = str(index[1]).replace('"', '""')
            columns = tuple(row[2] for row in conn.execute('PRAGMA index_info("' + name + '")'))
            if columns == wanted:
                return
    raise TriggerDeliveryError("trigger_database_schema_invalid")


def schema_status(conn):
    present = [_table_valid(conn, name) for name in _COLUMNS if name != "trigger_consumptions"]
    if not any(present):
        if _table_valid(conn, "trigger_consumptions"):
            _unique(conn, "trigger_consumptions", ("role_name", "fingerprint"))
        return False
    if not all(present) or not _table_valid(conn, "trigger_consumptions"):
        raise TriggerDeliveryError("trigger_database_schema_invalid")
    if [row[0] for row in conn.execute("SELECT version FROM trigger_delivery_schema LIMIT 2")] != [1]:
        raise TriggerDeliveryError("trigger_database_schema_invalid")
    for table, keys in (
        ("trigger_consumptions", ("role_name", "fingerprint")),
        ("trigger_deliveries", ("legacy_consumption_id",)),
        ("trigger_deliveries", ("scope", "role_name", "source_key")),
        ("trigger_delivery_attempts", ("delivery_id", "generation")),
    ):
        _unique(conn, table, keys)
    return True


def ensure_schema(conn):
    if schema_status(conn):
        return
    if not _table_valid(conn, "trigger_consumptions"):
        conn.execute(_LEGACY_DDL)
    for statement in _DDL:
        conn.execute(statement)
    cursor = conn.execute("INSERT INTO trigger_delivery_schema(version) VALUES (1)")
    if cursor.rowcount != 1 or not schema_status(conn):
        raise TriggerDeliveryError("trigger_schema_write_rejected")


def event(conn, row, from_state, reason, now, *, outcome=None, exit_code=None, cleanup_verified=None):
    values = (row["delivery_id"], row["generation"], row["attempt_id"], row["owner"],
              from_state, row["state"], reason, outcome, exit_code, cleanup_verified, now)
    cursor = conn.execute(
        "INSERT INTO trigger_delivery_transitions (delivery_id,generation,attempt_id,owner,"
        "from_state,to_state,reason,outcome,exit_code,cleanup_verified,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)", values,
    )
    saved = conn.execute(
        "SELECT delivery_id,generation,attempt_id,owner,from_state,to_state,reason,outcome,"
        "exit_code,cleanup_verified,created_at FROM trigger_delivery_transitions WHERE id=?",
        (cursor.lastrowid,),
    ).fetchone()
    if cursor.rowcount != 1 or saved is None or tuple(saved) != values:
        raise TriggerDeliveryError("trigger_transition_write_rejected")


def write_row(conn, old, changes):
    columns = tuple(changes)
    cursor = conn.execute(
        "UPDATE trigger_deliveries SET " + ",".join(key + "=?" for key in columns)
        + " WHERE delivery_id=? AND state=? AND generation=? AND owner IS ? AND attempt_id IS ?",
        (*changes.values(), old["delivery_id"], old["state"], old["generation"], old["owner"], old["attempt_id"]),
    )
    saved = conn.execute("SELECT * FROM trigger_deliveries WHERE delivery_id=?", (old["delivery_id"],)).fetchone()
    expected = dict(old)
    expected.update(changes)
    if cursor.rowcount != 1 or saved is None or dict(saved) != expected:
        raise TriggerDeliveryError("trigger_state_write_rejected")
    return dict(saved)
