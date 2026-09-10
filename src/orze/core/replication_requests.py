"""Immutable explicit requests mapping one source occurrence to one new task.

CALLING SPEC: get_request / request_for_task are bounded historical reads with
no DDL or current-execution permission. insert_request requires a caller-owned
write transaction, never starts/commits/rolls it back. The coordinator verifies
the adapter-specific source and creates the task in that same transaction.
Source phase is generic; request IDs are idempotency keys, not capabilities.
No configurations, artifact bytes, hardware or scientific verdicts live here.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from orze.core.execution_attempts import AttemptRef, _json, require_current

MAX_RECORD_BYTES = 16384
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {
    "schema", "request_id", "task_id", "source_ref", "scope", "database",
    "source_config_sha256", "source_file_sha256", "source_row_sha256",
    "source_terminal_sha256", "artifact_records_sha256", "execution_identity",
    "spec_fingerprint", "artifact_binding_sha256", "reason", "created_at",
    "request_sha256",
}
_SQL = """CREATE TABLE replication_requests (
    request_id TEXT NOT NULL COLLATE BINARY PRIMARY KEY,
    task_id TEXT NOT NULL COLLATE BINARY UNIQUE,
    record_json TEXT NOT NULL
)"""
_SELECT = """SELECT request_id, task_id,
    CASE WHEN typeof(record_json)='text'
         AND length(CAST(record_json AS BLOB)) <= 16384
         THEN record_json ELSE NULL END FROM main.replication_requests"""


class ReplicationError(ValueError):
    """Rejected/unavailable replication request; no fallback authorization."""


def token(value):
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ReplicationError("replication_token_invalid")
    return value


def canonical(value):
    try:
        return _json(value)
    except (ValueError, TypeError, RuntimeError, RecursionError) as exc:
        raise ReplicationError("replication_metadata_invalid") from exc


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def seal_record(value):
    value = dict(value)
    value["request_sha256"] = digest({k: v for k, v in value.items() if k != "request_sha256"})
    return validate_record(value)


def validate_record(value):
    if (type(value) is not dict or set(value) != _FIELDS
            or type(value["schema"]) is not int or value["schema"] != 1):
        raise ReplicationError("replication_record_invalid")
    token(value["request_id"])
    token(value["task_id"])
    ref = value["source_ref"]
    if (type(ref) is not dict or set(ref) != {"task_id", "phase", "attempt_id", "generation"}
            or type(ref["generation"]) is not int or not 0 < ref["generation"] < 2**63):
        raise ReplicationError("replication_source_ref_invalid")
    for key in ("task_id", "phase", "attempt_id"):
        token(ref[key])
    if ref["task_id"] == value["task_id"]:
        raise ReplicationError("replication_new_task_required")
    for key in ("scope", "database"):
        path = value[key]
        if (type(path) is not str or not path or len(path.encode("utf-8")) > 4096
                or not Path(path).is_absolute() or str(Path(path)) != path
                or ".." in Path(path).parts or any(ord(c) < 32 for c in path)):
            raise ReplicationError("replication_scope_invalid")
    for key in _FIELDS - {"schema", "request_id", "task_id", "source_ref", "scope",
                          "database", "reason", "created_at"}:
        if type(value[key]) is not str or _SHA.fullmatch(value[key]) is None:
            raise ReplicationError("replication_digest_invalid")
    for key, limit in (("reason", 1024), ("created_at", 128)):
        if (type(value[key]) is not str or not value[key].strip()
                or len(value[key].encode("utf-8")) > limit):
            raise ReplicationError("replication_description_invalid")
    if value["request_sha256"] != digest({k: v for k, v in value.items() if k != "request_sha256"}):
        raise ReplicationError("replication_record_digest_mismatch")
    raw = canonical(value)
    if len(raw.encode("utf-8")) > MAX_RECORD_BYTES:
        raise ReplicationError("replication_record_limit")
    return json.loads(raw)


def _schema(conn):
    rows = conn.execute(
        "SELECT type,sql,name FROM main.sqlite_master WHERE name=? COLLATE NOCASE",
        ("replication_requests",),
    ).fetchall()
    if not rows:
        return False
    normalize = lambda value: " ".join(str(value).strip().rstrip(";").split())
    if (len(rows) != 1 or rows[0][0] != "table" or rows[0][2] != "replication_requests"
            or normalize(rows[0][1]) != normalize(_SQL)):
        raise ReplicationError("replication_schema_incompatible")
    columns = conn.execute("PRAGMA main.table_xinfo(replication_requests)").fetchall()
    if [(r[1], r[2], r[3], r[4], r[5], r[6]) for r in columns] != [
            ("request_id", "TEXT", 1, None, 1, 0),
            ("task_id", "TEXT", 1, None, 0, 0),
            ("record_json", "TEXT", 1, None, 0, 0)]:
        raise ReplicationError("replication_schema_columns_invalid")
    identities = []
    for index in conn.execute("PRAGMA main.index_list(replication_requests)"):
        if not index[2]:
            continue
        keys = conn.execute(
            "SELECT name,coll,desc FROM pragma_index_xinfo(?, 'main') WHERE key=1 ORDER BY seqno",
            (index[1],),
        ).fetchall()
        if index[4] or any(row[1] != "BINARY" or row[2] for row in keys):
            raise ReplicationError("replication_schema_identity_invalid")
        identities.append((index[3], tuple(row[0] for row in keys)))
    if sorted(identities) != sorted([("pk", ("request_id",)), ("u", ("task_id",))]):
        raise ReplicationError("replication_schema_identity_invalid")
    return True


def ensure_schema(conn):
    if not conn.in_transaction:
        raise ReplicationError("replication_caller_transaction_required")
    if not _schema(conn):
        conn.execute(_SQL.replace("CREATE TABLE ", "CREATE TABLE main.", 1))
        if not _schema(conn):
            raise ReplicationError("replication_schema_not_created")


def _lookup(conn, field, value):
    token(value)
    if not _schema(conn):
        return None
    rows = conn.execute(_SELECT + f" WHERE {field}=? COLLATE BINARY LIMIT 2", (value,)).fetchall()
    if not rows:
        return None
    if len(rows) != 1 or type(rows[0][2]) is not str:
        raise ReplicationError("replication_record_unavailable")
    try:
        record = validate_record(json.loads(rows[0][2]))
        if canonical(record) != rows[0][2] or tuple(rows[0][:2]) != (record["request_id"], record["task_id"]):
            raise ValueError
        return record
    except (ValueError, TypeError, RecursionError) as exc:
        raise ReplicationError("replication_stored_record_invalid") from exc


def get_request(conn, request_id):
    return _lookup(conn, "request_id", request_id)


def request_for_task(conn, task_id):
    return _lookup(conn, "task_id", task_id)


def verify_source_row(conn, record):
    record = validate_record(record)
    row = require_current(conn, AttemptRef(**record["source_ref"]), states=("TERMINAL",))
    if digest(row) != record["source_row_sha256"] or digest(row["terminal"]) != record["source_terminal_sha256"]:
        raise ReplicationError("replication_source_changed")
    return row


def insert_request(conn, record):
    """Insert-only; caller must roll back the entire transaction on any error."""
    if not conn.in_transaction:
        raise ReplicationError("replication_caller_transaction_required")
    record = validate_record(record)
    verify_source_row(conn, record)
    ensure_schema(conn)
    cursor = conn.execute(
        "INSERT INTO main.replication_requests(request_id,task_id,record_json) VALUES(?,?,?)",
        (record["request_id"], record["task_id"], canonical(record)),
    )
    if cursor.rowcount != 1 or canonical(get_request(conn, record["request_id"])) != canonical(record):
        raise ReplicationError("replication_insert_not_applied")
    verify_source_row(conn, record)
