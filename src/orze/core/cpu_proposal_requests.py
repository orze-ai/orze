"""Bounded, historical records for normal CPU Policy proposals.

seal_record/validate_record only validate detached metadata. get_request and
records_view are read-only and create no schema. insert_request requires the
caller's transaction; it performs an insert and exact readback without commit,
rollback, Domain execution, source capture or task admission. The coordinator
must atomically create/check normal proposal outcome and its durable record.

The main-database table is new and exact, never migrated. Its primary key is
(scope, request_id); task_id is deliberately not unique because normal dedup
can relate multiple decisions to one task. Records are capped at 64 KiB and
embed the bounded B source snapshot, not a serialized live source capability.
Historical reads neither require the target to remain QUEUED nor authorize GO.
A compact at-most-32 record view advertises omitted rows, not a pagination or
execution capability. No second database, process runner or scientific verdict.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from orze.core.execution_attempts import _json
from orze.core.research_artifacts import _record as _artifact_record

MAX_RECORD_BYTES = 65536
MAX_VIEW_RECORDS = 32
_FIELDS = {"schema", "request_id", "task_id", "scope", "database", "decision",
           "config_sha256", "domain", "source_snapshot", "outcome",
           "admission_evidence", "created_at", "record_sha256"}
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REASONS = {"inserted": "proposal_admitted",
            "already_present_exact": "proposal_exact_replay",
            "config_duplicate": "proposal_config_duplicate",
            "conflict": "proposal_identity_conflict"}
_SQL = """CREATE TABLE cpu_proposal_requests (
    scope TEXT NOT NULL COLLATE BINARY,
    request_id TEXT NOT NULL COLLATE BINARY,
    task_id TEXT NOT NULL COLLATE BINARY,
    record_json TEXT NOT NULL,
    PRIMARY KEY (scope, request_id)
)"""
_SELECT = """SELECT scope,
    CASE WHEN typeof(request_id)='text' AND length(CAST(request_id AS BLOB))<=128
         THEN request_id ELSE NULL END,
    CASE WHEN typeof(task_id)='text' AND length(CAST(task_id AS BLOB))<=128
         THEN task_id ELSE NULL END,
    CASE WHEN typeof(record_json)='text' AND length(CAST(record_json AS BLOB))<=65536
         THEN record_json ELSE NULL END FROM main.cpu_proposal_requests"""


class ProposalRequestError(ValueError):
    """Invalid/unconfirmed metadata; no fallback admission authority."""


def _fail(reason):
    raise ProposalRequestError("cpu_proposal_request_" + reason)


def token(value):
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        _fail("token_invalid")
    return value


def _sha(value):
    if type(value) is not str or _SHA.fullmatch(value) is None:
        _fail("digest_invalid")


def _path(value):
    if (type(value) is not str or not value or len(value.encode()) > 4096
            or not Path(value).is_absolute() or str(Path(value)) != value
            or ".." in Path(value).parts or any(ord(c) < 32 for c in value)):
        _fail("scope_invalid")
    return value


def canonical(value):
    try:
        return _json(value)
    except (ValueError, TypeError, RuntimeError, RecursionError, UnicodeError, OverflowError) as exc:
        raise ProposalRequestError("cpu_proposal_request_metadata_invalid") from exc


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _sources(value, record, ids):
    if (type(value) is not dict or set(value) != {
            "schema", "scope", "database", "database_identity", "scope_identity", "inputs"}
            or type(value["schema"]) is not int or value["schema"] != 1
            or value["scope"] != record["scope"] or value["database"] != record["database"]
            or type(value["inputs"]) is not list or len(value["inputs"]) > 32):
        _fail("source_snapshot_invalid")
    for key in ("database_identity", "scope_identity"):
        witness = value[key]
        if (type(witness) is not list or len(witness) != 2
                or any(type(n) is not int or n < 0 for n in witness) or witness[1] == 0):
            _fail("source_snapshot_identity_invalid")
    actual_ids, size = [], 0
    for entry in value["inputs"]:
        if type(entry) is not dict or set(entry) != {"artifact", "source_sha256", "effect_sha256"}:
            _fail("source_entry_invalid")
        artifact = json.loads(_artifact_record(entry["artifact"]))
        if artifact["scope"] != record["scope"] or artifact["producer"]["phase"] != "action":
            _fail("source_scope_invalid")
        _sha(entry["source_sha256"])
        _sha(entry["effect_sha256"])
        actual_ids.append(artifact["artifact_id"])
        size += artifact["size_bytes"]
    if actual_ids != ids or size > 16 * 1024 * 1024 or len(canonical(value).encode()) > 32768:
        _fail("source_snapshot_mismatch_or_limit")


def _outcome(value, record):
    if (type(value) is not dict or set(value) != {
            "request_id", "task_id", "status", "reason", "existing_id"}
            or value["request_id"] != record["request_id"]
            or value["task_id"] != record["task_id"]
            or type(value["status"]) is not str or value["status"] not in _REASONS
            or value["reason"] != _REASONS[value["status"]]):
        _fail("outcome_invalid")
    if value["status"] == "config_duplicate":
        token(value["existing_id"])
        if value["existing_id"] == record["task_id"]:
            _fail("duplicate_identity_invalid")
    elif value["existing_id"] is not None:
        _fail("outcome_invalid")
    evidence = record["admission_evidence"]
    if type(evidence) is not dict or set(evidence) != {
            "task_id", "source_sha256", "transition_id", "transition_sha256"}:
        _fail("admission_evidence_invalid")
    target = value["existing_id"] if value["status"] == "config_duplicate" else record["task_id"]
    if evidence["task_id"] != target:
        _fail("admission_target_invalid")
    _sha(evidence["source_sha256"])
    if value["status"] == "inserted":
        transition = evidence["transition_id"]
        if type(transition) is not int or not 0 < transition < 2**63:
            _fail("admission_transition_invalid")
        _sha(evidence["transition_sha256"])
    elif evidence["transition_id"] is not None or evidence["transition_sha256"] is not None:
        _fail("admission_transition_invalid")


def seal_record(value):
    if type(value) is not dict:
        _fail("record_invalid")
    record = dict(value)
    record["record_sha256"] = digest({k: v for k, v in record.items() if k != "record_sha256"})
    return validate_record(record)


def validate_record(value):
    try:
        if (type(value) is not dict or set(value) != _FIELDS
                or type(value["schema"]) is not int or value["schema"] != 1):
            _fail("record_invalid")
        token(value["request_id"])
        token(value["task_id"])
        _path(value["scope"])
        _path(value["database"])
        from orze.core.research_interfaces import validate_proposal_decision
        decision = validate_proposal_decision(value["decision"])
        if (decision["request_id"] != value["request_id"] or decision["task_id"] != value["task_id"]
                or canonical(decision) != canonical(value["decision"])):
            _fail("decision_identity_invalid")
        _sha(value["config_sha256"])
        domain = value["domain"]
        if (type(domain) is not dict or set(domain) != {"declaration", "implementation_id"}
                or type(domain["declaration"]) is not dict
                or set(domain["declaration"]) != {"version", "kind", "config"}
                or type(domain["declaration"]["version"]) is not int
                or domain["declaration"]["version"] != 1
                or type(domain["declaration"]["config"]) is not dict):
            _fail("domain_invalid")
        token(domain["implementation_id"])
        token(domain["declaration"]["kind"])
        _sources(value["source_snapshot"], value, decision["domain_request"]["input_artifact_ids"])
        _outcome(value["outcome"], value)
        if (type(value["created_at"]) is not str or not value["created_at"].strip()
                or len(value["created_at"].encode()) > 128):
            _fail("created_at_invalid")
        _sha(value["record_sha256"])
        if value["record_sha256"] != digest({k: v for k, v in value.items() if k != "record_sha256"}):
            _fail("record_digest_mismatch")
        raw = canonical(value)
        if len(raw.encode()) > MAX_RECORD_BYTES:
            _fail("record_limit")
        return json.loads(raw)
    except ProposalRequestError:
        raise
    except (ValueError, TypeError, RuntimeError, KeyError, UnicodeError, OverflowError) as exc:
        raise ProposalRequestError("cpu_proposal_request_record_invalid") from exc


def _schema(conn):
    rows = conn.execute(
        "SELECT type,sql,name FROM main.sqlite_master WHERE name=? COLLATE NOCASE",
        ("cpu_proposal_requests",)).fetchall()
    if not rows:
        return False
    normalize = lambda value: " ".join(str(value).strip().rstrip(";").split())
    if (len(rows) != 1 or rows[0][0] != "table" or rows[0][2] != "cpu_proposal_requests"
            or normalize(rows[0][1]) != normalize(_SQL)):
        _fail("schema_incompatible")
    columns = conn.execute("PRAGMA main.table_xinfo(cpu_proposal_requests)").fetchall()
    if [(r[1], r[2], r[3], r[4], r[5], r[6]) for r in columns] != [
            ("scope", "TEXT", 1, None, 1, 0), ("request_id", "TEXT", 1, None, 2, 0),
            ("task_id", "TEXT", 1, None, 0, 0), ("record_json", "TEXT", 1, None, 0, 0)]:
        _fail("schema_columns_invalid")
    identities = []
    for index in conn.execute("PRAGMA main.index_list(cpu_proposal_requests)"):
        if not index[2]:
            continue
        keys = conn.execute(
            "SELECT name,coll,desc FROM pragma_index_xinfo(?, 'main') WHERE key=1 ORDER BY seqno",
            (index[1],)).fetchall()
        if index[4] or any(row[1] != "BINARY" or row[2] for row in keys):
            _fail("schema_identity_invalid")
        identities.append((index[3], tuple(row[0] for row in keys)))
    if identities != [("pk", ("scope", "request_id"))]:
        _fail("schema_identity_invalid")
    return True


def ensure_schema(conn):
    if not conn.in_transaction:
        _fail("caller_transaction_required")
    if not _schema(conn):
        conn.execute(_SQL.replace("CREATE TABLE ", "CREATE TABLE main.", 1))
        if not _schema(conn):
            _fail("schema_not_created")


def _decode(row):
    if len(row) != 4 or any(type(v) is not str for v in row):
        _fail("stored_record_unavailable")
    try:
        record = validate_record(json.loads(row[3]))
        if (canonical(record) != row[3]
                or tuple(row[:3]) != (record["scope"], record["request_id"], record["task_id"])):
            _fail("stored_record_mismatch")
        return record
    except (ValueError, TypeError, RecursionError) as exc:
        raise ProposalRequestError("cpu_proposal_request_stored_record_invalid") from exc


def get_request(conn, scope, request_id):
    _path(scope)
    token(request_id)
    if not _schema(conn):
        return None
    rows = conn.execute(_SELECT + " WHERE scope=? COLLATE BINARY AND request_id=? COLLATE BINARY LIMIT 2",
                        (scope, request_id)).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        _fail("record_identity_invalid")
    return _decode(rows[0])


def insert_request(conn, record):
    """Insert only; caller must roll back its entire transaction on any error."""
    if not conn.in_transaction:
        _fail("caller_transaction_required")
    record = validate_record(record)
    ensure_schema(conn)
    cursor = conn.execute(
        "INSERT INTO main.cpu_proposal_requests(scope,request_id,task_id,record_json) VALUES(?,?,?,?)",
        (record["scope"], record["request_id"], record["task_id"], canonical(record)))
    if (cursor.rowcount != 1 or canonical(get_request(conn, record["scope"], record["request_id"]))
            != canonical(record)):
        _fail("insert_not_applied")


def records_view(conn, scope, *, limit=MAX_VIEW_RECORDS):
    """Compact historical metadata, without source revalidation or admission."""
    _path(scope)
    if type(limit) is not int or not 1 <= limit <= MAX_VIEW_RECORDS:
        _fail("view_limit_invalid")
    if not _schema(conn):
        return {"requests": [], "more_available": False}
    rows = conn.execute(_SELECT + " WHERE scope=? COLLATE BINARY ORDER BY request_id COLLATE BINARY LIMIT ?",
                        (scope, limit + 1)).fetchall()
    records = [_decode(row) for row in rows[:limit]]
    return json.loads(canonical({"requests": [
        {key: record[key] for key in ("request_id", "task_id", "outcome", "record_sha256")}
        for record in records], "more_available": len(rows) > limit}))
