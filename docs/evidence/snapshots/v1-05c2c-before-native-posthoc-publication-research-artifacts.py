"""Insert-only artifact occurrence records in a caller-owned transaction.

CALLING SPEC:
    register_artifacts(conn, ref, records) -> tuple[artifact_id, ...]
        Requires an existing write transaction and the exact current RUNNING
        AttemptRef with an artifact_publication binding. Validates the whole
        declared output set before any INSERT. Any failure requires the caller
        to roll back its WHOLE transaction, including other adapter changes.
        This module never BEGINs, COMMITs, ROLLBACKs, or repairs old schemas.
    artifacts_for_attempt(conn, ref) -> list[dict]
    get_artifact(conn, artifact_id) -> dict | None
        Historical metadata reads: no DDL, filesystem IO, or current-owner
        requirement. Returned values are detached, not scientific qualification.

The publisher, not this SQLite module, creates and verifies independent file
snapshots before registration. An occurrence ID is not a content hash: equal
bytes in different attempts may have different IDs. API writes are insert-only;
this does not claim resistance to arbitrary direct SQL/filesystem modification.
The adapter must hold its effect lease and BEGIN IMMEDIATE. SQLite's Python API
does not expose whether an existing transaction was begun in that locking mode.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from orze.core.execution_attempts import AttemptRef, require_current


MAX_ARTIFACTS = 32
MAX_RECORD_BYTES = 16384
MAX_ARTIFACT_BYTES = 1 << 40
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {"schema", "artifact_id", "producer", "spec_fingerprint", "scope",
           "logical_name", "path", "content_sha256", "size_bytes"}
_PRODUCER_FIELDS = {"task_id", "phase", "attempt_id", "generation"}
_SQL = """CREATE TABLE research_artifacts (
    artifact_id TEXT NOT NULL COLLATE BINARY PRIMARY KEY,
    producer_task_id TEXT NOT NULL COLLATE BINARY,
    producer_phase TEXT NOT NULL COLLATE BINARY,
    producer_attempt_id TEXT NOT NULL COLLATE BINARY,
    producer_generation INTEGER NOT NULL CHECK (producer_generation > 0),
    logical_name TEXT NOT NULL COLLATE BINARY,
    record_json TEXT NOT NULL,
    UNIQUE (producer_attempt_id, logical_name)
)"""
_COLUMNS = [
    ("artifact_id", "TEXT", 1, 1), ("producer_task_id", "TEXT", 1, 0),
    ("producer_phase", "TEXT", 1, 0), ("producer_attempt_id", "TEXT", 1, 0),
    ("producer_generation", "INTEGER", 1, 0), ("logical_name", "TEXT", 1, 0),
    ("record_json", "TEXT", 1, 0),
]
_SELECT = """SELECT artifact_id, producer_task_id, producer_phase,
    producer_attempt_id, producer_generation, logical_name,
    CASE WHEN typeof(record_json)='text'
         AND length(CAST(record_json AS BLOB)) <= 16384
         THEN record_json ELSE NULL END FROM main.research_artifacts"""


class ResearchArtifactError(ValueError):
    """Invalid/unavailable artifact records; caller must roll back writes."""


def _token(value):
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ResearchArtifactError("artifact_token_invalid")
    return value


def _digest(value):
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ResearchArtifactError("artifact_digest_invalid")


def _absolute_path(value):
    try:
        if (type(value) is not str or not value or len(value) > 4096
                or len(value.encode("utf-8")) > 4096
                or any(ord(char) < 32 for char in value)):
            raise ValueError
        path = Path(value)
        if (not path.is_absolute() or str(path) != value
                or ".." in path.parts or "\\" in value):
            raise ValueError
    except (ValueError, UnicodeError) as exc:
        raise ResearchArtifactError("artifact_path_invalid") from exc
    return path


def _ref(ref):
    if type(ref) is not AttemptRef:
        raise ResearchArtifactError("artifact_producer_invalid")
    for value in (ref.task_id, ref.phase, ref.attempt_id):
        _token(value)
    if type(ref.generation) is not int or not 0 < ref.generation <= 2**63 - 1:
        raise ResearchArtifactError("artifact_producer_invalid")
    return asdict(ref)


def _record(record):
    if type(record) is not dict or set(record) != _FIELDS:
        raise ResearchArtifactError("artifact_record_fields_invalid")
    if type(record["schema"]) is not int or record["schema"] != 1:
        raise ResearchArtifactError("artifact_record_schema_invalid")
    artifact_id = _token(record["artifact_id"])
    producer = record["producer"]
    if type(producer) is not dict or set(producer) != _PRODUCER_FIELDS:
        raise ResearchArtifactError("artifact_producer_invalid")
    if type(producer["generation"]) is not int:
        raise ResearchArtifactError("artifact_producer_invalid")
    _ref(AttemptRef(**producer))
    _token(record["logical_name"])
    _digest(record["spec_fingerprint"])
    _digest(record["content_sha256"])
    _absolute_path(record["scope"])
    path = _absolute_path(record["path"])
    if path.parts[-2:] != (artifact_id, "content"):
        raise ResearchArtifactError("artifact_occurrence_path_invalid")
    size = record["size_bytes"]
    if type(size) is not int or not 0 <= size <= MAX_ARTIFACT_BYTES:
        raise ResearchArtifactError("artifact_size_invalid")
    try:
        encoded = json.dumps(record, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_RECORD_BYTES:
            raise ValueError
    except (ValueError, UnicodeError, TypeError, RecursionError) as exc:
        raise ResearchArtifactError("artifact_record_encoding_invalid") from exc
    return encoded


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ResearchArtifactError("artifact_record_duplicate_key")
            result[key] = value
        return result
    try:
        if type(raw) is not str:
            raise ValueError
        record = json.loads(raw, object_pairs_hook=pairs)
        if _record(record) != raw:
            raise ValueError
        return record
    except (ValueError, TypeError, RecursionError) as exc:
        raise ResearchArtifactError("artifact_stored_record_invalid") from exc


def _schema(conn):
    rows = conn.execute(
        "SELECT type,sql,name FROM main.sqlite_master WHERE name=? COLLATE NOCASE",
        ("research_artifacts",),
    ).fetchall()
    if not rows:
        return False
    normalize = lambda text: " ".join(str(text).strip().rstrip(";").split())
    if (len(rows) != 1 or rows[0][0] != "table" or rows[0][2] != "research_artifacts"
            or normalize(rows[0][1]) != normalize(_SQL)):
        raise ResearchArtifactError("artifact_schema_incompatible")
    columns = conn.execute("PRAGMA main.table_xinfo(research_artifacts)").fetchall()
    if ([(r[1], r[2], r[3], r[5]) for r in columns] != _COLUMNS
            or any(r[4] is not None or r[6] != 0 for r in columns)):
        raise ResearchArtifactError("artifact_schema_columns_invalid")
    identities = []
    for index in conn.execute("PRAGMA main.index_list(research_artifacts)").fetchall():
        if not index[2]:
            continue
        keys = conn.execute(
            "SELECT name,coll,desc FROM pragma_index_xinfo(?, 'main') "
            "WHERE key=1 ORDER BY seqno", (index[1],),
        ).fetchall()
        if index[4] or any(row[1] != "BINARY" or row[2] != 0 for row in keys):
            raise ResearchArtifactError("artifact_schema_identity_invalid")
        identities.append((index[3], tuple(row[0] for row in keys)))
    if sorted(identities) != sorted([
            ("pk", ("artifact_id",)), ("u", ("producer_attempt_id", "logical_name"))]):
        raise ResearchArtifactError("artifact_schema_identity_invalid")
    return True


def _row(raw):
    record = _decode(raw[6])
    producer = record["producer"]
    expected = (record["artifact_id"], producer["task_id"], producer["phase"],
                producer["attempt_id"], producer["generation"], record["logical_name"])
    if any(type(actual) is not type(wanted) or actual != wanted
           for actual, wanted in zip(raw[:6], expected)):
        raise ResearchArtifactError("artifact_record_identity_mismatch")
    return record


def artifacts_for_attempt(conn, ref: AttemptRef) -> list[dict]:
    """Read bounded historical records without granting current execution rights."""
    producer = _ref(ref)
    if not _schema(conn):
        return []
    rows = conn.execute(
        _SELECT + " WHERE producer_attempt_id=? COLLATE BINARY "
        "ORDER BY logical_name COLLATE BINARY LIMIT ?", (ref.attempt_id, MAX_ARTIFACTS + 1),
    ).fetchall()
    if len(rows) > MAX_ARTIFACTS:
        raise ResearchArtifactError("artifact_record_count_exceeded")
    records = [_row(row) for row in rows]
    if any(record["producer"] != producer for record in records):
        raise ResearchArtifactError("artifact_producer_identity_mismatch")
    return records


def get_artifact(conn, artifact_id: str) -> dict | None:
    _token(artifact_id)
    if not _schema(conn):
        return None
    rows = conn.execute(
        _SELECT + " WHERE artifact_id=? COLLATE BINARY LIMIT 2", (artifact_id,),
    ).fetchall()
    if len(rows) > 1:
        raise ResearchArtifactError("artifact_identity_ambiguous")
    return _row(rows[0]) if rows else None


def _binding(row):
    from orze.core.artifact_contract import validate_artifact_publication_binding
    return validate_artifact_publication_binding(row["binding"].get("artifact_publication"))


def register_artifacts(conn, ref: AttemptRef, records: list[dict]) -> tuple[str, ...]:
    """Insert an exact declared set; on any error caller rolls back everything."""
    if not conn.in_transaction:
        raise ResearchArtifactError("artifact_caller_transaction_required")
    producer = _ref(ref)
    if type(records) is not list or len(records) > MAX_ARTIFACTS:
        raise ResearchArtifactError("artifact_records_invalid")
    current = require_current(conn, ref, states=("RUNNING",))
    binding = _binding(current)
    outputs = binding["contract"]["outputs"]
    encoded = {}
    ids = set()
    captured = []
    for record in records:
        raw = _record(record)
        record = json.loads(raw)
        name, artifact_id = record["logical_name"], record["artifact_id"]
        if name in encoded or artifact_id in ids:
            raise ResearchArtifactError("artifact_batch_identity_duplicate")
        if (name not in outputs or record["producer"] != producer
                or record["scope"] != binding["scope"]
                or record["spec_fingerprint"] != binding["spec_fingerprint"]
                or record["path"] != str(Path(binding["root"]) / artifact_id / "content")
                or record["size_bytes"] > outputs[name]["max_bytes"]):
            raise ResearchArtifactError("artifact_record_binding_mismatch")
        encoded[name] = raw
        ids.add(artifact_id)
        captured.append(record)
    if set(encoded) != set(outputs):
        raise ResearchArtifactError("artifact_declared_outputs_incomplete")
    # Validate existing content/conflicts before the first INSERT as well.
    existing = {record["logical_name"]: _record(record)
                for record in artifacts_for_attempt(conn, ref)}
    if any(name not in encoded or encoded[name] != raw for name, raw in existing.items()):
        raise ResearchArtifactError("artifact_existing_record_conflict")
    for record in captured:
        other = get_artifact(conn, record["artifact_id"])
        if other is not None and _record(other) != encoded[record["logical_name"]]:
            raise ResearchArtifactError("artifact_existing_identity_conflict")
    if captured and not _schema(conn):
        conn.execute(_SQL)
        if not _schema(conn):
            raise ResearchArtifactError("artifact_schema_creation_failed")
    for record in captured:
        name = record["logical_name"]
        if name in existing:
            continue
        cursor = conn.execute(
            "INSERT INTO main.research_artifacts (artifact_id,producer_task_id,producer_phase,"
            "producer_attempt_id,producer_generation,logical_name,record_json) VALUES (?,?,?,?,?,?,?)",
            (record["artifact_id"], ref.task_id, ref.phase, ref.attempt_id,
             ref.generation, name, encoded[name]),
        )
        if cursor.rowcount != 1:
            raise ResearchArtifactError("artifact_insert_not_confirmed")
    actual = {record["logical_name"]: _record(record)
              for record in artifacts_for_attempt(conn, ref)}
    if actual != encoded:
        raise ResearchArtifactError("artifact_batch_readback_mismatch")
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    if canonical(require_current(conn, ref, states=("RUNNING",))) != canonical(current):
        raise ResearchArtifactError("artifact_publication_authority_changed")
    return tuple(record["artifact_id"] for record in captured)
