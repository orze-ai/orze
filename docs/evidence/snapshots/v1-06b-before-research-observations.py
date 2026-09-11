"""Bounded, insert-only observation records; no scientific or ranking inference.

CALLING SPEC:
    register_observations(conn, ref, records) -> tuple[observation_id, ...]
        Requires a caller-owned transaction, current RUNNING AttemptRef, and
        observation_publication binding. Validates the complete supplied batch
        and referenced artifact metadata before any INSERT. Failures require
        the caller to roll back its WHOLE transaction. No BEGIN/COMMIT/ROLLBACK.
    observations_for_attempt(conn, ref) -> list[dict]
    get_observation(conn, observation_id) -> dict | None
        Detached historical metadata; no DDL, filesystem access, or requirement
        that the producing attempt is still current. Missing tables stay absent.
    validate_observation_record(record) -> dict
        Detached metadata validation for the adapter's lock-free preparation;
        does not validate publication authority or referenced artifact existence.

The adapter owns file verification, protocol meaning and its validation claims.
An observation occurrence, its subject specification and its evaluator attempt
have different identities. Empty batches mean no observations, never zero scores.
valid/invalid/unknown records are stored without promoting any of them to a rank,
comparison result, statistical independence claim or scientific conclusion.
Caller must hold its effect lease and BEGIN IMMEDIATE; in_transaction alone
cannot establish SQLite's transaction locking mode. API writes are insert-only,
not protection against arbitrary direct SQL/filesystem modification. Adapters
must recheck records after any later terminal/lifecycle writes in the same tx.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict

from orze.core.execution_attempts import AttemptAuthorityError, AttemptRef, require_current
from orze.core.research_artifacts import get_artifact


MAX_OBSERVATIONS = 32
MAX_RECORD_BYTES = 32768
MAX_JSON_DEPTH = 16
MAX_JSON_NODES = 8192
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_FIELDS = {"schema", "observation_id", "evaluator", "scope", "spec_fingerprint",
           "protocol_fingerprint", "adapter_id", "input_artifact_ids",
           "result_artifact_ids", "name", "values", "validation", "comparison_scope"}
_REF_FIELDS = {"task_id", "phase", "attempt_id", "generation"}
_BINDING_FIELDS = ("adapter_id", "protocol_fingerprint", "spec_fingerprint",
                   "scope", "input_artifact_ids")
_SQL = """CREATE TABLE research_observations (
    observation_id TEXT NOT NULL COLLATE BINARY PRIMARY KEY,
    evaluator_task_id TEXT NOT NULL COLLATE BINARY,
    evaluator_phase TEXT NOT NULL COLLATE BINARY,
    evaluator_attempt_id TEXT NOT NULL COLLATE BINARY,
    evaluator_generation INTEGER NOT NULL CHECK (evaluator_generation > 0),
    name TEXT NOT NULL COLLATE BINARY,
    record_json TEXT NOT NULL,
    UNIQUE (evaluator_attempt_id, name)
)"""
_COLUMNS = [
    ("observation_id", "TEXT", 1, 1), ("evaluator_task_id", "TEXT", 1, 0),
    ("evaluator_phase", "TEXT", 1, 0), ("evaluator_attempt_id", "TEXT", 1, 0),
    ("evaluator_generation", "INTEGER", 1, 0), ("name", "TEXT", 1, 0),
    ("record_json", "TEXT", 1, 0),
]
_SELECT = """SELECT observation_id,evaluator_task_id,evaluator_phase,
    evaluator_attempt_id,evaluator_generation,name,
    CASE WHEN typeof(record_json)='text'
         AND length(CAST(record_json AS BLOB)) <= 32768
         THEN record_json ELSE NULL END FROM main.research_observations"""


class ResearchObservationError(ValueError):
    """Record contract unavailable or rejected; caller rolls back writes."""


def _token(value):
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ResearchObservationError("observation_identity_invalid")
    return value


def _ref(ref):
    if type(ref) is not AttemptRef:
        raise ResearchObservationError("observation_evaluator_invalid")
    for value in (ref.task_id, ref.phase, ref.attempt_id):
        _token(value)
    if type(ref.generation) is not int or not 0 < ref.generation <= 2**63 - 1:
        raise ResearchObservationError("observation_evaluator_invalid")
    return asdict(ref)


def _ids(values, *, required=False):
    if (type(values) is not list or len(values) > MAX_OBSERVATIONS
            or (required and not values)):
        raise ResearchObservationError("observation_artifact_ids_invalid")
    for value in values:
        _token(value)
    if len(set(values)) != len(values):
        raise ResearchObservationError("observation_artifact_ids_duplicate")


def _encode(value):
    nodes, string_bytes, active = 0, 0, set()

    def visit(item, depth):
        nonlocal nodes, string_bytes
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise ResearchObservationError("observation_json_complexity_limit")
        kind = type(item)
        if kind in (dict, list):
            if id(item) in active:
                raise ResearchObservationError("observation_json_recursive")
            active.add(id(item))
            if kind is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise ResearchObservationError("observation_json_key_invalid")
                    visit(key, depth + 1)
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
            active.remove(id(item))
        elif kind is str:
            if len(item) > MAX_RECORD_BYTES:
                raise ResearchObservationError("observation_json_byte_limit")
            string_bytes += len(item.encode("utf-8"))
            if string_bytes > MAX_RECORD_BYTES:
                raise ResearchObservationError("observation_json_byte_limit")
        elif kind is float:
            if not math.isfinite(item):
                raise ResearchObservationError("observation_json_nonfinite")
        elif kind is int:
            if item.bit_length() > MAX_RECORD_BYTES * 4:
                raise ResearchObservationError("observation_json_byte_limit")
        elif kind not in (bool, type(None)):
            raise ResearchObservationError("observation_json_value_invalid")

    try:
        visit(value, 0)
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
        if len(raw.encode("utf-8")) > MAX_RECORD_BYTES:
            raise ResearchObservationError("observation_json_byte_limit")
        return raw
    except (UnicodeError, RecursionError, TypeError, OverflowError) as exc:
        raise ResearchObservationError("observation_json_encoding_invalid") from exc


def _record(value):
    from orze.core.observation_contract import validate_observation_publication_binding
    if type(value) is not dict or set(value) != _FIELDS:
        raise ResearchObservationError("observation_record_fields_invalid")
    if type(value["schema"]) is not int or value["schema"] != 1:
        raise ResearchObservationError("observation_record_schema_invalid")
    _token(value["observation_id"])
    evaluator = value["evaluator"]
    if type(evaluator) is not dict or set(evaluator) != _REF_FIELDS:
        raise ResearchObservationError("observation_evaluator_invalid")
    try:
        _ref(AttemptRef(**evaluator))
    except AttemptAuthorityError as exc:
        raise ResearchObservationError("observation_evaluator_invalid") from exc
    validate_observation_publication_binding({key: value[key] for key in _BINDING_FIELDS})
    _ids(value["input_artifact_ids"])
    _ids(value["result_artifact_ids"], required=True)
    if type(value["name"]) is not str or _NAME.fullmatch(value["name"]) is None:
        raise ResearchObservationError("observation_name_invalid")
    if type(value["values"]) is not dict:
        raise ResearchObservationError("observation_values_invalid")
    validation = value["validation"]
    if (type(validation) is not dict or set(validation) != {"status", "reason_code"}
            or type(validation["status"]) is not str
            or validation["status"] not in {"valid", "invalid", "unknown"}):
        raise ResearchObservationError("observation_validation_invalid")
    _token(validation["reason_code"])
    comparison = value["comparison_scope"]
    if comparison is not None:
        if (type(comparison) is not str or not comparison.strip() or len(comparison) > 256
                or len(comparison.encode("utf-8")) > 256
                or any(ord(char) < 32 for char in comparison)):
            raise ResearchObservationError("observation_comparison_scope_invalid")
    return _encode(value)


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ResearchObservationError("observation_json_duplicate_key")
            result[key] = value
        return result
    try:
        if type(raw) is not str:
            raise ValueError
        record = json.loads(raw, object_pairs_hook=pairs)
        if _record(record) != raw:
            raise ValueError
        return record
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise ResearchObservationError("observation_stored_record_invalid") from exc


def validate_observation_record(record) -> dict:
    """Validate and detach metadata; not permission to publish an observation."""
    return json.loads(_record(record))


def _schema(conn):
    rows = conn.execute(
        "SELECT type,sql,name FROM main.sqlite_master WHERE name=? COLLATE NOCASE",
        ("research_observations",),
    ).fetchall()
    if not rows:
        return False
    normalize = lambda text: " ".join(str(text).strip().rstrip(";").split())
    if (len(rows) != 1 or rows[0][0] != "table" or rows[0][2] != "research_observations"
            or normalize(rows[0][1]) != normalize(_SQL)):
        raise ResearchObservationError("observation_schema_incompatible")
    columns = conn.execute("PRAGMA main.table_xinfo(research_observations)").fetchall()
    if ([(r[1], r[2], r[3], r[5]) for r in columns] != _COLUMNS
            or any(r[4] is not None or r[6] != 0 for r in columns)):
        raise ResearchObservationError("observation_schema_columns_invalid")
    identities = []
    for index in conn.execute("PRAGMA main.index_list(research_observations)").fetchall():
        if not index[2]:
            continue
        keys = conn.execute(
            "SELECT name,coll,desc FROM pragma_index_xinfo(?, 'main') WHERE key=1 ORDER BY seqno",
            (index[1],),
        ).fetchall()
        if index[4] or any(r[1] != "BINARY" or r[2] != 0 for r in keys):
            raise ResearchObservationError("observation_schema_identity_invalid")
        identities.append((index[3], tuple(r[0] for r in keys)))
    if sorted(identities) != sorted([
            ("pk", ("observation_id",)), ("u", ("evaluator_attempt_id", "name"))]):
        raise ResearchObservationError("observation_schema_identity_invalid")
    return True


def _row(raw):
    record = _decode(raw[6])
    evaluator = record["evaluator"]
    expected = (record["observation_id"], evaluator["task_id"], evaluator["phase"],
                evaluator["attempt_id"], evaluator["generation"], record["name"])
    if any(type(actual) is not type(wanted) or actual != wanted
           for actual, wanted in zip(raw[:6], expected)):
        raise ResearchObservationError("observation_record_identity_mismatch")
    return record


def observations_for_attempt(conn, ref: AttemptRef) -> list[dict]:
    evaluator = _ref(ref)
    if not _schema(conn):
        return []
    rows = conn.execute(
        _SELECT + " WHERE evaluator_attempt_id=? COLLATE BINARY ORDER BY name COLLATE BINARY LIMIT ?",
        (ref.attempt_id, MAX_OBSERVATIONS + 1),
    ).fetchall()
    if len(rows) > MAX_OBSERVATIONS:
        raise ResearchObservationError("observation_record_count_exceeded")
    records = [_row(row) for row in rows]
    if any(record["evaluator"] != evaluator for record in records):
        raise ResearchObservationError("observation_evaluator_identity_mismatch")
    return records


def get_observation(conn, observation_id: str) -> dict | None:
    _token(observation_id)
    if not _schema(conn):
        return None
    rows = conn.execute(
        _SELECT + " WHERE observation_id=? COLLATE BINARY LIMIT 2", (observation_id,),
    ).fetchall()
    if len(rows) > 1:
        raise ResearchObservationError("observation_identity_ambiguous")
    return _row(rows[0]) if rows else None


def _dependencies(conn, ref, binding, records):
    artifacts = {}

    def read(artifact_id):
        # Reused only within this pass. The post-write invocation starts fresh.
        if artifact_id not in artifacts:
            artifacts[artifact_id] = get_artifact(conn, artifact_id)
        return artifacts[artifact_id]

    for artifact_id in binding["input_artifact_ids"]:
        artifact = read(artifact_id)
        if (artifact is None or artifact["scope"] != binding["scope"]
                or artifact["spec_fingerprint"] != binding["spec_fingerprint"]):
            raise ResearchObservationError("observation_input_artifact_mismatch")
    for record in records:
        for artifact_id in record["result_artifact_ids"]:
            artifact = read(artifact_id)
            if (artifact is None or artifact["scope"] != binding["scope"]
                    or artifact["producer"] != asdict(ref)):
                raise ResearchObservationError("observation_result_artifact_mismatch")
    return artifacts


def register_observations(conn, ref: AttemptRef, records: list[dict]) -> tuple[str, ...]:
    from orze.core.observation_contract import validate_observation_publication_binding
    if not conn.in_transaction:
        raise ResearchObservationError("observation_caller_transaction_required")
    evaluator = _ref(ref)
    if type(records) is not list or len(records) > MAX_OBSERVATIONS:
        raise ResearchObservationError("observation_records_invalid")
    current = require_current(conn, ref, states=("RUNNING",))
    binding = validate_observation_publication_binding(current["binding"].get("observation_publication"))
    encoded, captured, ids = {}, [], set()
    for original in records:
        raw = _record(original)
        record = json.loads(raw)
        name, identity = record["name"], record["observation_id"]
        if name in encoded or identity in ids:
            raise ResearchObservationError("observation_batch_identity_duplicate")
        if (record["evaluator"] != evaluator
                or any(record[key] != binding[key] for key in _BINDING_FIELDS)):
            raise ResearchObservationError("observation_record_binding_mismatch")
        encoded[name] = raw
        captured.append(record)
        ids.add(identity)
    dependencies = _dependencies(conn, ref, binding, captured)
    existing = {record["name"]: _record(record) for record in observations_for_attempt(conn, ref)}
    if any(name not in encoded or raw != encoded[name] for name, raw in existing.items()):
        raise ResearchObservationError("observation_existing_record_conflict")
    for record in captured:
        other = get_observation(conn, record["observation_id"])
        if other is not None and _record(other) != encoded[record["name"]]:
            raise ResearchObservationError("observation_existing_identity_conflict")
    if captured and not _schema(conn):
        conn.execute(_SQL)
        if not _schema(conn):
            raise ResearchObservationError("observation_schema_creation_failed")
    for record in captured:
        if record["name"] in existing:
            continue
        cursor = conn.execute(
            "INSERT INTO main.research_observations (observation_id,evaluator_task_id,evaluator_phase,"
            "evaluator_attempt_id,evaluator_generation,name,record_json) VALUES (?,?,?,?,?,?,?)",
            (record["observation_id"], ref.task_id, ref.phase, ref.attempt_id,
             ref.generation, record["name"], encoded[record["name"]]),
        )
        if cursor.rowcount != 1:
            raise ResearchObservationError("observation_insert_not_confirmed")
    actual = {record["name"]: _record(record) for record in observations_for_attempt(conn, ref)}
    if actual != encoded:
        raise ResearchObservationError("observation_batch_readback_mismatch")
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    if (canonical(require_current(conn, ref, states=("RUNNING",))) != canonical(current)
            or canonical(_dependencies(conn, ref, binding, captured)) != canonical(dependencies)):
        raise ResearchObservationError("observation_publication_authority_changed")
    return tuple(record["observation_id"] for record in captured)
