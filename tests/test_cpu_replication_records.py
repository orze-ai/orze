"""Schema-2 CPU replication metadata: new requirements, not old product bugs.

Real SQLite attempts/transactions are used; source terminal rows below are
synthetic metadata, not executed CPU trees or closure/publication capabilities.
The unchanged coordinator remains responsible for adapter-specific authority.
"""
from copy import deepcopy
from dataclasses import asdict
import sqlite3

import pytest

from orze.core.cpu_action_contract import action_fingerprint
from orze.core.execution_attempts import (
    AttemptAuthorityError, create_attempt, current_attempt, finish_attempt,
    mark_running,
)
from orze.core.replication_requests import (
    ReplicationError, canonical, digest, get_request, insert_request,
    request_for_task, seal_record, validate_record,
)


@pytest.fixture
def cpu_record(tmp_path):
    action = {
        "version": 1, "adapter": "command", "purpose": "explicit CPU replica",
        "inputs": {"value": 3}, "command": ["python3", "-c", "pass"],
        "timeout_seconds": 1, "outputs": {},
    }
    db = tmp_path / "requests.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("BEGIN IMMEDIATE")
    ref = create_attempt(conn, "source-task", "action", "source-attempt", {
        "adapter": "native_cpu_action", "action_sha256": action_fingerprint(action),
    })
    mark_running(conn, ref)
    finish_attempt(conn, ref, {
        "outcome": "completed", "artifact_ids": [], "observation_ids": [],
    })
    conn.commit()
    row = current_attempt(conn, ref.task_id, ref.phase)
    record = {
        "schema": 2, "adapter": "native_cpu_action",
        "request_id": "cpu-request", "task_id": "replica-task",
        "source_ref": asdict(ref), "scope": str(tmp_path), "database": str(db),
        "source_config_sha256": digest({"action": action}),
        "source_row_sha256": digest(row),
        "source_terminal_sha256": digest(row["terminal"]),
        "artifact_records_sha256": digest({"records": []}),
        "observation_records_sha256": digest({"records": []}),
        "action_sha256": action_fingerprint(action),
        "domain_run_sha256": digest({"domain_run": None}),
        "spec_fingerprint": action_fingerprint(action),
        "artifact_binding_sha256": digest({"artifact_binding": None}),
        "reason": "explicit independent occurrence",
        "created_at": "2026-09-11T11:30:00Z",
    }
    # Seal manually here so the baseline fails in test bodies, not fixture setup.
    record["request_sha256"] = digest(record)
    try:
        yield conn, ref, record, action
    finally:
        conn.close()


def test_cpu_schema_two_seal_and_detached_validation(cpu_record):
    _, ref, raw, action = cpu_record
    record = seal_record(raw)
    assert record == raw
    assert record["source_ref"] == asdict(ref)
    assert record["action_sha256"] == action_fingerprint(action)
    assert record["domain_run_sha256"] == digest({"domain_run": None})
    assert record["domain_run_sha256"] != digest({})
    validated = validate_record(record)
    validated["source_ref"]["generation"] = 999
    assert record["source_ref"]["generation"] == ref.generation
    assert "source_file_sha256" not in record and "execution_identity" not in record


def test_cpu_schema_two_insert_read_and_duplicate_are_insert_only(cpu_record):
    conn, _, record, _ = cpu_record
    assert get_request(conn, record["request_id"]) is None
    conn.execute("BEGIN IMMEDIATE")
    insert_request(conn, record)
    assert conn.in_transaction
    conn.commit()
    before = list(conn.iterdump())
    assert get_request(conn, record["request_id"]) == record
    detached = request_for_task(conn, record["task_id"])
    detached["source_ref"]["generation"] = 999
    assert request_for_task(conn, record["task_id"]) == record
    assert list(conn.iterdump()) == before
    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.IntegrityError):
        insert_request(conn, record)
    assert conn.in_transaction
    conn.rollback()
    assert list(conn.iterdump()) == before


@pytest.mark.parametrize("field,value", [
    ("schema", True),
    ("schema", 2.0),
    ("schema", "2"),
    ("schema", 1),
    ("adapter", "training"),
    ("adapter", True),
    ("source_file_sha256", "1" * 64),
    ("execution_identity", "2" * 64),
    ("extra", None),
    ("action_sha256", "a" * 63),
    ("domain_run_sha256", "A" * 64),
    ("observation_records_sha256", True),
    ("artifact_records_sha256", "g" * 64),
])
def test_cpu_schema_two_rejects_wrong_shape_types_and_hashes(cpu_record, field, value):
    _, _, record, _ = cpu_record
    changed = deepcopy(record)
    changed[field] = value
    with pytest.raises(ReplicationError):
        seal_record(changed)


@pytest.mark.parametrize("field,value", [
    ("phase", "training"), ("phase", "posthoc"), ("phase", True),
    ("generation", True), ("generation", 1.0),
])
def test_cpu_schema_two_requires_exact_action_ref(cpu_record, field, value):
    _, _, record, _ = cpu_record
    changed = deepcopy(record)
    changed["source_ref"][field] = value
    with pytest.raises(ReplicationError):
        seal_record(changed)


def test_cpu_schema_two_requires_all_fields_and_exact_digest(cpu_record):
    _, _, record, _ = cpu_record
    missing = deepcopy(record)
    del missing["domain_run_sha256"]
    with pytest.raises(ReplicationError):
        seal_record(missing)
    changed = deepcopy(record)
    changed["action_sha256"] = "f" * 64
    with pytest.raises(ReplicationError, match="digest_mismatch"):
        validate_record(changed)


def test_cpu_schema_two_bounded_encoded_record(cpu_record):
    _, _, record, _ = cpu_record
    changed = deepcopy(record)
    # Each path remains within its 4096-byte bound, but JSON escaping makes
    # the whole record exceed 16 KiB. No filesystem path is touched.
    changed["scope"] = "/" + '"' * 3999
    changed["database"] = "/" + '"' * 3999
    with pytest.raises(ReplicationError, match="record_limit"):
        seal_record(changed)


def test_cpu_schema_two_same_table_accepts_v1_without_restricting_its_phase(cpu_record):
    conn, _, record, _ = cpu_record
    legacy = {k: v for k, v in record.items() if k not in {
        "adapter", "action_sha256", "domain_run_sha256", "observation_records_sha256",
    }}
    legacy.update(schema=1, request_id="v1-request", task_id="v1-replica",
                  source_file_sha256="1" * 64, execution_identity="2" * 64)
    legacy = seal_record(legacy)
    assert legacy["source_ref"]["phase"] == "action"
    conn.execute("BEGIN IMMEDIATE")
    insert_request(conn, legacy)
    insert_request(conn, record)
    conn.commit()
    assert get_request(conn, "v1-request") == legacy
    assert get_request(conn, "cpu-request") == record
    assert conn.execute("SELECT COUNT(*) FROM replication_requests").fetchone()[0] == 2


def test_cpu_schema_two_rollback_keeps_schema_and_request_uncommitted(cpu_record):
    conn, _, record, _ = cpu_record
    with pytest.raises(ReplicationError, match="transaction"):
        insert_request(conn, record)
    conn.execute("BEGIN IMMEDIATE")
    insert_request(conn, record)
    assert get_request(conn, record["request_id"]) == record
    conn.rollback()
    assert get_request(conn, record["request_id"]) is None
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='replication_requests'"
    ).fetchone() is None


@pytest.mark.parametrize("change", ["stale", "running", "row_hash", "terminal_hash"])
def test_cpu_schema_two_insert_requires_exact_current_terminal_source(cpu_record, change):
    conn, ref, record, _ = cpu_record
    changed = deepcopy(record)
    if change in {"stale", "running"}:
        conn.execute("BEGIN IMMEDIATE")
        newer = create_attempt(conn, ref.task_id, ref.phase, "new-source-attempt", {})
        mark_running(conn, newer)
        conn.commit()
        if change == "running":
            row = current_attempt(conn, ref.task_id, ref.phase)
            changed["source_ref"] = asdict(newer)
            changed["source_row_sha256"] = digest(row)
            changed["source_terminal_sha256"] = digest({"terminal": None})
    else:
        changed["source_" + ("row" if change == "row_hash" else "terminal") + "_sha256"] = "0" * 64
    changed = seal_record(changed)
    before = list(conn.iterdump())
    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises((AttemptAuthorityError, ReplicationError)):
        insert_request(conn, changed)
    assert conn.in_transaction
    conn.rollback()
    assert list(conn.iterdump()) == before
    assert get_request(conn, changed["request_id"]) is None


def test_cpu_schema_two_historical_mapping_is_not_source_authority(cpu_record):
    conn, ref, record, _ = cpu_record
    conn.execute("BEGIN IMMEDIATE")
    insert_request(conn, record)
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    create_attempt(conn, ref.task_id, ref.phase, "later-attempt", {})
    conn.commit()
    before = list(conn.iterdump())
    assert get_request(conn, record["request_id"]) == record
    assert list(conn.iterdump()) == before
    replacement = dict(record, request_id="later-request", task_id="later-replica")
    replacement = seal_record(replacement)
    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(AttemptAuthorityError):
        insert_request(conn, replacement)
    conn.rollback()
    assert list(conn.iterdump()) == before


def test_cpu_schema_two_stored_record_must_be_canonical(cpu_record):
    conn, _, record, _ = cpu_record
    conn.execute("BEGIN IMMEDIATE")
    insert_request(conn, record)
    conn.commit()
    conn.execute("UPDATE replication_requests SET record_json=?", (" " + canonical(record),))
    conn.commit()
    before = list(conn.iterdump())
    with pytest.raises(ReplicationError, match="stored_record_invalid"):
        get_request(conn, record["request_id"])
    assert list(conn.iterdump()) == before
