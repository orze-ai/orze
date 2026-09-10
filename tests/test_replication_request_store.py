"""Generic request-store mechanisms; no old API-absence regressions."""
from dataclasses import asdict
import sqlite3

import pytest

from orze.core.execution_attempts import create_attempt, current_attempt, finish_attempt, mark_running
from orze.core.replication_requests import (
    ReplicationError, digest, ensure_schema, get_request, insert_request,
    request_for_task, seal_record,
)


@pytest.fixture
def stored(tmp_path):
    conn = sqlite3.connect(tmp_path / "store.db")
    conn.row_factory = sqlite3.Row
    conn.execute("BEGIN IMMEDIATE")
    ref = create_attempt(conn, "task-original", "future_controller_adapter", "attempt-a", {})
    mark_running(conn, ref)
    finish_attempt(conn, ref, {"adapter_outcome": "done"})
    conn.commit()
    row = current_attempt(conn, ref.task_id, ref.phase)
    record = seal_record({
        "schema": 1, "request_id": "request-a", "task_id": "task-copy",
        "source_ref": asdict(ref), "scope": str(tmp_path), "database": str(tmp_path / "store.db"),
        "source_config_sha256": "1" * 64, "source_file_sha256": "2" * 64,
        "source_row_sha256": digest(row), "source_terminal_sha256": digest(row["terminal"]),
        "artifact_records_sha256": "3" * 64, "execution_identity": "4" * 64,
        "spec_fingerprint": "5" * 64, "artifact_binding_sha256": "6" * 64,
        "reason": "explicit control request", "created_at": "2026-09-10T00:00:00Z",
    })
    try:
        yield conn, ref, record
    finally:
        conn.close()


def test_generic_phase_and_detached_historical_read_do_not_grant_current_permission(stored):
    conn, ref, record = stored
    assert get_request(conn, "request-a") is None
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='replication_requests'").fetchone() is None
    conn.execute("BEGIN IMMEDIATE")
    insert_request(conn, record)
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    create_attempt(conn, ref.task_id, ref.phase, "attempt-b", {})
    conn.commit()
    snapshot = list(conn.iterdump())
    read = get_request(conn, "request-a")
    assert read == record
    read["source_ref"]["generation"] = 99
    assert request_for_task(conn, "task-copy") == record
    assert list(conn.iterdump()) == snapshot


def test_request_and_new_schema_follow_caller_rollback(stored):
    conn, _, record = stored
    conn.execute("BEGIN IMMEDIATE")
    insert_request(conn, record)
    assert conn.in_transaction and get_request(conn, "request-a") == record
    conn.rollback()
    assert get_request(conn, "request-a") is None
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='replication_requests'").fetchone() is None


def test_store_does_not_start_a_write_transaction(stored):
    conn, _, record = stored
    with pytest.raises(ReplicationError, match="transaction"):
        insert_request(conn, record)
    assert not conn.in_transaction


def test_duplicate_request_is_not_rewritten(stored):
    conn, _, record = stored
    conn.execute("BEGIN IMMEDIATE")
    insert_request(conn, record)
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.IntegrityError):
        insert_request(conn, record)
    assert conn.in_transaction
    conn.rollback()
    assert get_request(conn, "request-a") == record


def test_nocase_identity_schema_is_not_authoritative(stored):
    from orze.core.replication_requests import _SQL
    conn, _, _ = stored
    conn.execute(_SQL.replace("COLLATE BINARY", "COLLATE NOCASE"))
    conn.commit()
    snapshot = list(conn.iterdump())
    with pytest.raises(ReplicationError, match="schema"):
        get_request(conn, "request-a")
    assert list(conn.iterdump()) == snapshot


def test_trigger_ignored_insert_is_not_a_commit_receipt(stored):
    conn, _, record = stored
    conn.execute("BEGIN IMMEDIATE")
    ensure_schema(conn)
    conn.commit()
    conn.execute("CREATE TRIGGER refuse BEFORE INSERT ON replication_requests BEGIN SELECT RAISE(IGNORE); END")
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(ReplicationError, match="not_applied"):
        insert_request(conn, record)
    assert conn.in_transaction
    conn.rollback()
    assert get_request(conn, "request-a") is None
