"""First-green pure metadata/store requirements for CPU Policy proposals.

Real SQLite commits/rollback are exercised; synthetic source/evidence digests
are not actual source captures, admitted tasks, Domain execution or GO proof.
"""
from copy import deepcopy
import sqlite3

import pytest

from orze.core import cpu_proposal_requests as store


@pytest.fixture
def connection(tmp_path):
    conn = sqlite3.connect(tmp_path / "requests.db")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def record(tmp_path, *, status="inserted", key="request-1", task="proposed-task", inputs=()):
    scope = str(tmp_path / "results")
    request = {
        "version": 1, "purpose": "ordinary analysis proposal", "inputs": {},
        "timeout_seconds": 1, "outputs": {}, "input_artifact_ids": list(inputs),
        "payload": {"command": ["python3", "-c", "pass"]},
    }
    decision = {"kind": "Propose", "request_id": key, "task_id": task,
                "reason": "inspect recorded sources", "domain_request": request}
    entries = []
    for index, identity in enumerate(inputs):
        entries.append({"artifact": {
            "schema": 1, "artifact_id": identity,
            "producer": {"task_id": "source-" + str(index), "phase": "action",
                         "attempt_id": "attempt-" + str(index), "generation": 1},
            "spec_fingerprint": str(index + 1) * 64, "scope": scope,
            "logical_name": "result", "path": str(tmp_path / "artifacts" / identity / "content"),
            "content_sha256": "c" * 64, "size_bytes": 2,
        }, "source_sha256": "d" * 64, "effect_sha256": "e" * 64})
    reasons = {"inserted": "proposal_admitted",
               "already_present_exact": "proposal_exact_replay",
               "config_duplicate": "proposal_config_duplicate",
               "conflict": "proposal_identity_conflict"}
    owner = "existing-task" if status == "config_duplicate" else None
    return store.seal_record({
        "schema": 1, "request_id": key, "task_id": task, "scope": scope,
        "database": str(tmp_path / "requests.db"), "decision": decision,
        "config_sha256": "a" * 64,
        "domain": {"declaration": {"version": 1, "kind": "historical_domain", "config": {}},
                   "implementation_id": "fixture.domain.v1"},
        "source_snapshot": {"schema": 1, "scope": scope,
            "database": str(tmp_path / "requests.db"), "database_identity": [1, 2],
            "scope_identity": [1, 3], "inputs": entries},
        "outcome": {"request_id": key, "task_id": task, "status": status,
                    "reason": reasons[status], "existing_id": owner},
        "admission_evidence": {"task_id": owner or task, "source_sha256": "f" * 64,
            "transition_id": 1 if status == "inserted" else None,
            "transition_sha256": "b" * 64 if status == "inserted" else None},
        "created_at": "2026-09-11T12:00:00Z",
    })


@pytest.mark.parametrize("status", ["inserted", "already_present_exact", "config_duplicate", "conflict"])
def test_four_known_outcomes_are_detached_historical_metadata(tmp_path, monkeypatch, status):
    from orze.core import research_interfaces
    monkeypatch.setattr(research_interfaces, "_DOMAINS", {})
    value = record(tmp_path, status=status, inputs=("input-a", "input-b"))
    result = store.validate_record(value)
    assert result == value
    assert result["source_snapshot"]["inputs"][0]["artifact"]["spec_fingerprint"] != result[
        "source_snapshot"]["inputs"][1]["artifact"]["spec_fingerprint"]
    result["decision"]["domain_request"]["inputs"]["changed"] = True
    assert value["decision"]["domain_request"]["inputs"] == {}
    assert value["outcome"]["status"] == status


def test_missing_reads_do_not_create_table_or_transaction(connection, tmp_path):
    before = list(connection.iterdump())
    assert store.get_request(connection, str(tmp_path / "results"), "missing") is None
    assert store.records_view(connection, str(tmp_path / "results")) == {
        "requests": [], "more_available": False}
    assert list(connection.iterdump()) == before
    assert not connection.in_transaction


def test_insert_rollback_and_duplicate_do_not_commit_or_rewrite(connection, tmp_path):
    value = record(tmp_path)
    with pytest.raises(store.ProposalRequestError, match="transaction"):
        store.insert_request(connection, value)
    connection.execute("BEGIN IMMEDIATE")
    store.insert_request(connection, value)
    assert connection.in_transaction
    assert store.get_request(connection, value["scope"], value["request_id"]) == value
    connection.rollback()
    assert store.get_request(connection, value["scope"], value["request_id"]) is None
    connection.execute("BEGIN IMMEDIATE")
    store.insert_request(connection, value)
    connection.commit()
    before = list(connection.iterdump())
    connection.execute("BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_request(connection, value)
    assert connection.in_transaction
    connection.rollback()
    assert list(connection.iterdump()) == before


def test_scope_and_request_form_key_but_task_id_is_not_unique(connection, tmp_path):
    first = record(tmp_path, status="already_present_exact")
    second = record(tmp_path, status="already_present_exact", key="request-2")
    other = record(tmp_path / "other", status="already_present_exact")
    connection.execute("BEGIN IMMEDIATE")
    for value in (first, second, other):
        store.insert_request(connection, value)
    connection.commit()
    assert store.get_request(connection, first["scope"], first["request_id"]) == first
    assert store.get_request(connection, other["scope"], other["request_id"]) == other
    assert connection.execute("SELECT COUNT(DISTINCT task_id) FROM cpu_proposal_requests").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM cpu_proposal_requests").fetchone()[0] == 3


def test_compact_view_is_bounded_ordered_and_does_not_revalidate_current_sources(connection, tmp_path):
    connection.execute("BEGIN IMMEDIATE")
    for number in reversed(range(33)):
        store.insert_request(connection, record(tmp_path, key="r-" + str(number).zfill(2)))
    connection.commit()
    before = list(connection.iterdump())
    view = store.records_view(connection, str(tmp_path / "results"))
    assert view["more_available"] is True
    assert len(view["requests"]) == 32
    assert [item["request_id"] for item in view["requests"]] == ["r-" + str(n).zfill(2) for n in range(32)]
    assert all(set(item) == {"request_id", "task_id", "outcome", "record_sha256"} for item in view["requests"])
    view["requests"][0]["outcome"]["status"] = "changed"
    assert store.get_request(connection, str(tmp_path / "results"), "r-00")["outcome"]["status"] == "inserted"
    assert list(connection.iterdump()) == before


@pytest.mark.parametrize("fault", [
    "schema_bool", "extra", "domain_bool", "domain_extra", "request_mismatch",
    "task_mismatch", "unknown_outcome", "wrong_reason", "duplicate_without_owner",
    "bool_transition", "noninsert_transition", "bad_config_hash",
    "source_database", "source_identity_bool", "source_phase", "source_hash", "source_order",
])
def test_exact_record_and_source_shapes_reject_invalid_metadata(tmp_path, fault):
    value = record(tmp_path, inputs=("input-a", "input-b"))
    if fault == "schema_bool":
        value["schema"] = True
    elif fault == "extra":
        value["extra"] = None
    elif fault == "domain_bool":
        value["domain"]["declaration"]["version"] = True
    elif fault == "domain_extra":
        value["domain"]["authority"] = True
    elif fault == "request_mismatch":
        value["decision"]["request_id"] = "different"
    elif fault == "task_mismatch":
        value["admission_evidence"]["task_id"] = "different"
    elif fault == "unknown_outcome":
        value["outcome"]["status"] = "rejected"
    elif fault == "wrong_reason":
        value["outcome"]["reason"] = "anything"
    elif fault == "duplicate_without_owner":
        value["outcome"].update(status="config_duplicate", reason="proposal_config_duplicate")
    elif fault == "bool_transition":
        value["admission_evidence"]["transition_id"] = True
    elif fault == "noninsert_transition":
        value["outcome"].update(status="conflict", reason="proposal_identity_conflict")
    elif fault == "bad_config_hash":
        value["config_sha256"] = "A" * 64
    elif fault == "source_database":
        value["source_snapshot"]["database"] = str(tmp_path / "foreign.db")
    elif fault == "source_identity_bool":
        value["source_snapshot"]["database_identity"][0] = True
    elif fault == "source_phase":
        value["source_snapshot"]["inputs"][0]["artifact"]["producer"]["phase"] = "training"
    elif fault == "source_hash":
        value["source_snapshot"]["inputs"][0]["effect_sha256"] = "invalid"
    elif fault == "source_order":
        value["source_snapshot"]["inputs"].reverse()
    with pytest.raises(store.ProposalRequestError):
        store.seal_record(value)


def test_record_digest_and_complete_byte_bound_are_not_optional(tmp_path):
    value = record(tmp_path)
    value["decision"]["reason"] = "changed without resealing"
    with pytest.raises(store.ProposalRequestError, match="digest"):
        store.validate_record(value)
    value = record(tmp_path)
    value["domain"]["declaration"]["config"]["padding"] = "a" * 65000
    with pytest.raises(store.ProposalRequestError):
        store.seal_record(value)


@pytest.mark.parametrize("limit", [True, 0, 33])
def test_view_rejects_nonexact_or_oversized_limit(connection, tmp_path, limit):
    with pytest.raises(store.ProposalRequestError, match="limit"):
        store.records_view(connection, str(tmp_path / "results"), limit=limit)
    assert not connection.in_transaction


def test_incompatible_existing_table_is_never_migrated(connection, tmp_path):
    connection.execute(store._SQL.replace("COLLATE BINARY", "COLLATE NOCASE"))
    connection.commit()
    before = list(connection.iterdump())
    with pytest.raises(store.ProposalRequestError, match="schema"):
        store.get_request(connection, str(tmp_path / "results"), "request-1")
    assert list(connection.iterdump()) == before


def test_insert_ignore_is_not_an_admission_or_commit_receipt(connection, tmp_path):
    connection.execute("BEGIN IMMEDIATE")
    store.ensure_schema(connection)
    connection.commit()
    connection.execute("CREATE TRIGGER refuse BEFORE INSERT ON cpu_proposal_requests BEGIN SELECT RAISE(IGNORE); END")
    connection.commit()
    value = record(tmp_path)
    connection.execute("BEGIN IMMEDIATE")
    with pytest.raises(store.ProposalRequestError, match="insert_not_applied"):
        store.insert_request(connection, value)
    assert connection.in_transaction
    connection.rollback()
    assert store.get_request(connection, value["scope"], value["request_id"]) is None


@pytest.mark.parametrize("raw", ["{}", " " * 65537])
def test_stored_invalid_or_oversized_json_is_rejected_without_writes(connection, tmp_path, raw):
    value = record(tmp_path)
    connection.execute("BEGIN IMMEDIATE")
    store.insert_request(connection, value)
    connection.commit()
    connection.execute("UPDATE cpu_proposal_requests SET record_json=?", (raw,))
    connection.commit()
    before = list(connection.iterdump())
    with pytest.raises(store.ProposalRequestError):
        store.get_request(connection, value["scope"], value["request_id"])
    assert list(connection.iterdump()) == before
