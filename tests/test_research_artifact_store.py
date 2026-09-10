"""New B1 store mechanism tests, not missing-API old behavior regressions."""
import copy
import hashlib
import json
import sqlite3
from dataclasses import asdict

import pytest

from orze.core import research_artifacts as store
from orze.core.execution_attempts import (
    AttemptAuthorityError, AttemptRef, create_attempt, current_attempt,
    finish_attempt, mark_running,
)


def binding(tmp_path, outputs=None):
    return {
        "contract": {"version": 1, "outputs": outputs if outputs is not None else {
            "model": {"path": "worker/model.bin", "max_bytes": 100}}},
        "root": str(tmp_path / "independent-artifacts"),
        "scope": str(tmp_path / "results"),
        "spec_fingerprint": "a" * 64,
    }


def running(conn, publication, task="task-a", attempt="attempt-a"):
    ref = create_attempt(conn, task, "training", attempt, {
        "artifact_publication": publication, "pid": 0})
    mark_running(conn, ref)
    return ref


def record(ref, publication, *, name="model", artifact_id="artifact-a", data=b"same bytes"):
    return {
        "schema": 1, "artifact_id": artifact_id, "producer": asdict(ref),
        "spec_fingerprint": publication["spec_fingerprint"],
        "scope": publication["scope"], "logical_name": name,
        "path": f"{publication['root']}/{artifact_id}/content",
        "content_sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
    }


@pytest.fixture
def project(tmp_path):
    conn = sqlite3.connect(tmp_path / "catalog.db")
    conn.execute("BEGIN IMMEDIATE")
    publication = binding(tmp_path)
    ref = running(conn, publication)
    conn.commit()
    yield conn, ref, publication
    conn.close()


def test_missing_reads_are_readonly_and_do_not_create_schema():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("PRAGMA query_only=ON")
        ref = AttemptRef("task", "cpu", "attempt", 1)
        assert store.artifacts_for_attempt(conn, ref) == []
        assert store.get_artifact(conn, "artifact") is None
        assert conn.execute("SELECT * FROM sqlite_master").fetchall() == []
        assert not conn.in_transaction
    finally:
        conn.close()


def test_publication_requires_caller_transaction(project):
    conn, ref, publication = project
    with pytest.raises(store.ResearchArtifactError, match="caller_transaction"):
        store.register_artifacts(conn, ref, [record(ref, publication)])
    assert not conn.in_transaction
    assert store.get_artifact(conn, "artifact-a") is None


def test_schema_records_and_caller_data_roll_back_together(project):
    conn, ref, publication = project
    conn.execute("CREATE TABLE marker(value)")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("INSERT INTO marker VALUES ('uncommitted')")
    value = record(ref, publication)
    calls = []
    conn.set_trace_callback(calls.append)
    assert store.register_artifacts(conn, ref, [value]) == ("artifact-a",)
    assert store.get_artifact(conn, "artifact-a") == value
    assert conn.in_transaction
    assert not any(sql.split()[0].upper() in {"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT"}
                   for sql in calls)
    conn.set_trace_callback(None)
    conn.rollback()
    assert conn.execute("SELECT * FROM marker").fetchall() == []
    assert store.artifacts_for_attempt(conn, ref) == []
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='research_artifacts'").fetchall() == []


def test_exact_replay_is_write_free_and_returned_records_are_detached(project):
    conn, ref, publication = project
    value = record(ref, publication)
    conn.execute("BEGIN IMMEDIATE")
    store.register_artifacts(conn, ref, [value])
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    before = conn.total_changes
    assert store.register_artifacts(conn, ref, [copy.deepcopy(value)]) == ("artifact-a",)
    assert conn.total_changes == before
    detached = store.get_artifact(conn, "artifact-a")
    detached["producer"]["task_id"] = "foreign"
    assert store.get_artifact(conn, "artifact-a") == value
    conn.rollback()


def test_equal_content_has_distinct_occurrences_and_historical_reads(project):
    conn, old, publication = project
    conn.execute("BEGIN IMMEDIATE")
    first = record(old, publication)
    store.register_artifacts(conn, old, [first])
    finish_attempt(conn, old, {"outcome": "completed"})
    new = running(conn, publication, attempt="attempt-b")
    second = record(new, publication, artifact_id="artifact-b")
    assert new.generation == 2
    assert first["content_sha256"] == second["content_sha256"]
    assert first["spec_fingerprint"] == second["spec_fingerprint"]
    store.register_artifacts(conn, new, [second])
    conn.commit()
    conn.execute("PRAGMA query_only=ON")
    assert store.artifacts_for_attempt(conn, old) == [first]
    assert store.artifacts_for_attempt(conn, new) == [second]
    assert store.get_artifact(conn, "artifact-a") == first
    assert not conn.in_transaction


@pytest.mark.parametrize("state", ["LAUNCHING", "TERMINAL", "stale"])
def test_nonrunning_or_stale_reference_cannot_publish(project, state):
    conn, old, publication = project
    conn.execute("BEGIN IMMEDIATE")
    if state == "LAUNCHING":
        ref = create_attempt(conn, "task-b", "training", "attempt-b", {"artifact_publication": publication})
    else:
        finish_attempt(conn, old, {"outcome": "completed"})
        ref = old
        if state == "stale":
            running(conn, publication, attempt="attempt-b")
    before = conn.total_changes
    with pytest.raises(AttemptAuthorityError):
        store.register_artifacts(conn, ref, [record(ref, publication)])
    assert conn.total_changes == before and conn.in_transaction
    assert store.artifacts_for_attempt(conn, old) == []
    conn.rollback()


@pytest.mark.parametrize("mutation", [
    {"schema": True}, {"size_bytes": True}, {"size_bytes": -1}, {"size_bytes": 1.0},
    {"size_bytes": 101}, {"scope": "/different/results"}, {"spec_fingerprint": "b" * 64},
    {"content_sha256": "A" * 64}, {"logical_name": "undeclared"},
    {"path": "/worker/model.bin"}, {"path": "/root/../artifact-a/content"},
    {"extra": "not permitted"},
])
def test_invalid_record_rejected_before_insert(project, mutation):
    conn, ref, publication = project
    value = record(ref, publication)
    value.update(mutation)
    conn.execute("BEGIN IMMEDIATE")
    before = conn.total_changes
    with pytest.raises((store.ResearchArtifactError, AttemptAuthorityError)):
        store.register_artifacts(conn, ref, [value])
    assert conn.total_changes == before
    assert conn.in_transaction and store.artifacts_for_attempt(conn, ref) == []
    conn.rollback()


def test_producer_mismatch_and_bool_generation_are_not_equal(project):
    conn, ref, publication = project
    conn.execute("BEGIN IMMEDIATE")
    for change in ({"task_id": "foreign"}, {"generation": True}, {"generation": 1.0}):
        value = record(ref, publication)
        value["producer"].update(change)
        with pytest.raises((store.ResearchArtifactError, AttemptAuthorityError)):
            store.register_artifacts(conn, ref, [value])
    assert store.artifacts_for_attempt(conn, ref) == []
    conn.rollback()


def test_complete_declared_set_is_validated_before_first_insert(tmp_path):
    conn = sqlite3.connect(":memory:")
    try:
        publication = binding(tmp_path, {
            "model": {"path": "worker/model", "max_bytes": 20},
            "summary": {"path": "worker/summary", "max_bytes": 20}})
        conn.execute("BEGIN IMMEDIATE")
        ref = running(conn, publication)
        first = record(ref, publication)
        second = record(ref, publication, name="summary", artifact_id="artifact-b")
        for values in ([first], [first, first], [first, {**second, "size_bytes": 21}]):
            before = conn.total_changes
            with pytest.raises(store.ResearchArtifactError):
                store.register_artifacts(conn, ref, values)
            assert conn.total_changes == before
            assert store.artifacts_for_attempt(conn, ref) == []
        assert store.register_artifacts(conn, ref, [second, first]) == ("artifact-b", "artifact-a")
        assert store.artifacts_for_attempt(conn, ref) == [first, second]
    finally:
        conn.close()


def test_empty_declaration_allows_zero_records_without_schema(tmp_path):
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("BEGIN IMMEDIATE")
        publication = binding(tmp_path, {})
        ref = running(conn, publication)
        assert store.register_artifacts(conn, ref, []) == ()
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='research_artifacts'").fetchall()
        assert conn.in_transaction
    finally:
        conn.close()


def test_zero_byte_declared_artifact_is_preserved(project):
    conn, ref, publication = project
    value = record(ref, publication, data=b"")
    conn.execute("BEGIN IMMEDIATE")
    store.register_artifacts(conn, ref, [value])
    assert store.get_artifact(conn, "artifact-a")["size_bytes"] == 0
    conn.rollback()


def test_conflicting_name_id_or_content_never_overwrites(project):
    conn, ref, publication = project
    conn.execute("BEGIN IMMEDIATE")
    original = record(ref, publication)
    store.register_artifacts(conn, ref, [original])
    for replacement in (record(ref, publication, artifact_id="replacement"),
                        record(ref, publication, data=b"new")):
        before = conn.total_changes
        with pytest.raises(store.ResearchArtifactError, match="conflict"):
            store.register_artifacts(conn, ref, [replacement])
        assert conn.total_changes == before
        assert store.artifacts_for_attempt(conn, ref) == [original]
    peer = running(conn, publication, task="task-b", attempt="attempt-b")
    with pytest.raises(store.ResearchArtifactError, match="identity_conflict"):
        store.register_artifacts(conn, peer, [record(peer, publication)])
    assert store.get_artifact(conn, "artifact-a") == original
    conn.rollback()


@pytest.mark.parametrize("trigger", [
    "BEFORE INSERT ON research_artifacts BEGIN SELECT RAISE(IGNORE); END",
    "AFTER INSERT ON research_artifacts BEGIN UPDATE research_artifacts SET logical_name='changed'; END",
    "AFTER INSERT ON research_artifacts BEGIN UPDATE execution_attempts SET binding_json="
    "replace(binding_json, '\"pid\":0', '\"pid\":false'); END",
])
def test_unconfirmed_insert_or_changed_authority_requires_caller_rollback(project, trigger):
    conn, ref, publication = project
    conn.execute(store._SQL)
    conn.execute("CREATE TRIGGER sabotage " + trigger)
    conn.execute("CREATE TABLE marker(value)")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("INSERT INTO marker VALUES ('caller')")
    with pytest.raises((store.ResearchArtifactError, AttemptAuthorityError)):
        store.register_artifacts(conn, ref, [record(ref, publication)])
    assert conn.in_transaction
    assert conn.execute("SELECT * FROM marker").fetchall() == [("caller",)]
    conn.rollback()
    assert store.artifacts_for_attempt(conn, ref) == []
    assert conn.execute("SELECT * FROM marker").fetchall() == []
    assert current_attempt(conn, ref.task_id, ref.phase)["binding"]["pid"] == 0


def test_second_insert_failure_does_not_commit_first_and_caller_can_rollback(tmp_path):
    conn = sqlite3.connect(":memory:")
    try:
        publication = binding(tmp_path, {
            "model": {"path": "model", "max_bytes": 20},
            "summary": {"path": "summary", "max_bytes": 20}})
        conn.execute("BEGIN IMMEDIATE")
        ref = running(conn, publication)
        conn.execute(store._SQL)
        conn.execute("CREATE TRIGGER fail_second BEFORE INSERT ON research_artifacts "
                     "WHEN NEW.logical_name='summary' BEGIN SELECT RAISE(ABORT,'storage fault'); END")
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        values = [record(ref, publication), record(ref, publication, name="summary", artifact_id="artifact-b")]
        with pytest.raises(sqlite3.Error):
            store.register_artifacts(conn, ref, values)
        assert conn.in_transaction
        assert len(store.artifacts_for_attempt(conn, ref)) == 1
        conn.rollback()
        assert store.artifacts_for_attempt(conn, ref) == []
    finally:
        conn.close()


def test_main_namespace_not_temporary_shadow_is_published(project):
    conn, ref, publication = project
    conn.execute("CREATE TEMP TABLE research_artifacts(artifact_id TEXT)")
    conn.execute("BEGIN IMMEDIATE")
    value = record(ref, publication)
    store.register_artifacts(conn, ref, [value])
    assert conn.execute("SELECT * FROM temp.research_artifacts").fetchall() == []
    assert store.get_artifact(conn, "artifact-a") == value
    conn.rollback()


@pytest.mark.parametrize("variant", ["nocase", "column", "view"])
def test_existing_incompatible_schema_is_not_repaired(project, variant):
    conn, ref, publication = project
    if variant == "view":
        conn.execute("CREATE VIEW research_artifacts AS SELECT 1 AS artifact_id")
    else:
        sql = store._SQL.replace("COLLATE BINARY", "COLLATE NOCASE", 1) if variant == "nocase" else "CREATE TABLE research_artifacts(artifact_id TEXT)"
        conn.execute(sql)
    before = conn.execute("SELECT sql FROM sqlite_master WHERE name='research_artifacts'").fetchone()
    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(store.ResearchArtifactError, match="schema"):
        store.register_artifacts(conn, ref, [record(ref, publication)])
    assert conn.execute("SELECT sql FROM sqlite_master WHERE name='research_artifacts'").fetchone() == before
    assert conn.in_transaction
    conn.rollback()


@pytest.mark.parametrize("raw", [None, "[]", '{"schema":1,"schema":1}', " " * 20000])
def test_corrupt_or_oversized_stored_json_is_rejected_without_repair(project, raw):
    conn, ref, publication = project
    conn.execute("BEGIN IMMEDIATE")
    store.register_artifacts(conn, ref, [record(ref, publication)])
    conn.execute("UPDATE research_artifacts SET record_json=?", (raw if raw is not None else b"binary",))
    conn.commit()
    before = conn.total_changes
    with pytest.raises(store.ResearchArtifactError):
        store.get_artifact(conn, "artifact-a")
    with pytest.raises(store.ResearchArtifactError):
        store.artifacts_for_attempt(conn, ref)
    assert conn.total_changes == before and not conn.in_transaction
