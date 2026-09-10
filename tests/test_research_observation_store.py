"""New B2 SQLite mechanism acceptance; missing APIs are not old behavior reds."""
import copy
import hashlib
import sqlite3
from dataclasses import asdict

import pytest

from orze.core import research_observations as store
from orze.core.execution_attempts import (
    AttemptAuthorityError, AttemptRef, create_attempt, finish_attempt, mark_running,
)
from orze.core.research_artifacts import register_artifacts


def artifact_binding(tmp_path, spec, name):
    return {"contract": {"version": 1, "outputs": {name: {"path": name, "max_bytes": 1024}}},
            "root": str(tmp_path / "archive"), "scope": str(tmp_path / "results"),
            "spec_fingerprint": spec}


def artifact(ref, binding, identity, name):
    return {"schema": 1, "artifact_id": identity, "producer": asdict(ref),
            "spec_fingerprint": binding["spec_fingerprint"], "scope": binding["scope"],
            "logical_name": name, "path": f"{binding['root']}/{identity}/content",
            "content_sha256": hashlib.sha256(b"result").hexdigest(), "size_bytes": 6}


def new_eval(conn, tmp_path, binding, identity="eval-a"):
    publication = artifact_binding(tmp_path, "b" * 64, "report")
    ref = create_attempt(conn, "task-a", "generic-cpu-check", identity, {
        "observation_publication": binding, "artifact_publication": publication})
    mark_running(conn, ref)
    result_id = "artifact-" + identity
    register_artifacts(conn, ref, [artifact(ref, publication, result_id, "report")])
    return ref, result_id


def observation(ref, binding, result_id, *, identity="observation-a", name="quality"):
    return {"schema": 1, "observation_id": identity, "evaluator": asdict(ref),
            **copy.deepcopy(binding), "result_artifact_ids": [result_id], "name": name,
            "values": {"zero": 0, "negative": -2.5, "notes": [True, None, "雪"]},
            "validation": {"status": "valid", "reason_code": "adapter_checked"},
            "comparison_scope": "fixture_protocol_v1"}


@pytest.fixture
def project(tmp_path):
    conn = sqlite3.connect(tmp_path / "catalog.db")
    conn.execute("BEGIN IMMEDIATE")
    publication = artifact_binding(tmp_path, "a" * 64, "model")
    source = create_attempt(conn, "task-a", "training", "train-a", {"artifact_publication": publication})
    mark_running(conn, source)
    register_artifacts(conn, source, [artifact(source, publication, "artifact-input", "model")])
    finish_attempt(conn, source, {"outcome": "completed", "artifact_ids": ["artifact-input"]})
    binding = {"adapter_id": "cpu_check", "protocol_fingerprint": "c" * 64,
               "spec_fingerprint": "a" * 64, "scope": publication["scope"],
               "input_artifact_ids": ["artifact-input"]}
    ref, result_id = new_eval(conn, tmp_path, binding)
    conn.commit()
    yield conn, ref, binding, result_id, tmp_path
    conn.close()


def test_missing_history_reads_no_ddl_or_transactions():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("PRAGMA query_only=ON")
        assert store.observations_for_attempt(conn, AttemptRef("task", "cpu", "attempt", 1)) == []
        assert store.get_observation(conn, "observation") is None
        assert conn.execute("SELECT * FROM sqlite_master").fetchall() == []
        assert not conn.in_transaction
    finally:
        conn.close()


def test_caller_transaction_and_whole_rollback(project):
    conn, ref, binding, result_id, _ = project
    value = observation(ref, binding, result_id)
    with pytest.raises(store.ResearchObservationError, match="caller_transaction"):
        store.register_observations(conn, ref, [value])
    conn.execute("CREATE TABLE marker(value)")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("INSERT INTO marker VALUES (1)")
    statements = []
    conn.set_trace_callback(statements.append)
    assert store.register_observations(conn, ref, [value]) == ("observation-a",)
    assert conn.in_transaction
    assert not any(sql.split()[0].upper() in {"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT"}
                   for sql in statements)
    conn.set_trace_callback(None)
    conn.rollback()
    assert store.observations_for_attempt(conn, ref) == []
    assert conn.execute("SELECT * FROM marker").fetchall() == []


def test_zero_observations_are_not_zero_scores_or_a_table(project):
    conn, ref, _, _, _ = project
    conn.execute("BEGIN IMMEDIATE")
    assert store.register_observations(conn, ref, []) == ()
    assert store.observations_for_attempt(conn, ref) == []
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='research_observations'").fetchall() == []
    assert conn.in_transaction
    conn.rollback()


def test_multiple_named_results_preserve_validation_and_numeric_values(project):
    conn, ref, binding, result_id, _ = project
    values = []
    for index, status in enumerate(("valid", "invalid", "unknown")):
        value = observation(ref, binding, result_id, identity=f"observation-{index}", name=f"view-{index}")
        value["validation"]["status"] = status
        value["comparison_scope"] = None if status == "unknown" else "protocol-" + str(index)
        values.append(value)
    conn.execute("BEGIN IMMEDIATE")
    assert store.register_observations(conn, ref, values) == tuple(v["observation_id"] for v in values)
    assert store.observations_for_attempt(conn, ref) == values
    # API does not turn differing comparison scopes into a tie/comparison.
    assert [v["validation"]["status"] for v in values] == ["valid", "invalid", "unknown"]
    assert store.get_observation(conn, "observation-0")["values"]["zero"] == 0
    assert store.get_observation(conn, "observation-0")["values"]["negative"] == -2.5
    conn.rollback()


def test_detached_metadata_and_exact_replay_no_write(project):
    conn, ref, binding, result_id, _ = project
    value = observation(ref, binding, result_id)
    detached = store.validate_observation_record(value)
    detached["values"]["zero"] = 88
    assert value["values"]["zero"] == 0
    conn.execute("BEGIN IMMEDIATE")
    store.register_observations(conn, ref, [value])
    before = conn.total_changes
    store.register_observations(conn, ref, [copy.deepcopy(value)])
    assert conn.total_changes == before
    detached = store.get_observation(conn, "observation-a")
    detached["values"]["zero"] = 99
    assert store.get_observation(conn, "observation-a") == value
    for candidate in ({**value, "values": {"zero": 1}},
                      {**value, "observation_id": "replacement"}):
        with pytest.raises(store.ResearchObservationError, match="conflict"):
            store.register_observations(conn, ref, [candidate])
    assert conn.total_changes == before
    conn.rollback()


def test_new_attempt_same_subject_same_values_is_new_occurrence_and_old_is_readable(project):
    conn, old, binding, result_id, tmp_path = project
    conn.execute("BEGIN IMMEDIATE")
    first = observation(old, binding, result_id)
    store.register_observations(conn, old, [first])
    finish_attempt(conn, old, {"outcome": "failed", "observation_ids": ["observation-a"]})
    new, new_result_id = new_eval(conn, tmp_path, binding, "eval-b")
    second = observation(new, binding, new_result_id, identity="observation-b")
    store.register_observations(conn, new, [second])
    with pytest.raises(AttemptAuthorityError):
        store.register_observations(conn, old, [first])
    conn.commit()
    conn.execute("PRAGMA query_only=ON")
    assert store.observations_for_attempt(conn, old) == [first]
    assert store.observations_for_attempt(conn, new) == [second]
    assert first["values"] == second["values"]
    assert first["input_artifact_ids"] == second["input_artifact_ids"]
    assert first["evaluator"] != second["evaluator"]
    assert not conn.in_transaction


@pytest.mark.parametrize("change", [
    {"schema": True}, {"name": "not/a/name"}, {"values": []},
    {"scope": "/foreign"}, {"spec_fingerprint": "d" * 64},
    {"protocol_fingerprint": "d" * 64}, {"adapter_id": "foreign"},
    {"input_artifact_ids": []}, {"result_artifact_ids": []},
    {"result_artifact_ids": ["artifact-input"]},
    {"result_artifact_ids": ["missing"]},
    {"comparison_scope": " "}, {"comparison_scope": "雪" * 100},
    {"validation": {"status": "incomparable", "reason_code": "not_a_validity_status"}},
    {"extra": "not permitted"},
])
def test_invalid_or_unbound_records_never_insert(project, change):
    conn, ref, binding, result_id, _ = project
    value = observation(ref, binding, result_id)
    value.update(change)
    conn.execute("BEGIN IMMEDIATE")
    before = conn.total_changes
    with pytest.raises(ValueError):
        store.register_observations(conn, ref, [value])
    assert conn.total_changes == before and conn.in_transaction
    assert store.observations_for_attempt(conn, ref) == []
    conn.rollback()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), {1: "numeric-key"}, "雪" * 12000])
def test_values_are_bounded_finite_json(project, bad):
    conn, ref, binding, result_id, _ = project
    value = observation(ref, binding, result_id)
    value["values"] = {"value": bad}
    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(ValueError):
        store.register_observations(conn, ref, [value])
    assert store.observations_for_attempt(conn, ref) == []
    conn.rollback()


def test_depth_and_recursive_values_rejected_before_any_insert(project):
    conn, ref, binding, result_id, _ = project
    first = observation(ref, binding, result_id)
    second = observation(ref, binding, result_id, identity="observation-b", name="second")
    nested = {}
    for _ in range(16):
        nested = {"nested": nested}
    recursive = {}
    recursive["self"] = recursive
    conn.execute("BEGIN IMMEDIATE")
    for bad in (nested, recursive):
        second["values"] = bad
        before = conn.total_changes
        with pytest.raises(ValueError):
            store.register_observations(conn, ref, [first, second])
        assert conn.total_changes == before
        assert store.observations_for_attempt(conn, ref) == []
    conn.rollback()


def test_batch_duplicate_id_or_name_and_limit_fail_closed(project):
    conn, ref, binding, result_id, _ = project
    first = observation(ref, binding, result_id)
    variants = ([first, {**first, "name": "second"}],
                [first, {**first, "observation_id": "second"}], [first] * 33)
    conn.execute("BEGIN IMMEDIATE")
    for values in variants:
        with pytest.raises(ValueError):
            store.register_observations(conn, ref, values)
    assert store.observations_for_attempt(conn, ref) == []
    conn.rollback()


@pytest.mark.parametrize("trigger", [
    "BEFORE INSERT ON research_observations BEGIN SELECT RAISE(IGNORE); END",
    "AFTER INSERT ON research_observations BEGIN UPDATE research_observations SET name='changed'; END",
    "AFTER INSERT ON research_observations BEGIN DELETE FROM research_artifacts WHERE artifact_id='artifact-input'; END",
])
def test_write_or_dependency_change_requires_caller_rollback(project, trigger):
    conn, ref, binding, result_id, _ = project
    conn.execute(store._SQL)
    conn.execute("CREATE TRIGGER sabotage " + trigger)
    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(ValueError):
        store.register_observations(conn, ref, [observation(ref, binding, result_id)])
    assert conn.in_transaction
    conn.rollback()
    assert store.observations_for_attempt(conn, ref) == []
    assert conn.execute("SELECT artifact_id FROM research_artifacts WHERE artifact_id='artifact-input'").fetchone()


def test_sql_second_insert_error_preserves_caller_rollback(project):
    conn, ref, binding, result_id, _ = project
    conn.execute(store._SQL)
    conn.execute("CREATE TRIGGER second_error BEFORE INSERT ON research_observations "
                 "WHEN NEW.name='second' BEGIN SELECT RAISE(ABORT,'storage fault'); END")
    first = observation(ref, binding, result_id)
    second = observation(ref, binding, result_id, identity="second", name="second")
    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.Error):
        store.register_observations(conn, ref, [first, second])
    assert conn.in_transaction and store.observations_for_attempt(conn, ref) == [first]
    conn.rollback()
    assert store.observations_for_attempt(conn, ref) == []


def test_main_schema_and_binary_identities_not_temporary_shadow(project):
    conn, ref, binding, result_id, _ = project
    conn.execute("CREATE TEMP TABLE research_observations(name)")
    conn.execute("BEGIN IMMEDIATE")
    first = observation(ref, binding, result_id)
    second = observation(ref, binding, result_id, identity="Observation-a", name="Quality")
    store.register_observations(conn, ref, [first, second])
    assert store.get_observation(conn, "Observation-a") == second
    assert store.get_observation(conn, "observation-a") == first
    assert conn.execute("SELECT * FROM temp.research_observations").fetchall() == []
    conn.rollback()


def test_incompatible_schema_and_bad_stored_json_not_repaired(project):
    conn, ref, binding, result_id, _ = project
    conn.execute(store._SQL.replace("COLLATE BINARY", "COLLATE NOCASE", 1))
    with pytest.raises(ValueError, match="schema"):
        store.observations_for_attempt(conn, ref)
    conn.execute("DROP TABLE research_observations")
    conn.execute("BEGIN IMMEDIATE")
    store.register_observations(conn, ref, [observation(ref, binding, result_id)])
    conn.execute("UPDATE research_observations SET record_json=?", ("x" * 32769,))
    conn.commit()
    changes = conn.total_changes
    with pytest.raises(ValueError, match="stored_record"):
        store.get_observation(conn, "observation-a")
    assert conn.total_changes == changes and not conn.in_transaction
