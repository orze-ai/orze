"""Real SQLite metadata mechanisms; no process/source-byte proof is claimed.

Source artifacts are registered metadata, not simulated completed workers.
Native B integration must independently qualify their real closure and bytes.
"""
import copy
import json
import sqlite3

import pytest

from orze.core.cpu_observation_contract import cpu_observation_binding, cpu_observation_records
from orze.core.execution_attempts import (
    AttemptAuthorityError, create_attempt, finish_attempt, mark_running, require_current,
)
from orze.core import research_observations as store
from orze.core.research_artifacts import register_artifacts
from test_research_observation_store import artifact, artifact_binding


@pytest.fixture
def project(tmp_path):
    conn = sqlite3.connect(tmp_path / "catalog.db")
    conn.execute("BEGIN IMMEDIATE")
    sources = []
    for index in (1, 2):
        publication = artifact_binding(tmp_path, str(index) * 64, "result")
        ref = create_attempt(conn, "source-" + str(index), "action", "source-run-" + str(index),
                             {"artifact_publication": publication})
        mark_running(conn, ref)
        record = artifact(ref, publication, "input-" + str(index), "result")
        register_artifacts(conn, ref, [record])
        finish_attempt(conn, ref, {"outcome": "completed", "artifact_ids": [record["artifact_id"]]})
        sources.append(record)
    binding = cpu_observation_binding(adapter_id="domain.analysis.v1", protocol_fingerprint="c" * 64,
        spec_fingerprint="d" * 64, scope=str(tmp_path / "results"), input_artifacts=sources)
    result_publication = artifact_binding(tmp_path, "e" * 64, "measurements")
    ref = create_attempt(conn, "analysis", "action", "analysis-run", {
        "artifact_publication": result_publication, "observation_publication": binding})
    mark_running(conn, ref)
    output = artifact(ref, result_publication, "analysis-result", "measurements")
    register_artifacts(conn, ref, [output])
    conn.commit()
    yield conn, ref, binding, sources, output
    conn.close()


def records(ref, binding, count=2):
    claims = [{"name": "view-" + str(i), "values": {"zero": 0, "negative": -3.5},
        "validation": {"status": "valid" if i == 0 else "unknown", "reason_code": "domain_reported"},
        "comparison_scope": "protocol-A" if i == 0 else None} for i in range(count)]
    return cpu_observation_records(ref, binding, ["analysis-result"], claims)


def replace_publication(conn, ref, binding):
    current = require_current(conn, ref)
    updated = {**current["binding"], "observation_publication": binding}
    conn.execute("UPDATE execution_attempts SET binding_json=? WHERE attempt_id=?",
                 (json.dumps(updated, sort_keys=True, separators=(",", ":")), ref.attempt_id))


def test_mixed_specs_register_exact_batch_and_duplicate_has_no_write(project):
    conn, ref, binding, sources, output = project
    batch = records(ref, binding)
    conn.execute("BEGIN IMMEDIATE")
    assert store.register_observations(conn, ref, batch) == tuple(r["observation_id"] for r in batch)
    changes = conn.total_changes
    assert store.register_observations(conn, ref, batch) == tuple(r["observation_id"] for r in batch)
    assert conn.total_changes == changes
    assert store.observations_for_attempt(conn, ref) == batch
    assert len({sources[0]["spec_fingerprint"], sources[1]["spec_fingerprint"],
                output["spec_fingerprint"], binding["spec_fingerprint"]}) == 4
    conn.commit()
    assert [r["validation"]["status"] for r in store.observations_for_attempt(conn, ref)] == ["valid", "unknown"]


def test_zero_and_full_32_batch_are_valid_but_33_is_not(project):
    conn, ref, binding, _, _ = project
    conn.execute("BEGIN IMMEDIATE")
    assert store.register_observations(conn, ref, []) == ()
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='research_observations'").fetchall() == []
    batch = records(ref, binding, 32)
    assert len(store.register_observations(conn, ref, batch)) == 32
    before = conn.total_changes
    with pytest.raises(ValueError): store.register_observations(conn, ref, batch + [batch[0]])
    assert conn.total_changes == before
    conn.rollback()


def test_v1_single_subject_constraint_is_not_relaxed(project):
    conn, ref, binding, _, _ = project
    old = {key: value for key, value in binding.items() if key not in {"version", "input_artifact_bindings"}}
    old["spec_fingerprint"] = "1" * 64
    conn.execute("BEGIN IMMEDIATE")
    replace_publication(conn, ref, old)
    with pytest.raises(store.ResearchObservationError, match="input_artifact_mismatch"):
        store.register_observations(conn, ref, [])
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='research_observations'").fetchall() == []
    conn.rollback()


@pytest.mark.parametrize("field", ["producer", "spec_fingerprint", "content_sha256", "scope", "missing"])
def test_input_catalog_drift_is_rejected_before_insert(project, field):
    conn, ref, binding, sources, _ = project
    conn.execute("BEGIN IMMEDIATE")
    wrong = copy.deepcopy(binding)
    if field in {"producer", "spec_fingerprint", "content_sha256"}:
        item = wrong["input_artifact_bindings"]["input-1"]
        if field == "producer": item[field]["generation"] = 2
        else: item[field] = "f" * 64
        replace_publication(conn, ref, wrong)
    elif field == "scope":
        original = copy.deepcopy(sources[0]); original["scope"] = "/foreign"
        conn.execute("UPDATE research_artifacts SET record_json=? WHERE artifact_id='input-1'",
                     (json.dumps(original, sort_keys=True, separators=(",", ":")),))
    else: conn.execute("DELETE FROM research_artifacts WHERE artifact_id='input-1'")
    before = conn.total_changes
    with pytest.raises(store.ResearchObservationError, match="input_artifact_mismatch"):
        store.register_observations(conn, ref, records(ref, wrong))
    assert conn.total_changes == before
    conn.rollback()


@pytest.mark.parametrize("fault", ["foreign_result", "duplicate_name", "duplicate_id", "record_binding", "stale"])
def test_whole_batch_authority_and_identity_refuse_without_partial_records(project, fault):
    conn, ref, binding, _, _ = project
    batch = records(ref, binding)
    conn.execute("BEGIN IMMEDIATE")
    if fault == "foreign_result": batch[1]["result_artifact_ids"] = ["input-1"]
    elif fault == "duplicate_name": batch[1]["name"] = batch[0]["name"]
    elif fault == "duplicate_id": batch[1]["observation_id"] = batch[0]["observation_id"]
    elif fault == "record_binding": batch[1]["input_artifact_bindings"]["input-1"]["content_sha256"] = "f" * 64
    else:
        finish_attempt(conn, ref, {"outcome": "completed"})
        create_attempt(conn, ref.task_id, ref.phase, "replacement", {})
    before = conn.total_changes
    with pytest.raises((ValueError, AttemptAuthorityError)):
        store.register_observations(conn, ref, batch)
    assert conn.total_changes == before
    assert store.observations_for_attempt(conn, ref) == []
    conn.rollback()


def test_insert_trigger_dependency_drift_requires_whole_caller_rollback(project):
    conn, ref, binding, _, _ = project
    conn.execute("BEGIN IMMEDIATE")
    batch = records(ref, binding)
    store.register_observations(conn, ref, [])
    conn.execute(store._SQL)
    conn.execute("CREATE TABLE marker(value)")
    conn.execute("CREATE TRIGGER mutate_input AFTER INSERT ON research_observations BEGIN "
                 "DELETE FROM research_artifacts WHERE artifact_id='input-1'; END")
    conn.execute("INSERT INTO marker VALUES (1)")
    with pytest.raises(ValueError): store.register_observations(conn, ref, batch)
    assert conn.in_transaction
    conn.rollback()
    assert store.observations_for_attempt(conn, ref) == []
    assert conn.execute("SELECT COUNT(*) FROM research_artifacts WHERE artifact_id='input-1'").fetchone()[0] == 1
