"""New bounded metadata watches; SQLite fixtures do not claim OS closure."""
import copy
from dataclasses import asdict
import json

import pytest

from orze.core.cpu_observation_contract import cpu_observation_binding, cpu_observation_records
from orze.core.execution_attempts import AttemptAuthorityError, AttemptRef, create_attempt, mark_running
from orze.core.research_artifacts import register_artifacts
from orze.core.research_observations import observations_for_attempt, register_observations
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.cpu_action_sources import capture_sources, SourceHOLD
from orze.engine.execution_authority import execution_transaction
from orze.idea_lake import IdeaLake


@pytest.fixture
def observation_catalog(tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    folder = tmp_path / "results" / "idea-watch"
    folder.mkdir(parents=True)
    publication = cpu_observation_binding(adapter_id="test.domain", protocol_fingerprint="b" * 64,
        spec_fingerprint="a" * 64, scope=str(folder.parent), input_artifacts=[])
    artifacts = {"contract": {"version": 1, "outputs": {"value": {"path": "value", "max_bytes": 5}}},
        "root": str(tmp_path / "artifacts"), "scope": str(folder.parent), "spec_fingerprint": "c" * 64}
    with execution_transaction(lake, folder) as tx:
        ref = create_attempt(tx.conn, folder.name, "action", "attempt-watch",
            {"observation_publication": publication, "artifact_publication": artifacts})
        mark_running(tx.conn, ref)
        artifact = {"schema": 1, "artifact_id": "result-watch", "producer": asdict(ref),
            "logical_name": "value", "path": str(tmp_path / "artifacts" / "result-watch" / "content"),
            "content_sha256": "d" * 64, "size_bytes": 3, "scope": artifacts["scope"],
            "spec_fingerprint": artifacts["spec_fingerprint"]}
        register_artifacts(tx.conn, ref, [artifact])
    claims = ({"name": "metric", "values": {"value": -2},
        "validation": {"status": "valid", "reason_code": "domain_checked"},
        "comparison_scope": "domain-protocol"},)
    records = cpu_observation_records(ref, publication, [artifact["artifact_id"]], claims)
    yield lake, folder, ref, records
    lake.close()


def test_observation_watch_detaches_expected_set(observation_catalog):
    lake, folder, ref, records = observation_catalog
    supplied = copy.deepcopy(records)
    with execution_transaction(lake, folder) as tx:
        register_observations(tx.conn, ref, records)
        tx.watch_observations(ref, supplied)
        supplied[0]["values"]["value"] = 9
    assert observations_for_attempt(lake.conn, ref) == records


@pytest.mark.parametrize("case", ["foreign_ref", "too_many", "foreign_evaluator", "duplicate", "record_size"])
def test_observation_watch_bounds_without_foreign_write_authority(observation_catalog, case):
    lake, folder, ref, records = observation_catalog
    expected, watched = copy.deepcopy(records), ref
    if case == "foreign_ref":
        watched = AttemptRef("other-task", ref.phase, ref.attempt_id, ref.generation)
    elif case == "too_many":
        expected *= 33
    elif case == "foreign_evaluator":
        expected[0]["evaluator"]["attempt_id"] = "other-attempt"
    elif case == "duplicate":
        expected *= 2
    else:
        expected[0]["values"]["large"] = "x" * 32769
    with execution_transaction(lake, folder) as tx:
        with pytest.raises((AttemptAuthorityError, ValueError)):
            tx.watch_observations(watched, expected)
    assert observations_for_attempt(lake.conn, ref) == []


def test_empty_observation_watch_rejects_later_insert(observation_catalog):
    lake, folder, ref, records = observation_catalog
    with pytest.raises(AttemptAuthorityError, match="observations_changed"):
        with execution_transaction(lake, folder) as tx:
            tx.watch_observations(ref, [])
            register_observations(tx.conn, ref, records)
    assert observations_for_attempt(lake.conn, ref) == []


def test_observation_watch_checks_actual_postcommit_rewrite(observation_catalog):
    lake, folder, ref, records = observation_catalog
    with execution_transaction(lake, folder) as tx:
        register_observations(tx.conn, ref, records)
    original = lake.conn
    changed = copy.deepcopy(records[0])
    changed["values"]["value"] = 99
    calls = []

    class AfterCommit:
        def __getattr__(self, key):
            return getattr(original, key)

        def commit(self):
            original.commit()
            original.execute("UPDATE research_observations SET record_json=? WHERE observation_id=?",
                (json.dumps(changed, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
                 changed["observation_id"]))
            original.commit()
            calls.append("committed_observation_rewrite")

    lake.conn = AfterCommit()
    try:
        with pytest.raises(AttemptEffectInDoubt):
            with execution_transaction(lake, folder) as tx:
                tx.watch_observations(ref, records)
    finally:
        lake.conn = original
    assert calls == ["committed_observation_rewrite"]
    assert observations_for_attempt(lake.conn, ref) == [changed]


@pytest.mark.parametrize("case", ["same_lake", "foreign_lake", "plain_metadata"])
def test_source_watch_uses_strong_same_connection_capture(tmp_path, case):
    results = tmp_path / "results"
    folder = results / "idea-watch"
    folder.mkdir(parents=True)
    lake, other = IdeaLake(tmp_path / "lake.db"), IdeaLake(tmp_path / "other.db")
    try:
        prepared = capture_sources(other if case == "foreign_lake" else lake, results, [])
        if case == "plain_metadata":
            prepared = {"schema": 1, "inputs": []}
        with execution_transaction(lake, folder) as tx:
            if case == "same_lake":
                tx.watch_cpu_sources(prepared)
            else:
                with pytest.raises(SourceHOLD):
                    tx.watch_cpu_sources(prepared)
        assert not lake.conn.in_transaction
    finally:
        other.close()
        lake.close()
