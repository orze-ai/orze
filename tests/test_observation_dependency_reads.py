"""Bounded per-call lookup reuse, preserving fresh publication revalidation."""
import pytest

from orze.core import research_observations as store
from test_research_observation_store import observation, project


def test_shared_artifact_metadata_is_read_once_per_pass_but_final_pass_is_fresh(project):
    conn, ref, binding, result_id, _ = project
    values = [observation(ref, binding, result_id, identity=f"observation-{i}", name=f"view-{i:02}")
              for i in range(32)]
    conn.execute("BEGIN IMMEDIATE")
    reads = []
    def trace(sql):
        normalized = " ".join(sql.split())
        if "FROM main.research_artifacts WHERE artifact_id=" in normalized:
            reads.append(normalized)
    conn.set_trace_callback(trace)
    store.register_observations(conn, ref, values)
    conn.set_trace_callback(None)
    # Two distinct artifacts, one initial read and one fresh post-write read.
    assert len(reads) == 4
    assert len(store.observations_for_attempt(conn, ref)) == 32
    conn.rollback()

    # A per-invocation cache must not turn into a cache across publication passes.
    conn.execute(store._SQL)
    original = conn.execute("SELECT record_json FROM research_artifacts WHERE artifact_id='artifact-input'").fetchone()[0]
    conn.execute("CREATE TRIGGER change_dependency AFTER INSERT ON research_observations "
                 "BEGIN UPDATE research_artifacts SET record_json="
                 "replace(record_json, '\"size_bytes\":6', '\"size_bytes\":7') "
                 "WHERE artifact_id='artifact-input'; END")
    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(store.ResearchObservationError, match="authority_changed"):
        store.register_observations(conn, ref, values)
    assert conn.in_transaction
    conn.rollback()
    assert store.observations_for_attempt(conn, ref) == []
    assert conn.execute("SELECT record_json FROM research_artifacts WHERE artifact_id='artifact-input'").fetchone()[0] == original
