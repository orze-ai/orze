"""New recorded-evidence view mechanisms, not source/execution permissions."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from orze.core.execution_attempts import create_attempt, mark_running, finish_attempt
from orze.core.cpu_observation_contract import cpu_observation_binding, cpu_observation_records
from orze.core.research_artifacts import register_artifacts
from orze.core.research_observations import register_observations
from orze.engine import cpu_policy_evidence as evidence
from orze.engine import native_cpu_action as native
from orze.engine.artifact_publication import prepare_artifacts, verify_prepared_artifacts
from orze.engine.execution_authority import execution_transaction
from orze.engine.execution_catalog import bind_catalog
from orze.idea_lake import IdeaLake
from test_native_cpu_action import context, _finish
from test_cpu_action_sources import published


def test_empty_window_is_read_only_and_limits_are_exact(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    try:
        before = lake.conn.total_changes
        assert evidence.recorded_evidence(lake, results) == {
            "results": [], "unavailable": [], "more_available": False}
        assert lake.conn.total_changes == before
        assert lake.conn.execute("SELECT name FROM sqlite_master WHERE name='execution_attempts'").fetchall() == []
        for limit in (True, 0, 33, 1.0):
            with pytest.raises(ValueError):
                evidence.recorded_evidence(lake, results, limit=limit)
        lake.conn.execute("BEGIN")
        try:
            with pytest.raises(evidence.PolicyEvidenceHOLD):
                evidence.recorded_evidence(lake, results)
        finally:
            lake.conn.rollback()
    finally:
        lake.close()


def test_actual_confirmed_negative_cpu_result_is_not_a_zero_score(context):
    lake, results, _, cfg, create, _ = context
    action, permit = create("raise SystemExit(7)")
    handle = native.launch("idea-cpu", results, cfg, lake=lake, action=action,
                           permit=permit, admission=lambda: None)
    assert _finish(handle, results, cfg, lake, permit)["return_code"] == 7
    before = lake.conn.total_changes
    actual = evidence.recorded_evidence(lake, results)
    assert actual == {"results": [{"ref": asdict(handle.attempt_ref), "outcome": "failed",
        "artifact_records": [], "observation_records": []}], "unavailable": [], "more_available": False}
    assert lake.conn.total_changes == before


def test_peer_results_are_fresh_recorded_metadata_not_current_file_hashes(published, monkeypatch):
    lake, results, records, _ = published
    # Intentionally no current-content claim: the recorded digest is retained.
    Path(records[0]["path"]).write_bytes(b"changed after publication")
    from orze.engine import cpu_action_sources
    monkeypatch.setattr(cpu_action_sources, "_content", lambda *_: pytest.fail("large source hash in policy view"))
    peer = IdeaLake(lake.db_path)
    try:
        before = peer.conn.total_changes
        first = evidence.recorded_evidence(peer, results, limit=1)
        assert len(first["results"]) == 1 and first["more_available"] is True
        all_results = evidence.recorded_evidence(peer, results)
        assert len(all_results["results"]) == 2 and all_results["more_available"] is False
        assert [item["artifact_records"][0] for item in all_results["results"]] == records
        assert peer.conn.total_changes == before
    finally:
        peer.close()


def _recorded_observations(tmp_path, *, large):
    """Real metadata/effect transaction, explicitly no worker/closure claim.

    This small store fixture checks lossless observation reporting; the separate
    negative test above establishes the actual native CPU producer route.
    """
    results = tmp_path / "results"
    folder = results / "analysis"
    folder.mkdir(parents=True)
    lake = IdeaLake(tmp_path / "lake.db")
    publication = {"contract": {"version": 1, "outputs": {
        "result": {"path": "result", "max_bytes": 32}}}, "root": str(tmp_path / "artifacts"),
        "scope": str(results), "spec_fingerprint": "a" * 64}
    observations = cpu_observation_binding(adapter_id="test.metadata", protocol_fingerprint="b" * 64,
        spec_fingerprint="c" * 64, scope=str(results), input_artifacts=[])
    with execution_transaction(lake, folder) as tx:
        bind_catalog(lake, folder, tx.lease)
        ref = create_attempt(tx.conn, "analysis", "action", "analysis-attempt", {
            "artifact_publication": publication, "observation_publication": observations})
        mark_running(tx.conn, ref)
    work = folder / "_action_attempts" / ref.attempt_id / "work"
    work.mkdir(parents=True)
    (work / "result").write_bytes(b"declared metadata fixture")
    prepared = prepare_artifacts(ref, folder, publication, source_dir=work)
    artifacts = list(verify_prepared_artifacts(prepared, ref, folder, publication, source_dir=work))
    claims = [{"name": "measurement-" + str(index),
        "values": {"value": -(index + 1), **({"payload": "x" * 24000} if large else {})},
        "validation": {"status": status, "reason_code": "reported"}, "comparison_scope": None}
        for index, status in enumerate(("valid", "invalid", "unknown"))]
    batch = cpu_observation_records(ref, observations, [artifacts[0]["artifact_id"]], claims)
    with execution_transaction(lake, folder) as tx:
        ids = register_artifacts(tx.conn, ref, artifacts)
        observation_ids = register_observations(tx.conn, ref, batch)
        digest = tx.prepare(ref, {"operation": "metadata_view_fixture", "outcome": "completed"})
        finish_attempt(tx.conn, ref, {"outcome": "completed", "artifact_ids": list(ids),
            "observation_ids": list(observation_ids), "effect_receipt_sha256": digest})
    return lake, results, ref, batch


@pytest.mark.parametrize("large", [False, True])
def test_observation_claims_are_lossless_or_explicitly_unavailable(tmp_path, large):
    lake, results, ref, batch = _recorded_observations(tmp_path, large=large)
    try:
        actual = evidence.recorded_evidence(lake, results)
        if large:
            assert actual == {"results": [], "unavailable": [
                {"ref": asdict(ref), "reason": "evidence_result_limit"}], "more_available": True}
        else:
            assert actual["results"][0]["observation_records"] == batch
            assert [record["validation"]["status"] for record in batch] == ["valid", "invalid", "unknown"]
            assert actual["unavailable"] == []
    finally:
        lake.close()


@pytest.mark.parametrize("fault", ["ids", "effect", "between_reads"])
def test_unconfirmed_or_changed_record_is_explicitly_unavailable(published, monkeypatch, fault):
    lake, results, records, handles = published
    ref = handles[0].attempt_ref
    if fault == "ids":
        lake.conn.execute("UPDATE execution_attempts SET terminal_json=json_set(terminal_json,"
            "'$.artifact_ids',json('[]')) WHERE attempt_id=?", (ref.attempt_id,))
        lake.conn.commit()
    elif fault == "effect":
        path = results / ref.task_id / "_execution_effects" / ref.attempt_id / "committed.json"
        path.rename(path.with_name("saved-confirmation"))
    else:
        original = evidence._effect
        def change(*args):
            proof = original(*args)
            if args[0] == ref:
                lake.conn.execute("UPDATE research_artifacts SET record_json=json_set(record_json,"
                    "'$.content_sha256',?) WHERE artifact_id=?", ("0" * 64, records[0]["artifact_id"]))
                lake.conn.commit()
            return proof
        monkeypatch.setattr(evidence, "_effect", change)
    actual = evidence.recorded_evidence(lake, results)
    assert [item["ref"] for item in actual["results"]] == [asdict(handles[1].attempt_ref)]
    assert [item["ref"] for item in actual["unavailable"]] == [asdict(ref)]
    assert actual["unavailable"][0]["reason"]


def test_replaced_current_phase_does_not_adopt_historical_terminal(published):
    lake, results, _, handles = published
    lake.conn.execute("BEGIN IMMEDIATE")
    create_attempt(lake.conn, "source-0", "action", "new-current-attempt", {})
    lake.conn.commit()
    actual = evidence.recorded_evidence(lake, results)
    assert [item["ref"] for item in actual["results"]] == [asdict(handles[1].attempt_ref)]
    assert actual["unavailable"] == [] and actual["more_available"] is False
