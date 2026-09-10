"""Removing report configuration cannot downgrade a native protocol binding.

These metadata boundary tests use real catalog routing and a generic current
attempt record. Actual worker/lifecycle closure is covered by native tests.
"""
import pytest

from orze.core.execution_attempts import create_attempt, mark_running, finish_attempt
from orze.engine.attempt_effect_lock import attempt_effect_lock
from orze.engine.execution_catalog import bind_catalog
from orze.idea_lake import IdeaLake
from orze.reporting.evidence import qualify_result_artifacts


@pytest.mark.parametrize("mode", ["bound", "malformed_binding", "missing_catalog", "legacy_native"])
def test_stored_protocol_cannot_be_erased_by_omitting_current_config(tmp_path, mode):
    folder = tmp_path / "results" / "task"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text('{"status":"COMPLETED","score":999}')
    path = tmp_path / "catalog.db"
    lake = IdeaLake(path)
    try:
        with attempt_effect_lock(folder) as lease:
            bind_catalog(lake, folder, lease)
        binding = {"origin": "native_evaluation"}
        if mode != "legacy_native":
            binding["observation_publication"] = None if mode == "malformed_binding" else {
                "adapter_id": "orze.json_observations.v1", "protocol_fingerprint": "a" * 64,
                "spec_fingerprint": "b" * 64, "scope": str(folder.parent), "input_artifact_ids": []}
        lake.conn.execute("BEGIN IMMEDIATE")
        ref = create_attempt(lake.conn, "task", "evaluation", "eval-a", binding)
        mark_running(lake.conn, ref)
        finish_attempt(lake.conn, ref, {"outcome": "completed"})
        lake.conn.commit()
    finally:
        lake.close()
    if mode == "missing_catalog":
        path.rename(tmp_path / "archived-catalog.db")
    # Deliberately no observation_contract in the current display config.
    cfg = {"report": {"primary_metric": "score", "columns": [{"key": "score"}]}}
    _, _, value, reason = qualify_result_artifacts(folder, cfg)
    if mode == "legacy_native":
        assert value == 999
        assert reason == "local_artifacts_verified"
    else:
        assert value is None
        assert reason == ("observation_adapter_source_unverifiable" if mode == "missing_catalog"
                          else "observation_adapter_required")
