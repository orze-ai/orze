"""New action path/metadata mechanisms; no native command or GPU is run."""
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import pytest

from orze.core.cpu_action_contract import artifact_binding
from orze.core.execution_attempts import AttemptRef, create_attempt, mark_running
from orze.core.research_artifacts import artifacts_for_attempt, register_artifacts
from orze.core.research_observations import register_observations, observations_for_attempt
from orze.engine.artifact_publication import prepare_artifacts, verify_prepared_artifacts
from orze.engine.attempt_effect_lock import AttemptEffectBusy


def _action(outputs):
    return {"version": 1, "adapter": "command", "purpose": "Write a file",
            "inputs": {}, "command": ["not-executed"],
            "timeout_seconds": 1, "outputs": outputs}


def test_real_copy_and_registry_keep_action_ref_and_independent_inode(tmp_path):
    folder = tmp_path / "results" / "task-a"
    work = folder / "_action_attempts" / "attempt-a" / "work"
    work.mkdir(parents=True)
    source = work / "result.bin"
    source.write_bytes(b"original bytes")
    binding = artifact_binding({"_project_root": str(tmp_path)}, folder,
        _action({"result": {"path": "result.bin", "max_bytes": 64}}))
    conn = sqlite3.connect(tmp_path / "catalog.db")
    try:
        conn.execute("BEGIN IMMEDIATE")
        ref = create_attempt(conn, folder.name, "action", "attempt-a",
                             {"origin": "native_cpu_action", "artifact_publication": binding})
        mark_running(conn, ref)
        conn.commit()
        with source.open("r+b") as old_worker_fd:
            assert not conn.in_transaction
            prepared = prepare_artifacts(ref, folder, binding, source_dir=work)
            records = list(verify_prepared_artifacts(prepared, ref, folder, binding, source_dir=work))
            assert len(records) == 1 and records[0]["producer"] == asdict(ref)
            copied = Path(records[0]["path"])
            assert copied.stat().st_ino != source.stat().st_ino
            conn.execute("BEGIN IMMEDIATE")
            assert register_artifacts(conn, ref, records) == (records[0]["artifact_id"],)
            conn.commit()
            old_worker_fd.seek(0)
            old_worker_fd.write(b"changed bytes!")
            old_worker_fd.flush()
            assert copied.read_bytes() == b"original bytes"
            assert artifacts_for_attempt(conn, ref) == records
    finally:
        conn.close()


@pytest.mark.parametrize("location", ["default", "task-root", "other-attempt", "evaluation-work"])
def test_action_outputs_cannot_select_shared_or_other_attempt_paths(tmp_path, location):
    folder = tmp_path / "results" / "task-a"
    ref = AttemptRef(folder.name, "action", "attempt-a", 1)
    binding = artifact_binding({}, folder, _action({}))
    roots = {"default": None, "task-root": folder,
             "other-attempt": folder / "_action_attempts" / "attempt-b" / "work",
             "evaluation-work": folder / "_evaluation_attempts" / "attempt-a" / "work"}
    with pytest.raises(AttemptEffectBusy):
        prepare_artifacts(ref, folder, binding, source_dir=roots[location])
    assert not tmp_path.joinpath("results").exists()


def test_zero_artifacts_and_observations_need_no_output_file_or_registry_table(tmp_path):
    folder = tmp_path / "results" / "task-a"
    work = folder / "_action_attempts" / "attempt-a" / "work"
    publication = artifact_binding({}, folder, _action({}))
    observation = {"adapter_id": "orze.native_cpu_action.v1",
                   "protocol_fingerprint": "a" * 64,
                   "spec_fingerprint": publication["spec_fingerprint"],
                   "scope": publication["scope"], "input_artifact_ids": []}
    conn = sqlite3.connect(tmp_path / "catalog.db")
    try:
        conn.execute("BEGIN IMMEDIATE")
        ref = create_attempt(conn, folder.name, "action", "attempt-a",
            {"origin": "native_cpu_action", "artifact_publication": publication,
             "observation_publication": observation})
        mark_running(conn, ref)
        conn.commit()
        prepared = prepare_artifacts(ref, folder, publication, source_dir=work)
        assert json.loads(prepared.records_json) == []
        assert not folder.exists() and not Path(publication["root"]).exists()
        conn.execute("BEGIN IMMEDIATE")
        assert register_artifacts(conn, ref, []) == ()
        assert register_observations(conn, ref, []) == ()
        assert artifacts_for_attempt(conn, ref) == []
        assert observations_for_attempt(conn, ref) == []
        assert conn.execute("SELECT name FROM sqlite_master WHERE name IN "
                            "('research_artifacts','research_observations')").fetchall() == []
        conn.rollback()
    finally:
        conn.close()
