"""CPU grant separation: real target SQLite/claim/budget/worker boundaries.

The schema-1 source below is an explicit closed training-metadata fixture,
not a claim of native training or GPU execution. It exercises the existing
request store; that wrong-resource mapping must never authorize a CPU worker.
"""
from dataclasses import asdict
import json

import pytest

from orze.core.execution_attempts import create_attempt, current_attempt, finish_attempt, mark_running
from orze.core.replication_requests import digest, insert_request, request_for_task, seal_record
from orze.engine import native_cpu_action as native
from test_native_cpu_action import context, _finish


def _training_mapping(lake, results):
    conn = lake.conn
    conn.execute("BEGIN IMMEDIATE")
    ref = create_attempt(conn, "training-metadata-source", "training", "source-metadata-attempt", {})
    mark_running(conn, ref)
    finish_attempt(conn, ref, {"fixture": "closed metadata only; no GPU/process claim"})
    row = current_attempt(conn, ref.task_id, ref.phase)
    database = next(row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main")
    record = seal_record({
        "schema": 1, "request_id": "wrong-resource-request", "task_id": "idea-cpu",
        "source_ref": asdict(ref), "scope": str(results), "database": database,
        "source_config_sha256": "1" * 64, "source_file_sha256": "2" * 64,
        "source_row_sha256": digest(row), "source_terminal_sha256": digest(row["terminal"]),
        "artifact_records_sha256": "3" * 64, "execution_identity": "4" * 64,
        "spec_fingerprint": "5" * 64, "artifact_binding_sha256": "6" * 64,
        "reason": "controlled wrong-resource mapping", "created_at": "2026-09-11T00:00:00Z",
    })
    insert_request(conn, record)
    conn.commit()
    assert request_for_task(conn, "idea-cpu") == record
    return record


@pytest.mark.parametrize("training_mapping", [False, True])
def test_cpu_target_never_consumes_training_replication_mapping(context, training_mapping):
    lake, results, scope, cfg, create, processes = context
    marker = results.parent / "actual-cpu-executed"
    action, permit = create("from pathlib import Path; Path(" + repr(str(marker)) + ").write_text('executed')")
    if training_mapping:
        _training_mapping(lake, results)
    error = None
    terminal = None
    try:
        handle = native.launch("idea-cpu", results, cfg, lake=lake, action=action,
                               permit=permit, admission=lambda: None)
    except native.CPUActionHOLD as exc:
        error = exc
    else:
        terminal = _finish(handle, results, cfg, lake, permit)
    print(json.dumps({"training_mapping": training_mapping, "worker_marker": marker.exists(),
                      "returned_terminal": terminal is not None,
                      "terminal_outcome": None if terminal is None else terminal["outcome"],
                      "captured_real_processes": len(processes), "held": error is not None}))
    if training_mapping:
        assert error is not None, "CPU native launch accepted an existing schema-1 training request"
        assert not marker.exists()
        assert not processes
        assert current_attempt(lake.conn, "idea-cpu", "action") is None
    else:
        assert error is None
        assert marker.read_text() == "executed"
        assert terminal["outcome"] == "completed"
        assert terminal["process_tree"]["kind"] == "TREE_CLOSED"
