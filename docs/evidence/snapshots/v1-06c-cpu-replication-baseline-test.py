"""Explicit CPU replication: real SQLite/native CPU, no scientific workloads."""
import hashlib
import json

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import native_cpu_action as native
from orze.engine import replication
from test_native_cpu_action import context, _finish


@pytest.fixture
def completed(context):
    lake, results, scope, cfg, create, handles = context
    cfg.update(execution={"version": 1, "resource": "cpu", "slots": 1,
                          "wall_budget_seconds": 20},
               results_dir=str(results), idea_lake_db=str(lake.db_path))
    action, permit = create("pass")
    owner = native.launch("idea-cpu", results, cfg, lake=lake, action=action,
                          permit=permit, admission=lambda: None)
    assert _finish(owner, results, cfg, lake, permit)["outcome"] == "completed"
    yield lake, results, cfg, action, owner.attempt_ref


def test_public_request_copies_completed_cpu_without_config_salt(completed):
    lake, results, cfg, action, ref = completed
    before = lake.conn.execute("SELECT config FROM ideas WHERE idea_id='idea-cpu'").fetchone()[0]
    result = replication.request_replication("idea-cpu", results, cfg, lake,
                                            request_id="cpu-public-request")
    assert result["status"] == "created"
    assert result["task_id"] != "idea-cpu"
    row = lake.conn.execute("SELECT config,kind,status FROM ideas WHERE idea_id=?",
                            (result["task_id"],)).fetchone()
    assert tuple(row) == (before, "native_cpu_action", "queued")
    assert lake.conn.execute("SELECT sop_type,current_state FROM idea_state WHERE idea_id=?",
                            (result["task_id"],)).fetchone()[:] == ("action", "QUEUED")
    assert current_attempt(lake.conn, result["task_id"], "action") is None
