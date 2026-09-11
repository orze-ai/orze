"""Real CPU publication followed by same-writer terminal-trigger corruption.

The SQLite observer only records that an actual registered artifact exists.
No fake closure, artifact record, lifecycle, process or budget is installed.
"""
import json

import pytest

from test_native_cpu_action import context, _finish
from orze.core import cpu_action_budget as budget
from orze.core.execution_attempts import current_attempt
from orze.engine import native_cpu_action as native


def test_terminal_trigger_cannot_change_registered_artifact_and_release_budget(context):
    lake, results, scope, cfg, create, handles = context
    action, permit = create("from pathlib import Path; Path('result.txt').write_text('owned output')",
        outputs={"result": {"path": "result.txt", "max_bytes": 128}})
    handle = native.launch("idea-cpu", results, cfg, lake=lake, action=action,
                           permit=permit, admission=lambda: None)
    observed = []
    lake.conn.create_function("observe_registered_artifact", 1,
                              lambda raw: observed.append(json.loads(raw)) or 0)
    lake.conn.executescript("""
        CREATE TRIGGER cpu_terminal_artifact_change
        AFTER UPDATE OF state ON execution_attempts
        WHEN NEW.state='TERMINAL' AND NEW.phase='action'
        BEGIN
          SELECT observe_registered_artifact(record_json)
          FROM research_artifacts WHERE attempt_id=NEW.attempt_id;
          UPDATE research_artifacts
          SET record_json=json_set(record_json,'$.content_sha256',
              '0000000000000000000000000000000000000000000000000000000000000000')
          WHERE attempt_id=NEW.attempt_id;
        END;
    """)
    with pytest.raises(native.CPUActionHOLD):
        _finish(handle, results, cfg, lake, permit)
    assert len(observed) == 1
    assert observed[0]["producer"]["attempt_id"] == handle.attempt_ref.attempt_id
    assert observed[0]["content_sha256"] != "0" * 64
    row = current_attempt(lake.conn, "idea-cpu", "action")
    assert row["state"] == "RUNNING" and row["terminal"] is None
    assert lake.get_fsm_state("idea-cpu") == "IN_PROGRESS"
    assert budget.snapshot(lake, scope)["active_reservations"] == 1
    assert budget.snapshot(lake, scope)["reserved_wall_seconds"] == 2
    assert handle.process.closure_receipt()["event"] == "TREE_CLOSED"
