"""Draft adapter composition regressions; not historical API-absence reds."""
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import current_attempt, ensure_schema
from orze.engine import native_evaluation
from orze.engine.accounting import record_compute_start
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.idea_lake import IdeaLake


@pytest.fixture
def case(tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    idea = "idea-native-watch"
    folder = tmp_path / idea
    lake.insert(idea, "Native watch", "{}", "", status="queued")
    assert lake.record_state_transition(idea, "QUEUED", "CLAIMED")
    assert lake.record_state_transition(idea, "CLAIMED", "IN_PROGRESS")
    lake.conn.execute("BEGIN IMMEDIATE")
    ensure_schema(lake.conn)
    lake.conn.commit()
    try:
        yield lake, idea, folder
    finally:
        lake.close()


def _snapshot(lake):
    return {table: [tuple(row) for row in lake.conn.execute(
        f"SELECT * FROM main.{table} ORDER BY rowid")]
        for table in ("ideas", "idea_state", "idea_transitions", "idea_stage_state",
                      "idea_stage_transitions", "execution_attempts")}


def test_native_begin_rejects_attempt_insert_trigger_changing_prior_lifecycle(case):
    lake, idea, folder = case
    lake.conn.execute(
        "CREATE TRIGGER corrupt_launch AFTER INSERT ON execution_attempts BEGIN "
        "UPDATE idea_state SET current_state='FAILED' WHERE idea_id=NEW.task_id; "
        "UPDATE ideas SET status='failed' WHERE idea_id=NEW.task_id; END")
    lake.conn.commit()
    before = _snapshot(lake)
    with pytest.raises((AttemptEffectBusy, AttemptEffectInDoubt)):
        native_evaluation.begin(lake, folder, "attempt-A", 0)
    assert _snapshot(lake) == before
    assert current_attempt(lake.conn, idea, "evaluation") is None
    assert not lake.conn.in_transaction


def test_native_started_rejects_running_trigger_changing_prior_stage(case):
    lake, idea, folder = case
    ref = native_evaluation.begin(lake, folder, "attempt-A", 0)
    ep = SimpleNamespace(idea_id=idea, attempt_id=ref.attempt_id, attempt_ref=ref,
                         gpu=0, start_time=1234.5, process=SimpleNamespace(pid=12345))
    lake.conn.execute(
        "CREATE TRIGGER corrupt_started AFTER UPDATE ON execution_attempts "
        "WHEN NEW.state='RUNNING' BEGIN UPDATE idea_stage_state "
        "SET current_state='FAILED' WHERE idea_id=NEW.task_id AND stage='evaluation'; END")
    lake.conn.commit()
    before = _snapshot(lake)
    with pytest.raises((AttemptEffectBusy, AttemptEffectInDoubt)):
        native_evaluation.started(lake, ep, folder, record_compute_start)
    assert _snapshot(lake) == before
    assert current_attempt(lake.conn, idea, "evaluation")["state"] == "LAUNCHING"
    assert (folder / "_compute_receipts" / ref.attempt_id / "start.json").is_file()
    assert not lake.conn.in_transaction
