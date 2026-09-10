"""A later stage write must not invalidate an earlier stage receipt."""
import pytest

from orze.idea_lake import IdeaLake


@pytest.mark.parametrize("target", ["state", "audit"])
def test_global_transition_rechecks_each_stage_after_all_pipeline_writes(tmp_path, target):
    lake = IdeaLake(tmp_path / "ideas.db")
    try:
        idea = "idea-cross-stage"
        lake.insert(idea, "Cross stage", "{}", "", status="queued")
        assert lake.record_state_transition(idea, "QUEUED", "CLAIMED")
        mutation = (
            "UPDATE idea_stage_state SET current_state='FAILED' "
            "WHERE idea_id=NEW.idea_id AND stage='training';"
            if target == "state" else
            "UPDATE idea_stage_transitions SET reason='changed-receipt' "
            "WHERE idea_id=NEW.idea_id AND stage='training';"
        )
        lake.conn.execute(
            "CREATE TRIGGER late_stage AFTER INSERT ON idea_stage_transitions "
            "WHEN NEW.stage='evaluation' BEGIN " + mutation + " END")
        lake.conn.commit()
        tables = ("ideas", "idea_state", "idea_transitions",
                  "idea_stage_state", "idea_stage_transitions")
        before = {table: [tuple(row) for row in lake.conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid")] for table in tables}

        assert lake.record_state_transition(
            idea, "CLAIMED", "IN_PROGRESS", "expected") is False
        assert not lake.conn.in_transaction
        assert {table: [tuple(row) for row in lake.conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid")] for table in tables} == before
    finally:
        lake.close()
