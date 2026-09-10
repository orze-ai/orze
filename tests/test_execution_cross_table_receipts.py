"""The final attempt write cannot invalidate an earlier lifecycle receipt."""
import pytest

from orze.core.execution_attempts import create_attempt, finish_attempt, mark_running
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.execution_authority import execution_transaction, lifecycle_fence
from orze.idea_lake import IdeaLake


def test_attempt_terminal_trigger_cannot_invalidate_previously_accepted_lifecycle(tmp_path):
    idea = "idea-cross-receipts"
    folder = tmp_path / idea
    lake = IdeaLake(tmp_path / "lake.db")
    try:
        lake.insert(idea, "Cross receipt", "{}", "", status="queued")
        assert lake.record_state_transition(idea, "QUEUED", "CLAIMED")
        assert lake.record_state_transition(idea, "CLAIMED", "IN_PROGRESS")
        with execution_transaction(lake, folder):
            ref = create_attempt(lake.conn, idea, "training", "attempt-A", {})
            mark_running(lake.conn, ref)
        lake.conn.execute(
            "CREATE TRIGGER late_attempt AFTER UPDATE ON execution_attempts "
            "WHEN NEW.state='TERMINAL' BEGIN "
            "UPDATE idea_state SET current_state='FAILED' WHERE idea_id=NEW.task_id; "
            "UPDATE ideas SET status='failed' WHERE idea_id=NEW.task_id; END")
        lake.conn.commit()
        tables = ("ideas", "idea_state", "idea_transitions", "idea_stage_state",
                  "idea_stage_transitions", "execution_attempts")
        before = {table: [tuple(row) for row in lake.conn.execute(
            f"SELECT * FROM main.{table} ORDER BY rowid")] for table in tables}

        with pytest.raises(AttemptEffectInDoubt):
            with execution_transaction(lake, folder) as tx:
                digest = tx.prepare(ref, {"expected_outcome": "completed"})
                assert lake._record_state_transition_in_tx(
                    idea, "IN_PROGRESS", "COMPLETE", "first accepted receipt")
                expected_lifecycle = lifecycle_fence(lake, idea, "training")
                assert finish_attempt(lake.conn, ref, {
                    "outcome": "completed", "effect_receipt_sha256": digest,
                    "lifecycle": expected_lifecycle,
                }) == "committed"

        assert not lake.conn.in_transaction
        assert {table: [tuple(row) for row in lake.conn.execute(
            f"SELECT * FROM main.{table} ORDER BY rowid")] for table in tables} == before
        assert (folder / "_execution_effects" / ref.attempt_id / "prepared.json").is_file()
        assert not (folder / "_execution_effects" / ref.attempt_id / "committed.json").exists()
        assert (folder / "_attempt_effect.lock" / "lock.json").is_file()
    finally:
        lake.close()
