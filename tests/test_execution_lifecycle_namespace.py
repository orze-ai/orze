"""Native main-schema attempts cannot commit against TEMP lifecycle shadows.

This is a draft D2 composition regression, not an old-release API-absence red.
The shadow tables retain the real schema/PKs; CTAS would fail unrelated audit
readback and would not reproduce this namespace mismatch.
"""
import pytest

from orze.core.execution_attempts import (
    AttemptAuthorityError, create_attempt, finish_attempt, mark_running,
)
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.execution_authority import execution_transaction, lifecycle_fence
from orze.idea_lake import IdeaLake


def test_native_attempt_cannot_accept_terminal_lifecycle_written_only_to_temp(tmp_path):
    idea = "idea-shadow"
    folder = tmp_path / idea
    lake = IdeaLake(tmp_path / "lake.db")
    tables = ("ideas", "idea_state", "idea_transitions",
              "idea_stage_state", "idea_stage_transitions")
    try:
        lake.insert(idea, "Real task", "{}", "", status="queued")
        assert lake.record_state_transition(idea, "QUEUED", "CLAIMED")
        assert lake.record_state_transition(idea, "CLAIMED", "IN_PROGRESS")
        with execution_transaction(lake, folder):
            ref = create_attempt(lake.conn, idea, "training", "attempt-A", {})
            mark_running(lake.conn, ref)
        for table in tables:
            sql = lake.conn.execute(
                "SELECT sql FROM main.sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()[0]
            lake.conn.execute(sql.replace("CREATE TABLE", "CREATE TEMP TABLE", 1))
            lake.conn.execute(f"INSERT INTO temp.{table} SELECT * FROM main.{table}")
        lake.conn.commit()
        before = {(namespace, table): [tuple(row) for row in lake.conn.execute(
            f"SELECT * FROM {namespace}.{table} ORDER BY rowid")]
            for namespace in ("main", "temp") for table in tables}

        with pytest.raises((AttemptEffectBusy, AttemptEffectInDoubt)):
            with execution_transaction(lake, folder) as tx:
                digest = tx.prepare(ref, {"expected_outcome": "completed"})
                if not lake._record_state_transition_in_tx(
                        idea, "IN_PROGRESS", "COMPLETE", "native completion"):
                    raise AttemptAuthorityError("lifecycle_write_rejected")
                assert finish_attempt(lake.conn, ref, {
                    "outcome": "completed", "effect_receipt_sha256": digest,
                    "lifecycle": lifecycle_fence(lake, idea, "training"),
                }) == "committed"

        assert not lake.conn.in_transaction
        assert {(namespace, table): [tuple(row) for row in lake.conn.execute(
            f"SELECT * FROM {namespace}.{table} ORDER BY rowid")]
            for namespace in ("main", "temp") for table in tables} == before
        assert lake.conn.execute(
            "SELECT state FROM main.execution_attempts WHERE attempt_id=?",
            (ref.attempt_id,),
        ).fetchone()[0] == "RUNNING"
        assert not (folder / "_execution_effects" / ref.attempt_id / "committed.json").exists()
    finally:
        lake.close()
