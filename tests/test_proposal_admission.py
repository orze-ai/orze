"""New immutable admission mechanism; API absence is not an old behavior red."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from orze.idea_lake import IdeaLake


@pytest.fixture
def lake(tmp_path):
    instance = IdeaLake(str(tmp_path / "lake.db"))
    try:
        yield instance
    finally:
        instance.close()


def _admit(lake, idea_id="idea-a", **overrides):
    proposal = {
        "title": "Proposal", "config_yaml": "seed: 13\n", "raw_markdown": "source body",
        "status": "queued", "priority": "medium", "hypothesis": "test hypothesis",
        "if_absent": True,
    }
    proposal.update(overrides)
    return lake.insert(idea_id, **proposal)


def _snapshot(lake):
    return {
        table: [tuple(row) for row in lake.conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        for table in ("ideas", "idea_state", "idea_transitions", "idea_stage_state", "idea_stage_transitions")
    }


def test_new_proposal_has_one_audited_admission_and_real_queue_clocks(lake):
    result = _admit(lake)
    assert result["status"] == "inserted"
    assert lake.get("idea-a")["status"] == "queued"
    state = dict(lake.conn.execute("SELECT * FROM idea_state WHERE idea_id='idea-a'").fetchone())
    assert state["current_state"] == "QUEUED"
    assert state["first_queued_at"] == state["queued_at"] == state["updated_at"]
    assert state["started_at"] is state["terminal_at"] is state["completed_at"] is None
    history = lake.get_fsm_history("idea-a")
    assert len(history) == 1
    assert lake.conn.execute(
        "SELECT id FROM idea_transitions WHERE idea_id='idea-a'",
    ).fetchone()[0] == result["transition_id"]
    assert (history[0]["from_state"], history[0]["to_state"], history[0]["reason"]) == (
        "UNKNOWN", "QUEUED", "proposal_admitted",
    )


def test_exact_replay_after_execution_preserves_every_persisted_field(lake):
    assert _admit(lake)["status"] == "inserted"
    assert lake.record_state_transition("idea-a", "QUEUED", "CLAIMED", "test claim")
    assert lake.record_state_transition("idea-a", "CLAIMED", "IN_PROGRESS", "test start")
    assert lake.record_state_transition("idea-a", "IN_PROGRESS", "COMPLETE", "test finish")
    lake.conn.execute("UPDATE ideas SET eval_metrics=?, training_time=? WHERE idea_id=?",
                      ('{"score": 0}', 12.5, "idea-a"))
    lake.conn.commit()
    before = _snapshot(lake)
    assert _admit(lake)["status"] == "already_present_exact"
    assert _snapshot(lake) == before


@pytest.mark.parametrize("changed", [
    {"title": "Different"}, {"priority": "high"}, {"config_yaml": "seed: 99\n"},
    {"raw_markdown": "changed source"}, {"hypothesis": "different hypothesis"},
])
def test_same_id_requires_complete_source_identity_and_never_replaces(lake, changed):
    assert _admit(lake)["status"] == "inserted"
    before = _snapshot(lake)
    assert _admit(lake, **changed)["status"] == "conflict"
    assert _snapshot(lake) == before


def test_legacy_missing_hash_dedup_is_read_only_and_does_not_commit_a_repair(lake):
    lake.insert("idea-existing", "Legacy", "seed: 13\n", "legacy", status="queued")
    lake.conn.execute("UPDATE ideas SET config_hash=NULL, config_source_sha256=NULL")
    lake.conn.commit()
    before = _snapshot(lake)
    result = _admit(lake)
    assert result["status"] == "config_duplicate"
    assert result["existing_id"] == "idea-existing"
    assert _snapshot(lake) == before


def test_failed_config_does_not_block_a_new_task_under_existing_dedup_policy(lake):
    lake.insert("idea-failed", "Failed", "seed: 13\n", "failed", status="failed")
    assert _admit(lake)["status"] == "inserted"
    assert lake.get_all_ids() == {"idea-failed", "idea-a"}


@pytest.mark.parametrize("second_id,second_config,expected", [
    ("idea-a", "seed: 13\n", {"inserted", "already_present_exact"}),
    ("idea-a", "seed: 99\n", {"inserted", "conflict"}),
    ("idea-b", "seed: 13\n", {"inserted", "config_duplicate"}),
])
def test_two_connections_choose_one_winner_without_overwriting(lake, second_id, second_config, expected):
    barrier = threading.Barrier(2)

    def contend(idea_id, config_yaml):
        connection = IdeaLake(lake.db_path)
        try:
            barrier.wait(timeout=10)
            return _admit(connection, idea_id, config_yaml=config_yaml)
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(contend, "idea-a", "seed: 13\n")
        b = pool.submit(contend, second_id, second_config)
        results = [a.result(timeout=15), b.result(timeout=15)]
    assert {item["status"] for item in results} == expected
    assert lake.count() == 1
    assert len(lake.conn.execute("SELECT * FROM idea_state").fetchall()) == 1
    assert len(lake.conn.execute("SELECT * FROM idea_transitions").fetchall()) == 1
    inserted_index = next(index for index, item in enumerate(results) if item["status"] == "inserted")
    winner = lake.get(results[inserted_index]["idea_id"])
    assert winner["config"] == ("seed: 13\n" if inserted_index == 0 else second_config)


@pytest.mark.parametrize("table", ["ideas", "idea_state", "idea_transitions"])
def test_ignored_writes_roll_back_the_entire_admission(lake, table):
    lake.conn.executescript(f"""
        CREATE TRIGGER refuse_admission BEFORE INSERT ON {table}
        WHEN NEW.idea_id='idea-a' BEGIN SELECT RAISE(IGNORE); END;
    """)
    before = _snapshot(lake)
    result = _admit(lake)
    assert result["status"] == "rejected"
    assert _snapshot(lake) == before
    assert not lake.conn.in_transaction


def test_readback_detects_trigger_rewriting_new_state_and_rolls_back(lake):
    lake.conn.executescript("""
        CREATE TRIGGER corrupt_admission AFTER INSERT ON idea_state
        WHEN NEW.idea_id='idea-a'
        BEGIN UPDATE idea_state SET current_state='FAILED' WHERE idea_id=NEW.idea_id; END;
    """)
    before = _snapshot(lake)
    result = _admit(lake)
    assert result == {"status": "rejected", "reason": "proposal_readback_mismatch", "idea_id": "idea-a"}
    assert _snapshot(lake) == before


def test_active_caller_transaction_is_neither_committed_nor_rolled_back(lake):
    lake.conn.execute("BEGIN IMMEDIATE")
    lake.conn.execute("UPDATE id_sequence SET next_id=12345")
    result = _admit(lake)
    assert result["reason"] == "proposal_caller_transaction"
    assert lake.conn.in_transaction
    assert lake.conn.execute("SELECT next_id FROM id_sequence").fetchone()[0] == 12345
    assert lake.get("idea-a") is None
    lake.conn.rollback()
    assert lake.conn.execute("SELECT next_id FROM id_sequence").fetchone()[0] != 12345


def test_orphan_lifecycle_is_not_taken_over_as_a_new_task(lake):
    lake.insert("idea-a", "Removed", "seed: 44\n", "prior source", status="queued")
    lake.conn.execute("DELETE FROM ideas WHERE idea_id='idea-a'")
    lake.conn.commit()
    before = _snapshot(lake)
    assert _admit(lake)["reason"] == "proposal_orphan_lifecycle"
    assert _snapshot(lake) == before


def test_legacy_default_insert_still_replaces_as_an_explicit_import_update(lake):
    assert lake.insert("idea-a", "Old", "seed: 1\n", "old", status="queued") is None
    assert lake.insert("idea-a", "Updated", "seed: 2\n", "new", status="archived") is None
    assert lake.get("idea-a")["title"] == "Updated"
    assert lake.get("idea-a")["config"] == "seed: 2\n"


def test_nonqueued_admission_is_rejected_without_any_mutation(lake):
    before = _snapshot(lake)
    assert _admit(lake, status="completed")["reason"] == "proposal_requires_queued"
    assert _snapshot(lake) == before
