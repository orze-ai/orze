"""Independent evaluation retry is one audited, training-preserving transaction.

Missing retry_evaluation is a missing mechanism, not a behavioral regression.
The legacy-path control records why ordinary FAILED -> QUEUED is not this API.
"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from orze.idea_lake import IdeaLake


IDEA_ID = "idea-evaluation-retry"


@pytest.fixture
def project(tmp_path):
    lake = IdeaLake(tmp_path / "ideas.db")
    lake.insert(
        IDEA_ID, "Preserve title", "seed: 7\n", "Preserve research notes",
        status="queued", kind="train", parent="idea-parent",
        hypothesis="Preserve hypothesis", training_time=123.5,
        created_at="2001-02-03T04:05:06", eval_metrics={"diagnostic": "old failure"},
    )
    assert lake.record_state_transition(IDEA_ID, "QUEUED", "CLAIMED")
    assert lake.record_state_transition(IDEA_ID, "CLAIMED", "IN_PROGRESS")
    assert lake.record_stage_transition(
        IDEA_ID, "training", "IN_PROGRESS", "COMPLETE", "training_completed",
    )
    folder = tmp_path / "results" / IDEA_ID
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text('{"status":"COMPLETED","quality":0}')
    (folder / "checkpoint.pt").write_bytes(b"preserve-generated-artifact\x00\xff")
    try:
        yield lake, folder
    finally:
        lake.close()


def _fail_evaluation(lake):
    assert lake.record_stage_transition(
        IDEA_ID, "evaluation", "PENDING", "IN_PROGRESS", "evaluation_launched",
    )
    assert lake.record_state_transition(
        IDEA_ID, "IN_PROGRESS", "FAILED", "evaluation_output_invalid",
    )
    assert lake.get_stage_state(IDEA_ID, "training") == "COMPLETE"
    assert lake.get_stage_state(IDEA_ID, "evaluation") == "FAILED"


def _snapshot(lake):
    return {
        table: [dict(row) for row in lake.conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid",
        ).fetchall()]
        for table in (
            "ideas", "idea_state", "idea_stage_state",
            "idea_transitions", "idea_stage_transitions",
        )
    }


def _assert_only_evaluation_reopened(lake, before, reason="evaluation_retry_requested"):
    after = _snapshot(lake)
    assert lake.get_fsm_state(IDEA_ID) == "IN_PROGRESS"
    assert lake.get_stage_state(IDEA_ID, "training") == "COMPLETE"
    assert lake.get_stage_state(IDEA_ID, "evaluation") == "PENDING"
    assert lake.get(IDEA_ID)["status"] == "running"
    old_idea = dict(before["ideas"][0])
    old_idea["status"] = "running"
    assert after["ideas"] == [old_idea]
    old_training = [row for row in before["idea_stage_state"] if row["stage"] == "training"]
    new_training = [row for row in after["idea_stage_state"] if row["stage"] == "training"]
    assert new_training == old_training
    for table in ("idea_transitions", "idea_stage_transitions"):
        assert after[table][:-1] == before[table]
        assert len(after[table]) == len(before[table]) + 1
        assert after[table][-1]["reason"] == reason
    assert (after["idea_transitions"][-1]["from_state"],
            after["idea_transitions"][-1]["to_state"]) == ("FAILED", "IN_PROGRESS")
    assert (after["idea_stage_transitions"][-1]["stage"],
            after["idea_stage_transitions"][-1]["from_state"],
            after["idea_stage_transitions"][-1]["to_state"]) == (
                "evaluation", "FAILED", "PENDING",
            )


def test_retry_evaluation_preserves_training_artifacts_metadata_and_old_history(project):
    lake, folder = project
    _fail_evaluation(lake)
    before = _snapshot(lake)
    artifacts = {name: (folder / name).read_bytes() for name in ("metrics.json", "checkpoint.pt")}

    assert lake.retry_evaluation(IDEA_ID, reason="operator_fixed_evaluator") is True

    _assert_only_evaluation_reopened(lake, before, "operator_fixed_evaluator")
    assert {name: (folder / name).read_bytes() for name in artifacts} == artifacts


def test_repeated_pending_retry_is_idempotent_without_new_clocks_or_history(project):
    lake, _ = project
    _fail_evaluation(lake)
    assert lake.retry_evaluation(IDEA_ID) is True
    after_first = _snapshot(lake)

    assert lake.retry_evaluation(IDEA_ID) is True

    assert _snapshot(lake) == after_first


def test_first_evaluation_pending_is_not_a_previously_requested_retry(project):
    lake, _ = project
    before = _snapshot(lake)

    assert lake.retry_evaluation(IDEA_ID) is False

    assert _snapshot(lake) == before


def test_retry_request_cannot_interrupt_an_already_active_evaluator(project):
    lake, _ = project
    _fail_evaluation(lake)
    assert lake.retry_evaluation(IDEA_ID) is True
    assert lake.record_stage_transition(
        IDEA_ID, "evaluation", "PENDING", "IN_PROGRESS", "evaluation_relaunched",
    )
    before = _snapshot(lake)

    assert lake.retry_evaluation(IDEA_ID) is False

    assert _snapshot(lake) == before


@pytest.mark.parametrize("ineligible", [
    "training_failed", "evaluation_skipped", "global_complete",
    "global_queued", "status_conflict", "missing_fsm",
])
def test_ineligible_or_conflicting_lifecycle_cannot_be_repaired_by_retry(project, ineligible):
    lake, _ = project
    _fail_evaluation(lake)
    if ineligible == "training_failed":
        lake.conn.execute("UPDATE idea_stage_state SET current_state='FAILED' WHERE stage='training'")
    elif ineligible == "evaluation_skipped":
        lake.conn.execute("UPDATE idea_stage_state SET current_state='SKIPPED' WHERE stage='evaluation'")
    elif ineligible == "global_complete":
        lake.conn.execute("UPDATE idea_state SET current_state='COMPLETE'")
        lake.conn.execute("UPDATE ideas SET status='completed'")
        lake.conn.execute("UPDATE idea_stage_state SET current_state='COMPLETE' WHERE stage='evaluation'")
    elif ineligible == "global_queued":
        assert lake.record_state_transition(IDEA_ID, "FAILED", "QUEUED", "ordinary_training_retry")
    elif ineligible == "status_conflict":
        lake.conn.execute("UPDATE ideas SET status='queued'")
    else:
        lake.conn.execute("DELETE FROM idea_state")
    lake.conn.commit()
    before = _snapshot(lake)

    assert lake.retry_evaluation(IDEA_ID) is False

    assert _snapshot(lake) == before


def test_unknown_idea_is_not_inserted_or_given_synthetic_retry_history(project):
    lake, _ = project
    before = _snapshot(lake)

    assert lake.retry_evaluation("idea-unknown") is False

    assert _snapshot(lake) == before


@pytest.mark.parametrize("failure_point", [
    "global_state", "evaluation_audit", "global_audit", "legacy_status",
])
def test_sql_failure_rolls_back_every_retry_state_and_audit_write(project, failure_point):
    lake, _ = project
    _fail_evaluation(lake)
    events = {
        "global_state": "BEFORE UPDATE ON idea_state WHEN NEW.current_state='IN_PROGRESS'",
        "evaluation_audit": "BEFORE INSERT ON idea_stage_transitions WHEN NEW.stage='evaluation' AND NEW.to_state='PENDING'",
        "global_audit": "BEFORE INSERT ON idea_transitions WHEN NEW.to_state='IN_PROGRESS'",
        "legacy_status": "BEFORE UPDATE ON ideas WHEN NEW.status='running'",
    }
    lake.conn.execute(
        f"CREATE TRIGGER reject_retry_write {events[failure_point]} "
        "BEGIN SELECT RAISE(ABORT, 'injected retry transaction failure'); END",
    )
    lake.conn.commit()
    before = _snapshot(lake)

    with pytest.raises(sqlite3.DatabaseError, match="injected retry transaction failure"):
        lake.retry_evaluation(IDEA_ID)

    assert lake.conn.in_transaction is False
    assert _snapshot(lake) == before


def test_concurrent_connections_request_one_retry_without_duplicate_history(project):
    lake, _ = project
    _fail_evaluation(lake)
    before = _snapshot(lake)
    gate = Barrier(2)
    prepared = []

    def request():
        own_connection = IdeaLake(lake.db_path)
        try:
            gate.wait(timeout=5)
            return own_connection.retry_evaluation(IDEA_ID, prepare_artifacts=prepared.append)
        finally:
            own_connection.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(request) for _ in range(2)]
        assert [future.result(timeout=15) for future in futures] == [True, True]

    _assert_only_evaluation_reopened(lake, before)
    assert prepared == [before["idea_transitions"][-1]["id"]]


def test_artifact_preparation_runs_under_exclusive_write_lock_before_state_changes(project):
    lake, _ = project
    _fail_evaluation(lake)
    before = _snapshot(lake)
    prepared = []

    def prepare(failed_transition_id):
        assert lake.conn.in_transaction is True
        assert _snapshot(lake) == before
        prepared.append(failed_transition_id)
        other = sqlite3.connect(lake.db_path, timeout=0.01)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute("BEGIN IMMEDIATE")
        finally:
            other.close()

    assert lake.retry_evaluation(IDEA_ID, prepare_artifacts=prepare) is True

    assert prepared == [before["idea_transitions"][-1]["id"]]
    _assert_only_evaluation_reopened(lake, before)


def test_artifact_preparation_exception_leaves_database_unchanged(project):
    lake, _ = project
    _fail_evaluation(lake)
    before = _snapshot(lake)

    def prepare(_failed_transition_id):
        raise OSError("injected recoverable archive failure")

    with pytest.raises(OSError, match="injected recoverable archive failure"):
        lake.retry_evaluation(IDEA_ID, prepare_artifacts=prepare)

    assert lake.conn.in_transaction is False
    assert _snapshot(lake) == before


def test_idempotent_pending_request_does_not_prepare_artifacts_again(project):
    lake, _ = project
    _fail_evaluation(lake)
    prepared = []
    assert lake.retry_evaluation(IDEA_ID, prepare_artifacts=prepared.append) is True
    after_first = _snapshot(lake)

    assert lake.retry_evaluation(IDEA_ID, prepare_artifacts=prepared.append) is True

    assert len(prepared) == 1
    assert _snapshot(lake) == after_first


def test_legacy_queue_retry_is_a_training_retry_not_the_new_evaluation_api(project):
    """Baseline diagnostic: existing general retry intentionally resets both."""
    lake, _ = project
    _fail_evaluation(lake)

    assert lake.record_state_transition(IDEA_ID, "FAILED", "QUEUED", "ordinary_training_retry")

    assert lake.get_fsm_state(IDEA_ID) == "QUEUED"
    assert lake.get_stage_state(IDEA_ID, "training") == "PENDING"
    assert lake.get_stage_state(IDEA_ID, "evaluation") == "PENDING"
