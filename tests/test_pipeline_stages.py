import json

import pytest

from orze.idea_lake import IdeaLake


def _running_lake(tmp_path, idea_id="idea-pipeline"):
    lake = IdeaLake(tmp_path / "lake.db")
    lake.insert(idea_id, "pipeline", "{}", "", status="queued")
    assert lake.record_state_transition(idea_id, "QUEUED", "CLAIMED")
    assert lake.record_state_transition(idea_id, "CLAIMED", "IN_PROGRESS")
    return lake, idea_id


def test_training_and_evaluation_stages_are_independently_audited(tmp_path):
    lake, idea_id = _running_lake(tmp_path)

    assert lake.get_fsm_state(idea_id) == "IN_PROGRESS"
    assert lake.get_stage_state(idea_id, "training") == "IN_PROGRESS"
    assert lake.get_stage_state(idea_id, "evaluation") == "PENDING"

    assert lake.record_stage_transition(
        idea_id, "training", "IN_PROGRESS", "COMPLETE",
        "training_completed_evaluation_pending",
    )
    assert lake.get_fsm_state(idea_id) == "IN_PROGRESS"
    assert lake.record_stage_transition(
        idea_id, "evaluation", "PENDING", "IN_PROGRESS",
        "evaluation_launched",
    )
    assert lake.record_state_transition(
        idea_id, "IN_PROGRESS", "COMPLETE", "evaluation_validated",
    )

    assert lake.get_fsm_state(idea_id) == "COMPLETE"
    assert lake.get_stage_state(idea_id, "training") == "COMPLETE"
    assert lake.get_stage_state(idea_id, "evaluation") == "COMPLETE"
    assert [
        (row["stage"], row["from_state"], row["to_state"])
        for row in lake.get_stage_history(idea_id)
    ] == [
        ("training", "NOT_STARTED", "IN_PROGRESS"),
        ("evaluation", "NOT_STARTED", "PENDING"),
        ("training", "IN_PROGRESS", "COMPLETE"),
        ("evaluation", "PENDING", "IN_PROGRESS"),
        ("evaluation", "IN_PROGRESS", "COMPLETE"),
    ]
    lake.close()


def test_evaluation_failure_preserves_training_success(tmp_path):
    lake, idea_id = _running_lake(tmp_path)
    assert lake.record_stage_transition(
        idea_id, "training", "IN_PROGRESS", "COMPLETE",
        "training_completed_evaluation_pending",
    )
    assert lake.record_stage_transition(
        idea_id, "evaluation", "PENDING", "IN_PROGRESS",
        "evaluation_launched",
    )
    assert lake.record_state_transition(
        idea_id, "IN_PROGRESS", "FAILED", "evaluation_contract_failed",
    )

    assert lake.get_fsm_state(idea_id) == "FAILED"
    assert lake.get_stage_state(idea_id, "training") == "COMPLETE"
    assert lake.get_stage_state(idea_id, "evaluation") == "FAILED"
    lake.close()


def test_training_failure_skips_evaluation_and_retry_resets_both(tmp_path):
    lake, idea_id = _running_lake(tmp_path)
    assert lake.record_state_transition(
        idea_id, "IN_PROGRESS", "FAILED", "training_failed",
    )
    assert lake.get_stage_state(idea_id, "training") == "FAILED"
    assert lake.get_stage_state(idea_id, "evaluation") == "SKIPPED"

    assert lake.record_state_transition(
        idea_id, "FAILED", "QUEUED", "explicit_retry",
    )
    assert lake.get_stage_state(idea_id, "training") == "PENDING"
    assert lake.get_stage_state(idea_id, "evaluation") == "PENDING"
    assert lake.get_fsm_state(idea_id) == "QUEUED"
    lake.close()


def test_catch_up_with_evaluator_does_not_mark_pipeline_complete(tmp_path):
    lake, idea_id = _running_lake(tmp_path)
    idea_dir = tmp_path / "results" / idea_id
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED"}), encoding="utf-8")

    assert lake.catch_up_missing_terminals(
        tmp_path / "results", evaluation_required=True) == 1
    assert lake.get_fsm_state(idea_id) == "IN_PROGRESS"
    assert lake.get(idea_id)["status"] == "running"
    assert lake.get_stage_state(idea_id, "training") == "COMPLETE"
    assert lake.get_stage_state(idea_id, "evaluation") == "PENDING"
    lake.close()


def test_no_evaluator_closes_training_and_marks_eval_skipped(tmp_path):
    lake, idea_id = _running_lake(tmp_path)
    assert lake.record_state_transition(
        idea_id, "IN_PROGRESS", "COMPLETE", "training_completed",
    )
    assert lake.get_stage_state(idea_id, "training") == "COMPLETE"
    assert lake.get_stage_state(idea_id, "evaluation") == "SKIPPED"
    lake.close()


def test_stage_write_failure_rolls_back_global_launch_edge(tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    idea_id = "idea-atomic-stage"
    lake.insert(idea_id, "pipeline", "{}", "", status="queued")
    assert lake.record_state_transition(idea_id, "QUEUED", "CLAIMED")
    lake.conn.execute("""
        CREATE TRIGGER reject_stage_audit
        BEFORE INSERT ON idea_stage_transitions
        BEGIN
          SELECT RAISE(ABORT, 'injected stage audit failure');
        END
    """)
    lake.conn.commit()

    with pytest.raises(Exception, match="injected stage audit failure"):
        lake.record_state_transition(
            idea_id, "CLAIMED", "IN_PROGRESS", "training_launched")

    assert lake.get_fsm_state(idea_id) == "CLAIMED"
    assert lake.get_stage_state(idea_id, "training") == "NOT_STARTED"
    assert [row["to_state"] for row in lake.get_fsm_history(idea_id)] == [
        "CLAIMED"
    ]
    lake.close()
