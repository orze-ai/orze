"""Actual public CPU workflow for the original independently chosen holdout."""
import json

import pytest

from examples.holdout.testing import TASKS

pytest_plugins = ["examples.holdout.testing"]


def _by_task(run):
    return {item["evaluator"]["task_id"]: item for item in run["observations"]}


def test_same_core_publishes_producers_without_inventing_observations(holdout_runs):
    run = holdout_runs["workflow"]
    db = run["database"]
    assert len(db["execution_attempts"]) == len(db["cpu_action_reservations"]) == 7
    assert len(run["artifacts"]) == 6 and len(run["observations"]) == 4
    assert all(row["state"] == "SETTLED" for row in db["cpu_action_reservations"])
    for key in ("baseline_producer", "challenger_producer"):
        task = TASKS[key]
        attempt = next(row for row in db["execution_attempts"] if row["task_id"] == task)
        terminal = json.loads(attempt["terminal_json"])
        assert terminal["outcome"] == "completed" and terminal["observation_ids"] == []
        artifact = next(item for item in run["artifacts"] if item["producer"]["task_id"] == task)
        candidate = json.loads(run["artifact_contents"][artifact["artifact_id"]])
        assert set(candidate) == {"instance_id", "schedule"}
        assert all(set(entry) == {"job_id", "start"} for entry in candidate["schedule"])
        assert task not in _by_task(run)


def test_maximize_keeps_zero_lower_valid_and_protocol_specific_invalid(holdout_runs):
    run = holdout_runs["workflow"]
    obs = _by_task(run)
    baseline, recovered, v2 = [obs[TASKS[name]] for name in ("baseline_v1", "recovered_v1", "challenger_v2")]
    assert baseline["validation"]["status"] == recovered["validation"]["status"] == "valid"
    assert baseline["name"] == recovered["name"] == v2["name"] == "scheduled_value"
    assert baseline["values"]["direction"] == recovered["values"]["direction"] == "maximize"
    assert type(baseline["values"]["value"]) is int and baseline["values"]["value"] == 0
    # Lower feasible is valid; this does not claim a global optimum or gain.
    assert recovered["values"]["value"] > baseline["values"]["value"]
    assert v2["validation"] == {"status": "invalid", "reason_code": "capacity_overload"}
    assert "value" not in v2["values"]
    assert baseline["protocol_fingerprint"] == recovered["protocol_fingerprint"]
    assert baseline["comparison_scope"] == recovered["comparison_scope"]
    assert v2["protocol_fingerprint"] != recovered["protocol_fingerprint"]
    assert v2["comparison_scope"] != recovered["comparison_scope"]


@pytest.mark.parametrize("condition,status,reason", [
    ("boundary", "valid", "feasible"),
    ("duplicate_id", "invalid", "duplicate_job"),
    ("missing_prerequisite", "invalid", "missing_prerequisite"),
    ("overload", "invalid", "capacity_overload"),
    ("malformed", "invalid", "candidate_json_invalid"),
])
def test_declared_candidates_are_real_completed_evaluations_not_execution_holds(
        holdout_runs, condition, status, reason):
    run = holdout_runs[condition]
    assert len(run["artifacts"]) == len(run["database"]["execution_attempts"]) == 2
    assert len(run["observations"]) == 1
    observation = run["observations"][0]
    assert observation["validation"] == {"status": status, "reason_code": reason}
    assert ("value" in observation["values"]) is (status == "valid")
    assert all(row["state"] == "TERMINAL" and row["hold_reason"] is None
               and json.loads(row["terminal_json"])["outcome"] == "completed"
               for row in run["database"]["execution_attempts"])
    assert all(row["state"] == "SETTLED" for row in run["database"]["cpu_action_reservations"])
    producer = next(item for item in run["artifacts"] if item["logical_name"] == "candidate")
    assert observation["input_artifact_ids"] == [producer["artifact_id"]]
    if condition == "boundary":
        dataset = run["cfg"]["action_domain"]["config"]["instance"]
        jobs = {item["id"]: item for item in dataset["jobs"]}
        candidate = json.loads(run["artifact_contents"][producer["artifact_id"]])
        starts = {item["job_id"]: item["start"] for item in candidate["schedule"]}
        assert starts["b"] + jobs["b"]["duration"] == jobs["b"]["deadline"]
        assert starts["b"] + jobs["b"]["duration"] == starts["e"]
        assert jobs["b"]["demand"] == jobs["e"]["demand"] == dataset["capacity"]
        assert observation["values"]["value"] == sum(jobs[name]["value"] for name in starts)


def test_genuine_controller_invocations_and_native_work_are_counted_separately(holdout_runs):
    runs = list(holdout_runs.values())
    calls = [call for run in runs for call in run["calls"]]
    attempts = [row for run in runs for row in run["database"]["execution_attempts"]]
    reservations = [row for run in runs for row in run["database"]["cpu_action_reservations"]]
    assert len(calls) == 22
    identities = {(call["metadata"][0]["pid"], call["metadata"][0]["start_ticks"]) for call in calls}
    assert len(identities) == 22
    assert all(call["command"][1:3] == ["-m", "examples.holdout.testing"] for call in calls)
    assert all(call["controller_supervision"]["closure"]["wait_proof"] == "ECHILD_WALL"
               and call["exit_code"] == 0 for call in calls)
    assert len(attempts) == len(reservations) == 17
    assert len({row["attempt_id"] for row in attempts}) == 17
    assert sum(json.loads(row["permit_json"])["wall_limit_seconds"] for row in reservations) == 34
    outcomes = [json.loads(row["terminal_json"])["outcome"] for row in attempts]
    assert outcomes.count("failed") == 1 and outcomes.count("completed") == 16
    assert all(row["state"] == "SETTLED" for row in reservations)
