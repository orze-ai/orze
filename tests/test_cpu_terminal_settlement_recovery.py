"""A real controller crash must not permanently occupy a confirmed CPU slot."""
import json

import pytest

from cpu_terminal_recovery_helpers import admit, run_cli

pytest_plugins = ["cpu_terminal_recovery_helpers"]


@pytest.mark.parametrize("first_exit", [0, 7])
def test_restart_settles_confirmed_terminal_without_repeating_original_worker(make_crashed_project, first_exit):
    project = make_crashed_project(exit_code=first_exit)
    before = project["snapshots"][-1]
    db = before["database"]
    assert len(db["execution_attempts"]) == len(db["cpu_action_reservations"]) == 1
    original = db["execution_attempts"][0]
    terminal = json.loads(original["terminal_json"])
    assert original["state"] == "TERMINAL" and original["hold_reason"] is None
    assert terminal["outcome"] == ("completed" if first_exit == 0 else "failed")
    assert terminal["return_code"] == first_exit
    assert db["cpu_action_reservations"][0]["state"] == "BOUND"
    assert len(before["worker_events"]) == 1 and before["worker_events"][0]["tag"] == "first"
    assert len(db["research_artifacts"]) == int(first_exit == 0)
    assert db["research_observations"] == []
    admit(project, "idea-second", "second")
    second = run_cli(project, "restart")
    after = project["snapshots"][-1]
    # This is the original public product behavior requirement, not an API-absence test.
    statuses = {row["idea_id"]: row["status"] for row in after["database"]["ideas"]}
    assert statuses["idea-second"] == "completed", ("confirmed terminal still occupies slot", str(project["root"]))
    assert [row for row in after["database"]["execution_attempts"] if row["task_id"] == "idea-first"] == [original]
    assert len(after["database"]["execution_attempts"]) == 2
    assert all(row["state"] == "SETTLED" for row in after["database"]["cpu_action_reservations"])
    assert sum(int(json.loads(row["permit_json"])["reserved_nanoseconds"])
               for row in after["database"]["cpu_action_reservations"]) == 4000000000
    assert [event["tag"] for event in after["worker_events"]] == ["first", "second"]
    assert before["worker_events"][0] == after["worker_events"][0]
    assert all(after["files"][path] == raw for path, raw in before["files"].items())
    births = [tuple(call["controller_binding"]["worker"][k] for k in ("pid", "start_ticks"))
              for call in project["calls"]]
    assert len(set(births)) == 2
    assert project["calls"][0]["finished_monotonic"] <= second["started_monotonic"]
    run_cli(project, "second_restart_idle")
    idle = project["snapshots"][-1]
    for table in ("execution_attempts", "cpu_action_reservations", "research_artifacts",
                  "research_observations", "ideas", "idea_state", "idea_transitions",
                  "idea_stage_state", "idea_stage_transitions"):
        assert idle["database"][table] == after["database"][table]
    assert idle["files"] == after["files"] and idle["worker_events"] == after["worker_events"]
