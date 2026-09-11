"""Independent checks of captured real holdout product executions.

The session fixture owns all CPU invocations. These tests only read its complete
snapshots/artifacts: no additional worker, Policy implementation, Lake write,
retry authority, or replacement scientific evaluator is introduced here.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from examples.holdout.testing import TASKS

pytest_plugins = ["examples.holdout.testing"]


def _one(rows):
    assert len(rows) == 1
    return rows[0]


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _snapshot(run, label):
    return _one([s for s in run["snapshots"] if s["label"] == label])["database"]


def _entry(run, collection, label):
    return _one([item for item in run[collection] if item["label"] == label])


def _attempt(database, task):
    return _one([row for row in database["execution_attempts"] if row["task_id"] == task])


def _ref(row):
    return {key: row[key] for key in ("task_id", "phase", "attempt_id", "generation")}


def _task(database, task):
    return _one([row for row in database["ideas"] if row["idea_id"] == task])


def _observation(run, task):
    return _one([row for row in run["observations"] if row["evaluator"]["task_id"] == task])


def _cli_result(call):
    results = []
    for line in call["stdout"].splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if type(value) is dict and set(value) == {"status", "request_id", "task_id"}:
            results.append(value)
    return _one(results)


def test_public_admission_does_not_execute_or_reset_the_failed_occurrence(holdout_runs):
    for run in holdout_runs.values():
        for admission in run["admissions"]:
            before = _snapshot(run, admission["before_snapshot"])
            after = _snapshot(run, admission["after_snapshot"])
            for table in ("execution_attempts", "cpu_action_reservations",
                          "research_artifacts", "research_observations"):
                assert after[table] == before[table]
            assert _task(after, admission["task_id"])["config"] == admission["raw_config"]
    run = holdout_runs["workflow"]
    replay = _entry(run, "admissions", "failed_same_id_replay")
    assert replay["outcome"]["status"] == "already_present_exact"
    before, after = [_snapshot(run, replay[key]) for key in ("before_snapshot", "after_snapshot")]
    assert before == after
    failed = TASKS["failed_v1"]
    assert _task(after, failed)["status"] == "failed"
    for table, field in (("ideas", "idea_id"), ("idea_state", "idea_id"),
                         ("idea_stage_state", "idea_id"), ("execution_attempts", "task_id")):
        assert [row for row in run["database"][table] if row[field] == failed] == [
            row for row in after[table] if row[field] == failed]


def test_failure_is_closed_before_a_fresh_controller_repeats_identical_config(holdout_runs):
    run = holdout_runs["workflow"]
    failed_call, recovery_call = [_entry(run, "calls", label)
                                  for label in ("failed_v1", "recovered_v1")]
    before_recovery = _snapshot(run, recovery_call["before_snapshot"])
    old = _attempt(before_recovery, TASKS["failed_v1"])
    terminal = json.loads(old["terminal_json"])
    assert old["state"] == "TERMINAL" and old["hold_reason"] is None
    assert terminal["outcome"] == "failed" and terminal["return_code"] == 71
    reservation = _one([r for r in before_recovery["cpu_action_reservations"]
                        if r["task_id"] == TASKS["failed_v1"]])
    assert reservation["state"] == "SETTLED" and reservation["terminal_sha256"] == _digest(terminal)
    assert failed_call["finished_monotonic"] <= recovery_call["started_monotonic"]
    identities = [tuple(call["metadata"][0][key] for key in ("pid", "start_ticks"))
                  for call in (failed_call, recovery_call)]
    assert identities[0] != identities[1]
    assert failed_call["fault_injection"] == "partial_exit_71" and recovery_call["fault_injection"] is None
    admissions = [_entry(run, "admissions", label) for label in ("failed_v1", "recovered_v1")]
    assert admissions[0]["raw_config"] == admissions[1]["raw_config"]
    assert admissions[0]["domain_request"] == admissions[1]["domain_request"]
    recovered = _attempt(run["database"], TASKS["recovered_v1"])
    bindings = [json.loads(row["binding_json"]) for row in (old, recovered)]
    for key in ("action_sha256", "command_sha256", "inputs_sha256"):
        assert bindings[0][key] == bindings[1][key]
    assert _ref(old) != _ref(recovered) and json.loads(recovered["terminal_json"])["outcome"] == "completed"


def test_partial_output_is_real_but_never_an_eligible_publication(holdout_runs):
    run = holdout_runs["workflow"]
    failed = _attempt(run["database"], TASKS["failed_v1"])
    terminal, binding = [json.loads(failed[key]) for key in ("terminal_json", "binding_json")]
    partial = run["partial_output"]
    raw = partial["utf8"].encode("utf-8")
    assert Path(partial["path"]) == Path(binding["work_dir"]) / "evaluation.json"
    assert Path(partial["path"]).read_bytes() == raw
    assert len(raw) == partial["bytes"] > 0 and hashlib.sha256(raw).hexdigest() == partial["sha256"]
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    assert terminal["artifact_ids"] == terminal["observation_ids"] == []
    assert not [r for r in run["artifacts"] if r["producer"] == _ref(failed)]
    assert not [r for r in run["observations"] if r["evaluator"] == _ref(failed)]
    assert TASKS["failed_v1"] not in run["result_envelopes"]
    closure = terminal["process_tree"]
    assert closure["worker_returncode"] == 71 and closure["wait_proof"] == "ECHILD_WALL"
    assert closure["stop_requested"] is False and closure["forced_cleanup"] is False


def test_protocol_reassessment_reuses_source_and_preserves_the_v1_measurement(holdout_runs):
    run = holdout_runs["workflow"]
    source = _one([r for r in run["artifacts"] if r["producer"]["task_id"] == TASKS["challenger_producer"]])
    v1, v2 = [_observation(run, TASKS[key]) for key in ("recovered_v1", "challenger_v2")]
    expected = {source["artifact_id"]: {key: source[key] for key in
                                       ("producer", "spec_fingerprint", "content_sha256")}}
    for observation in (v1, v2):
        assert observation["input_artifact_ids"] == [source["artifact_id"]]
        assert observation["input_artifact_bindings"] == expected
    assert v1["spec_fingerprint"] == v2["spec_fingerprint"]
    assert v1["protocol_fingerprint"] != v2["protocol_fingerprint"]
    assert v1["comparison_scope"] != v2["comparison_scope"]
    first, second = [copy.deepcopy(_entry(run, "admissions", key)["domain_request"])
                     for key in ("recovered_v1", "challenger_v2")]
    assert first["payload"]["protocol"] == "schedule-feasibility-v1"
    assert second["payload"]["protocol"] == "schedule-feasibility-v2"
    second["payload"]["protocol"] = first["payload"]["protocol"]
    assert first == second
    call = _entry(run, "calls", "challenger_v2")
    before, after = [_snapshot(run, call[key]) for key in ("before_snapshot", "after_snapshot")]
    original = _one([json.loads(r["record_json"]) for r in before["research_observations"]
                     if json.loads(r["record_json"])["observation_id"] == v1["observation_id"]])
    assert original == v1
    for key in ("baseline_producer", "challenger_producer"):
        assert _attempt(before, TASKS[key]) == _attempt(after, TASKS[key]) == _attempt(run["database"], TASKS[key])
    assert len(after["execution_attempts"]) == len(before["execution_attempts"]) + 1


def test_replica_ack_binds_the_completed_evaluator_and_never_runs_at_admission(holdout_runs):
    run = holdout_runs["workflow"]
    call = _entry(run, "calls", "replicate_admit")
    before, after = [_snapshot(run, call[key]) for key in ("before_snapshot", "after_snapshot")]
    for table in ("execution_attempts", "cpu_action_reservations", "research_artifacts", "research_observations"):
        assert before[table] == after[table]
    record = json.loads(_one(after["replication_requests"])["record_json"])
    original = _attempt(before, TASKS["recovered_v1"])
    assert record["schema"] == 2 and record["adapter"] == "native_cpu_action"
    assert record["source_ref"] == _ref(original)
    assert record["request_sha256"] == _digest({k: v for k, v in record.items() if k != "request_sha256"})
    result = _cli_result(call)
    assert result == {"status": "created", "request_id": record["request_id"], "task_id": record["task_id"]}
    assert _task(after, record["task_id"])["status"] == "queued"
    assert _task(after, record["task_id"])["config"] == _task(before, TASKS["recovered_v1"])["config"]
    repeat = _attempt(run["database"], record["task_id"])
    assert _ref(repeat) != _ref(original)
    bindings = [json.loads(row["binding_json"]) for row in (original, repeat)]
    assert bindings[0]["action_sha256"] == bindings[1]["action_sha256"] == record["action_sha256"]
    observations = [_observation(run, task) for task in (TASKS["recovered_v1"], record["task_id"])]
    assert observations[0]["observation_id"] != observations[1]["observation_id"]
    for key in ("input_artifact_ids", "input_artifact_bindings", "values", "validation", "protocol_fingerprint"):
        assert observations[0][key] == observations[1][key]
    replay = _entry(run, "calls", "replicate_replay")
    assert _cli_result(replay) == {**result, "status": "already_requested"}
    assert _snapshot(run, replay["before_snapshot"]) == _snapshot(run, replay["after_snapshot"])


def test_idle_ticks_record_wait_without_new_work_or_a_durable_stop(holdout_runs):
    run = holdout_runs["workflow"]
    idle = [call for call in run["calls"] if call["label"].startswith("idle_")]
    assert [call["label"] for call in idle] == ["idle_0", "idle_1", "idle_2"]
    for call in idle:
        before, after = [_snapshot(run, call[key]) for key in ("before_snapshot", "after_snapshot")]
        for table in before:
            if table != "cpu_action_decisions":
                assert after[table] == before[table]
        previous = {row["decision_id"] for row in before["cpu_action_decisions"]}
        added = [json.loads(row["record_json"]) for row in after["cpu_action_decisions"]
                 if row["decision_id"] not in previous]
        assert len(added) == 1 and added[0]["kind"] == "Wait" and added[0]["reason"] == "queue_empty"
        assert all(row["stop_json"] is None for row in after["cpu_action_scopes"])


def test_native_closure_and_each_reserved_terminal_are_exactly_linked(holdout_runs):
    for run in holdout_runs.values():
        db = run["database"]
        for row in db["execution_attempts"]:
            ref, binding, terminal = _ref(row), json.loads(row["binding_json"]), json.loads(row["terminal_json"])
            closure = terminal["process_tree"]
            assert row["state"] == "TERMINAL" and row["hold_reason"] is None
            assert closure["event"] == "TREE_CLOSED" and closure["wait_proof"] == "ECHILD_WALL"
            assert closure["binding"] == binding["supervision"]
            assert closure["binding"]["identity"] == {"attempt_ref": ref, "scope": str(run["root"] / "results" / row["task_id"])}
            assert closure["stop_requested"] is False and closure["forced_cleanup"] is False
            assert type(closure["worker_returncode"]) is int and closure["worker_returncode"] == terminal["return_code"]
            reservation = _one([r for r in db["cpu_action_reservations"] if r["task_id"] == row["task_id"]])
            permit = json.loads(reservation["permit_json"])
            assert json.loads(reservation["ref_json"]) == ref and reservation["state"] == "SETTLED"
            assert reservation["terminal_sha256"] == _digest(terminal)
            assert permit["reservation_id"] == binding["reservation_id"] and permit["wall_limit_seconds"] == 2
            assert permit["reserved_nanoseconds"] == "2000000000"
            artifacts = [a for a in run["artifacts"] if a["producer"] == ref]
            observations = [o for o in run["observations"] if o["evaluator"] == ref]
            assert set(terminal["artifact_ids"]) == {a["artifact_id"] for a in artifacts}
            assert set(terminal["observation_ids"]) == {o["observation_id"] for o in observations}
            for observation in observations:
                assert set(observation["result_artifact_ids"]) <= set(terminal["artifact_ids"])


def test_declared_queue_work_keeps_invalid_values_absent_and_no_proposal_ledger(holdout_runs):
    for run in holdout_runs.values():
        assert run["cfg"]["action_policy"] == {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": .05, "config": {}}
        assert run["database"]["cpu_proposal_requests"] == []
        for observation in run["observations"]:
            assert observation["name"] == "scheduled_value" and observation["values"]["direction"] == "maximize"
            envelope = run["result_envelopes"][observation["evaluator"]["task_id"]]
            verdict = envelope["verdict"]
            assert observation["validation"] == {key: verdict[key] for key in ("status", "reason_code")}
            if verdict["status"] == "invalid":
                assert set(verdict) == {"status", "reason_code"} and "value" not in observation["values"]
            else:
                assert type(observation["values"]["value"]) is int and observation["values"]["value"] == verdict["scheduled_value"]
