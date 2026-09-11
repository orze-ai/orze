"""Actual four-run heterogeneous acceptance on the frozen Core interfaces."""
import copy
import json

import pytest

from examples.acceptance.common import digest
from examples.acceptance.policy import TASKS

pytest_plugins = ["examples.acceptance.testing"]


@pytest.mark.parametrize("domain", ["sorting", "compression"])
@pytest.mark.parametrize("variant,actions", [("default", 5), ("counterfactual", 3)])
def test_real_cli_closed_loop_and_exact_replica(acceptance_runs, domain, variant, actions):
    run = acceptance_runs[domain + "_" + variant]
    db = run["database"]
    assert len(db["execution_attempts"]) == actions
    assert len(db["cpu_action_reservations"]) == actions
    assert len(run["artifacts"]) == len(run["observations"]) == actions
    assert all(row["state"] == "SETTLED" for row in db["cpu_action_reservations"])
    assert all(row["status"] == "completed" for row in db["ideas"])
    terminals = [json.loads(row["terminal_json"]) for row in db["execution_attempts"]]
    assert all(row["state"] == "TERMINAL" for row in db["execution_attempts"])
    assert all(t["outcome"] == "completed" and t["process_tree"]["wait_proof"] == "ECHILD_WALL"
               and t["effect_receipt_sha256"] for t in terminals)
    assert len({row["attempt_id"] for row in db["execution_attempts"]}) == actions
    assert len({row["artifact_id"] for row in run["artifacts"]}) == actions
    assert len({row["observation_id"] for row in run["observations"]}) == actions

    assert len(db["replication_requests"]) == 1
    replica = json.loads(db["replication_requests"][0]["record_json"])
    source_id = TASKS["baseline" if variant == "default" else "challenger"]
    assert replica["source_ref"]["task_id"] == source_id
    ideas = {row["idea_id"]: row for row in db["ideas"]}
    assert replica["task_id"] != source_id
    assert ideas[replica["task_id"]]["config"] == ideas[source_id]["config"]
    sources = {row["producer"]["task_id"]: row for row in run["artifacts"]}
    assert sources[source_id]["spec_fingerprint"] == sources[replica["task_id"]]["spec_fingerprint"]
    selected = next(item["decision"] for item in run["trace"] if item["decision"]["kind"] == "Replicate")
    assert selected["source_ref"] == replica["source_ref"]
    assert selected["request_id"] == replica["request_id"]

    final = run["trace"][-1]
    assert final["decision"] == {"kind": "Stop", "reason": "confirmed_selection", "wakeup": None}
    assert final["budget"]["active_reservations"] == 0
    assert final["budget"]["remaining_wall_seconds"] == 10 - actions * 2
    assert any(item["ref"]["task_id"] == replica["task_id"] and item["outcome"] == "completed"
               for item in final["snapshot"]["recorded_evidence"]["results"])
    assert json.loads(db["cpu_action_scopes"][0]["stop_json"]) == final["decision"]
    assert run["stopped_rerun"]["exit_code"] == 75
    assert run["stopped_rerun"]["trace"] == []
    assert not (run["root"] / "ideas.md").exists() or (run["root"] / "ideas.md").read_text().strip() == ""


@pytest.mark.parametrize("domain,base_cost,unchecked_cost", [("sorting", 9, 29), ("compression", 137, 266)])
def test_real_invalid_unknown_analysis_and_valid_negative_remain_distinct(
        acceptance_runs, domain, base_cost, unchecked_cost):
    run = acceptance_runs[domain + "_default"]
    observations = {item["evaluator"]["task_id"]: item for item in run["observations"]}
    baseline = observations[TASKS["baseline"]]
    challenger = observations[TASKS["challenger"]]
    unknown = observations[TASKS["unchecked"]]
    analysis = observations[TASKS["analysis"]]
    assert baseline["values"]["cost"] == base_cost
    assert challenger["validation"]["status"] == "invalid"
    assert challenger["values"]["cost"] < base_cost
    assert unknown["validation"]["status"] == "unknown"
    assert unknown["values"]["cost"] == unchecked_cost
    assert analysis["validation"]["status"] == "valid"
    assert analysis["values"]["cost"] == unchecked_cost > base_cost
    assert analysis["comparison_scope"] == baseline["comparison_scope"]
    source_ids = [next(item["artifact_id"] for item in run["artifacts"]
                       if item["producer"]["task_id"] == TASKS[key])
                  for key in ("baseline", "challenger", "unchecked")]
    assert analysis["input_artifact_ids"] == source_ids
    assert set(analysis["input_artifact_bindings"]) == set(source_ids)
    proposal = next(item["decision"] for item in run["trace"]
                    if item["decision"]["kind"] == "Propose"
                    and item["decision"]["task_id"] == TASKS["analysis"])
    assert proposal["domain_request"]["input_artifact_ids"] == source_ids
    before_analysis = next(item for item in run["trace"] if item["decision"] == proposal)
    original_unknown = next(obs for result in before_analysis["snapshot"]["recorded_evidence"]["results"]
                            for obs in result["observation_records"] if obs["evaluator"]["task_id"] == TASKS["unchecked"])
    assert unknown == original_unknown
    assert analysis["observation_id"] != unknown["observation_id"]
    assert analysis["evaluator"] != unknown["evaluator"]


@pytest.mark.parametrize("domain", ["sorting", "compression"])
def test_dataset_only_counterfactual_changes_the_actual_selected_action(acceptance_runs, domain):
    default = acceptance_runs[domain + "_default"]
    counter = acceptance_runs[domain + "_counterfactual"]
    assert default["cfg"]["action_policy"] == counter["cfg"]["action_policy"]
    assert default["cfg"]["execution"] == counter["cfg"]["execution"]
    assert default["cfg"]["action_domain"]["kind"] == counter["cfg"]["action_domain"]["kind"]
    assert default["dataset_sha256"] != counter["dataset_sha256"]
    # Project paths necessarily differ; apart from them only dataset changes.
    left, right = copy.deepcopy(default["cfg"]), copy.deepcopy(counter["cfg"])
    for config in (left, right):
        for path in ("results_dir", "idea_lake_db", "ideas_file"):
            config.pop(path)
        config["action_domain"]["config"].pop("dataset")
    assert left == right
    assert {row["idea_id"] for row in counter["database"]["ideas"]}.isdisjoint(
        {TASKS["unchecked"], TASKS["analysis"]})
    assert not any(item["decision"]["kind"] == "Propose" and item["decision"]["task_id"]
                   in (TASKS["unchecked"], TASKS["analysis"]) for item in counter["trace"])
    challenger = next(item for item in counter["observations"] if item["evaluator"]["task_id"] == TASKS["challenger"])
    assert challenger["validation"]["status"] == "valid"
    baseline = next(item for item in counter["observations"] if item["evaluator"]["task_id"] == TASKS["baseline"])
    assert challenger["values"]["cost"] < baseline["values"]["cost"]
    assert next(item["decision"]["source_ref"]["task_id"] for item in counter["trace"]
                if item["decision"]["kind"] == "Replicate") == TASKS["challenger"]


def test_shared_session_counts_and_observed_timings_are_not_efficiency_claims(acceptance_runs):
    assert len(acceptance_runs) == 4
    assert sum(len(run["database"]["execution_attempts"]) for run in acceptance_runs.values()) == 16
    assert sum(sum(json.loads(row["permit_json"])["wall_limit_seconds"]
                   for row in run["database"]["cpu_action_reservations"])
               for run in acceptance_runs.values()) == 32
    for run in acceptance_runs.values():
        assert run["wall_seconds"] > 0
        assert all(item["snapshot_sha256"] == digest(item["snapshot"]) for item in run["trace"])
        values = [item["values"] for item in run["observations"]]
        assert all(v["worker_cpu_seconds"] >= 0 and v["worker_wall_seconds"] >= 0 for v in values)
        assert run["trace"][-1]["monotonic"] >= run["started_monotonic"]
