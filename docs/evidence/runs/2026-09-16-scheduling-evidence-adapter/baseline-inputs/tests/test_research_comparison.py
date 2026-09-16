"""Accounting controls, not new research runs or model-quality evidence."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from examples.research_comparison.protocol import digest, schedule, validate_protocol
from examples.research_comparison.report import METRICS, compare


def protocol():
    return {
        "schema": 1, "comparison_id": "offline-accounting-controls",
        "mode": "retrospective_offline", "repetitions": 2, "ordering": "task_and_repetition",
        "arms": {arm: {"artifact_sha256": digest(arm), "treatment_sha256": digest(arm)}
                 for arm in ("A", "B")},
        "shared": {key: digest(key) for key in ("model", "tools", "environment")},
        "verifier_sha256": digest("synthetic bookkeeping verifier"),
        "tasks": [{
            "id": name, "domain": "control", "role": role, "seeds": [7, 11],
            "inputs": {key: digest(key) for key in (
                "data", "evaluator", "instructions", "initial_history", "initial_memory")},
            "quality": {"direction": "minimize", "max_regression": 0,
                        "minimum_valid_observations": 1},
            "limits": {"provider_calls": 10, "provider_tokens": 1000,
                       "provider_cost_usd": 1, "reserved_seconds": 10,
                       "gpu_seconds": 0, "cli_wall_seconds": 30},
        } for name, role in (("target", "target"), ("negative", "negative_control"))],
    }


def records(p):
    result = []
    for slot in schedule(p):
        item = {**slot, "protocol_sha256": digest(p), "status": "completed",
                "metrics": {name: 1 for name in METRICS},
                "quality": {"valid": True, "confirmed": True, "score": 9,
                            "comparison_key": {"data": "same", "evaluator": "same"}},
                "observations": {"valid": 2, "invalid": 0, "unknown": 1},
                "native_outcomes": {"completed": 2, "failed": 0, "unknown": 0}}
        item["metrics"]["gpu_seconds"] = 0
        if item["arm"] == "B":
            item["metrics"]["native_actions"] = 0
        result.append(item)
    return result


def verify(record, task, arm):
    # Deliberately synthetic; production adapters must derive these from raw evidence.
    return {key: record[key] for key in (
        "status", "metrics", "quality", "observations", "native_outcomes")}


def test_schedule_keeps_seed_pairs_and_balances_each_task():
    p = protocol()
    items = schedule(p)
    assert len(items) == 8
    assert [item["arm"] for item in items if item["task_id"] == "target"] == ["A", "B", "B", "A"]
    assert [item["arm"] for item in items if item["task_id"] == "negative"] == ["B", "A", "A", "B"]
    assert len({item["run_id"] for item in items}) == 8


def test_all_pairs_must_pass_before_group_efficiency_is_reported():
    p = protocol()
    runs = records(p)
    good = compare(p, runs, verify=verify)
    assert good["all_pairs_qualified"] is True
    assert good["groups"]["target"]["median_paired_differences"]["native_actions"] == -1
    runs[1]["quality"]["score"] = 10
    bad = compare(p, runs, verify=verify)
    assert bad["groups"]["target"]["planned_pairs"] == 2
    assert bad["groups"]["target"]["qualified_pairs"] == 1
    assert "median_paired_differences" not in bad["groups"]["target"]
    assert bad["all_pairs_qualified"] is False


def test_missing_run_stays_in_denominator_and_no_cost_is_imputed():
    p = protocol()
    result = compare(p, records(p)[:-1], verify=verify)
    assert result["counts"] == {"planned_runs": 8, "provided_runs": 7, "missing_runs": 1}
    group = result["groups"]["negative"]
    assert group["planned_pairs"] == 2 and group["qualified_pairs"] == 1
    cost = group["cost_totals"]["B"]["provider_cost_usd"]
    assert cost == {"known_sum": 1, "known_runs": 1, "missing_runs": 1, "complete": False}


@pytest.mark.parametrize("change", ["duplicate", "unknown_id", "wrong_arm", "wrong_seed", "wrong_digest"])
def test_mislabeled_or_extra_inputs_are_not_silently_filtered(change):
    p = protocol()
    runs = records(p)
    if change == "duplicate":
        runs.append(copy.deepcopy(runs[0]))
    elif change == "unknown_id":
        runs[0]["run_id"] = "extra"
    elif change == "wrong_arm":
        runs[0]["arm"] = "B"
    elif change == "wrong_seed":
        runs[0]["seed"] = 999
    else:
        runs[0]["protocol_sha256"] = digest("different protocol")
    with pytest.raises(ValueError):
        compare(p, runs, verify=verify)


@pytest.mark.parametrize("status", ["failed", "unknown"])
def test_failed_attempt_costs_are_kept_without_success_only_speedups(status):
    p = protocol()
    runs = records(p)
    runs[0]["status"] = status
    runs[0]["metrics"]["provider_cost_usd"] = 0.75
    result = compare(p, runs, verify=verify)
    assert result["all_pairs_qualified"] is False
    assert result["groups"]["target"]["cost_totals"]["A"]["provider_cost_usd"]["known_sum"] == 1.75
    assert result["runs"][0]["status"] == status


def test_unknown_usage_is_not_zero_or_permission_to_claim_equal_budget():
    p = protocol()
    runs = records(p)
    runs[0]["metrics"]["provider_tokens"] = None
    result = compare(p, runs, verify=verify)
    assert result["all_pairs_qualified"] is False
    assert result["runs"][0]["budget_checks"]["provider_tokens"] == "unknown"
    assert result["groups"]["target"]["cost_totals"]["A"]["provider_tokens"]["complete"] is False


@pytest.mark.parametrize("metric", ["provider_calls", "provider_tokens", "provider_cost_usd",
                                   "reserved_seconds", "gpu_seconds", "cli_wall_seconds"])
def test_excess_budget_invalidates_pair_including_retries(metric):
    p = protocol()
    runs = records(p)
    runs[0]["metrics"][metric] = p["tasks"][0]["limits"][metric] + 1
    assert compare(p, runs, verify=verify)["all_pairs_qualified"] is False


def test_retrospective_unmeasured_cost_remains_unknown_but_measured_endpoints_survive():
    p = protocol()
    for task in p["tasks"]:
        task["limits"]["provider_tokens"] = None
    runs = records(p)
    for run in runs:
        run["metrics"]["provider_tokens"] = None
    result = compare(p, runs, verify=verify)
    assert result["all_pairs_qualified"] is True
    assert "provider_tokens" not in result["groups"]["target"]["median_paired_differences"]
    assert result["new_research_evidence"] is False


@pytest.mark.parametrize("field", ["valid", "confirmed"])
def test_unverified_quality_or_missing_confirmation_cannot_pass(field):
    p = protocol()
    runs = records(p)
    runs[0]["quality"][field] = False
    assert compare(p, runs, verify=verify)["all_pairs_qualified"] is False


def test_scope_mismatch_cannot_be_rescued_by_better_score():
    p = protocol()
    runs = records(p)
    runs[1]["quality"]["score"] = 0
    runs[1]["quality"]["comparison_key"]["data"] = "different"
    assert compare(p, runs, verify=verify)["all_pairs_qualified"] is False


def test_valid_coverage_loss_and_unknown_observations_are_reported():
    p = protocol()
    runs = records(p)
    runs[1]["observations"]["valid"] = 1
    runs[1]["observations"]["unknown"] = 2
    pair = compare(p, runs, verify=verify)["pairs"][0]
    assert pair["qualified"] is True
    assert pair["observation_differences"] == {"valid": -1, "invalid": 0, "unknown": 1}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "1"])
def test_invalid_measured_cost_is_unknown_and_never_a_speedup(value):
    p = protocol()
    runs = records(p)
    runs[0]["metrics"]["cli_wall_seconds"] = value
    result = compare(p, runs, verify=verify)
    assert result["all_pairs_qualified"] is False
    assert result["runs"][0]["status"] == "unknown"
    assert "median_paired_differences" not in result["groups"]["target"]


def test_verifier_failure_preserves_slot_and_does_not_trust_cached_metrics():
    p = protocol()
    runs = records(p)
    def broken(record, task, arm):
        if record["run_id"] == runs[0]["run_id"]:
            raise ValueError("source unavailable")
        return verify(record, task, arm)
    result = compare(p, runs, verify=broken)
    assert result["counts"]["provided_runs"] == 8
    assert result["runs"][0]["status"] == "unknown"
    assert result["groups"]["target"]["cost_totals"]["A"]["provider_cost_usd"]["complete"] is False


def test_prospective_protocol_requires_measurable_budgets_and_negative_control():
    p = protocol()
    p["mode"] = "prospective"
    validate_protocol(p)
    p["tasks"][0]["limits"]["provider_tokens"] = None
    with pytest.raises(ValueError):
        validate_protocol(p)
    p = protocol()
    p["tasks"] = p["tasks"][:1]
    with pytest.raises(ValueError):
        validate_protocol(p)


@pytest.mark.parametrize("mutation", ["duplicate_task", "missing_hash", "duplicate_seed", "bool_seed", "unknown_key"])
def test_protocol_cannot_silently_change_task_universe_or_inputs(mutation):
    p = protocol()
    if mutation == "duplicate_task":
        p["tasks"].append(copy.deepcopy(p["tasks"][0]))
    elif mutation == "missing_hash":
        del p["tasks"][0]["inputs"]["initial_memory"]
    elif mutation == "duplicate_seed":
        p["tasks"][0]["seeds"] = [7, 7]
    elif mutation == "bool_seed":
        p["tasks"][0]["seeds"][0] = True
    else:
        p["tasks"][0]["limtis"] = p["tasks"][0]["limits"]
    with pytest.raises(ValueError):
        validate_protocol(p)


def test_actual_historical_reports_are_requalified_instead_of_trusting_quality_flags():
    from examples.research_comparison.legacy import verify_acceptance
    fixtures = Path(__file__).with_name("fixtures")
    pinned = {"a": "602ce35eeb568bb0cafacd6679f07b4e23d80eb3488957fa92451f4cf087c954",
              "b": "6131d06b4c63f94e3fce383858be179333cba8641ed4271261b3f4e5f47848af"}
    measured = []
    for arm in ("a", "b"):
        raw = (fixtures / ("research_efficiency_smoke_sorting_" + arm + ".json")).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pinned[arm]
        record = json.loads(raw)
        record["metrics"] = {"native_actions": 0}
        output = verify_acceptance({"raw": record}, None, arm.upper())
        measured.append(output)
        assert output["metrics"]["provider_cost_usd"] is None
        record["controller_closure"]["worker_returncode"] = 1
        with pytest.raises(ValueError):
            verify_acceptance({"raw": record}, None, arm.upper())
    assert [item["metrics"]["native_actions"] for item in measured] == [5, 4]


def test_executed_order_cannot_be_rearranged_after_observing_results():
    p = protocol()
    runs = records(p)
    runs[0], runs[1] = runs[1], runs[0]
    with pytest.raises(ValueError, match="order"):
        compare(p, runs, verify=verify)


def test_maximize_direction_and_preregistered_tolerance():
    p = protocol()
    p["tasks"][0]["quality"].update(direction="maximize", max_regression=0.5)
    runs = records(p)
    runs[1]["quality"]["score"] = 8.5
    assert compare(p, runs, verify=verify)["pairs"][0]["qualified"] is True
    runs[1]["quality"]["score"] = 8.49
    assert compare(p, runs, verify=verify)["pairs"][0]["qualified"] is False


def test_lower_coverage_than_the_protocol_requires_invalidates_pair():
    p = protocol()
    p["tasks"][0]["quality"]["minimum_valid_observations"] = 2
    runs = records(p)
    runs[1]["observations"]["valid"] = 1
    assert compare(p, runs, verify=verify)["pairs"][0]["qualified"] is False


def test_medians_use_paired_differences_and_never_subtract_arm_medians():
    p = protocol()
    p["repetitions"] = 3
    for task in p["tasks"]:
        task["seeds"] = [7, 11, 13]
    runs = records(p)
    for run in runs:
        run["metrics"]["native_actions"] = (
            [1, 100, 101] if run["arm"] == "A" else [2, 3, 102])[run["repetition"]]
    group = compare(p, runs, verify=verify)["groups"]["target"]
    assert group["median_paired_differences"]["native_actions"] == 1
    assert group["median_by_arm"]["B"]["native_actions"] - group["median_by_arm"]["A"]["native_actions"] == -97


def test_plan_and_raw_measurements_are_not_modified_by_verifier():
    p = protocol()
    runs = records(p)
    before = copy.deepcopy((p, runs))
    def mutating(record, task, arm):
        result = verify(record, task, arm)
        record["extra"] = True
        task["limits"]["provider_calls"] = 0
        return result
    assert compare(p, runs, verify=mutating)["all_pairs_qualified"] is True
    assert (p, runs) == before


@pytest.mark.parametrize("payload", ['{"schema": 1, "schema": 2}', '{"value": NaN}', '{"value": Infinity}'])
def test_cli_input_rejects_ambiguous_or_nonfinite_json(payload):
    from examples.research_comparison.protocol import read_json
    with pytest.raises(ValueError):
        read_json(payload)


def test_check_command_does_not_execute_model_or_create_files(tmp_path, capsys):
    from examples.research_comparison.__main__ import main
    path = tmp_path / "protocol.json"
    p = protocol()
    path.write_text(json.dumps(p))
    assert main(["check", "--protocol", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["protocol_sha256"] == digest(p)
    assert output["schedule"] == schedule(p)
    assert list(tmp_path.iterdir()) == [path]
