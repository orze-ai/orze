"""Independent offline measurement checks, not additional research campaigns.

The two complete JSON fixtures are byte-exact copies of the initial actual
sorting/default smoke A/B reports. Their original absolute paths are historical
metadata only: no test opens those paths or launches a process. Counterfactual
edits exercise record consistency, not Core execution authority or cryptographic
attestation. The six-pair aggregation example below is synthetic bookkeeping,
not six measured replications or formal speed evidence.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from examples.acceptance.common import digest
from examples.research_efficiency.run import qualify, summarize


FIXTURES = Path(__file__).with_name("fixtures")
PINNED = {
    "A": ("research_efficiency_smoke_sorting_a.json",
          "602ce35eeb568bb0cafacd6679f07b4e23d80eb3488957fa92451f4cf087c954"),
    "B": ("research_efficiency_smoke_sorting_b.json",
          "6131d06b4c63f94e3fce383858be179333cba8641ed4271261b3f4e5f47848af"),
}


def _record(arm="A"):
    name, expected = PINNED[arm]
    raw = (FIXTURES / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    return json.loads(raw)


@pytest.mark.parametrize("arm,actions,reserved,analysis", [("A", 5, 10, 1), ("B", 4, 8, 0)])
def test_actual_development_reports_pass_without_claiming_formal_speed(arm, actions, reserved, analysis):
    record = _record(arm)
    before = copy.deepcopy(record)
    metrics = qualify(record)
    # JSON normalizes the action-signature tuple/list representation only.
    assert json.loads(json.dumps(metrics)) == record["metrics"]
    assert (metrics["native_actions"], metrics["reserved_seconds"],
            metrics["analysis_actions"]) == (actions, reserved, analysis)
    assert metrics["selection"]["candidate"] == "baseline"
    assert metrics["selection"]["cost"] == 9
    assert record == before


def test_settled_charge_must_belong_to_the_counted_full_attempt_ref():
    record = _record()
    reservations = record["database"]["cpu_action_reservations"]
    # A real, well-typed sibling Ref, not malformed JSON or an invented worker.
    assert reservations[0]["ref_json"] != reservations[1]["ref_json"]
    reservations[0]["ref_json"] = reservations[1]["ref_json"]
    with pytest.raises(ValueError):
        qualify(record)


def test_controller_closure_must_match_this_runs_captured_ready_binding():
    record = _record()
    foreign = _record("B")
    assert foreign["controller_binding"] != record["controller_binding"]
    # This is a genuine successful closure, but it belongs to the other CLI.
    record["controller_closure"] = foreign["controller_closure"]
    with pytest.raises(ValueError):
        qualify(record)


def test_selected_trace_ref_must_match_the_published_evaluator_and_real_attempt():
    record = _record()
    entry = record["trace"][-1]
    replica = json.loads(record["database"]["replication_requests"][0]["record_json"])
    selected = next(row for row in entry["snapshot"]["recorded_evidence"]["results"]
                    if row["ref"]["task_id"] == replica["task_id"])
    selected["ref"]["generation"] += 1
    # Keep the diagnostic snapshot self-consistent. This hash is not authority;
    # the untouched DB attempt and observation still name the original Ref.
    entry["snapshot_sha256"] = digest(entry["snapshot"])
    with pytest.raises(ValueError):
        qualify(record)


def test_negative_cli_elapsed_is_not_a_successful_speed_measurement():
    record = _record()
    assert record["finished_monotonic"] > record["started_monotonic"]
    record["wall_seconds"] = -1.0
    with pytest.raises(ValueError):
        qualify(record)


def test_failed_pair_stays_in_fixed_denominator_without_success_only_median():
    # Six copies exercise aggregation logic only; they are not executed runs.
    records = []
    for repetition in range(6):
        for arm in ("A", "B"):
            original = _record(arm)
            records.append({"repetition": repetition, "domain": "sorting",
                            "variant": "default", "arm": arm,
                            "quality_passed": True, "metrics": qualify(original)})
    positive = summarize(records)
    group = positive["groups"]["sorting_default"]
    assert group["pairs"] == group["quality_passed_pairs"] == 6
    assert group["median_paired_differences"]["native_actions"] == -1
    assert group["median_paired_differences"]["reserved_seconds"] == -2

    records[1]["quality_passed"] = False
    records[1].pop("metrics")
    summary = summarize(records)
    group = summary["groups"]["sorting_default"]
    assert len(summary["pairs"]) == 24
    assert group["pairs"] == 6 and group["quality_passed_pairs"] == 5
    assert "median_paired_differences" not in group and "median_by_arm" not in group
    assert summary["all_quality_passed"] is False
