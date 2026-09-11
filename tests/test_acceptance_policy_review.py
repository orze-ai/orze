"""Independent decision-logic review using the actual shared CLI trajectories.

The session plugin runs the four frozen product scenarios once. These tests
only copy their recorded metadata; the counterfactual edits below are NOT new
observations, verified source records, execution grants, or additional actions.
"""
import copy

import pytest


pytest_plugins = ["examples.acceptance.testing"]

_DECLARATION = {"version": 1, "kind": "acceptance", "idle": "stop",
                "wait_seconds": 0.05, "config": {}}


def _observations(snapshot):
    return [observation
            for result in snapshot["recorded_evidence"]["results"]
            for observation in result["observation_records"]]


def _observation(snapshot, *, candidate, operation):
    matches = [item for item in _observations(snapshot)
               if item["values"]["candidate"] == candidate
               and item["values"]["operation"] == operation]
    assert len(matches) == 1
    return matches[0]


def _decide(entry, snapshot):
    from examples.acceptance.policy import CommonPolicy
    return CommonPolicy(copy.deepcopy(_DECLARATION)).decide(
        snapshot, copy.deepcopy(entry["budget"]))


def _first_replication(run):
    return next(entry for entry in run["trace"]
                if entry["decision"]["kind"] == "Replicate")


@pytest.mark.parametrize("domain", ["sorting", "compression"])
def test_real_unknown_with_best_cost_still_requires_analysis(acceptance_runs, domain):
    run = acceptance_runs[domain + "_default"]
    entry = next(item for item in run["trace"]
                 if item["decision"]["kind"] == "Propose"
                 and any(obs["validation"]["status"] == "unknown"
                         for obs in _observations(item["snapshot"]))
                 and not any(obs["values"]["operation"] == "analyze"
                             for obs in _observations(item["snapshot"])))
    original = copy.deepcopy(entry["snapshot"])
    changed = copy.deepcopy(original)
    unknown = _observation(changed, candidate="unchecked", operation="measure")
    old_cost = unknown["values"]["cost"]
    assert unknown["validation"]["status"] == "unknown"
    unknown["values"]["cost"] = 0
    decision = _decide(entry, changed)
    assert decision == entry["decision"]
    assert decision["domain_request"]["payload"]["operation"] == "analyze"
    unknown["values"]["cost"] = old_cost
    assert changed == original
    assert entry["snapshot"] == original


@pytest.mark.parametrize("domain", ["sorting", "compression"])
def test_only_valid_cost_change_selects_exact_analysis_evaluator(acceptance_runs, domain):
    entry = _first_replication(acceptance_runs[domain + "_default"])
    original = copy.deepcopy(entry["snapshot"])
    changed = copy.deepcopy(original)
    baseline = _observation(changed, candidate="baseline", operation="measure")
    analysis = _observation(changed, candidate="unchecked", operation="analyze")
    unknown = _observation(changed, candidate="unchecked", operation="measure")
    assert baseline["validation"]["status"] == analysis["validation"]["status"] == "valid"
    assert analysis["values"]["cost"] > baseline["values"]["cost"] > 0
    assert entry["decision"]["source_ref"] == baseline["evaluator"]
    old_cost = analysis["values"]["cost"]
    analysis["values"]["cost"] = 0
    decision = _decide(entry, changed)
    assert decision["kind"] == "Replicate"
    # This repeats the analysis evaluator, NOT the still-unknown source run.
    assert decision["source_ref"] == analysis["evaluator"]
    assert decision["source_ref"] != unknown["evaluator"]
    assert unknown["validation"]["status"] == "unknown"
    analysis["values"]["cost"] = old_cost
    assert changed == original
    assert entry["snapshot"] == original


@pytest.mark.parametrize("field", ["protocol_fingerprint", "comparison_scope"])
def test_incomparable_best_observation_is_not_a_winner(acceptance_runs, field):
    entry = _first_replication(acceptance_runs["sorting_counterfactual"])
    original = copy.deepcopy(entry["snapshot"])
    changed = copy.deepcopy(original)
    challenger = _observation(changed, candidate="challenger", operation="measure")
    assert challenger["validation"]["status"] == "valid"
    assert entry["decision"]["source_ref"] == challenger["evaluator"]
    old_value = challenger[field]
    challenger[field] = "0" * 64 if old_value != "0" * 64 else "1" * 64
    try:
        decision = _decide(entry, changed)
    except ValueError:
        # Explicit refusal is allowed for an incompatible evidence window.
        pass
    else:
        assert decision["kind"] != "Stop"
        assert not (decision["kind"] == "Replicate"
                    and decision["source_ref"] == challenger["evaluator"])
    challenger[field] = old_value
    assert changed == original
    assert entry["snapshot"] == original


@pytest.mark.parametrize("domain", ["sorting", "compression"])
def test_actual_queued_replica_is_not_completed_stop_evidence(acceptance_runs, domain):
    run = acceptance_runs[domain + "_default"]
    replication_index = next(index for index, entry in enumerate(run["trace"])
                             if entry["decision"]["kind"] == "Replicate")
    entry = next(item for item in run["trace"][replication_index + 1:]
                 if item["snapshot"]["queue"])
    original = copy.deepcopy(entry["snapshot"])
    queued = {item["idea_id"] for item in original["queue"]}
    completed = {item["ref"]["task_id"]
                 for item in original["recorded_evidence"]["results"]}
    assert not queued.intersection(completed)
    decision = _decide(entry, copy.deepcopy(original))
    assert decision["kind"] == "Execute"
    assert decision["task_id"] in queued
    assert decision == entry["decision"]
    assert entry["snapshot"] == original
