"""Propose is a new explicit decision, not an execution or source permission."""
import copy

import pytest

from orze.core import research_interfaces as api

REF = {"task_id": "source", "phase": "action", "attempt_id": "attempt", "generation": 1}


def decision():
    return {"kind": "Propose", "request_id": "plan-1", "task_id": "idea-check",
            "reason": "check an explicit question",
            "domain_request": {"version": 1, "purpose": "check candidate",
                "inputs": {}, "timeout_seconds": 2, "outputs": {},
                "input_artifact_ids": [], "payload": {"command": ["true"]}}}


def snapshot():
    return {"queue": [], "active": False, "now": 10,
            "recorded_evidence": {"results": [{"ref": dict(REF), "outcome": "completed",
                "artifact_records": [{"artifact_id": "artifact-1", "producer": dict(REF)}]}],
                "unavailable": [], "more_available": False}}


def policy(monkeypatch, selected, *, domain=True, mutate=False):
    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))

    class Policy:
        def __init__(self, declaration):
            pass

        def decide(self, view, budget):
            if mutate:
                view["recorded_evidence"]["results"][0]["artifact_records"].append(
                    {"artifact_id": "unseen", "producer": dict(REF)})
            return copy.deepcopy(selected)

    api.register_policy("proposal_contract_fixture", "proposal.contract.v1", Policy)
    cfg = {"action_policy": {"version": 1, "kind": "proposal_contract_fixture",
                            "idle": "wait", "wait_seconds": 1}}
    if domain:
        cfg["action_domain"] = {"version": 1, "kind": "command", "config": {}}
    return api.BoundPolicy(api.capture_interfaces(cfg))


@pytest.mark.parametrize("with_source", [False, True])
def test_policy_can_propose_a_bounded_task_from_its_captured_view(monkeypatch, with_source):
    selected, view = decision(), snapshot()
    if with_source:
        selected["domain_request"]["input_artifact_ids"] = ["artifact-1"]
    before = copy.deepcopy(view)
    assert policy(monkeypatch, selected).decide(view, {}) == selected
    assert view == before


@pytest.mark.parametrize("fault", ["request_id", "task_id", "extra", "blank_reason",
                                   "large_reason", "no_domain", "missing_source",
                                   "failed_source", "producer_changed", "bool_generation",
                                   "mutated_view", "duplicate_source", "unknown_request_field"])
def test_proposal_decision_cannot_expand_its_captured_input_set(monkeypatch, fault):
    selected, view = decision(), snapshot()
    selected["domain_request"]["input_artifact_ids"] = ["artifact-1"]
    if fault in ("request_id", "task_id"):
        selected[fault] = "../bad"
    elif fault == "extra":
        selected["skip_dedup"] = True
    elif fault == "blank_reason":
        selected["reason"] = " "
    elif fault == "large_reason":
        selected["reason"] = "x" * 1025
    elif fault in ("missing_source", "mutated_view"):
        selected["domain_request"]["input_artifact_ids"] = ["unseen"]
    elif fault == "failed_source":
        view["recorded_evidence"]["results"][0]["outcome"] = "failed"
    elif fault == "producer_changed":
        view["recorded_evidence"]["results"][0]["artifact_records"][0]["producer"]["generation"] = 2
    elif fault == "bool_generation":
        view["recorded_evidence"]["results"][0]["ref"]["generation"] = True
        view["recorded_evidence"]["results"][0]["artifact_records"][0]["producer"]["generation"] = True
    elif fault == "duplicate_source":
        selected["domain_request"]["input_artifact_ids"] *= 2
    elif fault == "unknown_request_field":
        selected["domain_request"]["unbounded_callback"] = "no"
    before = copy.deepcopy(view)
    with pytest.raises(api.ResearchInterfaceError):
        policy(monkeypatch, selected, domain=fault != "no_domain",
               mutate=fault == "mutated_view").decide(view, {})
    assert view == before
