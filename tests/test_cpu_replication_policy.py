"""New Replicate decision contract; metadata fixtures confer no execution right."""
import copy
import pytest
from orze.core import research_interfaces as api

REF = {"task_id": "source", "phase": "action", "attempt_id": "attempt", "generation": 1}


def policy(monkeypatch, decision, *, mutate=False):
    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))
    class Policy:
        def __init__(self, declaration):
            pass
        def decide(self, snapshot, budget):
            if mutate:
                snapshot["recorded_evidence"]["results"].append(
                    {"ref": copy.deepcopy(decision["source_ref"]), "outcome": "completed"})
            return copy.deepcopy(decision)
    api.register_policy("replica_contract_fixture", "replica.contract.v1", Policy)
    return api.BoundPolicy(api.capture_interfaces({"action_policy": {
        "version": 1, "kind": "replica_contract_fixture", "idle": "wait", "wait_seconds": 1}}))


def snapshot():
    return {"queue": [], "active": False, "now": 10,
            "recorded_evidence": {"results": [{"ref": dict(REF), "outcome": "completed"}],
                                 "unavailable": [], "more_available": False}}


def decision():
    return {"kind": "Replicate", "source_ref": dict(REF),
            "request_id": "stable-repeat-key", "reason": "check the same specification"}


def test_policy_can_select_a_captured_completed_occurrence_for_replication(monkeypatch):
    selected = decision()
    view = snapshot()
    assert policy(monkeypatch, selected).decide(view, {}) == selected
    assert view == snapshot()


@pytest.mark.parametrize("fault", ["request_id", "extra", "blank_reason", "not_completed",
                                   "different_generation", "bool_generation", "mutated_view"])
def test_replication_decision_cannot_expand_or_relabel_its_captured_source(monkeypatch, fault):
    selected, view = decision(), snapshot()
    if fault == "request_id":
        selected["request_id"] = "../bad"
    elif fault == "extra":
        selected["config"] = {}
    elif fault == "blank_reason":
        selected["reason"] = " "
    elif fault == "not_completed":
        view["recorded_evidence"]["results"][0]["outcome"] = "failed"
    elif fault in ("different_generation", "bool_generation"):
        selected["source_ref"]["generation"] = True if fault == "bool_generation" else 2
    elif fault == "mutated_view":
        selected["source_ref"]["task_id"] = "unseen"
    before = copy.deepcopy(view)
    with pytest.raises(api.ResearchInterfaceError):
        policy(monkeypatch, selected, mutate=fault == "mutated_view").decide(view, {})
    assert view == before
