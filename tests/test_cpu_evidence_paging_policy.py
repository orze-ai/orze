"""Explicit paging decisions remain data-only and opt-in."""
import copy

import pytest

from orze.core import cpu_execution as execution
from orze.core import research_interfaces as api


REF = {"task_id": "later-source", "phase": "action", "attempt_id": "attempt-1", "generation": 1}


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))
    selected = {"decision": {"kind": "ReadEvidence", "cursor": "next-token"}, "mutate": False}

    class Policy:
        def __init__(self, declaration):
            pass

        def decide(self, snapshot, budget):
            if selected["mutate"]:
                snapshot["evidence_page"]["next_cursor"] = "forged-token"
            return copy.deepcopy(selected["decision"])

    api.register_policy("paging_test", "paging.test.v1", Policy)
    cfg = {"action_policy": {"version": 2, "kind": "paging_test", "idle": "wait",
                             "wait_seconds": .01, "evidence_page_size": 8}}
    return cfg, selected


def test_paged_policy_is_an_explicit_new_version(policy):
    cfg, _ = policy
    assert execution.action_policy(cfg) == cfg["action_policy"]
    assert api.BoundPolicy(api.capture_interfaces(cfg)).declaration == cfg["action_policy"]


@pytest.mark.parametrize("change", [{"version": 1}, {"evidence_page_size": True},
                                   {"evidence_page_size": 0}, {"evidence_page_size": 33},
                                   {"kind": "queue"}, {"evidence_page_size": 8.0}])
def test_invalid_or_implicit_paging_declarations_are_rejected(policy, change):
    cfg, _ = policy
    cfg["action_policy"].update(change)
    with pytest.raises(ValueError):
        execution.action_policy(cfg)


def test_next_cursor_is_checked_against_private_capture(policy):
    cfg, selected = policy
    bound = api.BoundPolicy(api.capture_interfaces(cfg))
    view = {"queue": [], "active": False, "now": 1,
            "evidence_page": {"next_cursor": "next-token"}}
    assert bound.decide(view, {}) == selected["decision"]
    selected.update(decision={"kind": "ReadEvidence", "cursor": "forged-token"}, mutate=True)
    with pytest.raises(api.ResearchInterfaceError):
        bound.decide(view, {})
    assert view["evidence_page"]["next_cursor"] == "next-token"


@pytest.mark.parametrize("refs", [[], [REF, REF], [{**REF, "generation": True}],
                                 [{**REF, "phase": "training"}], [{**REF, "extra": 1}]])
def test_invalid_selected_refs_are_not_query_authority(policy, refs):
    cfg, selected = policy
    selected["decision"] = {"kind": "SelectEvidence", "refs": refs}
    with pytest.raises(api.ResearchInterfaceError):
        api.BoundPolicy(api.capture_interfaces(cfg)).decide({"queue": []}, {})


def test_select_is_a_read_request_not_fabricated_source_metadata(policy):
    cfg, selected = policy
    selected["decision"] = {"kind": "SelectEvidence", "refs": [REF]}
    bound = api.BoundPolicy(api.capture_interfaces(cfg))
    assert bound.decide({"queue": []}, {}) == selected["decision"]
    selected["decision"]["artifact_records"] = []
    with pytest.raises(api.ResearchInterfaceError):
        bound.decide({"queue": []}, {})


def test_legacy_policy_cannot_enable_paging_by_mutating_snapshot(policy):
    cfg, selected = policy
    cfg["action_policy"].pop("evidence_page_size")
    cfg["action_policy"]["version"] = 1
    bound = api.BoundPolicy(api.capture_interfaces(cfg))
    with pytest.raises(api.ResearchInterfaceError):
        bound.decide({"queue": [], "evidence_page": {"next_cursor": "next-token"}}, {})
