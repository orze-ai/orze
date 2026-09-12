"""Pause is a quiescent invocation decision, not a stop-latch reset or refund."""
import copy
import json

import pytest

from orze.core import cpu_action_budget as budget
from orze.core import research_interfaces as api
from orze.idea_lake import IdeaLake


PAUSE = {"kind": "Pause", "reason": "evidence_unavailable", "wakeup": None}


def selected_policy(monkeypatch, decision, *, mutate=False):
    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))

    class Policy:
        def __init__(self, declaration):
            pass

        def decide(self, view, resources):
            if mutate:
                view["active"] = False
                resources["active_reservations"] = 0
            return copy.deepcopy(decision)

    api.register_policy("pause_review_fixture", "pause.review.v1", Policy)
    return api.BoundPolicy(api.capture_interfaces({"action_policy": {
        "version": 1, "kind": "pause_review_fixture", "idle": "wait", "wait_seconds": 1}}))


def test_explicit_quiescent_pause_is_a_valid_policy_decision(monkeypatch):
    view = {"queue": [], "active": False, "now": 10}
    resources = {"active_reservations": 0}
    assert selected_policy(monkeypatch, PAUSE).decide(view, resources) == PAUSE
    assert view == {"queue": [], "active": False, "now": 10}
    assert resources == {"active_reservations": 0}


@pytest.mark.parametrize("active,reservations", [(True, 0), (False, 1), (None, 0), (False, False)])
def test_callback_cannot_turn_active_or_unknown_state_into_quiescent_pause(
        monkeypatch, active, reservations):
    view = {"queue": [], "active": active, "now": 10}
    resources = {"active_reservations": reservations}
    with pytest.raises(api.ResearchInterfaceError):
        selected_policy(monkeypatch, PAUSE, mutate=True).decide(view, resources)
    assert view["active"] is active and resources["active_reservations"] is reservations


@pytest.mark.parametrize("change", [{"wakeup": 20}, {"reason": " "},
                                   {"reason": "x" * 129}, {"resume": True}])
def test_pause_does_not_accept_an_implicit_timer_or_unbounded_metadata(monkeypatch, change):
    with pytest.raises(api.ResearchInterfaceError):
        selected_policy(monkeypatch, {**PAUSE, **change}).decide(
            {"queue": [], "active": False, "now": 10}, {"active_reservations": 0})


@pytest.fixture
def ledger(tmp_path):
    directory = tmp_path / "results"
    directory.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    declaration = {"version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 10}
    scope = budget.initialize(lake, directory, declaration)
    try:
        yield lake, directory, declaration, scope
    finally:
        lake.close()


def test_pause_is_durable_without_permanently_stopping_the_resource_scope(ledger):
    lake, directory, declaration, scope = ledger
    record = budget.record_decision(lake, scope, PAUSE)
    raw = lake.conn.execute("SELECT record_json FROM cpu_action_decisions").fetchone()[0]
    assert json.loads(raw) == record
    assert {key: record[key] for key in PAUSE} == PAUSE
    assert lake.conn.execute("SELECT stop_json FROM cpu_action_scopes").fetchone()[0] is None
    peer = IdeaLake(lake.db_path)
    try:
        reopened = budget.initialize(peer, directory, declaration)
        assert reopened == scope
        assert not budget.snapshot(peer, reopened)["stopped"]
        assert budget.snapshot(peer, reopened)["reserved_wall_seconds"] == 0
        permit = budget.reserve(peer, reopened, "later-admitted-task", 2)
        assert permit is not None
        assert budget.require_permit(peer, permit) == permit
        assert budget.snapshot(peer, reopened)["reserved_wall_seconds"] == 2
    finally:
        peer.close()


def test_pause_cannot_label_a_live_reservation_quiescent_or_release_it(ledger):
    lake, _, _, scope = ledger
    permit = budget.reserve(lake, scope, "unconfirmed-work", 2)
    before = budget.snapshot(lake, scope)
    with pytest.raises(budget.CpuBudgetHOLD):
        budget.record_decision(lake, scope, PAUSE)
    assert budget.snapshot(lake, scope) == before
    assert budget.require_permit(lake, permit) == permit
    assert lake.conn.execute("SELECT count(*) FROM cpu_action_decisions").fetchone()[0] == 0


def test_pause_does_not_clear_an_existing_permanent_stop(ledger):
    lake, directory, declaration, scope = ledger
    stop = {"kind": "Stop", "reason": "operator_stop", "wakeup": None}
    budget.record_decision(lake, scope, stop)
    budget.record_decision(lake, scope, PAUSE)
    assert budget.initialize(lake, directory, declaration) == scope
    assert budget.snapshot(lake, scope)["stop"] == stop
    with pytest.raises(budget.CpuBudgetHOLD, match="stopped"):
        budget.reserve(lake, scope, "forbidden", 2)
