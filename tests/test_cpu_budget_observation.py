"""Research accounting observations must not initialize or grant a budget."""
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from orze.core import cpu_action_budget as budget
from orze.core.cpu_budget_observation import observe_cpu_budget, local_cpu_budget_hold
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    declaration = {"version": 1, "resource": "cpu", "slots": 2, "wall_budget_seconds": 10}
    yield lake, results, declaration
    lake.close()


def unchanged(lake):
    return "\n".join(lake.conn.iterdump())


def test_missing_namespace_is_unknown_and_no_database_writes(project):
    lake, results, declaration = project
    before = unchanged(lake)
    view = observe_cpu_budget(lake, results, declaration)
    assert view["availability"] == "unavailable" and view["reason"] == "cpu_budget_not_initialized"
    assert "accounting" not in view and "remaining_wall_seconds" not in view
    assert unchanged(lake) == before
    assert not local_cpu_budget_hold(lake, results, declaration)


@pytest.mark.parametrize("unlimited", [False, True])
def test_observation_matches_authoritative_accounting_and_detaches(project, unlimited):
    lake, results, declaration = project
    if unlimited:
        declaration = {**declaration, "version": 2, "wall_budget_seconds": None}
    scope = budget.initialize(lake, results, declaration)
    budget.reserve(lake, scope, "reserved-task", 3)
    before = unchanged(lake)
    view = observe_cpu_budget(lake, results, declaration)
    assert view["availability"] == "verified" and view["policy_sha256"] == scope["policy_sha256"]
    assert view["accounting"] == budget.snapshot(lake, scope)
    assert view["accounting"]["free_slots"] == 1
    assert view["accounting"]["remaining_wall_seconds"] == (None if unlimited else 7)
    view["accounting"]["free_slots"] = 999
    view["declaration"]["slots"] = 999
    assert observe_cpu_budget(lake, results, declaration)["accounting"]["free_slots"] == 1
    assert unchanged(lake) == before


def test_read_only_lake_handle_and_stop_are_observable(project):
    lake, results, declaration = project
    scope = budget.initialize(lake, results, declaration)
    budget.record_decision(lake, scope, {"kind": "Stop", "reason": "operator decision", "wakeup": None})
    before = unchanged(lake)
    with sqlite3.connect(Path(lake.db_path).as_uri() + "?mode=ro", uri=True) as conn:
        readonly = SimpleNamespace(conn=conn, db_path=lake.db_path)
        assert observe_cpu_budget(readonly, results, declaration)["accounting"]["stopped"] is True
    assert unchanged(lake) == before


@pytest.mark.parametrize("failure", ["partial_schema", "scope_policy", "bad_permit", "orphan", "trigger"])
def test_invalid_existing_accounting_never_becomes_uninitialized_or_free(project, failure):
    lake, results, declaration = project
    scope = budget.initialize(lake, results, declaration)
    budget.reserve(lake, scope, "held", 3)
    if failure == "partial_schema":
        lake.conn.execute("DROP TABLE cpu_action_decisions")
    elif failure == "scope_policy":
        declaration = {**declaration, "slots": 3}
    elif failure == "bad_permit":
        lake.conn.execute("UPDATE cpu_action_reservations SET permit_json='{}'")
    elif failure == "orphan":
        lake.conn.execute("DELETE FROM cpu_action_scopes")
    else:
        lake.conn.execute("CREATE TRIGGER hidden AFTER UPDATE ON cpu_action_scopes BEGIN SELECT 1; END")
    lake.conn.commit()
    before = unchanged(lake)
    with pytest.raises(budget.CpuBudgetHOLD):
        observe_cpu_budget(lake, results, declaration)
    assert unchanged(lake) == before


def test_other_scope_does_not_supply_this_scopes_quota(project):
    lake, results, declaration = project
    other = results.parent / "other"
    other.mkdir()
    budget.initialize(lake, other, declaration)
    before = unchanged(lake)
    assert observe_cpu_budget(lake, results, declaration)["availability"] == "unavailable"
    assert unchanged(lake) == before


def test_open_caller_transaction_is_not_committed_or_rolled_back(project):
    lake, results, declaration = project
    lake.conn.execute("BEGIN")
    with pytest.raises(budget.CpuBudgetHOLD, match="caller_transaction_active"):
        observe_cpu_budget(lake, results, declaration)
    assert lake.conn.in_transaction
    lake.conn.rollback()


def test_local_refusal_latch_can_be_rechecked_without_ledger_audit(project, monkeypatch):
    lake, results, declaration = project
    scope = budget.initialize(lake, results, declaration)
    assert not local_cpu_budget_hold(lake, results, declaration)
    monkeypatch.setattr(budget, "_HELD", {budget._key(scope)})
    assert local_cpu_budget_hold(lake, results, declaration)
    assert observe_cpu_budget(lake, results, declaration)["accounting"]["stopped"]
