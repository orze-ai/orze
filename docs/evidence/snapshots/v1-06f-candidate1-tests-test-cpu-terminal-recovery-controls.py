"""No-process compatibility controls; these are not TREE/crash evidence.

Reuse the unchanged pre-F budget fixture's genuine never-exec attempts.
Only initialize may add the new budget table to a private legacy-shaped DB.
"""
import json

import pytest

from orze.core import cpu_action_budget as budget
from test_cpu_action_budget import context, _intent, _not_started


def reservations(c):
    return [tuple(row) for row in c.lake.conn.execute(
        "SELECT * FROM cpu_action_reservations ORDER BY reservation_id")]


@pytest.mark.parametrize("state", ["RESERVED", "LAUNCHING", "NOT_STARTED"])
def test_open_or_never_exec_reservation_is_retained_without_automatic_settlement(context, state):
    c = context
    permit = budget.reserve(c.lake, c.scope, "control-task", 2)
    if state != "RESERVED":
        folder, ref = _intent(c, permit)
        if state == "NOT_STARTED":
            _not_started(c, folder, ref)
    before = reservations(c)
    result = budget.reconcile_confirmed_terminals(c.lake, c.scope)
    assert result["examined"] == 1
    assert result["settled"] == result["already_settled"] == []
    assert len(result["retained"]) == 1
    assert result["retained"][0]["reservation_id"] == permit["reservation_id"]
    assert isinstance(result["retained"][0]["reason"], str) and result["retained"][0]["reason"]
    assert reservations(c) == before
    status = budget.snapshot(c.lake, c.scope)
    assert status["active_reservations"] == 1 and status["reserved_wall_seconds"] == 2
    assert not status["stopped"]
    saved = [tuple(row) for row in c.lake.conn.execute("SELECT * FROM cpu_action_recovery")]
    expected = json.loads(json.dumps(result))
    result["retained"][0]["reason"] = "caller-mutated-detached-diagnostic"
    assert budget.reconcile_confirmed_terminals(c.lake, c.scope) == expected
    assert [tuple(row) for row in c.lake.conn.execute("SELECT * FROM cpu_action_recovery")] == saved


def test_explicit_legacy_not_started_settlement_after_stop_remains_compatible(context):
    c = context
    permit = budget.reserve(c.lake, c.scope, "legacy-no-exec", 2)
    folder, ref = _intent(c, permit)
    terminal = _not_started(c, folder, ref)
    budget.record_decision(c.lake, c.scope, {"kind": "Stop", "reason": "operator_stop", "wakeup": None})
    before = reservations(c)
    diagnostic = budget.reconcile_confirmed_terminals(c.lake, c.scope)
    assert diagnostic["settled"] == diagnostic["already_settled"] == []
    assert reservations(c) == before
    assert budget.settle(c.lake, permit, ref, terminal) == "settled"
    assert budget.settle(c.lake, permit, ref, terminal) == "duplicate"
    status = budget.snapshot(c.lake, c.scope)
    assert status["active_reservations"] == 0 and status["reserved_wall_seconds"] == 2
    assert status["stopped"] and status["stop"]["reason"] == "operator_stop"


def test_missing_new_table_is_not_silently_created_by_readonly_budget_api(context):
    c = context
    permit = budget.reserve(c.lake, c.scope, "existing-reservation", 2)
    before = reservations(c)
    # Deliberately form the former three-table shape in this private test DB.
    # No production migration, old committed history, or attempt is rewritten.
    c.lake.conn.execute("DROP TABLE main.cpu_action_recovery")
    c.lake.conn.commit()
    schema = [tuple(row) for row in c.lake.conn.execute(
        "SELECT type,name,sql FROM sqlite_master ORDER BY type,name")]
    with pytest.raises(budget.CpuBudgetHOLD, match="schema_invalid"):
        budget.snapshot(c.lake, c.scope)
    assert [tuple(row) for row in c.lake.conn.execute(
        "SELECT type,name,sql FROM sqlite_master ORDER BY type,name")] == schema
    assert reservations(c) == before
    assert not c.lake.conn.in_transaction
    assert budget.initialize(c.lake, c.results, c.declaration) == c.scope
    assert budget.initialize(c.lake, c.results, c.declaration) == c.scope
    assert reservations(c) == before
    assert budget.require_permit(c.lake, permit) == permit
