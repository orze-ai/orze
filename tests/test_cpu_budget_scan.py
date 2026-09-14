"""Budget scan metadata and SQL-cost controls; no fabricated process proof.

Synthetic SETTLED rows below exercise existing ledger metadata checks only.
They do not assert a real worker, TREE/effect receipt, or authorized settlement.
All writes are confined to this test's private SQLite catalog.
"""
from dataclasses import asdict
import hashlib
import json

import pytest

from orze.core import cpu_action_budget as budget
from orze.core.execution_attempts import AttemptRef
from orze.idea_lake import IdeaLake


@pytest.fixture
def ledger(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    scope = budget.initialize(lake, results, {
        "version": 2, "resource": "cpu", "slots": 2, "wall_budget_seconds": None})
    yield lake.conn, scope
    lake.close()


def add_row(conn, scope, index, *, state="SETTLED", slot=0, timeout=1):
    permit = {
        "schema": 1, "budget_scope": scope, "reservation_id": f"{index:048x}",
        "task_id": f"task-{index}", "slot": slot, "wall_limit_seconds": timeout,
        "reserved_nanoseconds": str(budget._ns(timeout, reservation=True)),
    }
    ref = None if state == "RESERVED" else budget._json(asdict(
        AttemptRef(permit["task_id"], "action", f"attempt-{index}", 1)))
    terminal = "a" * 64 if state == "SETTLED" else None
    conn.execute(
        "INSERT INTO main.cpu_action_reservations "
        "(reservation_id,scope,task_id,slot,permit_json,ref_json,state,terminal_sha256) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (permit["reservation_id"], scope["results_dir"], permit["task_id"], slot,
         budget._json(permit), ref, state, terminal))
    conn.commit()
    return permit


@pytest.mark.parametrize("size", [2, 100])
def test_totals_uses_one_select_for_all_validated_rows(ledger, size):
    conn, scope = ledger
    for index in range(size):
        add_row(conn, scope, index)
    before = tuple(conn.iterdump())
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        charged, active = budget._totals(conn, scope)
    finally:
        conn.set_trace_callback(None)
    assert charged == size * 1_000_000_000 and active == {}
    assert tuple(conn.iterdump()) == before
    reads = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert len(reads) == 1


def test_all_three_states_preserve_point_row_shape_and_active_slots(ledger):
    conn, scope = ledger
    permits = [
        add_row(conn, scope, 0, timeout=3),
        add_row(conn, scope, 1, state="BOUND", slot=1, timeout=4),
        add_row(conn, scope, 2, state="RESERVED", timeout=5),
    ]
    assert budget._totals(conn, scope) == (12_000_000_000, {1: "task-1", 0: "task-2"})
    for permit, state in zip(permits, ("SETTLED", "BOUND", "RESERVED")):
        row = budget._reservation(conn, permit)
        assert len(row) == 7 and row[:4] == (
            scope["results_dir"], permit["task_id"], permit["slot"], budget._json(permit))
        assert row[5] == state


def test_totals_keeps_python_integer_charge_and_filters_other_scope(ledger):
    conn, scope = ledger
    one = add_row(conn, scope, 0, timeout=1e100)
    two = add_row(conn, scope, 1, timeout=1e-9)
    other = json.loads(budget._json(scope))
    other["results_dir"] += "-other"
    other["policy_sha256"] = hashlib.sha256(budget._json({
        key: value for key, value in other.items() if key != "policy_sha256"
    }).encode()).hexdigest()
    add_row(conn, other, 2, timeout=3)
    charged, active = budget._totals(conn, scope)
    assert type(charged) is int
    assert charged == int(one["reserved_nanoseconds"]) + int(two["reserved_nanoseconds"])
    assert active == {}


@pytest.mark.parametrize("corruption", ["canonical", "reference_task", "terminal_marker"])
def test_scan_keeps_existing_row_rejections(ledger, corruption):
    conn, scope = ledger
    permit = add_row(conn, scope, 0, state="BOUND")
    if corruption == "canonical":
        conn.execute("UPDATE cpu_action_reservations SET permit_json=?",
                     (json.dumps(permit, indent=1),))
    elif corruption == "reference_task":
        ref = asdict(AttemptRef("different-task", "action", "attempt-0", 1))
        conn.execute("UPDATE cpu_action_reservations SET ref_json=?", (budget._json(ref),))
    else:
        conn.execute("UPDATE cpu_action_reservations SET terminal_sha256=?", ("b" * 64,))
    conn.commit()
    before = tuple(conn.iterdump())
    with pytest.raises(budget.CpuBudgetHOLD):
        budget._totals(conn, scope)
    assert tuple(conn.iterdump()) == before


def test_sql_reservation_id_is_not_replaced_by_permit_identity(ledger):
    conn, scope = ledger
    permit = add_row(conn, scope, 0)
    conn.execute("UPDATE cpu_action_reservations SET reservation_id=?", ("f" * 48,))
    conn.commit()
    before = tuple(conn.iterdump())
    with pytest.raises(budget.CpuBudgetHOLD, match="permit_changed"):
        budget._reservation(conn, permit)
    with pytest.raises(budget.CpuBudgetHOLD, match="permit_changed"):
        budget._totals(conn, scope)
    assert tuple(conn.iterdump()) == before
