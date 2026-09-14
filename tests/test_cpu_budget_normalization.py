"""Compare captured budget code with current readers, including bad history.

These are ledger metadata tests, not fabricated native execution receipts.
Execution and recovery are exercised by the existing product suites.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from orze.core import cpu_action_budget as budget
from orze.idea_lake import IdeaLake
from test_cpu_budget_scan import add_row, ledger
from test_cpu_budget_scan_review import CASES, metadata_case, result


BASELINE = Path(__file__).resolve().parents[1] / (
    "docs/evidence/runs/2026-09-14-budget-normalization/baseline/cpu_action_budget.py")
BASE_SHA = "5101beae489135f1162035da8483232a186faac4af5555012f7edb5acc21b23c"


def load_baseline():
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == BASE_SHA
    spec = importlib.util.spec_from_file_location("orze.core._normalization_baseline", BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case", CASES)
def test_captured_budget_and_current_reject_the_same_bad_history(tmp_path, case):
    old = load_baseline()
    lake, scope, permits = metadata_case(tmp_path, case, old)
    try:
        before = tuple(lake.conn.iterdump())
        assert result(lambda: budget._totals(lake.conn, scope)) == result(
            lambda: old._totals(lake.conn, scope))
        for permit in permits:
            assert result(lambda: budget._reservation(lake.conn, permit)) == result(
                lambda: old._reservation(lake.conn, permit))
        assert tuple(lake.conn.iterdump()) == before
    finally:
        lake.close()


@pytest.mark.parametrize("state,decodes", [("RESERVED", 1), ("BOUND", 2), ("SETTLED", 2)])
def test_scan_decodes_only_stored_documents(ledger, monkeypatch, state, decodes):
    conn, scope = ledger
    add_row(conn, scope, 1, state=state)
    original = budget._decode
    calls = []

    def counted(raw):
        calls.append(raw)
        return original(raw)

    monkeypatch.setattr(budget, "_decode", counted)
    expected_active = {} if state == "SETTLED" else {0: "task-1"}
    assert budget._totals(conn, scope) == (1_000_000_000, expected_active)
    assert len(calls) == decodes


@pytest.mark.parametrize("field,value", [
    ("schema", True), ("schema", 2), ("slot", True), ("slot", 2),
    ("wall_limit_seconds", False), ("wall_limit_seconds", 0),
    ("wall_limit_seconds", 1e-10), ("wall_limit_seconds", 1.0000000001),
    ("reserved_nanoseconds", "01"), ("reserved_nanoseconds", 1000000000),
    ("reservation_id", "x" * 48), ("task_id", ".."),
    ("budget_scope", []),
])
def test_parsed_permit_rejections_match_baseline(ledger, field, value):
    conn, scope = ledger
    permit = add_row(conn, scope, 1)
    permit[field] = value
    conn.execute("UPDATE cpu_action_reservations SET permit_json=?", (budget._json(permit),))
    conn.commit()
    old = load_baseline()
    actual = result(lambda: budget._totals(conn, scope))
    assert actual == result(lambda: old._totals(conn, scope))
    assert not actual["accepted"]


@pytest.mark.parametrize("version,wall", [(1, 10), (2, None)])
def test_public_normalizers_still_return_detached_canonical_values(ledger, version, wall):
    conn, scope = ledger
    old = load_baseline()
    scope = deepcopy(scope)
    scope["declaration"].update(version=version, wall_budget_seconds=wall)
    # Public normalization historically accepts JSON-serializable sequences.
    scope["database_identity"] = tuple(scope["database_identity"])
    scope["policy_sha256"] = hashlib.sha256(budget._json({
        key: value for key, value in scope.items() if key != "policy_sha256"
    }).encode()).hexdigest()
    permit = add_row(conn, scope, 1)
    before = deepcopy(permit)
    normalized = budget._permit(permit)
    assert normalized == old._permit(permit)
    assert isinstance(normalized["budget_scope"]["database_identity"], list)
    normalized["budget_scope"]["declaration"]["slots"] = 64
    assert permit == before


def test_snapshot_never_reuses_history_across_commit_rollback_stop_or_hold(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    peer = IdeaLake(lake.db_path)
    scope = budget.initialize(lake, results, {
        "version": 2, "resource": "cpu", "slots": 2, "wall_budget_seconds": None})
    try:
        assert budget.snapshot(lake, scope)["reserved_wall_seconds"] == 0
        permit = budget.reserve(peer, scope, "peer", 2)
        assert budget.snapshot(lake, scope)["reserved_wall_seconds"] == 2
        peer.conn.execute("UPDATE cpu_action_reservations SET permit_json='broken'")
        peer.conn.rollback()
        assert budget.require_permit(lake, permit) == permit
        peer.conn.execute("UPDATE cpu_action_reservations SET permit_json='broken'")
        peer.conn.commit()
        with pytest.raises(budget.CpuBudgetHOLD):
            budget.snapshot(lake, scope)
        with pytest.raises(budget.CpuBudgetHOLD):
            budget.require_permit(lake, permit)
        peer.conn.execute("UPDATE cpu_action_reservations SET permit_json=?", (budget._json(permit),))
        peer.conn.commit()
        assert budget.require_permit(lake, permit) == permit
        budget._HELD.add(budget._key(scope))
        assert budget.snapshot(lake, scope)["stopped"]
        with pytest.raises(budget.CpuBudgetHOLD, match="storage_unconfirmed"):
            budget.require_permit(lake, permit)
        budget._HELD.discard(budget._key(scope))
        budget.record_decision(peer, scope, {"kind": "Stop", "reason": "operator", "wakeup": None})
        assert budget.snapshot(lake, scope)["stopped"]
        with pytest.raises(budget.CpuBudgetHOLD, match="stopped"):
            budget.require_permit(lake, permit)
    finally:
        budget._HELD.discard(budget._key(scope))
        peer.close()
        lake.close()
