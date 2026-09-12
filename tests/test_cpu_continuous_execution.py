"""Explicit continuous CPU authorization, not unlimited action lifetime.

New v2 behavior requirements; baseline failures are not historical defects.
Uses temporary SQLite and genuine no-exec receipts, not simulated tree closure.
No workers, host discovery, GPU/provider, or edits to old fixtures.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import sqlite3
from threading import Barrier
from types import SimpleNamespace
import time

import pytest

from orze.core import cpu_action_budget as budget
from orze.core.cpu_execution import (
    CPUExecutionError, cpu_execution, execution_fingerprint, runtime_lease_seconds,
)
from orze.engine.cpu_phase import QueuePolicy
from orze.idea_lake import IdeaLake
from test_cpu_action_budget import _intent, _not_started


CONTINUOUS = {"version": 2, "resource": "cpu", "slots": 1,
              "wall_budget_seconds": None}


@pytest.fixture
def context(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    try:
        scope = budget.initialize(lake, results, CONTINUOUS)
        yield SimpleNamespace(lake=lake, results=results, scope=scope)
    finally:
        lake.close()


def test_v2_configuration_is_explicit_detached_and_pinned():
    cfg = {"execution": deepcopy(CONTINUOUS)}
    declaration = cpu_execution(cfg)
    assert declaration == CONTINUOUS and declaration is not cfg["execution"]
    assert runtime_lease_seconds(cfg, 2) == 2
    cfg["_cpu_execution_fingerprint"] = execution_fingerprint(cfg)
    cfg["execution"] = {**CONTINUOUS, "version": 1, "wall_budget_seconds": 10}
    with pytest.raises(CPUExecutionError, match="loaded CPU configuration changed"):
        cpu_execution(cfg)


@pytest.mark.parametrize("version,wall", [
    (1, None), (1, float("inf")), (2, 10), (2, 10**100),
    (2, float("inf")), (2, float("nan")), (True, None), (2.0, None),
])
def test_invalid_opt_in_rejected_by_config_and_budget_without_rows(tmp_path, version, wall):
    declaration = {**CONTINUOUS, "version": version, "wall_budget_seconds": wall}
    with pytest.raises(CPUExecutionError):
        cpu_execution({"execution": declaration})
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    try:
        with pytest.raises(budget.CpuBudgetHOLD):
            budget.initialize(lake, results, declaration)
        assert lake.conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'cpu_action_%'"
        ).fetchall() == []
    finally:
        lake.close()


def test_v1_finite_envelope_still_exhausts(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    declaration = {**CONTINUOUS, "version": 1, "slots": 2, "wall_budget_seconds": 3}
    try:
        assert cpu_execution({"execution": declaration}) == declaration
        scope = budget.initialize(lake, results, declaration)
        assert budget.reserve(lake, scope, "first", 3) is not None
        assert budget.reserve(lake, scope, "second", 1) is None
        view = budget.snapshot(lake, scope)
        assert view["remaining_wall_seconds"] == 0 and view["free_slots"] == 1
    finally:
        lake.close()


def test_continuous_settlement_preserves_charge_and_restart_identity(context):
    c = context
    for task, seconds in (("first", 2), ("second", 17)):
        permit = budget.reserve(c.lake, c.scope, task, seconds)
        assert permit["wall_limit_seconds"] == seconds
        folder, ref = _intent(c, permit)
        terminal = _not_started(c, folder, ref)
        assert budget.settle(c.lake, permit, ref, terminal) == "settled"
        assert budget.settle(c.lake, permit, ref, terminal) == "duplicate"
    peer = IdeaLake(c.lake.db_path)
    try:
        scope = budget.initialize(peer, c.results, deepcopy(CONTINUOUS))
        assert scope == c.scope
        view = budget.snapshot(peer, scope)
        assert view["remaining_wall_seconds"] is None
        assert view["reserved_wall_seconds"] == 19 and view["active_reservations"] == 0
        assert budget.reserve(peer, scope, "third", 23) is not None
        assert budget.snapshot(peer, scope)["reserved_wall_seconds"] == 42
        assert peer.conn.execute("SELECT count(*) FROM cpu_action_scopes").fetchone()[0] == 1
        assert peer.conn.execute("SELECT count(*) FROM cpu_action_reservations").fetchone()[0] == 3
    finally:
        peer.close()


@pytest.mark.parametrize("seconds", [None, True, 0, -1, float("inf"), float("nan")])
def test_continuous_action_still_needs_finite_positive_timeout(context, seconds):
    c = context
    with pytest.raises(budget.CpuBudgetHOLD):
        budget.reserve(c.lake, c.scope, "invalid", seconds)
    assert budget.snapshot(c.lake, c.scope)["reserved_wall_seconds"] == 0
    assert c.lake.conn.execute("SELECT count(*) FROM cpu_action_reservations").fetchone()[0] == 0


def test_runtime_lease_cannot_exceed_finite_action_timeout():
    cfg = {"execution": deepcopy(CONTINUOUS),
           "cpu_runtime_lease": {"version": 1, "ttl_seconds": 3}}
    assert cpu_execution(cfg) == CONTINUOUS
    with pytest.raises(CPUExecutionError, match="exceeds action timeout"):
        runtime_lease_seconds(cfg, 2)
    assert runtime_lease_seconds(cfg, 4) == 3


def test_continuous_stop_blocks_new_go_but_allows_confirmed_settlement(context):
    c = context
    permit = budget.reserve(c.lake, c.scope, "first", 2)
    folder, ref = _intent(c, permit)
    terminal = _not_started(c, folder, ref)
    stop = {"kind": "Stop", "reason": "research_finished", "wakeup": None}
    budget.record_decision(c.lake, c.scope, stop)
    budget.record_decision(c.lake, c.scope,
        {"kind": "Wait", "reason": "cannot_clear_stop", "wakeup": time.time() + 1})
    with pytest.raises(budget.CpuBudgetHOLD, match="stopped"):
        budget.require_permit(c.lake, permit, ref)
    with pytest.raises(budget.CpuBudgetHOLD, match="stopped"):
        budget.reserve(c.lake, c.scope, "second", 2)
    assert budget.settle(c.lake, permit, ref, terminal) == "settled"
    view = budget.snapshot(c.lake, c.scope)
    assert view["stop"] == stop and view["stopped"] is True
    assert view["reserved_wall_seconds"] == 2 and view["remaining_wall_seconds"] is None


def test_continuous_recovery_never_adopts_reserved_work(context):
    c = context
    permit = budget.reserve(c.lake, c.scope, "unlaunched", 2)
    recovered = budget.reconcile_confirmed_terminals(c.lake, c.scope)
    assert recovered["settled"] == []
    assert recovered["retained"] == [{"reservation_id": permit["reservation_id"], "reason": "reserved"}]
    assert budget.snapshot(c.lake, c.scope)["active_reservations"] == 1
    assert budget.reserve(c.lake, c.scope, "second", 2) is None
    assert not list(c.results.iterdir())


@pytest.mark.parametrize("initial_version", [1, 2])
def test_same_scope_cannot_change_authorization_mode(tmp_path, initial_version):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    initial = dict(CONTINUOUS) if initial_version == 2 else {
        **CONTINUOUS, "version": 1, "wall_budget_seconds": 10}
    other = dict(CONTINUOUS) if initial_version == 1 else {
        **CONTINUOUS, "version": 1, "wall_budget_seconds": 10}
    try:
        scope = budget.initialize(lake, results, initial)
        budget.reserve(lake, scope, "first", 2)
        before = tuple(lake.conn.execute("SELECT * FROM cpu_action_scopes").fetchone())
        with pytest.raises(budget.CpuBudgetHOLD, match="scope_binding_changed"):
            budget.initialize(lake, results, other)
        assert tuple(lake.conn.execute("SELECT * FROM cpu_action_scopes").fetchone()) == before
        assert budget.snapshot(lake, scope)["reserved_wall_seconds"] == 2
    finally:
        lake.close()


def test_continuous_parallel_reservations_still_share_one_slot(context):
    c = context
    barrier = Barrier(2)
    def reserve(task):
        lake = IdeaLake(c.lake.db_path)
        try:
            barrier.wait(timeout=3)
            return budget.reserve(lake, c.scope, task, 2)
        finally:
            lake.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        permits = list(pool.map(reserve, ("first", "second")))
    assert sum(value is not None for value in permits) == 1
    assert budget.snapshot(c.lake, c.scope)["reserved_wall_seconds"] == 2


def test_continuous_unknown_commit_does_not_grant_a_permit(context):
    c = context
    class RollbackCommit(sqlite3.Connection):
        def commit(self):
            self.rollback()
    c.lake.conn.close()
    c.lake.conn = sqlite3.connect(str(c.lake.db_path), factory=RollbackCommit)
    c.lake.conn.row_factory = sqlite3.Row
    with pytest.raises(budget.CpuBudgetHOLD):
        budget.reserve(c.lake, c.scope, "first", 2)
    view = budget.snapshot(c.lake, c.scope)
    assert view["stopped"] is True and view["reserved_wall_seconds"] == 0
    with pytest.raises(budget.CpuBudgetHOLD):
        budget.reserve(c.lake, c.scope, "second", 2)


@pytest.mark.parametrize("remaining,free,active,stopped,kind,reason", [
    (None, 1, 0, False, "Execute", None),
    (None, 0, 1, False, "Wait", "cpu_resource_or_budget_unavailable"),
    (None, 1, 0, True, "Stop", "scope_stopped"),
    (0, 1, 0, False, "Stop", "wall_envelope_exhausted"),
])
def test_queue_policy_separates_continuous_authorization_from_slot_and_stop(
        remaining, free, active, stopped, kind, reason):
    policy = QueuePolicy({"version": 1, "kind": "queue", "idle": "stop", "wait_seconds": 1})
    view = {"queue": [{"idea_id": "first", "action": {"timeout_seconds": 2}}],
            "active": bool(active), "now": time.time()}
    result = policy.decide(view, {"remaining_wall_seconds": remaining, "free_slots": free,
                                "active_reservations": active, "stopped": stopped})
    assert result["kind"] == kind
    if reason is None:
        assert result == {"kind": "Execute", "task_id": "first"}
    else:
        assert result["reason"] == reason
