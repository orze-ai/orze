"""Same-Lake CPU envelope mechanisms; no historical API-absence claims.

All storage is temporary SQLite. Fault connections alter only this test's
commit boundary; no GPU, provider, process discovery or real workload.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import sqlite3
from threading import Barrier
import time
from types import SimpleNamespace

import pytest

from orze.core import cpu_action_budget as budget
from orze.core.execution_attempts import create_attempt, finish_attempt, current_attempt
from orze.engine.execution_authority import execution_transaction
from orze.idea_lake import IdeaLake


@pytest.fixture
def context(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    declaration = {"version": 1, "resource": "cpu", "slots": 2, "wall_budget_seconds": 10}
    scope = budget.initialize(lake, results, declaration)
    yield SimpleNamespace(lake=lake, results=results, declaration=declaration, scope=scope)
    lake.close()


def test_legal_peer_after_commit_does_not_invalidate_own_reservation(context):
    c = context
    peer = IdeaLake(c.lake.db_path)
    observed = []

    class CommitPeer(sqlite3.Connection):
        def commit(self):
            super().commit()
            observed.append(budget.reserve(peer, c.scope, "peer", 2))

    c.lake.conn.close()
    c.lake.conn = sqlite3.connect(str(c.lake.db_path), factory=CommitPeer)
    c.lake.conn.row_factory = sqlite3.Row
    try:
        own = budget.reserve(c.lake, c.scope, "owner", 3)
        assert own is not None
        assert len(observed) == 1 and observed[0] is not None
        assert own["slot"] != observed[0]["slot"]
        status = budget.snapshot(peer, c.scope)
        assert status["active_reservations"] == 2
        assert status["reserved_wall_seconds"] == 5
        assert not status["stopped"]
    finally:
        peer.close()


def _counts(c):
    return {name: c.lake.conn.execute("SELECT count(*) FROM " + name).fetchone()[0]
            for name in ("cpu_action_scopes", "cpu_action_reservations", "cpu_action_decisions")}


def _intent(c, permit):
    folder = c.results / permit["task_id"]
    folder.mkdir()
    with execution_transaction(c.lake, folder) as tx:
        ref = create_attempt(tx.conn, permit["task_id"], "action", "attempt-" + permit["task_id"],
                             {"reservation_id": permit["reservation_id"]})
        tx.watch_attempt(ref)
    budget.bind(c.lake, permit, ref)
    return folder, ref


def _not_started(c, folder, ref):
    # The fixture never creates any process. This is a genuine no-exec control,
    # not a fabricated TREE_CLOSED receipt or process-tree simulation.
    with execution_transaction(c.lake, folder) as tx:
        digest = tx.prepare(ref, {"operation": "known_no_exec"})
        terminal = {"outcome": "not_started", "effect_receipt_sha256": digest}
        assert finish_attempt(tx.conn, ref, terminal, not_started=True) == "committed"
    return terminal


def test_initialize_is_idempotent_but_cannot_reset_same_scope_limits(context):
    c = context
    permit = budget.reserve(c.lake, c.scope, "task", 3)
    assert budget.initialize(c.lake, c.results, c.declaration) == c.scope
    prior = _counts(c)
    with pytest.raises(budget.CpuBudgetHOLD, match="scope_binding_changed"):
        budget.initialize(c.lake, c.results, {**c.declaration, "wall_budget_seconds": 100})
    assert _counts(c) == prior
    assert budget.require_permit(c.lake, permit) == permit
    assert budget.snapshot(c.lake, c.scope)["reserved_wall_seconds"] == 3


@pytest.mark.parametrize("field,value", [("version", True), ("resource", "gpu"),
    ("slots", True), ("slots", 65), ("wall_budget_seconds", float("nan")),
    ("wall_budget_seconds", float("inf")), ("wall_budget_seconds", False)])
def test_declaration_rejects_nonexact_or_nonfinite_without_mutating(context, field, value):
    c = context
    prior = _counts(c)
    with pytest.raises(budget.CpuBudgetHOLD):
        budget.initialize(c.lake, c.results, {**c.declaration, field: value})
    assert _counts(c) == prior


def test_begin_immediate_concurrent_admission_has_one_available_slot(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    scope = budget.initialize(lake, results, {
        "version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 10})
    barrier = Barrier(2)

    def worker(task):
        peer = IdeaLake(lake.db_path)
        try:
            barrier.wait(timeout=3)
            return budget.reserve(peer, scope, task, 3)
        finally:
            peer.close()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            permits = list(pool.map(worker, ("one", "two")))
        assert sum(p is not None for p in permits) == 1
        assert budget.snapshot(lake, scope)["active_reservations"] == 1
        assert budget.snapshot(lake, scope)["reserved_wall_seconds"] == 3
        assert list(results.iterdir()) == []
        assert lake.conn.execute("SELECT 1 FROM sqlite_master WHERE name='execution_attempts'").fetchone() is None
    finally:
        lake.close()


def test_insufficient_budget_and_slot_wait_have_zero_admission_effects(context):
    c = context
    before = _counts(c)
    assert budget.reserve(c.lake, c.scope, "too-long", 11) is None
    assert _counts(c) == before
    assert budget.reserve(c.lake, c.scope, "one", 2) is not None
    assert budget.reserve(c.lake, c.scope, "two", 2) is not None
    before = _counts(c)
    assert budget.reserve(c.lake, c.scope, "three", 1) is None
    assert _counts(c) == before
    assert list(c.results.iterdir()) == []
    assert c.lake.conn.execute("SELECT 1 FROM sqlite_master WHERE name='execution_attempts'").fetchone() is None


def test_wait_is_a_decision_without_debit_and_stop_is_sticky(context):
    c = context
    decision = {"kind": "Wait", "reason": "no_action", "wakeup": time.time() + 2}
    record = budget.record_decision(c.lake, c.scope, decision)
    assert all(record[k] == v for k, v in decision.items())
    assert budget.snapshot(c.lake, c.scope)["reserved_wall_seconds"] == 0
    permit = budget.reserve(c.lake, c.scope, "task", 2)
    budget.record_decision(c.lake, c.scope, {"kind": "Stop", "reason": "policy", "wakeup": None})
    budget.record_decision(c.lake, c.scope, decision)
    assert budget.snapshot(c.lake, c.scope)["stopped"]
    with pytest.raises(budget.CpuBudgetHOLD, match="stopped"):
        budget.reserve(c.lake, c.scope, "next", 2)
    with pytest.raises(budget.CpuBudgetHOLD, match="stopped"):
        budget.require_permit(c.lake, permit)
    assert _counts(c)["cpu_action_reservations"] == 1
    assert list(c.results.iterdir()) == []


def test_bind_full_reference_and_settle_no_exec_release_slot_without_refund(context):
    c = context
    permit = budget.reserve(c.lake, c.scope, "task", 4)
    folder, ref = _intent(c, permit)
    assert budget.bind(c.lake, permit, ref) == permit
    assert budget.require_permit(c.lake, permit, ref) == permit
    terminal = _not_started(c, folder, ref)
    budget.record_decision(c.lake, c.scope, {"kind": "Stop", "reason": "done", "wakeup": None})
    assert budget.settle(c.lake, permit, ref, terminal) == "settled"
    assert budget.settle(c.lake, permit, ref, terminal) == "duplicate"
    status = budget.snapshot(c.lake, c.scope)
    assert status["free_slots"] == 2 and status["active_reservations"] == 0
    assert status["reserved_wall_seconds"] == 4 and status["remaining_wall_seconds"] == 6
    assert status["stopped"]


def test_unconfirmed_or_changed_terminal_never_releases_slot(context):
    c = context
    permit = budget.reserve(c.lake, c.scope, "task", 4)
    folder, ref = _intent(c, permit)
    with pytest.raises(budget.CpuBudgetHOLD):
        budget.settle(c.lake, permit, ref, {"outcome": "not_started"})
    terminal = _not_started(c, folder, ref)
    with pytest.raises(budget.CpuBudgetHOLD, match="terminal_evidence_changed"):
        budget.settle(c.lake, permit, ref, {**terminal, "outcome": "different"})
    assert budget.snapshot(c.lake, c.scope)["active_reservations"] == 1
    assert budget.snapshot(c.lake, c.scope)["reserved_wall_seconds"] == 4


def test_permit_metadata_cannot_reassign_another_task(context):
    c = context
    permit = budget.reserve(c.lake, c.scope, "task", 4)
    changed = deepcopy(permit)
    changed["task_id"] = "different"
    with pytest.raises(budget.CpuBudgetHOLD, match="permit_changed"):
        budget.require_permit(c.lake, changed)
    assert budget.require_permit(c.lake, permit) == permit


def test_public_api_refuses_caller_transaction_without_committing_it(context):
    c = context
    c.lake.conn.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(budget.CpuBudgetHOLD, match="caller_transaction_active"):
            budget.reserve(c.lake, c.scope, "task", 2)
        assert c.lake.conn.in_transaction
    finally:
        c.lake.conn.rollback()
    assert budget.snapshot(c.lake, c.scope)["reserved_wall_seconds"] == 0


@pytest.mark.parametrize("settlement", [False, True])
def test_silent_commit_rollback_holds_instead_of_granting_or_releasing(context, settlement):
    c = context
    permit = folder = ref = terminal = None
    if settlement:
        permit = budget.reserve(c.lake, c.scope, "task", 3)
        folder, ref = _intent(c, permit)
        terminal = _not_started(c, folder, ref)
    fired = []

    class RollbackCommit(sqlite3.Connection):
        def commit(self):
            fired.append(True)
            self.rollback()

    c.lake.conn.close()
    c.lake.conn = sqlite3.connect(str(c.lake.db_path), factory=RollbackCommit)
    c.lake.conn.row_factory = sqlite3.Row
    with pytest.raises(budget.CpuBudgetHOLD):
        if settlement:
            budget.settle(c.lake, permit, ref, terminal)
        else:
            budget.reserve(c.lake, c.scope, "task", 3)
    assert fired == [True]
    status = budget.snapshot(c.lake, c.scope)
    assert status["stopped"]
    assert status["active_reservations"] == int(settlement)
    assert status["reserved_wall_seconds"] == (3 if settlement else 0)
    if settlement:
        assert current_attempt(c.lake.conn, ref.task_id, ref.phase)["state"] == "NOT_STARTED"


def test_reader_closes_independent_connection_even_on_validation_failure(context, monkeypatch):
    c = context
    real_connect = sqlite3.connect
    closed = []

    class Tracked(sqlite3.Connection):
        def close(self):
            closed.append(True)
            super().close()

    def connect(*args, **kwargs):
        return real_connect(*args, **kwargs, factory=Tracked)

    c.lake.conn.execute("UPDATE cpu_action_scopes SET binding_json='{}'")
    c.lake.conn.commit()
    monkeypatch.setattr(budget.sqlite3, "connect", connect)
    with pytest.raises(budget.CpuBudgetHOLD, match="scope_binding_changed"):
        budget.snapshot(c.lake, c.scope)
    assert closed == [True]
