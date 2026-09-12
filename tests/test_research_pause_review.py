"""Independent Pause transaction review; temporary SQLite and no worker runs.

Pause retires this quiescent invocation, not peer admission for the scope.
Transparent SQLite connection hooks inject the actual writer/COMMIT windows.
"""
import json
import sqlite3

import pytest

from orze.core import cpu_action_budget as budget
from orze.core import research_interfaces as api
from orze.idea_lake import IdeaLake
from test_cpu_product_loop import project, task
from test_research_pause import PAUSE, ledger, selected_policy


def _replace_connection(lake, kind):
    lake.conn.close()
    lake.conn = sqlite3.connect(str(lake.db_path), factory=kind)
    lake.conn.row_factory = sqlite3.Row


def test_peer_reservation_after_policy_check_prevents_recording_pause(ledger, monkeypatch):
    lake, _, _, scope = ledger
    snapshot = {"queue": [], "active": False, "now": 10}
    decision = selected_policy(monkeypatch, PAUSE).decide(snapshot, budget.snapshot(lake, scope))
    peer = IdeaLake(lake.db_path)
    try:
        permit = budget.reserve(peer, scope, "peer-admitted-after-snapshot", 2)
        with pytest.raises(budget.CpuBudgetHOLD, match="pause_requires_quiescence"):
            budget.record_decision(lake, scope, decision)
        assert lake.conn.execute("SELECT count(*) FROM cpu_action_decisions").fetchone()[0] == 0
        assert budget.require_permit(peer, permit) == permit
        assert budget.snapshot(peer, scope)["active_reservations"] == 1
        assert budget.snapshot(peer, scope)["stopped"] is False
    finally:
        peer.close()


def test_writer_quiescence_check_is_fenced_but_postcommit_peer_is_allowed(ledger):
    lake, _, _, scope = ledger
    peer = IdeaLake(lake.db_path)
    peer.conn.execute("PRAGMA busy_timeout=0")
    locked, admitted = [], []

    class WindowConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql.startswith("INSERT INTO main.cpu_action_decisions"):
                # The actual no-active check has returned. Another real writer
                # must still be excluded until this Pause transaction commits.
                assert self.in_transaction
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    peer.conn.execute("BEGIN IMMEDIATE")
                locked.append(True)
            return super().execute(sql, *args, **kwargs)

        def commit(self):
            super().commit()
            admitted.append(budget.reserve(peer, scope, "peer-after-pause-commit", 2))

    _replace_connection(lake, WindowConnection)
    try:
        record = budget.record_decision(lake, scope, PAUSE)
        assert locked == [True] and len(admitted) == 1 and admitted[0] is not None
        actual = json.loads(lake.conn.execute("SELECT record_json FROM cpu_action_decisions").fetchone()[0])
        assert record == actual and record["kind"] == "Pause"
        assert budget.require_permit(peer, admitted[0]) == admitted[0]
        assert budget.snapshot(peer, scope)["stopped"] is False
    finally:
        peer.close()


@pytest.mark.parametrize("fault,remaining_records", [
    ("rollback", 0), ("commit_raise", 1), ("commit_delete", 0),
])
def test_uncertain_pause_storage_never_returns_an_ack_or_reopens_authority(
        ledger, fault, remaining_records):
    lake, directory, declaration, scope = ledger
    fired = []

    class FaultConnection(sqlite3.Connection):
        def commit(self):
            fired.append(fault)
            if fault == "rollback":
                self.rollback()
                return
            super().commit()
            if fault == "commit_raise":
                raise sqlite3.OperationalError("injected lost COMMIT response")
            with sqlite3.connect(str(lake.db_path)) as peer:
                peer.execute("DELETE FROM cpu_action_decisions")

    _replace_connection(lake, FaultConnection)
    with pytest.raises(budget.CpuBudgetHOLD):
        budget.record_decision(lake, scope, PAUSE)
    assert fired == [fault]
    peer = IdeaLake(lake.db_path)
    try:
        assert peer.conn.execute("SELECT count(*) FROM cpu_action_decisions").fetchone()[0] == remaining_records
        stop = json.loads(peer.conn.execute("SELECT stop_json FROM cpu_action_scopes").fetchone()[0])
        assert stop == {"kind": "Stop", "reason": "budget_storage_unconfirmed", "wakeup": None}
        with pytest.raises(budget.CpuBudgetHOLD):
            budget.initialize(peer, directory, declaration)
        with pytest.raises(budget.CpuBudgetHOLD):
            budget.reserve(peer, scope, "not-authorized", 2)
        assert peer.conn.execute("SELECT count(*) FROM cpu_action_reservations").fetchone()[0] == 0
    finally:
        peer.close()


def test_peer_stop_after_pause_commit_remains_permanent(ledger):
    lake, _, _, scope = ledger
    peer = IdeaLake(lake.db_path)
    stop = {"kind": "Stop", "reason": "operator_stop", "wakeup": None}

    class StopAfterCommit(sqlite3.Connection):
        def commit(self):
            super().commit()
            budget.record_decision(peer, scope, stop)

    _replace_connection(lake, StopAfterCommit)
    try:
        record = budget.record_decision(lake, scope, PAUSE)
        assert record["kind"] == "Pause"
        assert budget.snapshot(peer, scope)["stop"] == stop
        with pytest.raises(budget.CpuBudgetHOLD, match="stopped"):
            budget.reserve(peer, scope, "forbidden", 2)
        decisions = [json.loads(row[0])["kind"] for row in peer.conn.execute(
            "SELECT record_json FROM cpu_action_decisions ORDER BY rowid")]
        assert decisions == ["Pause", "Stop"]
    finally:
        peer.close()


def test_actual_cli_pause_leaves_nonempty_queue_without_claim_or_attempt(project, monkeypatch):
    root, cfg, run = project
    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))

    class PausePolicy:
        def __init__(self, declaration):
            pass

        def decide(self, view, resources):
            assert len(view["queue"]) == 1 and resources["active_reservations"] == 0
            return dict(PAUSE)

    api.register_policy("pause_nonempty_review", "pause.nonempty.review.v1", PausePolicy)
    cfg["action_policy"]["kind"] = "pause_nonempty_review"
    task(root, outputs={}, program="raise AssertionError('Pause must never execute this worker')")
    assert run(once=False) in (None, 0)
    with sqlite3.connect(root / "lake.db") as conn:
        assert conn.execute("SELECT status FROM ideas").fetchall() == [("queued",)]
        assert conn.execute("SELECT count(*) FROM cpu_action_reservations").fetchone()[0] == 0
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='execution_attempts'").fetchall() == []
        assert conn.execute("SELECT stop_json FROM cpu_action_scopes").fetchone()[0] is None
        assert json.loads(conn.execute("SELECT record_json FROM cpu_action_decisions").fetchone()[0])["kind"] == "Pause"
    assert not list((root / "results").glob("idea-*/claim.json"))
