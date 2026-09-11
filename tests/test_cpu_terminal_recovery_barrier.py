"""Independent real-terminal recovery barriers, not unknown-worker adoption.

Each project first executes an actual native CPU worker and exits its owned
controller after confirmed TERMINAL/effect but before budget settlement. The
faults below affect only that private project's real filesystem/SQLite boundary.
No closure, lifecycle, reservation or process identity proof is fabricated.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from pathlib import Path
import sqlite3
from threading import Event
import time

import pytest

from cpu_terminal_recovery_helpers import run_cli, snapshot
from orze.core import cpu_action_budget as budget
from orze.core import idea_source_lock as locks
from orze.core.execution_attempts import AttemptRef
from orze.idea_lake import IdeaLake

pytest_plugins = ["cpu_terminal_recovery_helpers"]


@contextmanager
def opened(project):
    proof(project)
    lake = IdeaLake(project["root"] / "lake.db")
    try:
        scope = budget.initialize(lake, project["root"] / "results", project["cfg"]["execution"])
        yield lake, scope
    finally:
        lake.close()


def barrier(lake, scope):
    row = lake.conn.execute("SELECT binding_json,nonce,state,summary_json FROM main.cpu_action_recovery WHERE scope=?",
                            (scope["results_dir"],)).fetchone()
    return None if row is None else tuple(row)


def reservation(lake):
    return tuple(lake.conn.execute("SELECT state,ref_json,terminal_sha256 FROM main.cpu_action_reservations WHERE task_id='idea-first'").fetchone())


def proof(project):
    # Read the actual crash helper's confirmed boundary, never construct a tree.
    captured = project["calls"][0]["metadata"][1]
    assert captured["effect_confirmed"] and captured["effect_guard_absent"]
    return captured["permit"], AttemptRef(**captured["ref"]), captured["terminal"]


def unchanged_execution(project, original):
    current = snapshot(project, "independent-proof-after")
    for table in ("execution_attempts", "research_artifacts", "research_observations",
                  "ideas", "idea_state", "idea_transitions", "idea_stage_state", "idea_stage_transitions"):
        assert current["database"][table] == original["database"][table]
    assert current["worker_events"] == original["worker_events"]
    assert current["files"] == original["files"]
    reservations = {row["reservation_id"]: row for row in current["database"]["cpu_action_reservations"]}
    for row in original["database"]["cpu_action_reservations"]:
        for key in ("reservation_id", "scope", "task_id", "slot", "permit_json", "ref_json"):
            assert reservations[row["reservation_id"]][key] == row[key]


@pytest.mark.parametrize("which", ["effect", "recovery"])
def test_post_delete_fsync_uncertainty_blocks_fresh_restart_with_zero_bound(make_crashed_project, monkeypatch, which):
    project = make_crashed_project()
    original = project["snapshots"][-1]
    results = project["root"] / "results"
    target = results / "idea-first" / "_attempt_effect.lock" if which == "effect" else results / "_cpu_terminal_recovery.lock"
    fired = []
    real_sync = locks._sync_directory
    with opened(project) as (lake, scope):
        def fail_after_delete(path):
            if Path(path) == target.parent and not target.exists() and reservation(lake)[0] == "SETTLED":
                fired.append(barrier(lake, scope))
                raise OSError("owned test directory fsync failed after actual rmdir")
            return real_sync(path)

        with monkeypatch.context() as patch:
            patch.setattr(locks, "_sync_directory", fail_after_delete)
            with pytest.raises(budget.CpuBudgetHOLD):
                budget.reconcile_confirmed_terminals(lake, scope)
        assert len(fired) == 1 and fired[0][2] == "IN_PROGRESS"
        assert not target.exists()
        assert reservation(lake)[0] == "SETTLED"
        assert lake.conn.execute("SELECT count(*) FROM cpu_action_reservations WHERE state='BOUND'").fetchone()[0] == 0
        persisted = barrier(lake, scope)
        assert persisted[2:] == ("IN_PROGRESS", None)
    run_cli(project, "fresh_after_" + which + "_release_uncertainty", expected=75)
    with closing(IdeaLake(project["root"] / "lake.db")) as peer:
        assert barrier(peer, scope) == persisted
    unchanged_execution(project, original)


@pytest.mark.parametrize("operation", ["reserve", "require_permit"])
def test_in_progress_denies_already_running_peer_admission(make_crashed_project, monkeypatch, operation):
    project = make_crashed_project(slots=2)
    original = project["snapshots"][-1]
    real_release = locks._release
    checked = []
    with opened(project) as (lake, scope), closing(IdeaLake(project["root"] / "lake.db")) as peer:
        active = budget.reserve(peer, scope, "existing-peer", 1)
        assert budget.require_permit(peer, active) == active
        before = tuple(peer.conn.execute("SELECT * FROM cpu_action_reservations WHERE reservation_id=?",
                                         (active["reservation_id"],)).fetchone())

        def release(lease):
            if lease.lock_dir == Path(scope["results_dir"]) / "_cpu_terminal_recovery.lock":
                assert barrier(peer, scope)[2] == "IN_PROGRESS"
                with pytest.raises(budget.CpuBudgetHOLD):
                    if operation == "reserve":
                        budget.reserve(peer, scope, "must-not-admit", 1)
                    else:
                        budget.require_permit(peer, active)
                checked.append(True)
            return real_release(lease)

        with monkeypatch.context() as patch:
            patch.setattr(locks, "_release", release)
            result = budget.reconcile_confirmed_terminals(lake, scope)
        assert checked == [True]
        assert len(result["settled"]) == 1
        assert tuple(peer.conn.execute("SELECT * FROM cpu_action_reservations WHERE reservation_id=?",
                                       (active["reservation_id"],)).fetchone()) == before
        assert peer.conn.execute("SELECT count(*) FROM cpu_action_reservations").fetchone()[0] == 2
        assert barrier(peer, scope)[2] == "COMPLETE"
        assert budget.require_permit(peer, active) == active
    unchanged_execution(project, original)


def test_two_recovery_consumers_do_not_repeat_settlement_or_poison_owner(make_crashed_project, monkeypatch):
    project = make_crashed_project()
    original = project["snapshots"][-1]
    entered, release_owner = Event(), Event()
    real_release, real_settle = locks._release, budget._settle
    settlements = []
    with opened(project) as (lake, scope):
        def release(lease):
            if lease.lock_dir == Path(scope["results_dir"]) / "idea-first" / "_attempt_effect.lock":
                entered.set()
                if not release_owner.wait(5):
                    raise RuntimeError("test owner release timeout")
            return real_release(lease)

        def settle(*args, **kwargs):
            result = real_settle(*args, **kwargs)
            settlements.append(result)
            return result

        def owner():
            with closing(IdeaLake(project["root"] / "lake.db")) as private:
                return budget.reconcile_confirmed_terminals(private, scope)

        with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=1) as pool:
            patch.setattr(locks, "_release", release)
            patch.setattr(budget, "_settle", settle)
            future = pool.submit(owner)
            try:
                assert entered.wait(5)
                assert reservation(lake)[0] == "SETTLED"
                with pytest.raises(budget.CpuBudgetHOLD):
                    budget.reconcile_confirmed_terminals(lake, scope)
            finally:
                release_owner.set()
            result = future.result(timeout=5)
        assert settlements == ["settled"]
        assert len(result["settled"]) == 1
        assert barrier(lake, scope)[2] == "COMPLETE"
        assert not budget.snapshot(lake, scope)["stopped"]
    unchanged_execution(project, original)


def test_legal_peer_decision_after_complete_commit_preserves_exact_owner_result(make_crashed_project):
    project = make_crashed_project()
    original = project["snapshots"][-1]
    fired = []
    with opened(project) as (lake, scope), closing(IdeaLake(project["root"] / "lake.db")) as peer:
        class PeerCommit(sqlite3.Connection):
            def commit(self):
                complete = self.execute("SELECT state FROM cpu_action_recovery WHERE scope=?", (scope["results_dir"],)).fetchone()[0] == "COMPLETE"
                super().commit()
                if complete and not fired:
                    fired.append(budget.record_decision(peer, scope,
                        {"kind": "Wait", "reason": "legal_peer", "wakeup": time.time() + 2}))

        lake.conn.close()
        lake.conn = sqlite3.connect(str(lake.db_path), factory=PeerCommit)
        lake.conn.row_factory = sqlite3.Row
        result = budget.reconcile_confirmed_terminals(lake, scope)
        assert len(fired) == len(result["settled"]) == 1
        assert barrier(lake, scope)[2] == "COMPLETE"
        assert not budget.snapshot(lake, scope)["stopped"]
    unchanged_execution(project, original)


@pytest.mark.parametrize("stage,mode", [("IN_PROGRESS", "rollback"), ("COMPLETE", "rollback"), ("COMPLETE", "committed_raise")])
def test_recovery_transaction_unknown_is_not_an_ordinary_success(make_crashed_project, stage, mode):
    project = make_crashed_project()
    original = project["snapshots"][-1]
    fired = []
    with opened(project) as (lake, scope):
        prior = barrier(lake, scope)

        class FaultCommit(sqlite3.Connection):
            def commit(self):
                row = self.execute("SELECT state FROM cpu_action_recovery WHERE scope=?", (scope["results_dir"],)).fetchone()
                if row and row[0] == stage and not fired:
                    fired.append(row[0])
                    if mode == "rollback":
                        super().rollback()
                        return
                    super().commit()
                    raise OSError("actual COMPLETE committed; response lost")
                super().commit()

        lake.conn.close()
        lake.conn = sqlite3.connect(str(lake.db_path), factory=FaultCommit)
        lake.conn.row_factory = sqlite3.Row
        with pytest.raises(budget.CpuBudgetHOLD):
            budget.reconcile_confirmed_terminals(lake, scope)
        assert fired == [stage]
        if stage == "IN_PROGRESS":
            assert barrier(lake, scope) == prior
            assert reservation(lake)[0] == "BOUND"
        elif mode == "rollback":
            assert barrier(lake, scope)[2:] == ("IN_PROGRESS", None)
            assert reservation(lake)[0] == "SETTLED"
        else:
            complete = barrier(lake, scope)
            assert complete[2] == "COMPLETE" and complete[3] is not None
            assert reservation(lake)[0] == "SETTLED"
        durable_stop = lake.conn.execute("SELECT stop_json FROM cpu_action_scopes WHERE scope=?",
                                        (scope["results_dir"],)).fetchone()[0]
    # A real new interpreter cannot inherit the parent test's sticky HOLD set.
    # Only a durable COMPLETE response-loss case may recheck normally.
    run_cli(project, "fresh_after_" + stage.lower() + "_" + mode,
            expected=0 if mode == "committed_raise" and durable_stop is None else 75)
    unchanged_execution(project, original)


def test_existing_unqualified_gate_owner_is_preserved(make_crashed_project):
    project = make_crashed_project()
    original = project["snapshots"][-1]
    with opened(project) as (lake, scope):
        prior = barrier(lake, scope)
        with locks.idea_source_lock(Path(scope["results_dir"]) / "_cpu_terminal_recovery.lock") as owner:
            assert owner is not None
            with pytest.raises(budget.CpuBudgetHOLD):
                budget.reconcile_confirmed_terminals(lake, scope)
            assert locks.idea_source_lock_owned(owner)
            assert barrier(lake, scope) == prior
            assert reservation(lake)[0] == "BOUND"
    unchanged_execution(project, original)


def test_recovery_refuses_caller_transaction_without_consuming_it(make_crashed_project):
    project = make_crashed_project()
    original = project["snapshots"][-1]
    with opened(project) as (lake, scope):
        prior = barrier(lake, scope)
        lake.conn.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(budget.CpuBudgetHOLD):
                budget.reconcile_confirmed_terminals(lake, scope)
            assert lake.conn.in_transaction
            assert barrier(lake, scope) == prior
            assert reservation(lake)[0] == "BOUND"
        finally:
            lake.conn.rollback()
    unchanged_execution(project, original)


def test_actual_peer_nonce_replacement_cannot_be_completed_by_old_owner(make_crashed_project, monkeypatch):
    project = make_crashed_project()
    original = project["snapshots"][-1]
    real_release = locks._release
    replaced = []
    with opened(project) as (lake, scope), closing(IdeaLake(project["root"] / "lake.db")) as peer:
        def release(lease):
            if lease.lock_dir == Path(scope["results_dir"]) / "_cpu_terminal_recovery.lock":
                old = barrier(peer, scope)
                replacement = "f" * 48 if old[1] != "f" * 48 else "e" * 48
                peer.conn.execute("UPDATE cpu_action_recovery SET nonce=? WHERE scope=?",
                                  (replacement, scope["results_dir"]))
                peer.conn.commit()
                replaced.append(replacement)
            return real_release(lease)

        with monkeypatch.context() as patch:
            patch.setattr(locks, "_release", release)
            with pytest.raises(budget.CpuBudgetHOLD):
                budget.reconcile_confirmed_terminals(lake, scope)
        assert len(replaced) == 1
        assert barrier(lake, scope)[1:] == (replaced[0], "IN_PROGRESS", None)
        assert reservation(lake)[0] == "SETTLED"
    unchanged_execution(project, original)
