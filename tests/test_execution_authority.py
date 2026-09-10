"""New coordinator mechanism tests, plus explicit e358dc draft regressions.

The draft-only failures are transaction cleanup/ownership and contradictory
lifecycle fences, not failures of a historical published coordinator API.
"""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import (
    AttemptAuthorityError, create_attempt, current_attempt,
    finish_attempt, mark_running,
)
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.execution_authority import execution_transaction, lifecycle_fence
from orze.idea_lake import IdeaLake


class FaultConnection(sqlite3.Connection):
    fail_rollback = False
    commit_fault = None
    fail_restore = False
    before_commit = None
    after_commit = None

    def execute(self, sql, parameters=(), /):
        if sql == "PRAGMA busy_timeout=7000" and self.fail_restore:
            raise sqlite3.OperationalError("synthetic timeout restoration failure")
        return super().execute(sql, parameters)

    def commit(self):
        if self.before_commit:
            self.before_commit()
        if self.commit_fault == "before":
            raise sqlite3.OperationalError("synthetic commit-before failure")
        if self.commit_fault == "noop":
            return
        super().commit()
        if self.commit_fault == "after":
            raise sqlite3.OperationalError("synthetic commit-after failure")
        if self.after_commit:
            self.after_commit()

    def rollback(self):
        if self.fail_rollback:
            raise sqlite3.OperationalError("synthetic rollback failure")
        return super().rollback()


@pytest.fixture
def case(tmp_path):
    path = tmp_path / "ideas.db"
    lake = IdeaLake(str(path))
    task = "idea-authority"
    lake.insert(task, "Authority fixture", "seed: 13", "", status="queued")
    assert lake.record_state_transition(task, "QUEUED", "CLAIMED")
    assert lake.record_state_transition(task, "CLAIMED", "IN_PROGRESS")
    lake.conn.close()
    lake.conn = sqlite3.connect(path, factory=FaultConnection)
    lake.conn.row_factory = sqlite3.Row
    lake.conn.execute("PRAGMA busy_timeout=7000")
    lake.conn.execute("BEGIN IMMEDIATE")
    ref = create_attempt(lake.conn, task, "training", "attempt-A", {})
    mark_running(lake.conn, ref)
    lake.conn.commit()
    folder = tmp_path / "results" / task
    folder.mkdir(parents=True)
    c = SimpleNamespace(lake=lake, conn=lake.conn, folder=folder, ref=ref,
                        db=path, lock=folder / "_attempt_effect.lock")
    try:
        yield c
    finally:
        lake.close()


def _close(tx, c, *, valid_binding=True):
    digest = tx.prepare(c.ref, {"write": "diagnostic"})
    (c.folder / "diagnostic.txt").write_text("controlled synthetic effect")
    assert c.lake._record_state_transition_in_tx(
        c.ref.task_id, "IN_PROGRESS", "COMPLETE")
    terminal = {
        "effect_receipt_sha256": digest if valid_binding else "0" * 64,
        "lifecycle": lifecycle_fence(c.lake, c.ref.task_id, "training"),
    }
    assert finish_attempt(c.conn, c.ref, terminal) == "committed"
    return digest


def _assert_hold(c):
    assert c.lock.is_dir(), "uncertain cleanup must retain the real owner directory"
    with pytest.raises((AttemptEffectBusy, AttemptEffectInDoubt)):
        with execution_transaction(c.lake, c.folder):
            pytest.fail("an uncertain operation must not admit another writer")


def test_prepare_and_bound_terminal_commit_before_filesystem_confirmation(case):
    c = case
    prepared = c.folder / "_execution_effects" / c.ref.attempt_id / "prepared.json"
    confirmed = prepared.with_name("committed.json")
    events = []

    def before():
        assert prepared.exists() and not confirmed.exists()
        assert current_attempt(c.conn, c.ref.task_id, c.ref.phase)["state"] == "TERMINAL"
        events.append("sql_commit_started")

    def after():
        assert not confirmed.exists()
        peer = sqlite3.connect(c.db)
        try:
            assert current_attempt(peer, c.ref.task_id, c.ref.phase)["state"] == "TERMINAL"
        finally:
            peer.close()
        events.append("sql_committed")

    c.conn.before_commit, c.conn.after_commit = before, after
    with execution_transaction(c.lake, c.folder) as tx:
        digest = _close(tx, c)
    assert events == ["sql_commit_started", "sql_committed"]
    assert json.loads(confirmed.read_text())["prepared_sha256"] == digest
    assert not c.conn.in_transaction
    assert c.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 7000
    assert not c.lock.exists()


def test_existing_caller_transaction_is_never_committed_or_rolled_back(case):
    c = case
    c.conn.execute("BEGIN IMMEDIATE")
    c.conn.execute("UPDATE ideas SET title='caller-owned'")
    with pytest.raises(AttemptEffectBusy, match="caller_transaction"):
        with execution_transaction(c.lake, c.folder):
            pytest.fail("must not enter")
    assert c.conn.in_transaction
    assert c.conn.execute("SELECT title FROM ideas").fetchone()[0] == "caller-owned"
    assert not c.lock.exists()
    c.conn.rollback()


def test_unprepared_body_failure_rolls_back_and_releases_own_guard(case):
    c = case
    with pytest.raises(ValueError, match="synthetic body"):
        with execution_transaction(c.lake, c.folder):
            c.conn.execute("UPDATE ideas SET title='temporary'")
            raise ValueError("synthetic body")
    assert not c.conn.in_transaction and not c.lock.exists()
    assert c.conn.execute("SELECT title FROM ideas").fetchone()[0] == "Authority fixture"
    assert c.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 7000


def test_wrong_terminal_receipt_binding_keeps_intent_and_rolls_back_sql(case):
    c = case
    with pytest.raises(AttemptEffectInDoubt):
        with execution_transaction(c.lake, c.folder) as tx:
            _close(tx, c, valid_binding=False)
    _assert_hold(c)
    assert current_attempt(c.conn, c.ref.task_id, c.ref.phase)["state"] == "RUNNING"
    assert c.lake.get_fsm_state(c.ref.task_id) == "IN_PROGRESS"
    assert (c.folder / "diagnostic.txt").exists()
    assert not list(c.folder.glob("_execution_effects/*/committed.json"))


@pytest.mark.parametrize("prepared", [False, True])
def test_rollback_failure_cannot_mask_hold_or_release_effect_owner(case, prepared):
    c = case
    with pytest.raises(AttemptEffectInDoubt):
        with execution_transaction(c.lake, c.folder) as tx:
            if prepared:
                tx.prepare(c.ref, {"write": "future"})
            c.conn.execute("UPDATE ideas SET title='rollback-uncertain'")
            c.conn.fail_rollback = True
            raise ValueError("primary callback failure")
    _assert_hold(c)


@pytest.mark.parametrize("fault", ["before", "after", "noop"])
def test_any_unconfirmed_commit_is_hold_even_without_file_prepare(case, fault):
    c = case
    with pytest.raises(AttemptEffectInDoubt):
        with execution_transaction(c.lake, c.folder):
            c.conn.execute("UPDATE ideas SET title='commit-uncertain'")
            c.conn.commit_fault = fault
    _assert_hold(c)


def test_callback_committing_then_raising_does_not_escape_as_ordinary_failure(case):
    c = case
    with pytest.raises(AttemptEffectInDoubt):
        with execution_transaction(c.lake, c.folder):
            c.conn.execute("UPDATE ideas SET title='callback-committed'")
            c.conn.commit()
            raise ValueError("callback failed after owning commit")
    _assert_hold(c)
    assert c.conn.execute("SELECT title FROM ideas").fetchone()[0] == "callback-committed"


@pytest.mark.parametrize("prepared", [False, True])
def test_timeout_restore_failure_never_downgrades_uncertainty_to_busy(case, prepared):
    c = case
    with pytest.raises(AttemptEffectInDoubt):
        with execution_transaction(c.lake, c.folder) as tx:
            if prepared:
                tx.prepare(c.ref, {})
            c.conn.fail_restore = True
            raise ValueError("primary body error")
    _assert_hold(c)


def test_confirmation_io_failure_keeps_committed_sql_but_blocks_replay(case, monkeypatch):
    c = case
    from orze.engine import attempt_effect_receipts as receipts
    original = receipts._publish

    def publish(path, raw):
        if path.name == "committed.json":
            raise OSError("synthetic confirmation IO failure")
        return original(path, raw)

    monkeypatch.setattr(receipts, "_publish", publish)
    with pytest.raises(AttemptEffectInDoubt):
        with execution_transaction(c.lake, c.folder) as tx:
            _close(tx, c)
    _assert_hold(c)
    assert current_attempt(c.conn, c.ref.task_id, c.ref.phase)["state"] == "TERMINAL"
    assert c.lake.get_fsm_state(c.ref.task_id) == "COMPLETE"
    assert not list(c.folder.glob("_execution_effects/*/committed.json"))


@pytest.mark.parametrize("conflict", ["legacy", "global_audit", "stage_state", "stage_audit"])
def test_lifecycle_fence_rejects_contradictory_authoritative_rows(case, conflict):
    c = case
    if conflict == "legacy":
        c.conn.execute("UPDATE ideas SET status='queued'")
    elif conflict == "global_audit":
        c.conn.execute("UPDATE idea_transitions SET to_state='FAILED' "
                       "WHERE id=(SELECT MAX(id) FROM idea_transitions)")
    elif conflict == "stage_state":
        c.conn.execute("UPDATE idea_stage_state SET current_state='FAILED' WHERE stage='training'")
    else:
        c.conn.execute("UPDATE idea_stage_transitions SET to_state='FAILED' "
                       "WHERE id=(SELECT MAX(id) FROM idea_stage_transitions WHERE stage='training')")
    c.conn.commit()
    with pytest.raises(AttemptAuthorityError):
        lifecycle_fence(c.lake, c.ref.task_id, c.ref.phase)


def test_valid_lifecycle_fence_is_a_readonly_transition_identity_snapshot(case):
    c = case
    changes = c.conn.total_changes
    fence = lifecycle_fence(c.lake, c.ref.task_id, c.ref.phase)
    assert fence["legacy_status"] == "running"
    assert fence["global_state"] == fence["phase_state"] == "IN_PROGRESS"
    assert type(fence["global_transition_id"]) is int
    assert type(fence["phase_transition_id"]) is int
    assert c.conn.total_changes == changes and not c.conn.in_transaction
