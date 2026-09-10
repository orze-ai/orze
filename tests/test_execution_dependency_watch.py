"""New closed-source dependency API acceptance, not historical behavior reds."""
import sqlite3

import pytest

from orze.core.execution_attempts import StaleAttempt, create_attempt, finish_attempt
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.execution_authority import execution_transaction, lifecycle_fence
from test_execution_authority import case


def _closed(c):
    c.conn.execute("BEGIN IMMEDIATE")
    c.conn.execute("UPDATE execution_attempts SET binding_json=?", ('{"version":1}',))
    assert finish_attempt(c.conn, c.ref, {
        "lifecycle": lifecycle_fence(c.lake, c.ref.task_id, "training")}) == "committed"
    c.conn.commit()


def test_dependency_allows_later_lifecycle_but_attempt_watch_does_not(case):
    c = case
    _closed(c)
    with execution_transaction(c.lake, c.folder) as tx:
        tx.watch_dependency(c.ref)
        assert c.lake._record_state_transition_in_tx(c.ref.task_id, "IN_PROGRESS", "FAILED")
    assert c.lake.get_fsm_state(c.ref.task_id) == "FAILED"
    with pytest.raises(AttemptEffectBusy):
        with execution_transaction(c.lake, c.folder) as tx:
            tx.watch_attempt(c.ref)


def test_dependency_requires_closed_current_and_rejects_next_generation(case):
    c = case
    with pytest.raises(AttemptEffectBusy):
        with execution_transaction(c.lake, c.folder) as tx:
            tx.watch_dependency(c.ref)
    _closed(c)
    with pytest.raises((AttemptEffectBusy, StaleAttempt)):
        with execution_transaction(c.lake, c.folder) as tx:
            tx.watch_dependency(c.ref)
            create_attempt(c.conn, c.ref.task_id, "training", "attempt-B", {})
    assert not c.lock.exists()


def test_dependency_postcommit_exact_numeric_change_retains_hold(case):
    c = case
    _closed(c)

    def after():
        c.conn.execute("UPDATE execution_attempts SET binding_json='{" + '"version":true' + "}'")
        sqlite3.Connection.commit(c.conn)

    with pytest.raises(AttemptEffectInDoubt):
        with execution_transaction(c.lake, c.folder) as tx:
            tx.watch_dependency(c.ref)
            c.conn.after_commit = after
    assert c.lock.is_dir()
