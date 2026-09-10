"""New watch/identity mechanism acceptance, not old-API-absence regressions."""
import sqlite3
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import create_attempt, current_attempt, mark_running
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.execution_authority import canonical_identity_equal, execution_transaction


class WatchedConnection(sqlite3.Connection):
    after_commit = None

    def commit(self):
        super().commit()
        if self.after_commit is not None:
            callback, self.after_commit = self.after_commit, None
            callback()


@pytest.fixture
def case(tmp_path):
    conn = sqlite3.connect(tmp_path / "attempts.db", factory=WatchedConnection)
    lake = SimpleNamespace(conn=conn)
    folder = tmp_path / "task-generic"
    with execution_transaction(lake, folder):
        ref = create_attempt(conn, folder.name, "tool", "attempt-A", {"version": 1})
        mark_running(conn, ref)
    try:
        yield lake, folder, ref
    finally:
        conn.close()


def test_watch_generic_phase_does_not_invent_lifecycle_or_stage(case):
    lake, folder, ref = case
    with execution_transaction(lake, folder) as tx:
        tx.watch_attempt(ref)
    assert current_attempt(lake.conn, ref.task_id, ref.phase)["state"] == "RUNNING"
    assert lake.conn.execute(
        "SELECT name FROM sqlite_master WHERE name IN ('ideas','idea_state','idea_stage_state')"
    ).fetchall() == []
    assert not (folder / "_attempt_effect.lock").exists()


def test_watch_rejects_later_same_transaction_mutation_and_rolls_back(case):
    lake, folder, ref = case
    with pytest.raises(AttemptEffectBusy):
        with execution_transaction(lake, folder) as tx:
            tx.watch_attempt(ref)
            lake.conn.execute(
                "UPDATE execution_attempts SET binding_json='{" + '"version":2' + "}'")
    assert current_attempt(lake.conn, ref.task_id, ref.phase)["binding"] == {"version": 1}
    assert not lake.conn.in_transaction
    assert not (folder / "_attempt_effect.lock").exists()


def test_watch_rejects_postcommit_mutation_and_retains_hold(case):
    lake, folder, ref = case

    def after():
        lake.conn.execute("UPDATE execution_attempts SET binding_json=?", ('{"version":2}',))
        sqlite3.Connection.commit(lake.conn)

    with pytest.raises(AttemptEffectInDoubt):
        with execution_transaction(lake, folder) as tx:
            tx.watch_attempt(ref)
            lake.conn.after_commit = after
    assert current_attempt(lake.conn, ref.task_id, ref.phase)["binding"] == {"version": 2}
    assert not lake.conn.in_transaction
    assert (folder / "_attempt_effect.lock").is_dir()


def test_watch_retains_value_snapshot_not_mutable_caller_dictionary(case):
    lake, folder, ref = case
    with execution_transaction(lake, folder) as tx:
        value = current_attempt(lake.conn, ref.task_id, ref.phase)
        tx.watch_attempt(ref)
        value["binding"]["version"] = 999
        value["state"] = "IN_DOUBT"
    assert current_attempt(lake.conn, ref.task_id, ref.phase)["binding"] == {"version": 1}


def test_shared_identity_comparison_is_strict_bounded_json_object_only():
    assert canonical_identity_equal({"a": [True, None], "id": 2}, {"id": 2, "a": [True, None]})
    assert not canonical_identity_equal({"id": 2}, {"id": 2.0})
    assert not canonical_identity_equal({"id": 1}, {"id": True})
    recursive = {}
    recursive["self"] = recursive
    invalid = [None, [], {1: "x"}, {"x": float("nan")}, {"x": object()},
               {"x": "a" * 65537}, {"x": [0] * 2049}, recursive]
    deep = {}
    for _ in range(34):
        deep = {"nested": deep}
    invalid.append(deep)
    for value in invalid:
        assert not canonical_identity_equal(value, value)
