"""Dedicated CPU replication read-only watches on actual SQLite metadata.

Every grant comes from a real completed CPU action and public request route.
Faults change only the fixture's database; no verifier/callback is replaced.
"""
import copy

import pytest

from orze.core.replication_requests import ReplicationError, request_for_task
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.execution_authority import execution_transaction
from test_native_cpu_replication import context, replica


@pytest.mark.parametrize("role", ["source", "target"])
def test_watch_allows_only_exact_read_roles_and_detaches_expected_record(replica, role):
    c = replica
    task_id = c.record["source_ref"]["task_id"] if role == "source" else c.record["task_id"]
    folder = c.results / task_id
    folder.mkdir(exist_ok=True)
    supplied = copy.deepcopy(c.record)
    with execution_transaction(c.lake, folder) as tx:
        tx.watch_cpu_replication(supplied)
        supplied["reason"] = "caller mutated detached input after registration"
    assert request_for_task(c.lake.conn, c.record["task_id"]) == c.record
    assert c.lake.conn.execute("SELECT COUNT(*) FROM execution_attempts WHERE task_id=?",
                              (c.record["task_id"],)).fetchone()[0] == 0


def test_unrelated_task_guard_does_not_receive_replication_authority(replica):
    c = replica
    unrelated = c.results / "unrelated-task"
    unrelated.mkdir()
    with pytest.raises(AttemptEffectBusy) as caught:
        with execution_transaction(c.lake, unrelated) as tx:
            tx.watch_cpu_replication(c.record)
    assert str(caught.value.__cause__) == "execution_cpu_replication_scope_invalid"
    assert request_for_task(c.lake.conn, c.record["task_id"]) == c.record


@pytest.mark.parametrize("schema", [True, 1])
def test_watch_rejects_non_cpu_schema_before_admitting_metadata(replica, schema):
    c = replica
    supplied = {**c.record, "schema": schema}
    with pytest.raises(AttemptEffectBusy) as caught:
        with execution_transaction(c.lake, c.results / c.source.idea_id) as tx:
            tx.watch_cpu_replication(supplied)
    assert str(caught.value.__cause__) == "execution_cpu_replication_watch_invalid"
    assert request_for_task(c.lake.conn, c.record["task_id"]) == c.record


def test_watch_rechecks_actual_request_before_commit_and_rolls_back(replica):
    c = replica
    with pytest.raises(ReplicationError):
        with execution_transaction(c.lake, c.results / c.source.idea_id) as tx:
            tx.watch_cpu_replication(c.record)
            tx.conn.execute("UPDATE replication_requests SET record_json='{}'")
    assert request_for_task(c.lake.conn, c.record["task_id"]) == c.record
    assert not c.lake.conn.in_transaction


def test_watch_rechecks_actual_request_after_real_commit_and_holds(replica):
    c = replica
    original = c.lake.conn
    commits = []

    class CommitMutation:
        def __getattr__(self, name):
            return getattr(original, name)

        def commit(self):
            original.commit()
            commits.append("actual writer committed")
            original.execute("UPDATE replication_requests SET record_json='{}'")
            original.commit()
            commits.append("controlled postcommit mutation committed")

    c.lake.conn = CommitMutation()
    try:
        with pytest.raises(AttemptEffectInDoubt):
            with execution_transaction(c.lake, c.results / c.source.idea_id) as tx:
                tx.watch_cpu_replication(c.record)
    finally:
        c.lake.conn = original
    assert commits == ["actual writer committed", "controlled postcommit mutation committed"]
    assert original.execute("SELECT record_json FROM replication_requests").fetchone()[0] == "{}"
    assert not original.in_transaction
