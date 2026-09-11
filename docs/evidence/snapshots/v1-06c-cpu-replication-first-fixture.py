"""Explicit CPU replication: real SQLite/native CPU, no scientific workloads."""
import hashlib
import json
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import native_cpu_action as native
from orze.engine import replication
from orze.core.replication_requests import ReplicationError, get_request
from orze.core.evaluation_retry_state import open_existing_lake
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.scheduler import claim
from test_native_cpu_action import context, _finish


@pytest.fixture
def completed(context):
    lake, results, scope, cfg, create, handles = context
    cfg.update(execution={"version": 1, "resource": "cpu", "slots": 1,
                          "wall_budget_seconds": 20},
               results_dir=str(results), idea_lake_db=str(lake.db_path))
    action, permit = create("pass")
    owner = native.launch("idea-cpu", results, cfg, lake=lake, action=action,
                          permit=permit, admission=lambda: None)
    assert _finish(owner, results, cfg, lake, permit)["outcome"] == "completed"
    yield lake, results, cfg, action, owner.attempt_ref


def test_public_request_copies_completed_cpu_without_config_salt(completed):
    lake, results, cfg, action, ref = completed
    before = lake.conn.execute("SELECT config FROM ideas WHERE idea_id='idea-cpu'").fetchone()[0]
    result = replication.request_replication("idea-cpu", results, cfg, lake,
                                            request_id="cpu-public-request")
    assert result["status"] == "created"
    assert result["task_id"] != "idea-cpu"
    row = lake.conn.execute("SELECT config,kind,status FROM ideas WHERE idea_id=?",
                            (result["task_id"],)).fetchone()
    assert tuple(row) == (before, "native_cpu_action", "queued")
    assert lake.conn.execute("SELECT sop_type,current_state FROM idea_state WHERE idea_id=?",
                            (result["task_id"],)).fetchone()[:] == ("action", "QUEUED")
    assert current_attempt(lake.conn, result["task_id"], "action") is None


def _request(c, key="cpu-request", **kwargs):
    lake, results, cfg, action, ref = c
    return replication.request_replication("idea-cpu", results, cfg, lake,
                                          request_id=key, **kwargs)


def test_same_request_claimed_replay_preserves_all_rows(completed):
    lake, results, cfg, action, ref = completed
    first = _request(completed)
    assert claim(first["task_id"], results, None, lake, resource="cpu")
    before = list(lake.conn.iterdump())
    assert _request(completed) == {**first, "status": "already_requested"}
    assert list(lake.conn.iterdump()) == before


def test_distinct_requests_keep_same_spec_and_ordinary_dedup(completed):
    lake, results, cfg, action, ref = completed
    first, second = _request(completed, "one"), _request(completed, "two")
    a, b = get_request(lake.conn, "one"), get_request(lake.conn, "two")
    assert first["task_id"] != second["task_id"]
    assert a["action_sha256"] == b["action_sha256"] == a["spec_fingerprint"]
    assert a["domain_run_sha256"] == b["domain_run_sha256"]
    raw = lake.conn.execute("SELECT config FROM ideas WHERE idea_id='idea-cpu'").fetchone()[0]
    assert len({row[0] for row in lake.conn.execute("SELECT config FROM ideas")}) == 1
    assert lake.insert("ordinary-copy", "same", raw, "", kind="native_cpu_action",
                       if_absent=True)["status"] == "config_duplicate"


def test_two_real_connections_one_request_one_target(completed):
    lake, results, cfg, action, ref = completed
    def request():
        peer = open_existing_lake(lake.db_path)
        try:
            return replication.request_replication("idea-cpu", results, cfg, peer, request_id="same")
        except ReplicationError as exc:
            if "attempt_effect_lock_unavailable" not in str(exc.__cause__) + str(exc):
                raise
            return None
        finally:
            peer.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(lambda _: request(), range(2)))
    assert any(item is not None for item in replies)
    replies = [request() if item is None else item for item in replies]
    assert len({item["task_id"] for item in replies}) == 1
    assert lake.conn.execute("SELECT count(*) FROM replication_requests").fetchone()[0] == 1
    assert lake.conn.execute("SELECT count(*) FROM ideas").fetchone()[0] == 2


@pytest.mark.parametrize("change", ["reason", "expected_ref"])
def test_same_key_or_captured_source_conflict_has_no_mutation(completed, change):
    lake, results, cfg, action, ref = completed
    _request(completed)
    before = list(lake.conn.iterdump())
    args = {"reason": "different"} if change == "reason" else {
        "expected_source_ref": {**asdict(ref), "generation": ref.generation + 1}}
    with pytest.raises(ReplicationError):
        _request(completed, **args)
    assert list(lake.conn.iterdump()) == before


@pytest.mark.parametrize("change", ["config", "terminal", "confirmation"])
def test_changed_source_cannot_reauthorize_existing_request(completed, change):
    from orze.engine.cpu_replication import verify_record
    lake, results, cfg, action, ref = completed
    _request(completed)
    record = get_request(lake.conn, "cpu-request")
    if change == "config":
        lake.conn.execute("UPDATE ideas SET config=config||'\n# changed' WHERE idea_id='idea-cpu'")
        lake.conn.commit()
    elif change == "terminal":
        row = current_attempt(lake.conn, "idea-cpu", "action")
        terminal = {**row["terminal"], "outcome": "failed"}
        lake.conn.execute("UPDATE execution_attempts SET terminal_json=? WHERE task_id='idea-cpu'",
                          (json.dumps(terminal),))
        lake.conn.commit()
    else:
        (results / "idea-cpu" / "_execution_effects" / ref.attempt_id / "committed.json").unlink()
    before = list(lake.conn.iterdump())
    with pytest.raises((ReplicationError, AttemptEffectInDoubt)):
        verify_record(lake, results, record)
    assert list(lake.conn.iterdump()) == before


def test_action_target_scope_and_train_adapter_cannot_reuse_cpu_grant(completed):
    from orze.engine.cpu_replication import authorization, verify_record
    lake, results, cfg, action, ref = completed
    target = _request(completed)["task_id"]
    record = get_request(lake.conn, "cpu-request")
    assert authorization(lake, target, results, cfg, action=action) == record
    assert authorization(lake, "idea-cpu", results, cfg, action=action) is None
    with pytest.raises(ReplicationError, match="action_or_domain"):
        authorization(lake, target, results, cfg, action={**action, "purpose": "changed"})
    with pytest.raises(ReplicationError, match="training_adapter"):
        replication.replication_authorization(lake, target, results / target, cfg, "a" * 64)
    with pytest.raises((ReplicationError, AttemptEffectInDoubt)):
        verify_record(lake, results.parent, record)
    lake.conn.execute("UPDATE ideas SET config=config||'\n#changed' WHERE idea_id=?", (target,))
    lake.conn.commit()
    with pytest.raises(ReplicationError, match="target_changed"):
        authorization(lake, target, results, cfg, action=action)


def test_request_insert_trigger_mutation_rolls_back_task_and_request(completed):
    from orze.core.replication_requests import ensure_schema
    lake, results, cfg, action, ref = completed
    lake.conn.execute("BEGIN IMMEDIATE")
    ensure_schema(lake.conn)
    lake.conn.execute("CREATE TRIGGER corrupt_replica AFTER INSERT ON replication_requests "
        "BEGIN UPDATE ideas SET config=config||'\n#fault' WHERE idea_id=NEW.task_id; END")
    lake.conn.commit()
    with pytest.raises(ReplicationError, match="target_readback"):
        _request(completed)
    assert lake.conn.execute("SELECT count(*) FROM ideas").fetchone()[0] == 1
    assert lake.conn.execute("SELECT count(*) FROM replication_requests").fetchone()[0] == 0


def test_silent_commit_rollback_never_reports_created(completed):
    lake, results, cfg, action, ref = completed
    original = lake.conn
    calls = []
    class RollbackCommit:
        def __getattr__(self, name):
            return getattr(original, name)
        def commit(self):
            calls.append("rollback-instead-of-commit")
            original.rollback()
    lake.conn = RollbackCommit()
    try:
        with pytest.raises(AttemptEffectInDoubt):
            _request(completed)
        assert calls == ["rollback-instead-of-commit"]
        assert original.execute("SELECT count(*) FROM ideas").fetchone()[0] == 1
        assert get_request(original, "cpu-request") is None
    finally:
        lake.conn = original
