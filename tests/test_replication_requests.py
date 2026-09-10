"""New explicit replication mechanisms, not historical missing-API bugs."""
from copy import deepcopy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from orze.core.evaluation_retry_state import open_existing_lake
from orze.core.execution_attempts import current_attempt
from orze.core.replication_requests import get_request, request_for_task
from orze.engine.replication import (
    ReplicationError, request_replication, replication_authorization,
)
from orze.engine.scheduler import claim
from test_artifact_snapshot_contract import project, _launch, _output, _poll
from test_native_training_caller_boundaries import case as native_case


@pytest.fixture
def completed(project):
    c = project
    tp, folder = _launch(c)
    _output(tp, folder)
    assert _poll(c, tp)[0] == [(tp.idea_id, 0)]
    c.source_tp = tp
    return c


def request(c, request_id="repeat-one", *, lake=None, cfg=None):
    return request_replication(c.idea, c.results, cfg or c.cfg,
                               lake or c.lake, request_id=request_id)


def test_explicit_request_creates_exact_config_new_queued_task_atomically(completed):
    c = completed
    source = c.lake.conn.execute("SELECT config FROM ideas WHERE idea_id=?", (c.idea,)).fetchone()[0]
    result = request(c)
    assert result["status"] == "created" and result["task_id"] != c.idea
    record = get_request(c.lake.conn, "repeat-one")
    assert request_for_task(c.lake.conn, result["task_id"]) == record
    row = c.lake.conn.execute("SELECT config,status,parent FROM ideas WHERE idea_id=?", (result["task_id"],)).fetchone()
    assert tuple(row) == (source, "queued", c.idea)
    assert c.lake.get_fsm_state(result["task_id"]) == "QUEUED"
    assert current_attempt(c.lake.conn, result["task_id"], "training") is None
    assert record["execution_identity"] == c.source_tp.execution_identity
    assert not (c.results / result["task_id"]).exists()


def test_exact_request_replay_does_not_reset_claimed_task(completed):
    c = completed
    result = request(c)
    assert claim(result["task_id"], c.results, 0, lake=c.lake)
    before = list(c.lake.conn.iterdump())
    second = request(c)
    assert second == {**result, "status": "already_requested"}
    assert list(c.lake.conn.iterdump()) == before


def test_distinct_requests_create_distinct_tasks_without_config_salt(completed):
    c = completed
    a, b = request(c, "one"), request(c, "two")
    assert a["task_id"] != b["task_id"]
    rows = c.lake.conn.execute("SELECT config FROM ideas ORDER BY idea_id").fetchall()
    assert len(rows) == 3 and len({row[0] for row in rows}) == 1
    assert get_request(c.lake.conn, "one")["spec_fingerprint"] == get_request(c.lake.conn, "two")["spec_fingerprint"]


def test_two_connections_same_request_have_one_mapping(completed):
    c = completed
    def run():
        lake = open_existing_lake(c.lake.db_path)
        try:
            return request(c, lake=lake)
        finally:
            lake.close()
    # Filesystem source guard may reject simultaneous acquisition. Both callers
    # retry the same request key, never allocate an alternative request/task.
    def contend():
        try:
            return run()
        except ReplicationError as exc:
            if "attempt_effect_lock_unavailable" not in str(exc):
                raise
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: contend(), range(2)))
    assert any(result is not None for result in results)
    results = [run() if result is None else result for result in results]
    assert len({r["task_id"] for r in results}) == 1
    assert c.lake.conn.execute("SELECT COUNT(*) FROM replication_requests").fetchone()[0] == 1
    assert c.lake.conn.execute("SELECT COUNT(*) FROM ideas").fetchone()[0] == 2


@pytest.mark.parametrize("changed", ["config", "file", "script", "scope"])
def test_source_identity_or_scope_change_rejects_without_admission(completed, changed):
    c = completed
    cfg = deepcopy(c.cfg)
    if changed == "config":
        c.lake.conn.execute("UPDATE ideas SET config='seed: 99' WHERE idea_id=?", (c.idea,))
        c.lake.conn.commit()
    elif changed == "file":
        (c.results / c.idea / "idea_config.yaml").write_text("seed: 99\n")
    elif changed == "script":
        from pathlib import Path
        Path(cfg["train_script"]).write_text("# changed execution\n")
    else:
        cfg["results_dir"] = str(c.results.parent / "another")
    with pytest.raises(ReplicationError):
        request(c, cfg=cfg)
    assert c.lake.conn.execute("SELECT COUNT(*) FROM ideas").fetchone()[0] == 1


def test_failed_or_unconfirmed_source_is_not_an_authorization(completed):
    c = completed
    row = current_attempt(c.lake.conn, c.idea, "training")
    terminal = row["terminal"]
    terminal["outcome"] = "failed"
    c.lake.conn.execute("UPDATE execution_attempts SET terminal_json=? WHERE attempt_id=?", (
        json.dumps(terminal, sort_keys=True, separators=(",", ":")), row["attempt_id"]))
    c.lake.conn.commit()
    with pytest.raises(ReplicationError):
        request(c)


def test_unbound_old_task_cannot_manufacture_replication(completed):
    c = completed
    c.lake.insert("idea-old", "Old", "seed: 13", "", status="completed")
    with pytest.raises(ReplicationError):
        request_replication("idea-old", c.results, c.cfg, c.lake, request_id="old")


def test_caller_transaction_is_never_committed_or_rolled_back(completed):
    c = completed
    c.lake.conn.execute("BEGIN IMMEDIATE")
    c.lake.conn.execute("UPDATE ideas SET title='caller-owned'")
    with pytest.raises(ReplicationError, match="caller_transaction"):
        request(c)
    assert c.lake.conn.in_transaction
    assert c.lake.conn.execute("SELECT title FROM ideas").fetchone()[0] == "caller-owned"
    c.lake.conn.rollback()


@pytest.mark.parametrize("table", ["ideas", "idea_state", "replication_requests"])
def test_noop_insert_rolls_back_entire_request_and_task(completed, table):
    from orze.core.replication_requests import ensure_schema
    c = completed
    c.lake.conn.execute("BEGIN IMMEDIATE")
    ensure_schema(c.lake.conn)
    c.lake.conn.commit()
    c.lake.conn.execute(f"CREATE TRIGGER reject_insert BEFORE INSERT ON {table} BEGIN SELECT RAISE(IGNORE); END")
    c.lake.conn.commit()
    with pytest.raises(ReplicationError):
        request(c)
    assert not c.lake.conn.in_transaction
    assert c.lake.conn.execute("SELECT COUNT(*) FROM ideas").fetchone()[0] == 1
    assert c.lake.conn.execute("SELECT COUNT(*) FROM replication_requests").fetchone()[0] == 0
    assert c.lake.conn.execute("SELECT COUNT(*) FROM idea_state").fetchone()[0] == 1


def test_read_missing_request_table_does_not_create_schema(completed):
    c = completed
    before = list(c.lake.conn.iterdump())
    assert get_request(c.lake.conn, "none") is None
    assert request_for_task(c.lake.conn, c.idea) is None
    assert list(c.lake.conn.iterdump()) == before


def test_launch_authorization_requires_real_claim_and_exact_target(completed):
    c = completed
    result = request(c)
    task = result["task_id"]
    assert claim(task, c.results, 0, lake=c.lake)
    folder = c.results / task
    source_config = c.lake.conn.execute("SELECT config FROM ideas WHERE idea_id=?", (task,)).fetchone()[0]
    (folder / "idea_config.yaml").write_text(source_config)
    claim_id = json.loads((folder / "claim.json").read_text())["attempt_id"]
    auth = replication_authorization(c.lake, task, folder, c.cfg,
                                    c.source_tp.execution_identity, claim_id=claim_id)
    assert auth["request_id"] == "repeat-one" and auth["task_id"] == task
    with pytest.raises(ReplicationError):
        replication_authorization(c.lake, task, folder, c.cfg,
                                  c.source_tp.execution_identity, claim_id="wrong")
    (folder / "idea_config.yaml").write_text("seed: 99\n")
    with pytest.raises(ReplicationError):
        replication_authorization(c.lake, task, folder, c.cfg,
                                  c.source_tp.execution_identity, claim_id=claim_id)


def test_ordinary_identical_config_admission_still_rejects(completed):
    c = completed
    request(c)
    result = c.lake.insert("idea-ordinary", "Duplicate", "seed: 13", "", status="queued", if_absent=True)
    assert result["status"] == "config_duplicate"
    assert request_for_task(c.lake.conn, "idea-ordinary") is None


@pytest.mark.parametrize("request_id", ["../escape", "", True])
def test_invalid_request_id_has_no_writes(completed, request_id):
    c = completed
    before = list(c.lake.conn.iterdump())
    with pytest.raises(ReplicationError):
        request(c, request_id)
    assert list(c.lake.conn.iterdump()) == before
