"""New read-only demand API acceptance, not historical missing-API reds.

Real temporary IdeaLake and SQLite rows are observed without scheduler,
configuration parsing, result artifacts or process/provider interaction.
"""
from dataclasses import FrozenInstanceError
import importlib
import os
from pathlib import Path
import sqlite3

import pytest

from orze.idea_lake import IdeaLake


@pytest.fixture
def lake(tmp_path):
    instance = IdeaLake(tmp_path / "ideas.db")
    try:
        yield instance
    finally:
        instance.close()


def _module():
    return importlib.import_module("orze.core.research_demand")


def _load(lake):
    return _module().load_persistent_demand(lake.db_path)


def _queued(lake, name="idea-queued"):
    lake.insert(name, name, "strategy: nonexistent\nseed: 13", "source", status="queued")
    return name


def _running(lake, name="idea-running"):
    _queued(lake, name)
    assert lake.record_state_transition(name, "QUEUED", "CLAIMED", "test claim")
    assert lake.record_state_transition(name, "CLAIMED", "IN_PROGRESS", "test training")
    return name


def _training_done(lake, name="idea-evaluation"):
    _running(lake, name)
    assert lake.record_stage_transition(name, "training", "IN_PROGRESS", "COMPLETE", "training finished")
    return name


def _files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _nullable_stages(lake):
    rows = list(lake.conn.execute("SELECT idea_id,stage,current_state,updated_at FROM idea_stage_state"))
    lake.conn.execute("DROP TABLE idea_stage_state")
    lake.conn.execute("CREATE TABLE idea_stage_state (idea_id TEXT NOT NULL, stage TEXT NOT NULL, "
                      "current_state TEXT, updated_at TEXT, PRIMARY KEY(idea_id,stage))")
    lake.conn.executemany("INSERT INTO idea_stage_state VALUES (?,?,?,?)", rows)
    lake.conn.commit()


def test_empty_catalog_is_known_zero_and_nested_snapshot_is_immutable(lake):
    snapshot = _load(lake)
    assert snapshot.available and snapshot.complete
    assert snapshot.queue_count == snapshot.waiting_count == snapshot.counts.total == 0
    with pytest.raises(FrozenInstanceError):
        snapshot.complete = False
    with pytest.raises(FrozenInstanceError):
        snapshot.counts.queued = 999


def test_observation_is_one_closed_read_transaction_without_config_or_admission(lake, monkeypatch):
    _queued(lake)
    lake.conn.execute("UPDATE ideas SET config=?, title=?", ("x" * 200_000, "y" * 100_000))
    lake.conn.commit()
    before = _files(Path(lake.db_path).parent)
    module = _module()
    original = module._open_authoritative_lifecycle
    connections, statements = [], []

    def opening(path):
        connection, reason = original(path)
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        connection.set_trace_callback(statements.append)
        connections.append(connection)
        return connection, reason

    monkeypatch.setattr(module, "_open_authoritative_lifecycle", opening)
    monkeypatch.setattr(IdeaLake, "__init__", lambda *a, **k: pytest.fail("no bootstrap"))
    monkeypatch.setattr(IdeaLake, "get_queue", lambda *a, **k: pytest.fail("no limited dispatch query"))
    snapshot = _load(lake)
    assert snapshot.queue_count == snapshot.waiting_count == 1
    assert _files(Path(lake.db_path).parent) == before
    assert sum(s.strip().upper() == "BEGIN" for s in statements) == 1
    assert all(s.split(None, 1)[0].upper() in {"SELECT", "PRAGMA", "BEGIN", "WITH"} for s in statements)
    assert not any("i.config" in s.lower() or "i.title" in s.lower() or "eval_metrics" in s.lower() for s in statements)
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    lake.conn.execute("UPDATE ideas SET status='failed'")
    lake.conn.execute("UPDATE idea_state SET current_state='FAILED'")
    lake.conn.commit()
    assert snapshot.queue_count == 1  # Detached value, not a live view.


@pytest.mark.parametrize("kind,count", [("queued", 2507), ("evaluation", 257)])
def test_full_persistent_counts_are_not_dispatch_or_retry_query_limits(lake, kind, count):
    names = [f"idea-bulk-{index}" for index in range(count)]
    status, state = ("queued", "QUEUED") if kind == "queued" else ("running", "IN_PROGRESS")
    lake.conn.executemany("INSERT INTO ideas (idea_id,title,config,raw_markdown,status) VALUES (?,?,'{}','',?)",
                          [(name, name, status) for name in names])
    lake.conn.executemany("INSERT INTO idea_state (idea_id,current_state) VALUES (?,?)",
                          [(name, state) for name in names])
    if kind == "evaluation":
        lake.conn.executemany("INSERT INTO idea_stage_state (idea_id,stage,current_state,updated_at) VALUES (?,?,?,'now')",
                              [(name, stage, value) for name in names for stage, value in
                               (("training", "COMPLETE"), ("evaluation", "PENDING"))])
    lake.conn.commit()
    snapshot = _load(lake)
    assert snapshot.available and snapshot.complete
    assert snapshot.waiting_count == snapshot.counts.total == count
    assert snapshot.queue_count == (count if kind == "queued" else 0)


@pytest.mark.parametrize("retry", [False, True])
def test_first_evaluation_and_explicit_retry_are_both_persistent_pending(lake, retry):
    name = _training_done(lake)
    if retry:
        assert lake.record_stage_transition(name, "evaluation", "PENDING", "IN_PROGRESS", "eval start")
        assert lake.record_state_transition(name, "IN_PROGRESS", "FAILED", "eval failed")
        assert lake.retry_evaluation(name)
    snapshot = _load(lake)
    assert snapshot.complete and snapshot.counts.evaluation_pending == 1
    assert snapshot.queue_count == 0 and snapshot.waiting_count == 1
    assert snapshot.counts.in_progress == 0


def test_future_eval_pending_during_training_is_not_waiting_work(lake):
    _running(lake)
    _queued(lake, "idea-claimed")
    assert lake.record_state_transition("idea-claimed", "QUEUED", "CLAIMED", "claim")
    lake.insert("idea-finished", "done", "{}", "", status="completed")
    snapshot = _load(lake)
    assert snapshot.complete and snapshot.waiting_count == 0
    assert snapshot.counts.in_progress == snapshot.counts.claimed == snapshot.counts.inactive == 1
    assert snapshot.counts.evaluation_pending == 0 and snapshot.counts.total == 3


def test_conflicts_preserve_queued_lower_bound_but_do_not_claim_exact_zero(lake):
    _queued(lake, "idea-good")
    _queued(lake, "idea-conflict")
    _queued(lake, "idea-no-state")
    lake.conn.execute("UPDATE idea_state SET current_state='FAILED' WHERE idea_id='idea-conflict'")
    lake.conn.execute("DELETE FROM idea_state WHERE idea_id='idea-no-state'")
    lake.conn.commit()
    before = _files(Path(lake.db_path).parent)
    snapshot = _load(lake)
    assert snapshot.available and not snapshot.complete
    assert snapshot.counts.queued == 1 and snapshot.counts.unknown == 2
    assert snapshot.queue_count is None and snapshot.waiting_count is None
    assert _files(Path(lake.db_path).parent) == before


@pytest.mark.parametrize("kind", ["queued", "evaluation"])
def test_recorded_null_stage_is_unknown_not_missing_history(lake, kind):
    name = _queued(lake) if kind == "queued" else _training_done(lake)
    _nullable_stages(lake)
    lake.conn.execute("INSERT OR REPLACE INTO idea_stage_state VALUES (?,'evaluation',NULL,'now')", (name,))
    lake.conn.commit()
    snapshot = _load(lake)
    assert snapshot.available and not snapshot.complete and snapshot.counts.unknown == 1
    assert snapshot.queue_count is None and snapshot.waiting_count is None
    assert snapshot.counts.queued == snapshot.counts.evaluation_pending == 0


@pytest.mark.parametrize("history", ["missing_table", "missing_rows"])
def test_missing_stage_history_keeps_queued_and_complete_compatibility(lake, history):
    _queued(lake)
    lake.insert("idea-complete", "complete", "{}", "", status="completed")
    if history == "missing_table":
        lake.conn.execute("DROP TABLE idea_stage_state")
    else:
        lake.conn.execute("DELETE FROM idea_stage_state")
    lake.conn.commit()
    snapshot = _load(lake)
    assert snapshot.complete and snapshot.queue_count == snapshot.waiting_count == 1
    assert snapshot.counts.inactive == 1


@pytest.mark.parametrize("kind", ["missing", "invalid_bytes", "incompatible_schema"])
def test_unknown_database_is_not_created_repaired_or_called_empty(tmp_path, kind):
    path = tmp_path / "no-parent" / "ideas.db"
    if kind != "missing":
        path = tmp_path / "existing.db"
        if kind == "invalid_bytes":
            path.write_bytes(b"not a database")
        else:
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE unrelated (data TEXT)")
    before = _files(tmp_path)
    snapshot = _module().load_persistent_demand(path)
    assert not snapshot.available and not snapshot.complete
    assert snapshot.queue_count is None and snapshot.waiting_count is None
    assert _files(tmp_path) == before
    assert not (tmp_path / "no-parent").exists()


@pytest.mark.parametrize("kind", ["leaf", "parent", "hardlink"])
def test_redirected_catalog_is_unavailable_without_any_writes(lake, tmp_path, kind):
    _queued(lake)
    path = tmp_path / "redirect.db"
    if kind == "leaf":
        path.symlink_to(lake.db_path)
    elif kind == "parent":
        directory = tmp_path / "redirect"
        directory.symlink_to(tmp_path, target_is_directory=True)
        path = directory / Path(lake.db_path).name
    else:
        os.link(lake.db_path, path)
    before = Path(lake.db_path).read_bytes()
    snapshot = _module().load_persistent_demand(path)
    assert not snapshot.available and snapshot.queue_count is None
    assert Path(lake.db_path).read_bytes() == before


@pytest.mark.parametrize("kind", ["duplicate_ideas", "duplicate_stage", "bad_stage_schema"])
def test_structural_ambiguity_discards_the_entire_partial_read(lake, kind):
    _queued(lake)
    if kind == "duplicate_ideas":
        lake.conn.execute("CREATE TABLE old_ideas AS SELECT * FROM ideas")
        lake.conn.execute("DROP TABLE ideas")
        lake.conn.execute("ALTER TABLE old_ideas RENAME TO ideas")
        lake.conn.execute("INSERT INTO ideas SELECT * FROM ideas")
    else:
        lake.conn.execute("DROP TABLE idea_stage_state")
        if kind == "bad_stage_schema":
            lake.conn.execute("CREATE TABLE idea_stage_state (idea_id TEXT)")
        else:
            lake.conn.execute("CREATE TABLE idea_stage_state (idea_id TEXT,stage TEXT,current_state TEXT)")
            lake.conn.executemany("INSERT INTO idea_stage_state VALUES ('idea-queued','training','PENDING')", [(), ()])
    lake.conn.commit()
    before = _files(Path(lake.db_path).parent)
    snapshot = _load(lake)
    assert not snapshot.available and snapshot.queue_count is None
    assert snapshot.counts.queued == 0  # No trusted lower bound from an ambiguous join.
    assert _files(Path(lake.db_path).parent) == before


def test_interrupted_read_closes_connection_and_does_not_return_partial_counts(lake, monkeypatch):
    _queued(lake)
    module = _module()
    original = module._open_authoritative_lifecycle
    connections = []

    def opening(path):
        connection, reason = original(path)
        connection.set_authorizer(lambda action, *_: sqlite3.SQLITE_DENY
                                 if action == sqlite3.SQLITE_READ else sqlite3.SQLITE_OK)
        connections.append(connection)
        return connection, reason

    monkeypatch.setattr(module, "_open_authoritative_lifecycle", opening)
    snapshot = _load(lake)
    assert not snapshot.available and snapshot.waiting_count is None
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
