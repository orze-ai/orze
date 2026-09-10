"""Acceptance tests for the new closed, read-only catalog value adapter.

This is a new mechanism suite, not a claim that a missing former API was a
behavioral failure. Public CLI/admin regressions are covered independently.
"""

import os
import sqlite3
from pathlib import Path

import pytest

from orze.idea_lake import IdeaLake
from orze.reporting import catalog


@pytest.fixture
def lake(tmp_path):
    instance = IdeaLake(tmp_path / "ideas.db")
    try:
        yield instance
    finally:
        instance.close()


def _insert(lake, idea_id="idea-good", status="completed"):
    lake.insert(idea_id, "Preserve display title", "huge_config: not_needed", "raw",
                status=status, priority="high", eval_metrics={"quality": 999})


def _observe_connections(monkeypatch):
    original = catalog._open_authoritative_lifecycle
    connections, statements = [], []

    def observe(path):
        connection, reason = original(path)
        if connection is not None:
            assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
            connection.set_trace_callback(statements.append)
            connections.append(connection)
        return connection, reason

    monkeypatch.setattr(catalog, "_open_authoritative_lifecycle", observe)
    return connections, statements


def _assert_closed(connections):
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def _file_snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def test_snapshot_uses_read_transaction_closes_connection_and_keeps_copied_values(
    lake, monkeypatch,
):
    _insert(lake)
    before = Path(lake.db_path).read_bytes()
    connections, statements = _observe_connections(monkeypatch)

    def forbidden_initialization(*_args, **_kwargs):
        raise AssertionError("snapshot must not initialize or bootstrap IdeaLake")

    monkeypatch.setattr(IdeaLake, "__init__", forbidden_initialization)
    monkeypatch.setattr(IdeaLake, "_ensure_schema", forbidden_initialization)

    snapshot = catalog.load_catalog_snapshot(lake.db_path)

    assert snapshot.available
    _assert_closed(connections)
    assert Path(lake.db_path).read_bytes() == before
    assert any(statement.strip().upper() == "BEGIN" for statement in statements)
    assert all(statement.lstrip().split(None, 1)[0].upper() in
               {"BEGIN", "SELECT", "PRAGMA"} for statement in statements)
    records = snapshot.get_metadata_index()
    assert records["idea-good"]["title"] == "Preserve display title"
    assert records["idea-good"]["priority"] == "high"
    assert "config" not in records["idea-good"]
    assert "eval_metrics" not in records["idea-good"]
    assert snapshot.get_lifecycle_counts() == {"COMPLETED": 1}
    # Returned adapter copies must not be live DB handles or mutable aliases.
    records["idea-good"]["title"] = "caller changed copy"
    snapshot.get_lifecycle_counts()["COMPLETED"] = 99
    assert snapshot.get_metadata_index()["idea-good"]["title"] == "Preserve display title"
    assert snapshot.get_lifecycle_counts() == {"COMPLETED": 1}

    lake.conn.execute("BEGIN IMMEDIATE")
    lake.conn.execute("UPDATE ideas SET title='New title', status='failed'")
    lake.conn.execute("UPDATE idea_state SET current_state='FAILED'")
    lake.conn.commit()
    assert snapshot.get_metadata_index()["idea-good"]["title"] == "Preserve display title"
    assert snapshot.get_lifecycle_counts() == {"COMPLETED": 1}
    refreshed = catalog.load_catalog_snapshot(lake.db_path)
    assert refreshed.get_metadata_index()["idea-good"]["title"] == "New title"
    assert refreshed.get_lifecycle_counts() == {"FAILED": 1}
    _assert_closed(connections)


@pytest.mark.parametrize("database", ["missing", "incompatible"])
def test_unavailable_database_never_creates_bootstraps_or_migrates(tmp_path, database):
    path = tmp_path / "uncreated-parent" / "ideas.db"
    if database == "incompatible":
        path = tmp_path / "incompatible.db"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE unrelated (payload TEXT)")
            connection.execute("INSERT INTO unrelated VALUES ('preserve')")
    before = _file_snapshot(tmp_path)

    snapshot = catalog.load_catalog_snapshot(path)

    assert not snapshot.available
    assert snapshot.reason != "authoritative_lifecycle_loaded"
    assert snapshot.get_metadata_index() == {}
    assert snapshot.get_lifecycle_counts() == {}
    assert _file_snapshot(tmp_path) == before


@pytest.mark.parametrize("redirect", ["leaf_symlink", "ancestor_symlink", "hardlink"])
def test_redirected_database_is_rejected_without_changing_files(tmp_path, redirect):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    db = real_parent / "ideas.db"
    original = IdeaLake(db)
    _insert(original)
    original.close()
    if redirect == "leaf_symlink":
        path = tmp_path / "redirect.db"
        path.symlink_to(db)
    elif redirect == "ancestor_symlink":
        link = tmp_path / "redirect"
        link.symlink_to(real_parent, target_is_directory=True)
        path = link / "ideas.db"
    else:
        path = tmp_path / "hardlink.db"
        os.link(db, path)
    before = _file_snapshot(tmp_path)

    snapshot = catalog.load_catalog_snapshot(path)

    assert not snapshot.available
    assert snapshot.reason == "authoritative_lifecycle_database_redirected"
    assert snapshot.records == {}
    assert _file_snapshot(tmp_path) == before


def test_bad_rows_are_unknown_without_discarding_good_rows_or_changing_pipeline_basis(lake):
    _insert(lake, "idea-good")
    _insert(lake, "idea-mirror-conflict")
    _insert(lake, "idea-stage-conflict")
    _insert(lake, "idea-no-fsm", status="queued")
    lake.conn.execute(
        "UPDATE ideas SET status='queued' WHERE idea_id='idea-mirror-conflict'",
    )
    lake.conn.execute("DELETE FROM idea_state WHERE idea_id='idea-no-fsm'")
    # Deliberate pre-existing corruption: normal lifecycle writers correctly
    # refuse to start evaluation once the global task is complete.
    lake.conn.execute(
        "INSERT INTO idea_stage_state (idea_id, stage, current_state, updated_at) "
        "VALUES ('idea-stage-conflict', 'evaluation', 'IN_PROGRESS', '2000-01-01')",
    )
    lake.conn.commit()

    snapshot = catalog.load_catalog_snapshot(lake.db_path)

    assert snapshot.available
    records = snapshot.get_metadata_index()
    assert len(records) == 4
    assert records["idea-good"]["lifecycle_state"] == "COMPLETE"
    for idea_id in ("idea-mirror-conflict", "idea-stage-conflict", "idea-no-fsm"):
        assert records[idea_id]["lifecycle_state"] == "UNKNOWN"
        assert records[idea_id]["lifecycle_reason"] != "lifecycle_agreed"
    assert snapshot.unknown_count == 3
    assert records["idea-mirror-conflict"]["status"] == "queued"
    assert records["idea-mirror-conflict"]["fsm_state"] == "COMPLETE"
    # Pipeline counts intentionally retain the existing audited-FSM definition;
    # admin task agreement is a separate derived per-record state.
    assert snapshot.get_lifecycle_counts() == lake.get_lifecycle_counts()
    assert snapshot.get_lifecycle_counts() == {"COMPLETED": 3, "QUEUED": 1}


def test_unrecorded_optional_stages_and_training_only_skipped_evaluation_are_compatible(lake):
    _insert(lake, "idea-historical")
    _insert(lake, "idea-training-only", status="queued")
    for old, new in (("QUEUED", "CLAIMED"), ("CLAIMED", "IN_PROGRESS"),
                     ("IN_PROGRESS", "COMPLETE")):
        assert lake.record_state_transition("idea-training-only", old, new, "fixture")

    with_stages = catalog.load_catalog_snapshot(lake.db_path)

    assert with_stages.available
    historical = with_stages.records["idea-historical"]
    assert historical["lifecycle_state"] == "COMPLETE"
    assert historical["training_state"] is None
    assert historical["evaluation_state"] is None
    current = with_stages.records["idea-training-only"]
    assert current["lifecycle_state"] == "COMPLETE"
    assert current["training_state"] == "COMPLETE"
    assert current["evaluation_state"] == "SKIPPED"
    # A genuinely older read-only schema need not acquire a new stage table.
    lake.conn.execute("DROP TABLE idea_stage_state")
    lake.conn.commit()
    before = Path(lake.db_path).read_bytes()

    without_stages = catalog.load_catalog_snapshot(lake.db_path)

    assert without_stages.available
    assert without_stages.unknown_count == 0
    assert all(record["training_state"] is None and record["evaluation_state"] is None
               for record in without_stages.records.values())
    assert Path(lake.db_path).read_bytes() == before
    assert lake.conn.execute(
        "SELECT name FROM sqlite_master WHERE name='idea_stage_state'",
    ).fetchone() is None


@pytest.mark.parametrize("broken", ["malformed_stage_schema", "duplicate_join_identity"])
def test_broken_stage_catalog_is_unavailable_and_closes_connection(lake, monkeypatch, broken):
    _insert(lake)
    lake.conn.execute("DROP TABLE idea_stage_state")
    if broken == "malformed_stage_schema":
        lake.conn.execute("CREATE TABLE idea_stage_state (idea_id TEXT, stage TEXT)")
    else:
        lake.conn.execute(
            "CREATE TABLE idea_stage_state (idea_id TEXT, stage TEXT, current_state TEXT)",
        )
        lake.conn.executemany(
            "INSERT INTO idea_stage_state VALUES (?, ?, ?)",
            [("idea-good", "evaluation", "COMPLETE")] * 2,
        )
    lake.conn.commit()
    before = Path(lake.db_path).read_bytes()
    connections, _ = _observe_connections(monkeypatch)

    snapshot = catalog.load_catalog_snapshot(lake.db_path)

    assert not snapshot.available
    assert snapshot.reason == "authoritative_catalog_invalid"
    assert snapshot.get_metadata_index() == {}
    assert snapshot.get_lifecycle_counts() == {}
    assert Path(lake.db_path).read_bytes() == before
    _assert_closed(connections)
