import sqlite3

import pytest

import orze.idea_lake as idea_lake_module
from orze.idea_lake import IdeaLake, _schema_bootstrap_cache_key


def test_same_unchanged_database_reuses_completed_schema_bootstrap(tmp_path):
    path = tmp_path / "idea_lake.db"
    first = IdeaLake(path)
    assert first.schema_bootstrap_cache_hit is False
    first.close()

    second = IdeaLake(path)
    assert second.schema_bootstrap_cache_hit is True
    second.close()


def test_different_database_does_not_reuse_schema_bootstrap(tmp_path):
    first = IdeaLake(tmp_path / "first.db")
    first.close()

    second = IdeaLake(tmp_path / "second.db")
    assert second.schema_bootstrap_cache_hit is False
    second.close()


def test_schema_change_forces_full_bootstrap_and_repairs_index(tmp_path):
    path = tmp_path / "idea_lake.db"
    lake = IdeaLake(path)
    lake.close()

    conn = sqlite3.connect(path)
    conn.execute("DROP INDEX idx_trigger_consumptions_role")
    conn.commit()
    conn.close()

    repaired = IdeaLake(path)
    assert repaired.schema_bootstrap_cache_hit is False
    assert repaired.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
        ("idx_trigger_consumptions_role",),
    ).fetchone()
    repaired.close()


def test_missing_migration_marker_forces_reconciliation(tmp_path):
    path = tmp_path / "idea_lake.db"
    lake = IdeaLake(path)
    lake.conn.execute(
        "DELETE FROM schema_migrations WHERE name = 'partial_is_failed_v1'"
    )
    lake.conn.commit()
    lake.close()

    repaired = IdeaLake(path)
    assert repaired.schema_bootstrap_cache_hit is False
    assert repaired.conn.execute(
        "SELECT 1 FROM schema_migrations WHERE name = ?",
        ("partial_is_failed_v1",),
    ).fetchone()
    repaired.close()


def test_missing_sequence_row_forces_full_bootstrap(tmp_path):
    path = tmp_path / "idea_lake.db"
    lake = IdeaLake(path)
    lake.conn.execute("DELETE FROM id_sequence")
    lake.conn.commit()
    lake.close()

    repaired = IdeaLake(path)
    assert repaired.schema_bootstrap_cache_hit is False
    assert repaired.conn.execute(
        "SELECT next_id FROM id_sequence"
    ).fetchone()[0] == 1
    repaired.close()


def test_memory_and_uri_databases_are_never_cache_keys():
    assert _schema_bootstrap_cache_key(":memory:") is None
    assert _schema_bootstrap_cache_key("file:shared?mode=memory") is None


def test_failed_fresh_bootstrap_rolls_back_all_schema(tmp_path, monkeypatch):
    path = tmp_path / "idea_lake.db"
    monkeypatch.setattr(
        idea_lake_module,
        "_SCHEMA_SQL",
        idea_lake_module._SCHEMA_SQL + "\nCREATE TABLE invalid (;;",
    )

    with pytest.raises(sqlite3.OperationalError):
        IdeaLake(path)

    conn = sqlite3.connect(path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    conn.close()
    assert tables == []
