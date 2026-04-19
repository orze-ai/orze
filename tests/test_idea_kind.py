"""Tests for the F8 idea.kind field."""

import sqlite3

import pytest

from orze.idea_lake import ALLOWED_KINDS, IdeaLake


def test_fresh_db_has_kind_column(tmp_path):
    lake = IdeaLake(tmp_path / "f.db")
    cols = {r[1] for r in lake.conn.execute("PRAGMA table_info(ideas)")}
    assert "kind" in cols
    # default is 'train'
    lake.insert("idea-001", "t", "x: 1", "raw")
    assert lake.get("idea-001")["kind"] == "train"


def test_allowed_kinds_contract():
    assert {"train", "posthoc_eval", "tta_sweep",
            "agg_search", "bundle_combine", "audit"} <= ALLOWED_KINDS


def test_insert_rejects_unknown_kind(tmp_path):
    lake = IdeaLake(tmp_path / "f.db")
    with pytest.raises(ValueError):
        lake.insert("idea-002", "t", "x: 1", "raw", kind="bogus")


def test_insert_accepts_all_allowed_kinds(tmp_path):
    lake = IdeaLake(tmp_path / "f.db")
    for k in sorted(ALLOWED_KINDS):
        lake.insert(f"idea-{k}", f"t-{k}", "x: 1", "raw", kind=k)
        assert lake.get(f"idea-{k}")["kind"] == k


def test_migration_from_legacy_db(tmp_path):
    """A pre-F8 DB (no kind column) should gain one on open with default train."""
    db_path = tmp_path / "legacy.db"
    # Build a legacy-shaped schema by hand — no 'kind' column.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE ideas (
            idea_id TEXT PRIMARY KEY,
            id_num INTEGER,
            title TEXT NOT NULL,
            priority TEXT DEFAULT 'medium',
            category TEXT DEFAULT 'architecture',
            parent TEXT,
            hypothesis TEXT,
            config TEXT NOT NULL,
            raw_markdown TEXT NOT NULL,
            config_summary TEXT,
            eval_metrics TEXT,
            status TEXT DEFAULT 'archived',
            training_time REAL,
            archived_at TEXT,
            created_at TEXT,
            approach_family TEXT DEFAULT 'other'
        )"""
    )
    conn.execute(
        "INSERT INTO ideas (idea_id, title, config, raw_markdown) "
        "VALUES ('idea-leg', 'legacy', 'x: 1', 'raw')"
    )
    conn.commit()
    conn.close()

    # Opening via IdeaLake should migrate and backfill kind='train'.
    lake = IdeaLake(db_path)
    cols = {r[1] for r in lake.conn.execute("PRAGMA table_info(ideas)")}
    assert "kind" in cols
    row = lake.get("idea-leg")
    assert row["kind"] == "train"
    # Subsequent inserts honor kind.
    lake.insert("idea-ph", "ph", "x: 1", "raw", kind="posthoc_eval")
    assert lake.get("idea-ph")["kind"] == "posthoc_eval"


def test_yaml_kind_field_accepted(tmp_path):
    """Reading a kind: from YAML idea config should round-trip through insert."""
    import yaml
    idea_yaml = """
title: TTA hflip
kind: tta_sweep
config:
  tta: hflip
"""
    obj = yaml.safe_load(idea_yaml)
    lake = IdeaLake(tmp_path / "f.db")
    lake.insert(
        "idea-hflip", obj["title"], idea_yaml, "raw",
        kind=obj.get("kind", "train"),
    )
    assert lake.get("idea-hflip")["kind"] == "tta_sweep"
