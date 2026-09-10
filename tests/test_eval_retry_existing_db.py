"""Evaluation retry opens only an already existing, compatible authority DB."""

import os
import sqlite3

import pytest

from orze.idea_lake import IdeaLake


def _open(path):
    from orze.core.evaluation_retry_state import open_existing_lake
    return open_existing_lake(path)


@pytest.mark.parametrize("nested", [False, True])
def test_missing_database_is_not_created_and_has_no_new_sidecars(tmp_path, nested):
    path = tmp_path / "missing" / "ideas.db" if nested else tmp_path / "ideas.db"
    before = set(tmp_path.rglob("*"))

    with pytest.raises((ValueError, OSError, sqlite3.Error)):
        _open(path)

    assert set(tmp_path.rglob("*")) == before
    assert not path.exists()


@pytest.mark.parametrize("redirect", ["file_symlink", "ancestor_symlink", "hardlink"])
def test_redirected_existing_database_is_rejected_without_content_changes(tmp_path, redirect):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    db = real_parent / "ideas.db"
    lake = IdeaLake(db)
    lake.close()
    if redirect == "file_symlink":
        target = tmp_path / "redirect.db"
        target.symlink_to(db)
    elif redirect == "ancestor_symlink":
        link = tmp_path / "redirect"
        link.symlink_to(real_parent, target_is_directory=True)
        target = link / "ideas.db"
    else:
        target = tmp_path / "hardlink.db"
        os.link(db, target)
    before = db.read_bytes()

    with pytest.raises(ValueError):
        _open(target)

    assert db.read_bytes() == before


def test_nonregular_database_target_is_not_opened(tmp_path):
    target = tmp_path / "database-directory"
    target.mkdir()

    with pytest.raises(ValueError):
        _open(target)

    assert list(target.iterdir()) == []


def test_incompatible_existing_schema_is_not_bootstrapped_or_migrated(tmp_path):
    db = tmp_path / "incompatible.db"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE unrelated (payload TEXT)")
    connection.execute("INSERT INTO unrelated VALUES ('preserve')")
    connection.commit()
    connection.close()
    before = db.read_bytes()

    with pytest.raises(ValueError):
        _open(db)

    assert db.read_bytes() == before
    check = sqlite3.connect(db)
    try:
        assert check.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [
            ("unrelated",),
        ]
    finally:
        check.close()


def test_existing_wal_database_is_rejected_without_changing_its_journal_policy(tmp_path):
    db = tmp_path / "wal.db"
    lake = IdeaLake(db)
    assert lake.conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    try:
        with pytest.raises(ValueError):
            _open(db)
        assert lake.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        lake.close()


def test_existing_lake_supports_retry_without_constructor_or_schema_bootstrap(
    tmp_path, monkeypatch,
):
    db = tmp_path / "ideas.db"
    original = IdeaLake(db)
    original.insert("idea-existing", "Existing", "{}", "", status="queued")
    assert original.record_state_transition("idea-existing", "QUEUED", "CLAIMED")
    assert original.record_state_transition("idea-existing", "CLAIMED", "IN_PROGRESS")
    assert original.record_stage_transition("idea-existing", "training", "IN_PROGRESS", "COMPLETE", "training_completed")
    assert original.record_stage_transition("idea-existing", "evaluation", "PENDING", "IN_PROGRESS", "evaluation_launched")
    assert original.record_state_transition("idea-existing", "IN_PROGRESS", "FAILED", "evaluation_failed")
    original.close()

    def forbidden_bootstrap(*_args, **_kwargs):
        raise AssertionError("existing retry authority must not be bootstrapped")

    monkeypatch.setattr(IdeaLake, "__init__", forbidden_bootstrap)
    monkeypatch.setattr(IdeaLake, "_ensure_schema", forbidden_bootstrap)
    lake = _open(db)
    try:
        assert isinstance(lake, IdeaLake)
        assert lake.get_fsm_state("idea-existing") == "FAILED"
        assert lake.retry_evaluation("idea-existing") is True
        assert lake.get_fsm_state("idea-existing") == "IN_PROGRESS"
        assert lake.get_stage_state("idea-existing", "training") == "COMPLETE"
        assert lake.get_stage_state("idea-existing", "evaluation") == "PENDING"
    finally:
        lake.close()
