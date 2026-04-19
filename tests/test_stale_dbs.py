"""Tests for F6 — stale zero-byte DB relocation."""
from pathlib import Path

from orze.engine.stale_dbs import relocate_zero_byte_dbs


def test_moves_zero_byte_files(tmp_path):
    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "queue.db").write_bytes(b"")
    (cwd / "orze.db").write_bytes(b"")
    # Non-empty file should NOT be moved
    (cwd / "idea_lake.db").write_bytes(b"legit data")

    stale = tmp_path / "results" / "_stale"
    moved = relocate_zero_byte_dbs(cwd, stale)

    names_moved = sorted(src.name for src, _ in moved)
    assert names_moved == ["orze.db", "queue.db"]
    assert not (cwd / "queue.db").exists()
    assert (cwd / "idea_lake.db").exists()
    dests = sorted(d.name for _, d in moved)
    assert any(d.startswith("queue.db.") for d in dests)


def test_idempotent_on_clean_cwd(tmp_path):
    cwd = tmp_path / "p"
    cwd.mkdir()
    stale = tmp_path / "results" / "_stale"
    assert relocate_zero_byte_dbs(cwd, stale) == []
    # Calling again — still no-op.
    assert relocate_zero_byte_dbs(cwd, stale) == []


def test_skips_files_not_in_whitelist(tmp_path):
    cwd = tmp_path / "p"
    cwd.mkdir()
    (cwd / "something_else.db").write_bytes(b"")
    stale = tmp_path / "_stale"
    assert relocate_zero_byte_dbs(cwd, stale) == []
    assert (cwd / "something_else.db").exists()
