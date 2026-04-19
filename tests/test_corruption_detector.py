"""Tests for F3 — corruption detector tolerates legitimate post-ingest wipe,
stale .corrupt.* files are migrated to _corrupt_ideas/ and pruned."""
import time
from pathlib import Path
from types import SimpleNamespace

from orze.engine import roles


def _make_rp(ideas_path: Path, pre_size: int, pre_count: int):
    return SimpleNamespace(
        role_name="testrole",
        cycle_num=1,
        ideas_pre_size=pre_size,
        ideas_pre_count=pre_count,
        ideas_consumed_during_run=0,
        ideas_md_mtime_pre=0.0,
        writes_ideas_file=True,
    )


def test_shrink_after_ingest_is_not_corruption(tmp_path, caplog):
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n\n## idea-aaa: x\n", encoding="utf-8")
    rp = _make_rp(ideas, pre_size=ideas.stat().st_size, pre_count=1)

    # Simulate the ingest-then-wipe sequence.
    ideas.write_text("# Ideas\n\n", encoding="utf-8")
    roles.mark_ingest(ideas)

    # Backup exists so if detector misfires, it would "restore".
    (tmp_path / "ideas.md.safe").write_text(
        "# Ideas\n\n## idea-aaa: x\n", encoding="utf-8")

    caplog.set_level("ERROR")
    roles._check_ideas_integrity(str(ideas), rp)
    # No corruption-error logged.
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("CORRUPTION DETECTED" in m for m in msgs), msgs
    # File stayed wiped.
    assert ideas.read_text(encoding="utf-8").strip() == "# Ideas"


def test_shrink_without_ingest_IS_corruption(tmp_path, caplog):
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n\n## idea-aaa: x\n", encoding="utf-8")
    rp = _make_rp(ideas, pre_size=ideas.stat().st_size, pre_count=1)

    # Backup with the full content
    (tmp_path / "ideas.md.safe").write_text(
        "# Ideas\n\n## idea-aaa: x\n", encoding="utf-8")

    # Rogue truncate — NO ingest was recorded.
    ideas.write_text("# Ideas\n\n", encoding="utf-8")

    caplog.set_level("ERROR")
    roles._check_ideas_integrity(str(ideas), rp)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("CORRUPTION DETECTED" in m for m in msgs), msgs
    # Restored.
    assert "idea-aaa" in ideas.read_text(encoding="utf-8")


def test_cleanup_stale_corrupt_files_migrates_and_prunes(tmp_path):
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n", encoding="utf-8")

    # 7 stray .corrupt.* files at cwd, varying mtimes
    for i in range(7):
        p = tmp_path / f"ideas.md.corrupt.{1000 + i}"
        p.write_text(f"junk {i}", encoding="utf-8")
        # Ensure distinct mtime
        import os as _os
        _os.utime(p, (1000 + i, 1000 + i))

    moved = roles.cleanup_stale_corrupt_files(ideas, keep=5)
    assert moved == 7
    archive = tmp_path / "_corrupt_ideas"
    remaining = sorted(archive.iterdir())
    assert len(remaining) == 5, remaining
    # Oldest should be pruned — highest-numbered survive.
    assert all(int(p.name.split(".")[-1]) >= 1002 for p in remaining)
