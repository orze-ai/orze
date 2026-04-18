"""Tests for orze.engine.upgrade_cleanup.

Verifies the two important behaviours:
  1. Version change detected → listed garbage files deleted, stamp refreshed.
  2. Same version (or first boot) → nothing is deleted, stamp gets written.

The module is patched-in-place — we swap ``_current_versions`` rather
than touching real package metadata, so the tests are hermetic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orze.engine import upgrade_cleanup
from orze.engine.upgrade_cleanup import (
    STAMP_FILENAME,
    _GARBAGE_FILES,
    _GARBAGE_GLOBS,
    check_and_clean,
)


def _seed_garbage(results_dir: Path) -> list:
    """Create every file the module is expected to scrub, plus a couple
    of trigger files and one file that MUST be left alone. Returns the
    list of paths seeded as garbage."""
    garbage = []
    for name in _GARBAGE_FILES:
        p = results_dir / name
        p.write_text("stale", encoding="utf-8")
        garbage.append(p)
    # _trigger_* glob coverage — write two distinct triggers
    for name in ("_trigger_professor", "_trigger_engineer"):
        p = results_dir / name
        p.write_text("stale", encoding="utf-8")
        garbage.append(p)
    # A file that must NEVER be touched by cleanup.
    (results_dir / "report.md").write_text("valuable", encoding="utf-8")
    (results_dir / "idea_lake.db").write_text("valuable", encoding="utf-8")
    (results_dir / "_retrospection.txt").write_text("valuable",
                                                    encoding="utf-8")
    return garbage


_UNSET = object()


@pytest.fixture
def fake_versions(monkeypatch):
    """Let each test set the "installed" versions directly.

    ``None`` is a valid value (means "package not installed"), so the
    sentinel ``_UNSET`` is used to distinguish "don't override" from
    "override to None".
    """
    versions = {"orze": "3.4.11", "orze_pro": "0.7.10"}

    def _set(orze=_UNSET, orze_pro=_UNSET):
        if orze is not _UNSET:
            versions["orze"] = orze
        if orze_pro is not _UNSET:
            versions["orze_pro"] = orze_pro

    monkeypatch.setattr(upgrade_cleanup, "_current_versions",
                        lambda: dict(versions))
    return _set


class TestFirstBoot:
    def test_missing_stamp_writes_stamp_and_cleans_nothing(self, tmp_path,
                                                          fake_versions):
        _seed_garbage(tmp_path)

        result = check_and_clean(tmp_path)

        assert result["upgraded"] is False
        assert result["from"] is None
        assert result["cleaned"] == []

        # Stamp exists and is parseable.
        stamp = json.loads((tmp_path / STAMP_FILENAME).read_text("utf-8"))
        assert stamp["orze"] == "3.4.11"
        assert stamp["orze_pro"] == "0.7.10"
        assert "stamped_at" in stamp

        # Every seeded garbage file is still there — first boot must be safe.
        for name in _GARBAGE_FILES:
            assert (tmp_path / name).exists(), name
        assert (tmp_path / "_trigger_professor").exists()

    def test_missing_results_dir_returns_noop(self, tmp_path, fake_versions):
        missing = tmp_path / "does_not_exist"
        result = check_and_clean(missing)
        assert result["upgraded"] is False
        assert result["cleaned"] == []
        assert not missing.exists()


class TestSameVersion:
    def test_nothing_cleaned(self, tmp_path, fake_versions):
        # Pre-populate stamp with the versions we'll report next call.
        (tmp_path / STAMP_FILENAME).write_text(
            json.dumps({"orze": "3.4.11", "orze_pro": "0.7.10",
                        "stamped_at": "2026-04-18T00:00:00"}),
            encoding="utf-8")
        _seed_garbage(tmp_path)

        result = check_and_clean(tmp_path)

        assert result["upgraded"] is False
        assert result["cleaned"] == []
        for name in _GARBAGE_FILES:
            assert (tmp_path / name).exists(), name


class TestUpgradeDetected:
    def test_orze_version_bump_cleans_all_garbage(self, tmp_path,
                                                  fake_versions):
        (tmp_path / STAMP_FILENAME).write_text(
            json.dumps({"orze": "3.3.5", "orze_pro": "0.7.10",
                        "stamped_at": "2026-04-10T00:00:00"}),
            encoding="utf-8")
        seeded = _seed_garbage(tmp_path)

        result = check_and_clean(tmp_path)

        assert result["upgraded"] is True
        assert result["from"]["orze"] == "3.3.5"
        assert result["to"]["orze"] == "3.4.11"

        # Every seeded garbage file is gone.
        for p in seeded:
            assert not p.exists(), p

        # Garbage files deleted exactly cover the names in cleaned.
        expected_names = set(_GARBAGE_FILES) | {
            "_trigger_professor", "_trigger_engineer"}
        assert set(result["cleaned"]) == expected_names

        # Valuable files were NOT touched.
        assert (tmp_path / "report.md").exists()
        assert (tmp_path / "idea_lake.db").exists()
        assert (tmp_path / "_retrospection.txt").exists()

        # Stamp refreshed to new version.
        stamp = json.loads((tmp_path / STAMP_FILENAME).read_text("utf-8"))
        assert stamp["orze"] == "3.4.11"
        assert stamp["orze_pro"] == "0.7.10"

    def test_orze_pro_version_bump_alone_triggers_cleanup(self, tmp_path,
                                                          fake_versions):
        """The real-world case: orze unchanged, orze-pro upgraded."""
        (tmp_path / STAMP_FILENAME).write_text(
            json.dumps({"orze": "3.4.11", "orze_pro": "0.7.6",
                        "stamped_at": "2026-04-18T10:00:00"}),
            encoding="utf-8")
        (tmp_path / ".pause_research").write_text("stale", "utf-8")

        result = check_and_clean(tmp_path)

        assert result["upgraded"] is True
        assert ".pause_research" in result["cleaned"]
        assert not (tmp_path / ".pause_research").exists()

    def test_orze_pro_removed_triggers_cleanup(self, tmp_path, fake_versions):
        """orze-pro uninstalled since last boot → upgrade path fires."""
        (tmp_path / STAMP_FILENAME).write_text(
            json.dumps({"orze": "3.4.11", "orze_pro": "0.7.10",
                        "stamped_at": "2026-04-18T10:00:00"}),
            encoding="utf-8")
        (tmp_path / ".pause_research").write_text("stale", "utf-8")
        fake_versions(orze_pro=None)

        result = check_and_clean(tmp_path)

        assert result["upgraded"] is True
        assert not (tmp_path / ".pause_research").exists()

    def test_downgrade_also_triggers_cleanup(self, tmp_path, fake_versions):
        """Any version delta counts — downgrades ship incompatible state
        too."""
        (tmp_path / STAMP_FILENAME).write_text(
            json.dumps({"orze": "3.4.11", "orze_pro": "0.7.10",
                        "stamped_at": "2026-04-18T10:00:00"}),
            encoding="utf-8")
        (tmp_path / "_fsm_state.json").write_text("{}", "utf-8")
        fake_versions(orze="3.4.10")

        result = check_and_clean(tmp_path)

        assert result["upgraded"] is True
        assert "_fsm_state.json" in result["cleaned"]


class TestResilience:
    def test_corrupt_stamp_treated_as_first_boot(self, tmp_path,
                                                 fake_versions):
        (tmp_path / STAMP_FILENAME).write_text("{not valid json",
                                                encoding="utf-8")
        (tmp_path / ".pause_research").write_text("stale", "utf-8")

        result = check_and_clean(tmp_path)

        # Corrupt stamp → treat as first boot: do NOT scrub.
        assert result["upgraded"] is False
        assert result["cleaned"] == []
        assert (tmp_path / ".pause_research").exists()
        # Stamp is now valid JSON.
        stamp = json.loads((tmp_path / STAMP_FILENAME).read_text("utf-8"))
        assert stamp["orze"] == "3.4.11"

    def test_missing_garbage_files_are_not_errors(self, tmp_path,
                                                  fake_versions):
        (tmp_path / STAMP_FILENAME).write_text(
            json.dumps({"orze": "3.3.5", "orze_pro": "0.7.10",
                        "stamped_at": "2026-04-10T00:00:00"}),
            encoding="utf-8")
        # Only seed ONE garbage file — the rest are missing.
        (tmp_path / ".pause_research").write_text("stale", "utf-8")

        result = check_and_clean(tmp_path)

        assert result["upgraded"] is True
        assert result["cleaned"] == [".pause_research"]

    def test_glob_does_not_match_unrelated_files(self, tmp_path,
                                                  fake_versions):
        (tmp_path / STAMP_FILENAME).write_text(
            json.dumps({"orze": "3.3.5", "orze_pro": "0.7.10",
                        "stamped_at": "2026-04-10T00:00:00"}),
            encoding="utf-8")
        # These look vaguely related but must not match the scrubber.
        (tmp_path / "trigger_notes.md").write_text("keep me", "utf-8")
        (tmp_path / "_triggered_by.json").write_text("keep me", "utf-8")
        (tmp_path / "_trigger_thinker").write_text("zap me", "utf-8")

        result = check_and_clean(tmp_path)

        assert "_trigger_thinker" in result["cleaned"]
        assert "trigger_notes.md" not in result["cleaned"]
        assert "_triggered_by.json" not in result["cleaned"]
        assert (tmp_path / "trigger_notes.md").exists()
        assert (tmp_path / "_triggered_by.json").exists()
        assert not (tmp_path / "_trigger_thinker").exists()

    def test_archived_triggers_are_preserved(self, tmp_path, fake_versions):
        """User-archived trigger files (``_trigger_*.archived.*``,
        ``_trigger_*.bak``) are history, not garbage. They must survive
        an upgrade-triggered cleanup."""
        (tmp_path / STAMP_FILENAME).write_text(
            json.dumps({"orze": "3.3.5", "orze_pro": "0.7.10",
                        "stamped_at": "2026-04-10T00:00:00"}),
            encoding="utf-8")
        (tmp_path / "_trigger_engineer").write_text("live", "utf-8")
        (tmp_path / "_trigger_bug_fixer.archived.20260418").write_text(
            "archived", "utf-8")
        (tmp_path / "_trigger_professor.bak").write_text("backup", "utf-8")

        result = check_and_clean(tmp_path)

        assert "_trigger_engineer" in result["cleaned"]
        assert "_trigger_bug_fixer.archived.20260418" not in result["cleaned"]
        assert "_trigger_professor.bak" not in result["cleaned"]
        assert not (tmp_path / "_trigger_engineer").exists()
        assert (tmp_path
                / "_trigger_bug_fixer.archived.20260418").exists()
        assert (tmp_path / "_trigger_professor.bak").exists()


def test_garbage_file_list_covers_the_real_pain_points():
    """Regression guard: the original bug report was a stuck
    ``.pause_research`` after an orze-pro upgrade, with a stale FSM
    state file. If someone shrinks the list, this test fails loudly."""
    assert ".pause_research" in _GARBAGE_FILES
    assert "_fsm_state.json" in _GARBAGE_FILES
    assert "_trigger_*" in _GARBAGE_GLOBS
