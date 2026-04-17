"""Test that check_active_roles respects the writes_ideas_file flag."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orze.engine import roles as roles_mod
from orze.engine.process import RoleProcess


def _make_finished_proc(ret: int = 0):
    """Build a mock subprocess that has already exited with `ret`."""
    proc = MagicMock()
    proc.poll.return_value = ret
    return proc


def _make_rp(role_name: str, writes_ideas_file: bool,
             tmp_path: Path, ideas_pre_size: int = 100) -> RoleProcess:
    log_fh = open(tmp_path / f"{role_name}.log", "w")
    rp = RoleProcess(
        role_name=role_name,
        process=_make_finished_proc(0),
        start_time=time.time() - 5,
        log_path=tmp_path / f"{role_name}.log",
        timeout=600.0,
        lock_dir=tmp_path / f"{role_name}_lock",
        cycle_num=1,
        _log_fh=log_fh,
        ideas_pre_size=ideas_pre_size,
        ideas_pre_count=0,
        writes_ideas_file=writes_ideas_file,
    )
    rp.lock_dir.mkdir(exist_ok=True)
    return rp


def test_strategy_role_skips_soft_failure_check(tmp_path, caplog):
    """writes_ideas_file=False → no soft-failure warning even without ideas.md."""
    caplog.set_level(logging.WARNING)
    # Reset module-level counter to isolate test.
    roles_mod._consecutive_soft_failures.clear()

    rp = _make_rp("engineer", writes_ideas_file=False, tmp_path=tmp_path)
    active = {"engineer": rp}

    # No ideas.md exists at all — but engineer shouldn't care.
    finished = roles_mod.check_active_roles(
        active, ideas_file=str(tmp_path / "ideas.md"))

    assert finished == [("engineer", True)]
    assert "soft failure" not in caplog.text.lower()
    assert "engineer" not in roles_mod._consecutive_soft_failures


def test_research_role_still_gets_soft_failure_warning(tmp_path, caplog):
    """writes_ideas_file=True (default) → warning when ideas.md unchanged."""
    caplog.set_level(logging.WARNING)
    roles_mod._consecutive_soft_failures.clear()

    ideas = tmp_path / "ideas.md"
    ideas.write_text("x" * 100)  # match ideas_pre_size

    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path,
                  ideas_pre_size=100)
    active = {"research": rp}

    finished = roles_mod.check_active_roles(active, ideas_file=str(ideas))

    assert finished == [("research", True)]
    assert "soft failure" in caplog.text.lower()
    assert roles_mod._consecutive_soft_failures["research"] == 1


def test_research_role_appended_ideas_no_warning(tmp_path, caplog):
    """research that actually appends to ideas.md gets no warning."""
    caplog.set_level(logging.WARNING)
    roles_mod._consecutive_soft_failures.clear()

    ideas = tmp_path / "ideas.md"
    ideas.write_text("x" * 500)  # larger than pre_size

    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path,
                  ideas_pre_size=100)
    active = {"research": rp}

    finished = roles_mod.check_active_roles(active, ideas_file=str(ideas))

    assert finished == [("research", True)]
    assert "soft failure" not in caplog.text.lower()


def test_default_writes_ideas_file_is_true(tmp_path):
    """RoleProcess without explicit flag defaults to True (back-compat)."""
    rp = RoleProcess(
        role_name="x",
        process=_make_finished_proc(0),
        start_time=time.time(),
        log_path=tmp_path / "x.log",
        timeout=600.0,
        lock_dir=tmp_path / "lock",
        cycle_num=1,
    )
    assert rp.writes_ideas_file is True
