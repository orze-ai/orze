"""Test check_active_roles outcome tuples, soft-failure tracking, timeouts."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orze.engine import roles as roles_mod
from orze.engine.process import RoleProcess
from orze.engine.roles import (
    OUTCOME_OK,
    OUTCOME_TIMEOUT,
    OUTCOME_ERROR,
    OUTCOME_SOFT_FAILURE,
    is_success,
)


def _make_finished_proc(ret: int = 0):
    """Build a mock subprocess that has already exited with `ret`."""
    proc = MagicMock()
    proc.poll.return_value = ret
    return proc


def _make_running_proc():
    """Build a mock subprocess that is still running (poll returns None)."""
    proc = MagicMock()
    proc.poll.return_value = None
    return proc


def _make_rp(role_name: str, writes_ideas_file: bool,
             tmp_path: Path, ideas_pre_size: int = 100,
             process=None, start_time: float | None = None,
             timeout: float = 600.0) -> RoleProcess:
    log_fh = open(tmp_path / f"{role_name}.log", "w")
    rp = RoleProcess(
        role_name=role_name,
        process=process if process is not None else _make_finished_proc(0),
        start_time=start_time if start_time is not None else time.time() - 5,
        log_path=tmp_path / f"{role_name}.log",
        timeout=timeout,
        lock_dir=tmp_path / f"{role_name}_lock",
        cycle_num=1,
        _log_fh=log_fh,
        ideas_pre_size=ideas_pre_size,
        ideas_pre_count=0,
        writes_ideas_file=writes_ideas_file,
    )
    rp.lock_dir.mkdir(exist_ok=True)
    return rp


def test_strategy_role_ok_outcome(tmp_path, caplog):
    """writes_ideas_file=False exit 0 → OUTCOME_OK, no soft-failure warning."""
    caplog.set_level(logging.WARNING)
    roles_mod._consecutive_soft_failures.clear()

    rp = _make_rp("engineer", writes_ideas_file=False, tmp_path=tmp_path)
    active = {"engineer": rp}

    finished = roles_mod.check_active_roles(
        active, ideas_file=str(tmp_path / "ideas.md"))

    assert finished == [("engineer", OUTCOME_OK)]
    assert "soft failure" not in caplog.text.lower()
    assert "engineer" not in roles_mod._consecutive_soft_failures


def test_research_role_soft_failure_outcome(tmp_path, caplog):
    """writes_ideas_file=True + ideas.md unchanged → OUTCOME_SOFT_FAILURE."""
    caplog.set_level(logging.WARNING)
    roles_mod._consecutive_soft_failures.clear()

    ideas = tmp_path / "ideas.md"
    ideas.write_text("x" * 100)  # match ideas_pre_size

    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path,
                  ideas_pre_size=100)
    active = {"research": rp}

    finished = roles_mod.check_active_roles(active, ideas_file=str(ideas))

    assert finished == [("research", OUTCOME_SOFT_FAILURE)]
    assert "soft failure" in caplog.text.lower()
    assert roles_mod._consecutive_soft_failures["research"] == 1


def test_research_role_appended_ideas_ok_outcome(tmp_path, caplog):
    """research that actually appends to ideas.md → OUTCOME_OK."""
    caplog.set_level(logging.WARNING)
    roles_mod._consecutive_soft_failures.clear()

    ideas = tmp_path / "ideas.md"
    ideas.write_text("x" * 500)  # larger than pre_size

    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path,
                  ideas_pre_size=100)
    active = {"research": rp}

    finished = roles_mod.check_active_roles(active, ideas_file=str(ideas))

    assert finished == [("research", OUTCOME_OK)]
    assert "soft failure" not in caplog.text.lower()


def test_role_error_outcome(tmp_path, caplog):
    """Non-zero exit code → OUTCOME_ERROR regardless of writes_ideas_file."""
    caplog.set_level(logging.WARNING)
    roles_mod._consecutive_soft_failures.clear()

    rp = _make_rp("engineer", writes_ideas_file=False, tmp_path=tmp_path,
                  process=_make_finished_proc(ret=1))
    active = {"engineer": rp}

    finished = roles_mod.check_active_roles(
        active, ideas_file=str(tmp_path / "ideas.md"))

    assert finished == [("engineer", OUTCOME_ERROR)]


def test_role_timeout_outcome(tmp_path, caplog):
    """Still-running process past its timeout → OUTCOME_TIMEOUT + killed."""
    caplog.set_level(logging.WARNING)
    roles_mod._consecutive_soft_failures.clear()

    running = _make_running_proc()

    # Patch _terminate_and_reap so we don't actually try to kill a MagicMock.
    import orze.engine.roles as mod
    original_terminate = mod._terminate_and_reap
    mod._terminate_and_reap = lambda *a, **kw: None
    try:
        rp = _make_rp("engineer", writes_ideas_file=False, tmp_path=tmp_path,
                      process=running,
                      start_time=time.time() - 1200,  # 20 min
                      timeout=600.0)
        active = {"engineer": rp}

        finished = roles_mod.check_active_roles(
            active, ideas_file=str(tmp_path / "ideas.md"))
    finally:
        mod._terminate_and_reap = original_terminate

    assert finished == [("engineer", OUTCOME_TIMEOUT)]
    assert "engineer" not in active


def test_is_success_helper():
    """is_success is True only for OUTCOME_OK; accepts bool for back-compat."""
    assert is_success(OUTCOME_OK) is True
    assert is_success(OUTCOME_TIMEOUT) is False
    assert is_success(OUTCOME_ERROR) is False
    assert is_success(OUTCOME_SOFT_FAILURE) is False
    # Back-compat: callers still handling bool-return must keep working.
    assert is_success(True) is True
    assert is_success(False) is False


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


def test_cross_daemon_consumption_credits_via_mtime(tmp_path, caplog):
    """Multi-daemon case: ideas.md wiped by ANOTHER daemon during this
    role's run. ideas_consumed_during_run stays 0 (only the consuming
    daemon bumps it), and post-wipe file size happens to match the
    pre-role snapshot (both are small). Without the mtime fallback
    this false-positives as "ideas.md was not modified". With the
    mtime fallback, the role is credited."""
    caplog.set_level(logging.WARNING)
    roles_mod._consecutive_soft_failures.clear()

    import os
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# header\n")
    pre_size = ideas.stat().st_size
    pre_mtime = ideas.stat().st_mtime

    # Simulate another daemon wiping the file after our role started.
    # Rewrite to the same byte-length (so the size/count fallback cannot
    # rescue the check) and advance mtime via os.utime. Using explicit
    # utime rather than sleep+write avoids flakiness on filesystems
    # with coarse mtime resolution.
    ideas.write_text("# header\n")
    os.utime(str(ideas), (pre_mtime + 5.0, pre_mtime + 5.0))

    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path,
                  ideas_pre_size=pre_size)
    rp.ideas_md_mtime_pre = pre_mtime
    active = {"research": rp}

    finished = roles_mod.check_active_roles(active, ideas_file=str(ideas))

    assert finished == [("research", OUTCOME_OK)]
    assert "soft failure" not in caplog.text.lower()


def test_role_stall_kills_zero_output_role(tmp_path, caplog):
    """A running role whose log hasn't grown for longer than
    role_stall_minutes is killed and returns OUTCOME_TIMEOUT — before
    the wall-clock timeout would fire. Backfills _stall_since by
    invoking the check twice with the log size held constant."""
    caplog.set_level(logging.WARNING)
    roles_mod._consecutive_soft_failures.clear()

    running = _make_running_proc()
    import orze.engine.roles as mod
    original_terminate = mod._terminate_and_reap
    mod._terminate_and_reap = lambda *a, **kw: None
    try:
        rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path,
                      process=running,
                      start_time=time.time() - 60,  # 1 min elapsed (< timeout)
                      timeout=1200.0)
        # Pretend the log was already observed 400s ago with zero bytes
        # and hasn't grown since — this collapses the two-call handshake
        # the real polling loop would do.
        rp._last_log_size = 0
        rp._stall_since = time.time() - 400
        active = {"research": rp}

        finished = roles_mod.check_active_roles(
            active, ideas_file=str(tmp_path / "ideas.md"),
            role_stall_minutes=5)
    finally:
        mod._terminate_and_reap = original_terminate

    assert finished == [("research", OUTCOME_TIMEOUT)]
    assert "role stall" in caplog.text.lower()
    assert "research" not in active


def test_role_stall_does_not_kill_active_role(tmp_path, caplog):
    """A running role whose log is growing on this poll is NOT killed —
    _is_role_stalled must reset _stall_since when it sees progress."""
    caplog.set_level(logging.WARNING)
    roles_mod._consecutive_soft_failures.clear()

    running = _make_running_proc()
    import orze.engine.roles as mod
    original_terminate = mod._terminate_and_reap
    mod._terminate_and_reap = lambda *a, **kw: None
    try:
        rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path,
                      process=running,
                      start_time=time.time() - 60,
                      timeout=1200.0)
        # Log is growing: current size (100) > _last_log_size (50). Stall
        # timer was armed a long time ago but must be reset by the poll.
        rp.log_path.write_text("x" * 100)
        rp._last_log_size = 50
        rp._stall_since = time.time() - 400
        active = {"research": rp}

        finished = roles_mod.check_active_roles(
            active, ideas_file=str(tmp_path / "ideas.md"),
            role_stall_minutes=5)
    finally:
        mod._terminate_and_reap = original_terminate

    assert finished == []  # still running
    assert "research" in active
    assert rp._stall_since == 0.0  # reset on progress


def test_unchanged_mtime_still_soft_failures(tmp_path, caplog):
    """Guard: if ideas.md mtime hasn't advanced past ideas_md_mtime_pre
    and no other signal fired, we still want the soft-failure outcome.
    Prevents the mtime fallback from silently hiding all real cases."""
    caplog.set_level(logging.WARNING)
    roles_mod._consecutive_soft_failures.clear()

    ideas = tmp_path / "ideas.md"
    ideas.write_text("x" * 100)
    pre_size = ideas.stat().st_size
    pre_mtime = ideas.stat().st_mtime

    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path,
                  ideas_pre_size=pre_size)
    rp.ideas_md_mtime_pre = pre_mtime  # matches current mtime exactly
    active = {"research": rp}

    finished = roles_mod.check_active_roles(active, ideas_file=str(ideas))

    assert finished == [("research", OUTCOME_SOFT_FAILURE)]
