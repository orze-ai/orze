"""Tests for determinism hardening wins #1 and #2 (docs/determinism_hardening.md).

#1 — _is_launcher_paused uses a single canonical path, resolves cwd-independent,
     emits [PAUSE_CHECK] every call.
#2 — _ideas_modified_credits returns structured credit signals; the call site
     in check_active_roles logs [SOFT_FAILURE_REASON] / [SOFT_FAILURE_CHECK]
     with all three credits so c1196-style silent skips are observable.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orze.engine import roles as roles_mod
from orze.engine.launcher import (
    _is_launcher_paused,
    _resolve_pause_flag_path,
)
from orze.engine.process import RoleProcess
from orze.engine.roles import (
    OUTCOME_OK,
    OUTCOME_SOFT_FAILURE,
    _fmt_credits,
    _ideas_modified_credits,
)


# ---------------------------------------------------------------------------
# #1 — pause-flag determinism
# ---------------------------------------------------------------------------

def test_pause_flag_default_path_under_results_dir(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    path = _resolve_pause_flag_path({}, results)
    assert path == (results / "_launcher_paused.flag").resolve()


def test_pause_flag_config_override_absolute(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    explicit = tmp_path / "pause.flag"
    cfg = {"launcher": {"paused_flag_path": str(explicit)}}
    path = _resolve_pause_flag_path(cfg, results)
    assert path == explicit.resolve()


def test_pause_flag_config_override_relative_resolved_against_results(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    cfg = {"launcher": {"paused_flag_path": "pause.flag"}}
    path = _resolve_pause_flag_path(cfg, results)
    assert path == (results / "pause.flag").resolve()


def test_pause_check_cwd_independent(tmp_path, monkeypatch):
    """Regression for c1135: cwd shifts must not flip pause detection."""
    results = tmp_path / "results"
    results.mkdir()
    cwd_a = tmp_path / "a"
    cwd_b = tmp_path / "b"
    cwd_a.mkdir()
    cwd_b.mkdir()
    # Drop a flag in cwd_a only — under the old multi-path check this would
    # have toggled True/False depending on cwd. Now it must be False
    # regardless because the canonical path is results_dir/<flag>.
    (cwd_a / "_launcher_paused.flag").touch()

    monkeypatch.chdir(cwd_a)
    assert _is_launcher_paused({}, results) is False
    monkeypatch.chdir(cwd_b)
    assert _is_launcher_paused({}, results) is False


def test_pause_check_canonical_flag_detected(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "_launcher_paused.flag").touch()
    assert _is_launcher_paused({}, results) is True


def test_pause_check_config_paused_short_circuits(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    assert _is_launcher_paused({"launcher": {"paused": True}}, results) is True


def test_pause_check_emits_structured_log(tmp_path, caplog):
    results = tmp_path / "results"
    results.mkdir()
    caplog.set_level(logging.INFO, logger="orze")
    _is_launcher_paused({}, results)
    assert "[PAUSE_CHECK]" in caplog.text
    assert "present=False" in caplog.text
    assert "config_paused=False" in caplog.text


# ---------------------------------------------------------------------------
# #2 — soft-failure credit instrumentation
# ---------------------------------------------------------------------------

def _make_finished_proc(ret: int = 0):
    proc = MagicMock()
    proc.poll.return_value = ret
    return proc


def _make_rp(role_name: str, writes_ideas_file: bool, tmp_path: Path,
             ideas_pre_size: int = 100, ideas_pre_count: int = 0) -> RoleProcess:
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
        ideas_pre_count=ideas_pre_count,
        writes_ideas_file=writes_ideas_file,
    )
    rp.lock_dir.mkdir(exist_ok=True)
    return rp


def test_credits_consumed_signal(tmp_path):
    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path)
    rp.ideas_consumed_during_run = 3
    ideas = tmp_path / "ideas.md"
    ideas.write_text("x" * 100)
    c = _ideas_modified_credits(str(ideas), rp)
    assert c["modified"] is True
    assert c["credit"] == "consumed"
    assert c["consumed"] == 3


def test_credits_mtime_signal(tmp_path):
    ideas = tmp_path / "ideas.md"
    ideas.write_text("x" * 100)
    pre_mtime = ideas.stat().st_mtime
    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path,
                  ideas_pre_size=ideas.stat().st_size)
    rp.ideas_md_mtime_pre = pre_mtime
    # Bump mtime without changing content
    os.utime(str(ideas), (pre_mtime + 10.0, pre_mtime + 10.0))
    c = _ideas_modified_credits(str(ideas), rp)
    assert c["modified"] is True
    assert c["credit"] == "mtime"
    assert c["mtime_delta"] > 0


def test_credits_size_signal(tmp_path):
    ideas = tmp_path / "ideas.md"
    ideas.write_text("x" * 500)
    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path,
                  ideas_pre_size=100)
    c = _ideas_modified_credits(str(ideas), rp)
    assert c["modified"] is True
    assert c["credit"] == "size"
    assert c["size_pre"] == 100
    assert c["size_post"] == 500


def test_credits_no_pre_snapshot_signal(tmp_path):
    ideas = tmp_path / "ideas.md"
    ideas.write_text("x" * 50)
    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path,
                  ideas_pre_size=0)
    c = _ideas_modified_credits(str(ideas), rp)
    assert c["modified"] is True
    assert c["credit"] == "no_pre_snapshot"


def test_credits_missing_file_signal(tmp_path):
    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path)
    c = _ideas_modified_credits(str(tmp_path / "absent.md"), rp)
    assert c["modified"] is False
    assert c["credit"] == "missing"


def test_credits_none_signal_when_truly_unchanged(tmp_path):
    ideas = tmp_path / "ideas.md"
    ideas.write_text("x" * 100)
    pre_mtime = ideas.stat().st_mtime
    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path,
                  ideas_pre_size=100, ideas_pre_count=0)
    rp.ideas_md_mtime_pre = pre_mtime
    c = _ideas_modified_credits(str(ideas), rp)
    assert c["modified"] is False
    assert c["credit"] == "none"


def test_fmt_credits_renders_all_fields(tmp_path):
    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path)
    rp.ideas_consumed_during_run = 2
    ideas = tmp_path / "ideas.md"
    ideas.write_text("hello")
    c = _ideas_modified_credits(str(ideas), rp)
    s = _fmt_credits(c)
    assert "credit=consumed" in s
    assert "consumed=2" in s
    assert "size=" in s
    assert "count=" in s


def test_soft_failure_log_includes_credits(tmp_path, caplog):
    """The call site must surface the credit dict — c1196 was unfixable
    because only the verdict was logged."""
    caplog.set_level(logging.WARNING, logger="orze")
    roles_mod._consecutive_soft_failures.clear()
    ideas = tmp_path / "ideas.md"
    ideas.write_text("x" * 100)
    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path,
                  ideas_pre_size=100, ideas_pre_count=0)
    rp.ideas_md_mtime_pre = ideas.stat().st_mtime
    active = {"research": rp}
    finished = roles_mod.check_active_roles(active, ideas_file=str(ideas))
    assert finished == [("research", OUTCOME_SOFT_FAILURE)]
    assert "[SOFT_FAILURE_REASON]" in caplog.text
    assert "credit=none" in caplog.text
    assert "consumed=0" in caplog.text


def test_ok_log_includes_credits(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="orze")
    roles_mod._consecutive_soft_failures.clear()
    ideas = tmp_path / "ideas.md"
    ideas.write_text("x" * 500)  # bigger than pre_size → size credit
    rp = _make_rp("research", writes_ideas_file=True, tmp_path=tmp_path,
                  ideas_pre_size=100)
    active = {"research": rp}
    finished = roles_mod.check_active_roles(active, ideas_file=str(ideas))
    assert finished == [("research", OUTCOME_OK)]
    assert "[SOFT_FAILURE_CHECK]" in caplog.text
    assert "credit=size" in caplog.text
