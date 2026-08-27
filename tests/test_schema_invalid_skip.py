"""F2: argparse 'unrecognized arguments' must NOT trigger the LLM fix loop."""
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from orze.engine.failure import (
    _build_executor_fix_cmd,
    _is_argparse_schema_invalid,
    _run_bounded_executor,
    _try_executor_fix,
    _mark_lake_failure,
)


def test_executor_fix_is_bounded_without_implicit_permission_bypass(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    cmd = _build_executor_fix_cmd(
        "claude", "prompt", "sonnet", {}, project_root=project,
        audit_log=project / "policy.jsonl")
    assert "--dangerously-skip-permissions" not in cmd
    assert cmd[cmd.index("--max-turns") + 1] == "20"
    assert "Bash" in cmd[cmd.index("--allowedTools") + 1].split(",")
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    settings = json.loads(cmd[cmd.index("--settings") + 1])
    assert settings["sandbox"]["enabled"] is True
    assert settings["sandbox"]["failIfUnavailable"] is True
    assert settings["sandbox"]["allowUnsandboxedCommands"] is False


@pytest.mark.parametrize("value", [True, False, None])
def test_executor_fix_rejects_permission_bypass_key(tmp_path, value):
    with pytest.raises(ValueError, match="dangerously_skip_permissions"):
        _build_executor_fix_cmd(
            "claude", "prompt", "sonnet",
            {"dangerously_skip_permissions": value},
            project_root=tmp_path, audit_log=tmp_path / "policy.jsonl")


def test_executor_timeout_reaps_escaped_descendant(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    parent_script = tmp_path / "parent.py"
    parent_script.write_text(
        "import os,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)'],start_new_session=True)\n"
        "Path(os.environ['CHILD_PID_FILE']).write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["CHILD_PID_FILE"] = str(child_pid_file)

    with pytest.raises(subprocess.TimeoutExpired):
        _run_bounded_executor(
            [sys.executable, str(parent_script)], timeout=0.5,
            env=env, cwd=tmp_path)

    deadline = time.time() + 2
    while not child_pid_file.exists() and time.time() < deadline:
        time.sleep(0.01)
    assert child_pid_file.exists()
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    from orze.engine.process import process_is_running
    assert not process_is_running(child_pid)


@pytest.mark.parametrize(
    ("returncode", "stdout", "accepted"),
    [
        (0, "changed something\n", False),
        (1, "FIX_APPLIED\n", False),
        (0, "FIX_APPLIED\n", True),
    ],
)
def test_executor_accepts_only_successful_attested_fix(
        tmp_path, monkeypatch, returncode, stdout, accepted):
    results = tmp_path / "results"
    idea_dir = results / "idea-fix"
    idea_dir.mkdir(parents=True)
    (idea_dir / "train_output.log").write_text("failure\n", encoding="utf-8")
    observed = []

    def run(cmd, **kwargs):
        observed.append((cmd, kwargs))
        return subprocess.CompletedProcess(
            cmd, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr("orze.engine.failure._run_bounded_executor", run)
    fixes = {}
    result = _try_executor_fix(
        "idea-fix", "runtime failure", results,
        {
            "max_fix_attempts": 1,
            "agent_tool_policy": {"enabled": True},
            "_project_root": str(tmp_path),
        },
        fixes,
    )

    assert result is accepted
    assert fixes == {"idea-fix": 1}
    assert observed


def test_executor_direct_call_fails_closed_when_policy_disabled(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_dir = results / "idea-fix"
    idea_dir.mkdir(parents=True)
    (idea_dir / "train_output.log").write_text("failure\n", encoding="utf-8")
    monkeypatch.setattr(
        "orze.engine.failure._run_bounded_executor",
        lambda *args, **kwargs: pytest.fail("executor must not launch"),
    )

    assert not _try_executor_fix(
        "idea-fix", "runtime failure", results,
        {
            "max_fix_attempts": 1,
            "agent_tool_policy": {"enabled": False},
        },
        {},
    )


def test_classifier_positive():
    err = "train.py: error: unrecognized arguments: --foo --bar"
    assert _is_argparse_schema_invalid(err, 2)


def test_classifier_wrong_exit_code():
    err = "train.py: error: unrecognized arguments: --foo"
    assert not _is_argparse_schema_invalid(err, 1)
    assert not _is_argparse_schema_invalid(err, 0)


def test_classifier_log_tail_path():
    log = "...some traceback...\nERROR: error: unrecognized arguments: --foo\n"
    assert _is_argparse_schema_invalid("exit code 2", 2, log)


def test_classifier_other_errors_not_skipped():
    assert not _is_argparse_schema_invalid("CUDA out of memory", 1)
    assert not _is_argparse_schema_invalid("ImportError: No module named foo", 1)


def _make_lake(tmp_path: Path) -> Path:
    db = tmp_path / "idea_lake.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(textwrap.dedent("""
        CREATE TABLE ideas (
            idea_id TEXT PRIMARY KEY,
            title TEXT,
            config TEXT,
            raw_markdown TEXT,
            eval_metrics TEXT,
            status TEXT
        );
        INSERT INTO ideas (idea_id, title, config, raw_markdown, status)
            VALUES ('idea-001', 't', '{}', '', 'running');
    """))
    conn.commit()
    conn.close()
    return db


def test_try_executor_fix_skips_and_marks_lake(tmp_path, caplog):
    db = _make_lake(tmp_path)
    results_dir = tmp_path / "results"
    idea_dir = results_dir / "idea-001"
    idea_dir.mkdir(parents=True)
    log = idea_dir / "train_output.log"
    log.write_text("train.py: error: unrecognized arguments: --novel-key\n")

    cfg = {
        "max_fix_attempts": 5,
        "idea_lake_db": str(db),
        "executor_fix": {"claude_bin": "claude", "model": "sonnet"},
    }

    fix_counts = {}
    with caplog.at_level("INFO"):
        result = _try_executor_fix(
            "idea-001", "exit code 2", results_dir, cfg, fix_counts,
            exit_code=2,
        )
    assert result is False
    assert "idea-001" not in fix_counts  # no LLM attempt counted
    assert any("[SKIP-FIX] idea-001 — schema_invalid" in m
               for m in caplog.messages)

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT status, eval_metrics FROM ideas WHERE idea_id='idea-001'"
    ).fetchone()
    conn.close()
    assert row[0] == "failed"
    em = json.loads(row[1])
    assert em["failure_reason"] == "schema_invalid"


def test_mark_schema_invalid_idempotent(tmp_path):
    db = _make_lake(tmp_path)
    cfg = {"idea_lake_db": str(db)}
    _mark_lake_failure("idea-001", cfg, tmp_path, "schema_invalid")
    _mark_lake_failure("idea-001", cfg, tmp_path, "schema_invalid")
    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT status FROM ideas WHERE idea_id='idea-001'").fetchone()
    conn.close()
    assert row[0] == "failed"


def test_mark_schema_invalid_missing_db_silent(tmp_path):
    cfg = {"idea_lake_db": str(tmp_path / "nonexistent.db")}
    _mark_lake_failure("idea-001", cfg, tmp_path, "schema_invalid")
