"""F2: argparse 'unrecognized arguments' must NOT trigger the LLM fix loop."""
import json
import sqlite3
import textwrap
from pathlib import Path

from orze.engine.failure import (
    _is_argparse_schema_invalid,
    _try_executor_fix,
    _mark_lake_failure,
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
