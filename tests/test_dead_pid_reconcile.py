"""F7: dead-PID reconciler for 'running' idea rows."""
import json
import sqlite3
import textwrap
from pathlib import Path

from orze.engine.lifecycle import (
    reconcile_running_dead_pids,
    _running_idea_pids,
)


def _make_lake_with_running(tmp_path: Path, idea_ids: list) -> Path:
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
    """))
    for iid in idea_ids:
        conn.execute(
            "INSERT INTO ideas (idea_id, title, config, raw_markdown, status)"
            " VALUES (?, 't', '{}', '', 'running')",
            (iid,))
    conn.commit()
    conn.close()
    return db


def test_orphaned_running_marked_failed(tmp_path, monkeypatch):
    db = _make_lake_with_running(tmp_path, ["idea-orphan"])
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cfg = {
        "results_dir": str(results_dir),
        "idea_lake_db": str(db),
    }
    # No process found.
    monkeypatch.setattr(
        "orze.engine.lifecycle._running_idea_pids", lambda: set())

    n = reconcile_running_dead_pids(cfg)
    assert n == 1

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT status, eval_metrics FROM ideas "
        "WHERE idea_id='idea-orphan'").fetchone()
    conn.close()
    assert row[0] == "failed"
    em = json.loads(row[1])
    assert em["failure_reason"] == "orphaned_pid"


def test_alive_running_not_touched(tmp_path, monkeypatch):
    db = _make_lake_with_running(tmp_path, ["idea-alive"])
    cfg = {
        "results_dir": str(tmp_path / "results"),
        "idea_lake_db": str(db),
    }
    monkeypatch.setattr(
        "orze.engine.lifecycle._running_idea_pids",
        lambda: {"idea-alive"})
    n = reconcile_running_dead_pids(cfg)
    assert n == 0

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT status FROM ideas WHERE idea_id='idea-alive'").fetchone()
    conn.close()
    assert row[0] == "running"


def test_other_host_claim_skipped(tmp_path, monkeypatch):
    db = _make_lake_with_running(tmp_path, ["idea-other-host"])
    results_dir = tmp_path / "results"
    idea_dir = results_dir / "idea-other-host"
    idea_dir.mkdir(parents=True)
    (idea_dir / "claim.json").write_text(json.dumps({
        "claimed_by": "some-other-host", "claimed_at": "x", "pid": 1, "gpu": 0,
    }))
    cfg = {
        "results_dir": str(results_dir),
        "idea_lake_db": str(db),
    }
    monkeypatch.setattr(
        "orze.engine.lifecycle._running_idea_pids", lambda: set())

    n = reconcile_running_dead_pids(cfg)
    assert n == 0

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT status FROM ideas "
        "WHERE idea_id='idea-other-host'").fetchone()
    conn.close()
    assert row[0] == "running"


def test_missing_db_no_crash(tmp_path):
    cfg = {
        "results_dir": str(tmp_path),
        "idea_lake_db": str(tmp_path / "no.db"),
    }
    assert reconcile_running_dead_pids(cfg) == 0


def test_running_idea_pids_finds_self(tmp_path):
    """Sanity: _running_idea_pids should find a real process whose cmdline
    contains '--idea-id <id>'. We launch a short sleep with such cmdline.
    """
    import subprocess
    import sys
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; sys.argv = ['x', '--idea-id', 'idea-self-test']; "
         "time.sleep(2)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # The subprocess' actual cmdline is python -c '...', not our argv
        # patch (sys.argv inside the child won't change /proc cmdline).
        # So instead launch with --idea-id directly on the python cmdline:
        proc.terminate()
        proc.wait(timeout=5)

        proc2 = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(2)",
             "--idea-id", "idea-self-test"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            import time as _t
            _t.sleep(0.2)
            ids = _running_idea_pids()
            assert "idea-self-test" in ids
        finally:
            proc2.terminate()
            try:
                proc2.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc2.kill()
    finally:
        if proc.poll() is None:
            proc.kill()
