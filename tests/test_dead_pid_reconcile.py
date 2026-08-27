"""F7: dead-PID reconciler for 'running' idea rows."""
import json
import sqlite3
from pathlib import Path

from orze.engine.lifecycle import (
    reconcile_running_dead_pids,
    _running_idea_pids,
)
from orze.idea_lake import IdeaLake


def _make_lake_with_running(tmp_path: Path, idea_ids: list) -> Path:
    db = tmp_path / "idea_lake.db"
    lake = IdeaLake(str(db))
    for iid in idea_ids:
        lake.insert(iid, "t", "{}", "", status="running")
    lake.close()
    return db


def test_imported_running_status_initializes_matching_fsm(tmp_path):
    db = _make_lake_with_running(tmp_path, ["idea-running"])
    lake = IdeaLake(str(db))
    assert lake.get("idea-running")["status"] == "running"
    assert lake.get_fsm_state("idea-running") == "IN_PROGRESS"
    lake.close()


def test_running_without_claim_is_requeued(tmp_path, monkeypatch):
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
    assert n == 0

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT status, eval_metrics FROM ideas "
        "WHERE idea_id='idea-orphan'").fetchone()
    conn.close()
    assert row[0] == "queued"
    assert row[1] is None
    lake = IdeaLake(str(db))
    assert lake.get_fsm_state("idea-orphan") == "QUEUED"
    lake.close()


def test_alive_running_not_touched(tmp_path, monkeypatch):
    db = _make_lake_with_running(tmp_path, ["idea-alive"])
    results_dir = tmp_path / "results"
    idea_dir = results_dir / "idea-alive"
    idea_dir.mkdir(parents=True)
    (idea_dir / "claim.json").write_text(json.dumps({
        "claimed_by": __import__("socket").gethostname(),
        "pid": __import__("os").getpid(),
    }))
    cfg = {
        "results_dir": str(results_dir),
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


def test_failed_orphan_without_claim_stays_terminal_and_is_idempotent(
        tmp_path, monkeypatch):
    db = _make_lake_with_running(tmp_path, ["idea-failed-orphan"])
    em = json.dumps({"failure_reason": "orphaned_pid", "liveness_misses": 2})
    lake = IdeaLake(str(db))
    lake.conn.execute(
        "UPDATE ideas SET eval_metrics=? WHERE idea_id=?",
        (em, "idea-failed-orphan"))
    lake.conn.commit()
    assert lake.set_status("idea-failed-orphan", "failed")
    lake.close()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cfg = {"results_dir": str(results_dir), "idea_lake_db": str(db)}
    monkeypatch.setattr(
        "orze.engine.lifecycle._running_idea_pids", lambda: set())

    assert reconcile_running_dead_pids(cfg) == 0
    assert reconcile_running_dead_pids(cfg) == 0

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT status, eval_metrics FROM ideas WHERE idea_id=?",
        ("idea-failed-orphan",),
    ).fetchone()
    conn.close()
    assert row == ("failed", em)


def test_failed_orphan_is_not_silently_upgraded_by_late_metrics(
        tmp_path, monkeypatch):
    db = _make_lake_with_running(tmp_path, ["idea-late-complete"])
    lake = IdeaLake(str(db))
    lake.conn.execute(
        "UPDATE ideas SET eval_metrics=? WHERE idea_id=?",
        (json.dumps({"failure_reason": "orphaned_pid"}),
         "idea-late-complete"),
    )
    lake.conn.commit()
    assert lake.set_status("idea-late-complete", "failed")
    lake.close()
    results_dir = tmp_path / "results"
    idea_dir = results_dir / "idea-late-complete"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED"}))
    cfg = {"results_dir": str(results_dir), "idea_lake_db": str(db)}
    monkeypatch.setattr(
        "orze.engine.lifecycle._running_idea_pids", lambda: set())

    assert reconcile_running_dead_pids(cfg) == 0

    conn = sqlite3.connect(str(db))
    status = conn.execute(
        "SELECT status FROM ideas WHERE idea_id=?",
        ("idea-late-complete",),
    ).fetchone()[0]
    conn.close()
    assert status == "failed"


def test_submission_file_alone_cannot_complete_dead_training(
        tmp_path, monkeypatch):
    import os
    import socket
    import time

    db = _make_lake_with_running(tmp_path, ["idea-submission-only"])
    results_dir = tmp_path / "results"
    idea_dir = results_dir / "idea-submission-only"
    idea_dir.mkdir(parents=True)
    claim = idea_dir / "claim.json"
    claim.write_text(json.dumps({
        "claimed_by": socket.gethostname(), "pid": 999999999,
    }), encoding="utf-8")
    (idea_dir / "submission.csv").write_text("id,prediction\n1,x\n")
    old = time.time() - 600
    os.utime(claim, (old, old))
    cfg = {"results_dir": str(results_dir), "idea_lake_db": str(db)}
    monkeypatch.setattr(
        "orze.engine.lifecycle._running_idea_pids", lambda: set())

    assert reconcile_running_dead_pids(cfg) == 0
    lake = IdeaLake(str(db))
    assert lake.get("idea-submission-only")["eval_metrics"][
        "liveness_misses"
    ] == 1
    assert lake.get_fsm_state("idea-submission-only") == "IN_PROGRESS"
    lake.close()
    assert reconcile_running_dead_pids(cfg) == 1
    lake = IdeaLake(str(db))
    assert lake.get("idea-submission-only")["status"] == "failed"
    assert lake.get_fsm_state("idea-submission-only") == "FAILED"
    lake.close()


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
