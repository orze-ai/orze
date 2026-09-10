"""Public read-to-publication races cannot let legacy recovery replace B."""
import json
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace

import pytest

from orze.engine import accounting, failure, lifecycle, scheduler, training_attempts
from orze.idea_lake import IdeaLake


@pytest.mark.parametrize("entry", ["startup", "dead_pid"])
def test_native_admission_after_legacy_read_is_rechecked_before_any_publication(tmp_path, monkeypatch, entry):
    db = tmp_path / "ideas.db"
    lake = IdeaLake(db)
    idea = "idea-migration"
    lake.insert(idea, "Legacy", "seed: 1", "", status="running")
    folder = tmp_path / "results" / idea
    folder.mkdir(parents=True)
    claim = folder / "claim.json"
    claim.write_text(json.dumps({"claimed_by": socket.gethostname(), "pid": 999999999}))
    metrics = folder / "metrics.json"
    metrics.write_text('{"status":"COMPLETED"}')
    os.utime(claim, (1, 1))
    os.utime(metrics, (1, 1))
    cfg = {"results_dir": str(folder.parent), "idea_lake_db": str(db)}
    monkeypatch.setattr(lifecycle, "_running_idea_pids", lambda: set())
    monkeypatch.setattr(lifecycle, "process_is_running", lambda *args: False)
    original_read = Path.read_text
    captured = {}

    def snapshot():
        files = {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
        tables = {table: [tuple(row) for row in lake.conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
                  for table in ("ideas", "idea_state", "idea_transitions", "idea_stage_state",
                                "idea_stage_transitions", "execution_attempts")}
        return files, tables

    def read_then_admit(path, *args, **kwargs):
        data = original_read(path, *args, **kwargs)
        if path == metrics and not captured:
            captured["entered"] = True
            failure._reset_idea_for_retry(folder, release_claim=True, lake=lake)
            assert lake.record_state_transition(idea, "IN_PROGRESS", "QUEUED")
            assert scheduler.claim(idea, folder.parent, 0, lake=lake)
            owner = json.loads(original_read(claim))
            process = SimpleNamespace(pid=987654321)
            tp = SimpleNamespace(idea_id=idea, attempt_id=owner["attempt_id"], gpu=0,
                                 process=process, start_time=time.time(), execution_identity="a" * 64)
            tp.attempt_ref = training_attempts.begin(lake, tp, folder)
            accounting.record_compute_start(tp, folder)
            training_attempts.started(lake, tp, folder,
                                      {"pid": process.pid, "pgid": process.pid, "start_ticks": 1234})
            metrics.write_text('{"status":"IN_PROGRESS","owner":"B"}')
            captured["snapshot"] = snapshot()
        return data

    monkeypatch.setattr(Path, "read_text", read_then_admit)
    try:
        if entry == "startup":
            lifecycle.reconcile_stale_running(cfg)
        else:
            assert lifecycle.reconcile_running_dead_pids(cfg) == 0
        assert captured.get("entered"), "must pass the initial legacy gate before B is admitted"
        assert snapshot() == captured["snapshot"]
    finally:
        lake.close()
