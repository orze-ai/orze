import json
import os
import socket
import subprocess
import sys

from orze.engine.lifecycle import reconcile_stale_running
from orze.engine.process import (
    capture_process_identity, process_group_members, process_is_running,
    terminate_recorded_process_group,
)
from orze.idea_lake import IdeaLake


def test_dead_local_in_progress_claim_is_audited_and_released(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-0001"
    idea_dir.mkdir(parents=True)
    db_path = tmp_path / "ideas.db"
    lake = IdeaLake(str(db_path))
    lake.insert("idea-0001", "test", "{}", "", status="running")
    assert lake.record_state_transition("idea-0001", "QUEUED", "CLAIMED")
    assert lake.record_state_transition("idea-0001", "CLAIMED", "IN_PROGRESS")
    lake.close()
    (idea_dir / "claim.json").write_text(json.dumps({
        "claimed_by": socket.gethostname(),
        "pid": 999999999,
    }))

    reconcile_stale_running({
        "results_dir": str(results),
        "idea_lake_db": str(db_path),
    })

    lake = IdeaLake(str(db_path))
    assert lake.get("idea-0001")["status"] == "queued"
    assert lake.get_fsm_state("idea-0001") == "QUEUED"
    history = lake.get_fsm_history("idea-0001")
    assert history[-1]["from_state"] == "IN_PROGRESS"
    assert history[-1]["to_state"] == "QUEUED"
    assert history[-1]["reason"] == "startup_recover_orphan_terminated"
    assert not (idea_dir / "claim.json").exists()
    assert list(idea_dir.glob("claim.recovered.*.json"))
    lake.close()


def test_missing_owner_pid_does_not_wedge_recovery(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-0001"
    idea_dir.mkdir(parents=True)
    db_path = tmp_path / "ideas.db"
    lake = IdeaLake(str(db_path))
    lake.insert("idea-0001", "test", "{}", "", status="running")
    assert lake.record_state_transition("idea-0001", "QUEUED", "CLAIMED")
    assert lake.record_state_transition("idea-0001", "CLAIMED", "IN_PROGRESS")
    lake.close()
    (idea_dir / "claim.json").write_text(json.dumps({
        "claimed_by": socket.gethostname(),
    }))

    reconcile_stale_running({
        "results_dir": str(results),
        "idea_lake_db": str(db_path),
    })

    lake = IdeaLake(str(db_path))
    assert lake.get("idea-0001")["status"] == "queued"
    assert lake.get_fsm_state("idea-0001") == "QUEUED"
    lake.close()


def test_reused_owner_pid_is_rejected_by_start_identity(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-0001"
    idea_dir.mkdir(parents=True)
    db_path = tmp_path / "ideas.db"
    lake = IdeaLake(str(db_path))
    lake.insert("idea-0001", "test", "{}", "", status="running")
    assert lake.record_state_transition("idea-0001", "QUEUED", "CLAIMED")
    assert lake.record_state_transition("idea-0001", "CLAIMED", "IN_PROGRESS")
    lake.close()
    identity = capture_process_identity(os.getpid())
    (idea_dir / "claim.json").write_text(json.dumps({
        "claimed_by": socket.gethostname(),
        "pid": os.getpid(),
        "owner_start_ticks": identity["start_ticks"] + 1,
    }))

    reconcile_stale_running({
        "results_dir": str(results),
        "idea_lake_db": str(db_path),
    })

    lake = IdeaLake(str(db_path))
    assert lake.get("idea-0001")["status"] == "queued"
    assert lake.get_fsm_state("idea-0001") == "QUEUED"
    lake.close()


def test_orphan_trainer_group_is_proven_stopped_before_requeue(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-0001"
    idea_dir.mkdir(parents=True)
    db_path = tmp_path / "ideas.db"
    lake = IdeaLake(str(db_path))
    lake.insert("idea-0001", "test", "{}", "", status="running")
    assert lake.record_state_transition("idea-0001", "QUEUED", "CLAIMED")

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)",
         "--idea-id", "idea-0001"],
        preexec_fn=os.setpgrp,
    )
    try:
        identity = capture_process_identity(proc.pid)
        assert lake.record_state_transition(
            "idea-0001", "CLAIMED", "IN_PROGRESS", pid=proc.pid)
        lake.close()
        (idea_dir / "claim.json").write_text(json.dumps({
            "claimed_by": socket.gethostname(),
            "pid": 999999999,
            "trainer_pid": identity["pid"],
            "trainer_pgid": identity["pgid"],
            "trainer_start_ticks": identity["start_ticks"],
        }))

        reconcile_stale_running({
            "results_dir": str(results),
            "idea_lake_db": str(db_path),
        })
        proc.wait(timeout=5)

        assert not process_is_running(proc.pid, identity["start_ticks"])
        recovery = json.loads((idea_dir / "recovery.json").read_text())
        assert recovery["termination_attempted"] is True
        assert recovery["trainer_proven_stopped"] is True
        lake = IdeaLake(str(db_path))
        assert lake.get_fsm_state("idea-0001") == "QUEUED"
        assert lake.get_fsm_history("idea-0001")[-1]["reason"] == (
            "startup_recover_orphan_terminated")
        lake.close()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, 9)
            proc.wait(timeout=5)


def test_live_local_claim_is_not_reconciled(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-0001"
    idea_dir.mkdir(parents=True)
    db_path = tmp_path / "ideas.db"
    lake = IdeaLake(str(db_path))
    lake.insert("idea-0001", "test", "{}", "", status="running")
    lake.close()
    (idea_dir / "claim.json").write_text(json.dumps({
        "claimed_by": socket.gethostname(),
        "pid": os.getpid(),
    }))

    reconcile_stale_running({
        "results_dir": str(results),
        "idea_lake_db": str(db_path),
    })

    lake = IdeaLake(str(db_path))
    assert lake.get("idea-0001")["status"] == "running"
    assert (idea_dir / "claim.json").exists()
    lake.close()


def test_termination_proves_entire_group_empty_when_child_ignores_term():
    parent_code = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c',"
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)',"
        "'--idea-id','idea-group']); time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", parent_code, "--idea-id", "idea-group"],
        preexec_fn=os.setpgrp,
    )
    try:
        identity = capture_process_identity(proc.pid)
        for _ in range(50):
            if len(process_group_members(identity["pgid"])) >= 2:
                break
            __import__("time").sleep(0.02)
        assert len(process_group_members(identity["pgid"])) >= 2
        assert terminate_recorded_process_group(
            identity["pid"], identity["pgid"], identity["start_ticks"],
            "idea-group", timeout=0.2,
        )
        proc.wait(timeout=5)
        assert process_group_members(identity["pgid"]) == []
    finally:
        if process_group_members(proc.pid):
            os.killpg(proc.pid, 9)
        if proc.poll() is None:
            proc.wait(timeout=5)


def test_no_claim_recovery_wal_is_consumed_after_rename_crash(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-0001"
    idea_dir.mkdir(parents=True)
    db_path = tmp_path / "ideas.db"
    lake = IdeaLake(str(db_path))
    lake.insert("idea-0001", "test", "{}", "", status="running")
    assert lake.record_state_transition("idea-0001", "QUEUED", "CLAIMED")
    assert lake.record_state_transition("idea-0001", "CLAIMED", "IN_PROGRESS")
    lake.close()
    (idea_dir / "recovery.json").write_text(json.dumps({
        "trainer_proven_stopped": True,
        "trainer_pgid": None,
        "target_state": "QUEUED",
    }))

    reconcile_stale_running({
        "results_dir": str(results),
        "idea_lake_db": str(db_path),
    })

    lake = IdeaLake(str(db_path))
    assert lake.get("idea-0001")["status"] == "queued"
    assert lake.get_fsm_state("idea-0001") == "QUEUED"
    lake.close()


def test_no_claim_completed_metrics_finalize_instead_of_requeue(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-0001"
    idea_dir.mkdir(parents=True)
    db_path = tmp_path / "ideas.db"
    lake = IdeaLake(str(db_path))
    lake.insert("idea-0001", "test", "{}", "", status="running")
    assert lake.record_state_transition("idea-0001", "QUEUED", "CLAIMED")
    assert lake.record_state_transition("idea-0001", "CLAIMED", "IN_PROGRESS")
    lake.close()
    (idea_dir / "metrics.json").write_text(json.dumps({"status": "COMPLETED"}))

    reconcile_stale_running({
        "results_dir": str(results),
        "idea_lake_db": str(db_path),
    })

    lake = IdeaLake(str(db_path))
    assert lake.get("idea-0001")["status"] == "completed"
    assert lake.get_fsm_state("idea-0001") == "COMPLETE"
    lake.close()


def test_dead_claim_with_partial_metrics_finalizes_failed(tmp_path):
    """A dead evaluation cannot remain IN_PROGRESS after daemon recovery."""
    results = tmp_path / "results"
    idea_dir = results / "idea-0001"
    idea_dir.mkdir(parents=True)
    db_path = tmp_path / "ideas.db"
    lake = IdeaLake(str(db_path))
    lake.insert("idea-0001", "test", "{}", "", status="running")
    assert lake.record_state_transition("idea-0001", "QUEUED", "CLAIMED")
    assert lake.record_state_transition("idea-0001", "CLAIMED", "IN_PROGRESS")
    lake.close()
    (idea_dir / "claim.json").write_text(json.dumps({
        "claimed_by": socket.gethostname(),
        "pid": 999999999,
    }))
    (idea_dir / "metrics.json").write_text(json.dumps({
        "status": "IN_PROGRESS",
        "wer_ami": 8.72,
    }))

    reconcile_stale_running({
        "results_dir": str(results),
        "idea_lake_db": str(db_path),
    })

    lake = IdeaLake(str(db_path))
    assert lake.get("idea-0001")["status"] == "failed"
    assert lake.get_fsm_state("idea-0001") == "FAILED"
    history = lake.get_fsm_history("idea-0001")
    assert history[-1]["reason"] == "startup_recover_failed_output"
    lake.close()
    recovery = json.loads((idea_dir / "recovery.json").read_text())
    assert recovery["metrics_status_after_stop"] == "FAILED"
    assert recovery["target_state"] == "FAILED"


def test_queue_reconcile_preserves_recovery_evidence_directory(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-0001"
    idea_dir.mkdir(parents=True)
    (idea_dir / "recovery.json").write_text(json.dumps({
        "trainer_proven_stopped": True,
        "target_state": "QUEUED",
    }))
    (idea_dir / "attempts.jsonl").write_text('{"pid": 123}\n')
    (idea_dir / "claim.recovered.1.json").write_text("{}")
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert("idea-0001", "test", "{}", "", status="queued")

    assert lake.reconcile_statuses(str(results)) == 0

    assert idea_dir.exists()
    assert (idea_dir / "recovery.json").exists()
    assert (idea_dir / "attempts.jsonl").exists()
    assert lake.get("idea-0001")["status"] == "queued"
    lake.close()
