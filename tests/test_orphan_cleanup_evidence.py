"""Stale-claim cleanup is identity-safe, auditable, and non-destructive."""

import json
import os
import socket
import time

from orze.engine.scheduler import cleanup_orphans, get_unclaimed, run_cleanup
from orze.idea_lake import IdeaLake


def _lake(tmp_path, idea_id):
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert(idea_id, "fixture", "{}", "", status="queued")
    assert lake.record_state_transition(idea_id, "QUEUED", "CLAIMED")
    assert lake.record_state_transition(idea_id, "CLAIMED", "IN_PROGRESS")
    return lake


def _stale_claim(idea_dir, *, host=None):
    idea_dir.mkdir(parents=True)
    claim = idea_dir / "claim.json"
    claim.write_text(json.dumps({
        "attempt_id": "a" * 32,
        "claimed_by": host or socket.gethostname(),
        "claimed_at": "2020-01-01T00:00:00",
        "pid": 999999991,
        "owner_start_ticks": 123,
        "gpu": 4,
    }), encoding="utf-8")
    old = time.time() - 7200
    os.utime(claim, (old, old))
    return claim


def test_cross_host_stale_claim_and_directory_are_never_mutated(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_dir = results / "idea-remote"
    claim = _stale_claim(idea_dir, host="other-host")
    evidence = idea_dir / "checkpoint.bin"
    evidence.write_bytes(b"evidence")
    monkeypatch.setattr(
        "orze.engine.scheduler.process_is_running", lambda *args: False)

    assert cleanup_orphans(results, 1) == 0

    assert claim.exists()
    assert evidence.read_bytes() == b"evidence"
    assert not list(idea_dir.glob("claim.orphan.*.json"))


def test_dead_local_claim_is_archived_and_directory_becomes_claimable(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_id = "idea-local"
    idea_dir = results / idea_id
    _stale_claim(idea_dir)
    evidence = idea_dir / "checkpoint.bin"
    evidence.write_bytes(b"preserve-me")
    lake = _lake(tmp_path, idea_id)
    monkeypatch.setattr(
        "orze.engine.scheduler.process_is_running", lambda *args: False)

    assert cleanup_orphans(results, 1, lake=lake) == 1

    assert idea_dir.exists()
    assert evidence.read_bytes() == b"preserve-me"
    assert not (idea_dir / "claim.json").exists()
    assert len(list(idea_dir.glob("claim.orphan.*.json"))) == 1
    assert len(list(idea_dir.glob("recovery.orphan.*.json"))) == 1
    assert lake.get_fsm_state(idea_id) == "QUEUED"
    assert get_unclaimed({
        idea_id: {"priority": "high", "config": {}},
    }, results, lake=lake) == [idea_id]
    lake.close()


def test_partial_metrics_are_archived_before_requeue(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_id = "idea-partial"
    idea_dir = results / idea_id
    _stale_claim(idea_dir)
    (idea_dir / "metrics.json").write_text(json.dumps({
        "status": "PARTIAL", "last_step": 10,
    }), encoding="utf-8")
    lake = _lake(tmp_path, idea_id)
    monkeypatch.setattr(
        "orze.engine.scheduler.process_is_running", lambda *args: False)

    assert cleanup_orphans(results, 1, lake=lake) == 1

    assert not (idea_dir / "metrics.json").exists()
    archived = list(idea_dir.glob("metrics.orphan.*.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text())["status"] == "PARTIAL"
    assert lake.get_fsm_state(idea_id) == "QUEUED"
    assert get_unclaimed({
        idea_id: {"priority": "high", "config": {}},
    }, results, lake=lake) == [idea_id]
    lake.close()


def test_terminal_metrics_are_preserved_and_reconciled(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_id = "idea-complete"
    idea_dir = results / idea_id
    _stale_claim(idea_dir)
    metrics = {"status": "COMPLETED", "score": 1.0}
    (idea_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8")
    lake = _lake(tmp_path, idea_id)
    monkeypatch.setattr(
        "orze.engine.scheduler.process_is_running", lambda *args: False)

    assert cleanup_orphans(results, 1, lake=lake) == 1

    assert json.loads((idea_dir / "metrics.json").read_text()) == metrics
    assert lake.get_fsm_state(idea_id) == "COMPLETE"
    assert lake.get(idea_id)["status"] == "completed"
    assert not (idea_dir / "claim.json").exists()
    lake.close()


def test_stale_but_live_identity_is_not_reclaimed(tmp_path, monkeypatch):
    results = tmp_path / "results"
    claim = _stale_claim(results / "idea-live")
    monkeypatch.setattr(
        "orze.engine.scheduler.process_is_running", lambda *args: True)

    assert cleanup_orphans(results, 1) == 0
    assert claim.exists()


def test_cleanup_patterns_cannot_delete_lifecycle_or_compute_evidence(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-evidence"
    receipt = idea_dir / "_compute_receipts" / ("a" * 32) / "terminal.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    protected = [
        idea_dir / "claim.orphan.1.json",
        idea_dir / "metrics.orphan.1.json",
        idea_dir / "recovery.orphan.1.json",
        idea_dir / "artifact_preflight.json",
        idea_dir / "interruption.json",
    ]
    for path in protected:
        path.write_text("{}", encoding="utf-8")
    disposable_json = idea_dir / "scratch.json"
    disposable_log = idea_dir / "scratch.log"
    disposable_json.write_text("{}", encoding="utf-8")
    disposable_log.write_text("temporary", encoding="utf-8")

    run_cleanup(results, {
        "cleanup": {"patterns": ["**/*.json", "*.log"]},
        "gc": {},
    })

    assert receipt.exists()
    assert all(path.exists() for path in protected)
    assert not disposable_json.exists()
    assert not disposable_log.exists()
