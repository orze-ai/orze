"""Queue reporting and recovery must agree with the audited lifecycle."""

import json
import socket

from orze.engine.scheduler import _count_statuses
from orze.idea_lake import IdeaLake


def test_imported_statuses_initialize_matching_audited_state(tmp_path):
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    expected = {
        "queued": "QUEUED",
        "running": "IN_PROGRESS",
        "completed": "COMPLETE",
        "partial": "FAILED",
        "failed": "FAILED",
        "skipped": "SKIPPED",
        "archived": "ARCHIVED",
    }
    for status, state in expected.items():
        idea_id = f"idea-{status}"
        lake.insert(idea_id, status, "{}", "", status=status)
        assert lake.get_fsm_state(idea_id) == state
    lake.close()


def test_status_counts_use_audited_fsm_not_filesystem_guesses(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    for status in (
        "queued", "running", "completed", "failed", "skipped", "archived"
    ):
        lake.insert(f"idea-{status}", status, "{}", "", status=status)

    # Deliberately contradict every filesystem heuristic. The state ledger,
    # not whichever file happened to flush first, remains authoritative.
    queued_dir = results / "idea-queued"
    queued_dir.mkdir()
    (queued_dir / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED"}), encoding="utf-8"
    )
    running_dir = results / "idea-running"
    running_dir.mkdir()
    (running_dir / "metrics.json").write_text(
        json.dumps({"status": "FAILED"}), encoding="utf-8"
    )

    counts = _count_statuses(
        {"idea-queued": {}, "idea-running": {}}, results, lake=lake
    )
    assert counts == {
        "QUEUED": 1,
        "IN_PROGRESS": 1,
        "COMPLETED": 1,
        "FAILED": 1,
        "SKIPPED": 1,
        "ARCHIVED": 1,
    }
    lake.close()


def test_legacy_status_tamper_cannot_requeue_completed_fsm(tmp_path):
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert("idea-complete", "complete", "{}", "", status="completed")
    lake.conn.execute(
        "UPDATE ideas SET status = 'queued' WHERE idea_id = 'idea-complete'"
    )
    lake.conn.commit()

    assert lake.get_queue() == []
    assert _count_statuses({}, tmp_path / "results", lake=lake) == {
        "COMPLETED": 1
    }
    lake.close()


def test_filesystem_reconcile_records_direct_terminal_catchup(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-complete"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED", "score": 0.9}), encoding="utf-8"
    )
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert("idea-complete", "complete", "{}", "", status="queued")

    assert lake.reconcile_statuses(str(results)) == 1
    assert lake.get("idea-complete")["status"] == "completed"
    assert lake.get_fsm_state("idea-complete") == "COMPLETE"
    history = lake.get_fsm_history("idea-complete")
    assert [(row["from_state"], row["to_state"], row["reason"])
            for row in history] == [
        ("QUEUED", "COMPLETE", "reconcile_filesystem_completed")
    ]
    lake.close()


def test_terminal_reconciliation_rejects_conflicting_evidence(tmp_path):
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert("idea-failed", "failed", "{}", "", status="failed")

    assert not lake.reconcile_terminal_state(
        "idea-failed", "COMPLETE", "reconcile_conflicting_metrics"
    )
    assert lake.get_fsm_state("idea-failed") == "FAILED"
    assert lake.get("idea-failed")["status"] == "failed"
    assert lake.get_fsm_history("idea-failed") == []
    lake.close()


def test_orphan_directory_and_claim_evidence_are_preserved(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-orphan"
    idea_dir.mkdir(parents=True)
    claim = {"claimed_by": socket.gethostname(), "pid": 999999999}
    (idea_dir / "claim.json").write_text(json.dumps(claim), encoding="utf-8")
    (idea_dir / "train_output.log").write_text("last trainer output\n")
    (idea_dir / "checkpoint.pt").write_bytes(b"checkpoint evidence")
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert("idea-orphan", "orphan", "{}", "", status="queued")

    assert lake.reconcile_statuses(str(results)) == 0
    assert json.loads((idea_dir / "claim.json").read_text()) == claim
    assert (idea_dir / "train_output.log").read_text() == "last trainer output\n"
    assert (idea_dir / "checkpoint.pt").read_bytes() == b"checkpoint evidence"
    assert lake.get_fsm_state("idea-orphan") == "QUEUED"
    lake.close()


def test_corrupt_claim_fails_closed_without_deleting_evidence(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-corrupt"
    idea_dir.mkdir(parents=True)
    (idea_dir / "claim.json").write_text("{not-json", encoding="utf-8")
    (idea_dir / "train_output.log").write_text("evidence", encoding="utf-8")
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert("idea-corrupt", "corrupt", "{}", "", status="queued")

    assert lake.reconcile_statuses(str(results)) == 0
    assert (idea_dir / "claim.json").read_text() == "{not-json"
    assert (idea_dir / "train_output.log").read_text() == "evidence"
    assert lake.get_fsm_state("idea-corrupt") == "QUEUED"
    lake.close()


def test_partial_complete_migration_is_audited_and_idempotent(tmp_path):
    db = tmp_path / "ideas.db"
    lake = IdeaLake(str(db))
    lake.insert("idea-partial", "partial", "{}", "", status="partial")
    # Recreate the historical bad classification and make the repair pending.
    lake.conn.execute(
        "UPDATE idea_state SET current_state = 'COMPLETE' "
        "WHERE idea_id = 'idea-partial'"
    )
    lake.conn.execute(
        "DELETE FROM schema_migrations WHERE name = 'partial_is_failed_v1'"
    )
    lake.conn.commit()
    lake.close()

    lake = IdeaLake(str(db))
    assert lake.get_fsm_state("idea-partial") == "FAILED"
    history = lake.get_fsm_history("idea-partial")
    assert [(row["from_state"], row["to_state"], row["reason"])
            for row in history] == [
        ("COMPLETE", "FAILED", "migration_partial_not_complete")
    ]
    lake.close()

    lake = IdeaLake(str(db))
    assert len(lake.get_fsm_history("idea-partial")) == 1
    lake.close()
