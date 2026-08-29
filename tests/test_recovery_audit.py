import json
import os

import orze.engine.recovery_audit as recovery_module
from orze.engine.process import capture_process_identity
from orze.engine.recovery_audit import audit_recovery_state
from orze.idea_lake import IdeaLake


def _terminal_lake(tmp_path):
    db_path = tmp_path / "lake.db"
    lake = IdeaLake(str(db_path))
    for idea_id, terminal in (
        ("idea-complete", "COMPLETE"),
        ("idea-failed", "FAILED"),
    ):
        lake.insert(idea_id, "audit", "seed: 1\n", "", status="queued")
        assert lake.record_state_transition(idea_id, "QUEUED", "CLAIMED")
        assert lake.record_state_transition(
            idea_id, "CLAIMED", "IN_PROGRESS"
        )
        if terminal == "COMPLETE":
            assert lake.record_stage_transition(
                idea_id, "training", "IN_PROGRESS", "COMPLETE", "trained"
            )
        assert lake.record_state_transition(
            idea_id, "IN_PROGRESS", terminal, "test_terminal"
        )
    lake.close()
    return db_path


def test_recovery_audit_verifies_consistent_terminal_pipeline(tmp_path):
    db_path = _terminal_lake(tmp_path)

    receipt = audit_recovery_state(db_path, tmp_path / "results")

    assert receipt["status"] == "VERIFIED"
    assert receipt["counts"]["ideas"] == 2
    assert receipt["counts"]["active_states"] == 0
    assert receipt["checks"]["global_state_universe_exact"]["passed"] is True
    assert receipt["counts"]["missing_global_states"] == 0
    assert receipt["counts"]["orphan_global_states"] == 0
    assert receipt["counts"]["global_transition_history_missing"] == 0
    assert receipt["contradiction_idea_ids"] == []
    assert receipt["evidence_gap_idea_ids"] == []
    assert receipt["rank_claim_proven"] is False


def test_recovery_audit_rejects_live_process_in_terminal_state(tmp_path):
    db_path = _terminal_lake(tmp_path)
    results = tmp_path / "results"
    idea_dir = results / "idea-complete"
    idea_dir.mkdir(parents=True)
    identity = capture_process_identity(os.getpid())
    (idea_dir / "claim.json").write_text(json.dumps({
        "claimed_by": "test-host",
        "pid": identity["pid"],
        "owner_start_ticks": identity["start_ticks"],
    }), encoding="utf-8")

    receipt = audit_recovery_state(
        db_path, results, hostname="test-host"
    )

    assert receipt["status"] == "FAILED"
    assert receipt["counts"]["live_process_terminal_states"] == 1
    assert receipt["contradiction_idea_ids"] == ["idea-complete"]


def test_recovery_audit_rejects_active_state_without_claim(tmp_path):
    db_path = tmp_path / "lake.db"
    lake = IdeaLake(str(db_path))
    lake.insert("idea-active", "active", "{}", "", status="queued")
    assert lake.record_state_transition("idea-active", "QUEUED", "CLAIMED")
    lake.close()

    receipt = audit_recovery_state(db_path, tmp_path / "results")

    assert receipt["status"] == "FAILED"
    assert receipt["counts"]["dead_process_active_states"] == 1
    assert receipt["contradiction_idea_ids"] == ["idea-active"]


def test_recovery_audit_marks_missing_stage_schema_unverified(tmp_path):
    db_path = tmp_path / "legacy.db"
    lake = IdeaLake(str(db_path))
    lake.insert("idea-legacy", "legacy", "{}", "", status="failed")
    lake.close()
    connection = __import__("sqlite3").connect(db_path)
    connection.execute("DROP TABLE idea_stage_state")
    connection.execute("DROP TABLE idea_stage_transitions")
    connection.commit()
    connection.close()

    receipt = audit_recovery_state(db_path, tmp_path / "results")

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["required_tables"]["passed"] is False
    assert receipt["evidence_gap_idea_ids"] == ["idea-legacy"]


def test_recovery_audit_does_not_follow_claim_symlink(tmp_path):
    db_path = tmp_path / "lake.db"
    lake = IdeaLake(str(db_path))
    lake.insert("idea-active", "active", "{}", "", status="queued")
    assert lake.record_state_transition("idea-active", "QUEUED", "CLAIMED")
    lake.close()
    results = tmp_path / "results"
    idea_dir = results / "idea-active"
    idea_dir.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (idea_dir / "claim.json").symlink_to(outside)

    receipt = audit_recovery_state(db_path, results)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["evidence_gap_idea_ids"] == ["idea-active"]


def test_recovery_audit_rejects_transition_ledger_divergence(tmp_path):
    db_path = _terminal_lake(tmp_path)
    connection = __import__("sqlite3").connect(db_path)
    connection.execute(
        "UPDATE idea_transitions SET to_state = 'FAILED' "
        "WHERE idea_id = 'idea-complete' AND id = ("
        "SELECT MAX(id) FROM idea_transitions WHERE idea_id = 'idea-complete')"
    )
    connection.commit()
    connection.close()

    receipt = audit_recovery_state(db_path, tmp_path / "results")

    assert receipt["status"] == "FAILED"
    assert receipt["checks"]["transition_ledgers_match_current_state"][
        "passed"
    ] is False
    assert receipt["contradiction_idea_ids"] == ["idea-complete"]


def test_recovery_audit_requires_terminal_global_transition_history(tmp_path):
    db_path = _terminal_lake(tmp_path)
    connection = __import__("sqlite3").connect(db_path)
    connection.execute(
        "DELETE FROM idea_transitions WHERE idea_id = ?",
        ("idea-complete",),
    )
    connection.commit()
    connection.close()

    receipt = audit_recovery_state(db_path, tmp_path / "results")

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["transition_ledgers_match_current_state"][
        "passed"
    ] is False
    assert receipt["checks"]["transition_ledgers_match_current_state"][
        "missing_idea_ids"
    ] == ["idea-complete"]
    assert receipt["counts"]["global_transition_history_missing"] == 1
    assert receipt["contradiction_idea_ids"] == []
    assert receipt["evidence_gap_idea_ids"] == ["idea-complete"]


def test_recovery_audit_rejects_orphan_global_state_row(tmp_path):
    db_path = _terminal_lake(tmp_path)
    connection = __import__("sqlite3").connect(db_path)
    connection.execute(
        "INSERT INTO idea_state (idea_id, current_state) VALUES (?, ?)",
        ("idea-orphan-state", "COMPLETE"),
    )
    connection.commit()
    connection.close()

    receipt = audit_recovery_state(db_path, tmp_path / "results")

    assert receipt["status"] == "FAILED"
    assert receipt["checks"]["global_state_universe_exact"][
        "passed"
    ] is False
    assert receipt["checks"]["global_state_universe_exact"][
        "orphan_idea_ids"
    ] == ["idea-orphan-state"]
    assert receipt["counts"]["orphan_global_states"] == 1
    assert receipt["contradiction_idea_ids"] == ["idea-orphan-state"]


def test_recovery_audit_rejects_missing_global_state_row(tmp_path):
    db_path = _terminal_lake(tmp_path)
    connection = __import__("sqlite3").connect(db_path)
    connection.execute(
        "DELETE FROM idea_state WHERE idea_id = ?", ("idea-complete",)
    )
    connection.commit()
    connection.close()

    receipt = audit_recovery_state(db_path, tmp_path / "results")

    assert receipt["status"] == "FAILED"
    assert receipt["checks"]["global_state_universe_exact"][
        "passed"
    ] is False
    assert receipt["checks"]["global_state_universe_exact"][
        "missing_idea_ids"
    ] == ["idea-complete"]
    assert receipt["counts"]["missing_global_states"] == 1
    assert "idea-complete" in receipt["contradiction_idea_ids"]


def test_recovery_audit_rejects_auditor_source_change(tmp_path, monkeypatch):
    db_path = _terminal_lake(tmp_path)
    original = recovery_module._sha256
    source_calls = 0

    def changed_after_start(path):
        nonlocal source_calls
        if path == recovery_module.Path(recovery_module.__file__):
            source_calls += 1
            if source_calls > 1:
                return "0" * 64
        return original(path)

    monkeypatch.setattr(recovery_module, "_sha256", changed_after_start)

    receipt = audit_recovery_state(db_path, tmp_path / "results")

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["reason"] == "recovery_auditor_source_changed"
