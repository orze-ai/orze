"""Prospective decision receipts must bound the autonomous outcome loop."""

import json
import sqlite3
import threading

import pytest

from orze.core.decision_batches import (
    admit_decision_contract,
    reconcile_decision_batches,
    stage_decision_contract,
    validate_idea_decision_admission,
)
from orze.core.fs import locked_append


def _cfg(tmp_path):
    return {
        "idea_lake_db": str(tmp_path / ".orze" / "idea_lake.db"),
        "report": {
            "primary_metric": "score",
            "sort": "descending",
            "min_datasets": 0,
            "columns": [{"key": "score"}],
        },
        "research_policy": {
            "require_batch_decision_contract": True,
            "max_decision_batch": 2,
            "min_decision_effect": 0.1,
        },
    }


def _contract(on_failure="redirect_family", threshold=0.7):
    return {
        "uncertainty": "Whether this bounded family improves qualified score.",
        "metric": "score",
        "baseline": 0.5,
        "comparator": "gt",
        "threshold": threshold,
        "on_failure": on_failure,
        "max_experiments": 2,
    }


def _ideas():
    return [
        {"idea_id": "idea-alpha", "approach_family": "architecture"},
        {"idea_id": "idea-beta", "approach_family": "data"},
    ]


def _create_lake(tmp_path, rows):
    db_path = tmp_path / ".orze" / "idea_lake.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        PRAGMA locking_mode=NORMAL;
        CREATE TABLE ideas (
            idea_id TEXT PRIMARY KEY,
            status TEXT,
            approach_family TEXT
        );
        CREATE TABLE idea_state (
            idea_id TEXT PRIMARY KEY,
            current_state TEXT
        );
    """)
    conn.executemany(
        "INSERT INTO ideas VALUES (?, ?, ?)",
        [(idea_id, status, family)
         for idea_id, status, family, _ in rows],
    )
    conn.executemany(
        "INSERT INTO idea_state VALUES (?, ?)",
        [(idea_id, state) for idea_id, _, _, state in rows],
    )
    conn.commit()
    conn.close()
    return db_path


def _replace_lifecycle(db_path, rows):
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM ideas")
    conn.execute("DELETE FROM idea_state")
    conn.executemany(
        "INSERT INTO ideas VALUES (?, ?, ?)",
        [(idea_id, status, family)
         for idea_id, status, family, _ in rows],
    )
    conn.executemany(
        "INSERT INTO idea_state VALUES (?, ?)",
        [(idea_id, state) for idea_id, _, _, state in rows],
    )
    conn.commit()
    conn.close()


def _write_score(results_dir, idea_id, score):
    idea_dir = results_dir / idea_id
    idea_dir.mkdir(parents=True, exist_ok=True)
    (idea_dir / "metrics.json").write_text(json.dumps({
        "status": "COMPLETED", "score": score,
    }), encoding="utf-8")


def _write_tainted_score(results_dir, idea_id, score):
    idea_dir = results_dir / idea_id
    idea_dir.mkdir(parents=True, exist_ok=True)
    (idea_dir / "metrics.json").write_text(json.dumps({
        "status": "COMPLETED",
        "score": score,
        "tainted_leakage": True,
    }), encoding="utf-8")


def _stage_and_admit(tmp_path, *, on_failure="redirect_family",
                     threshold=0.7):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cfg = _cfg(tmp_path)
    path, payload = stage_decision_contract(
        results_dir, cfg, 7, _contract(on_failure, threshold), _ideas())
    staged = reconcile_decision_batches(results_dir, cfg)
    assert staged["allow_new_batch"] is False
    assert staged["reason"] == "decision_contract_staged_unresolved"
    admit_decision_contract(path, payload, 2, cfg)
    return results_dir, cfg, path


def test_admitted_batch_blocks_until_every_idea_is_terminal(tmp_path):
    _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "queued", "data", "QUEUED"),
    ])
    results_dir, cfg, _ = _stage_and_admit(tmp_path)
    gate = reconcile_decision_batches(results_dir, cfg)
    assert gate["allow_new_batch"] is False
    assert gate["reason"] == "decision_contract_batch_pending"
    assert gate["pending_receipts"] == 1


def test_queue_append_and_receipt_admission_share_ingestion_lock(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cfg = _cfg(tmp_path)
    path, payload = stage_decision_contract(
        results_dir, cfg, 3, _contract(), _ideas())
    ideas_path = tmp_path / "ideas.md"
    ideas_lock = results_dir / ".ideas_md.lock"

    def finalize():
        admit_decision_contract(path, payload, 2, cfg)

    assert locked_append(
        ideas_path, "two receipt-bound ideas\n", ideas_lock,
        after_append=finalize,
    ) is True
    assert ideas_path.read_text(encoding="utf-8") == (
        "two receipt-bound ideas\n")
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == (
        "admitted")
    assert validate_idea_decision_admission(
        results_dir, cfg, "idea-alpha") is None


def test_qualified_threshold_success_releases_next_batch(tmp_path):
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "queued", "data", "QUEUED"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path, threshold=0.7)
    _replace_lifecycle(db_path, [
        ("idea-alpha", "completed", "architecture", "COMPLETE"),
        ("idea-beta", "failed", "data", "FAILED"),
    ])
    _write_score(results_dir, "idea-alpha", 0.8)
    gate = reconcile_decision_batches(results_dir, cfg)
    assert gate["allow_new_batch"] is True
    assert gate["blocked_families"] == ()
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["status"] == "succeeded"
    assert receipt["qualified_success_count"] == 1
    assert receipt["terminal_count"] == 2
    assert "score" not in receipt
    assert validate_idea_decision_admission(
        results_dir, cfg, "idea-alpha"
    ) == "decision_contract_launch_not_admitted"
    with pytest.raises(ValueError, match="idea_reused"):
        stage_decision_contract(
            results_dir, cfg, 8, _contract(threshold=0.8), _ideas())


def test_failed_batch_redirects_and_mechanically_blocks_families(tmp_path):
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "running", "data", "IN_PROGRESS"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path, threshold=0.9)
    _replace_lifecycle(db_path, [
        ("idea-alpha", "completed", "architecture", "COMPLETE"),
        ("idea-beta", "failed", "data", "FAILED"),
    ])
    _write_score(results_dir, "idea-alpha", 0.8)
    gate = reconcile_decision_batches(results_dir, cfg)
    assert gate["allow_new_batch"] is True
    assert gate["blocked_families"] == ("architecture", "data")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed_redirect"
    assert receipt["qualified_success_count"] == 0
    next_ideas = [
        {"idea_id": "idea-gamma", "approach_family": "architecture"},
        {"idea_id": "idea-delta", "approach_family": "data"},
    ]
    with pytest.raises(ValueError, match="redirect_not_applied"):
        stage_decision_contract(
            results_dir, cfg, 8, _contract(), next_ideas)


def test_tainted_apparent_success_cannot_satisfy_contract(tmp_path):
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "running", "data", "IN_PROGRESS"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path, threshold=0.9)
    _replace_lifecycle(db_path, [
        ("idea-alpha", "completed", "architecture", "COMPLETE"),
        ("idea-beta", "failed", "data", "FAILED"),
    ])
    _write_tainted_score(results_dir, "idea-alpha", 1.0)
    gate = reconcile_decision_batches(results_dir, cfg)
    assert gate["blocked_families"] == ("architecture", "data")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed_redirect"
    assert receipt["qualified_success_count"] == 0


def test_unrecognized_family_cannot_bypass_redirect_by_relabeling(tmp_path):
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "invented_family", "IN_PROGRESS"),
        ("idea-beta", "running", "invented_family", "IN_PROGRESS"),
    ])
    ideas = [dict(idea, approach_family="invented_family") for idea in _ideas()]
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cfg = _cfg(tmp_path)
    path, payload = stage_decision_contract(
        results_dir, cfg, 7, _contract(threshold=0.9), ideas)
    admit_decision_contract(path, payload, 2, cfg)
    _replace_lifecycle(db_path, [
        ("idea-alpha", "failed", "invented_family", "FAILED"),
        ("idea-beta", "failed", "invented_family", "FAILED"),
    ])
    gate = reconcile_decision_batches(results_dir, cfg)
    assert gate["blocked_families"] == ("other",)
    next_ideas = [
        {"idea_id": "idea-gamma", "approach_family": "invented_family"},
        {"idea_id": "idea-delta", "approach_family": "invented_family"},
    ]
    with pytest.raises(ValueError, match="redirect_not_applied"):
        stage_decision_contract(
            results_dir, cfg, 8, _contract(), next_ideas)


def test_failed_stop_action_persistently_stops_new_batches(tmp_path):
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "running", "data", "IN_PROGRESS"),
    ])
    results_dir, cfg, path = _stage_and_admit(
        tmp_path, on_failure="stop_branch", threshold=0.9)
    _replace_lifecycle(db_path, [
        ("idea-alpha", "failed", "architecture", "FAILED"),
        ("idea-beta", "skipped", "data", "SKIPPED"),
    ])
    gate = reconcile_decision_batches(results_dir, cfg)
    assert gate["allow_new_batch"] is False
    assert gate["reason"] == "decision_contract_stop_active"
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == (
        "failed_stopped")
    assert reconcile_decision_batches(results_dir, cfg)["reason"] == (
        "decision_contract_stop_active")


def test_read_only_reconciliation_predicts_without_mutating(tmp_path):
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "running", "data", "IN_PROGRESS"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path, threshold=0.9)
    _replace_lifecycle(db_path, [
        ("idea-alpha", "failed", "architecture", "FAILED"),
        ("idea-beta", "failed", "data", "FAILED"),
    ])
    before = path.read_bytes()
    directory_mtime = path.parent.stat().st_mtime_ns
    gate = reconcile_decision_batches(results_dir, cfg, apply=False)
    assert gate["allow_new_batch"] is True
    assert gate["blocked_families"] == ("architecture", "data")
    assert path.read_bytes() == before
    assert path.parent.stat().st_mtime_ns == directory_mtime
    assert json.loads(before)["status"] == "admitted"


def test_missing_or_conflicting_authority_fails_closed(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cfg = _cfg(tmp_path)
    path, payload = stage_decision_contract(
        results_dir, cfg, 1, _contract(), _ideas())
    admit_decision_contract(path, payload, 2, cfg)
    assert reconcile_decision_batches(results_dir, cfg)["reason"] == (
        "authoritative_lifecycle_database_unavailable")

    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "completed", "architecture", "FAILED"),
        ("idea-beta", "failed", "data", "FAILED"),
    ])
    assert db_path.is_file()
    assert reconcile_decision_batches(results_dir, cfg)["reason"] == (
        "authoritative_lifecycle_state_conflict")


def test_corrupt_or_redirected_receipt_fails_closed(tmp_path):
    _create_lake(tmp_path, [
        ("idea-alpha", "queued", "architecture", "QUEUED"),
        ("idea-beta", "queued", "data", "QUEUED"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["identity_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert reconcile_decision_batches(results_dir, cfg)["reason"] == (
        "decision_receipt_identity_invalid")


def test_redirected_receipt_file_fails_closed(tmp_path):
    _create_lake(tmp_path, [
        ("idea-alpha", "queued", "architecture", "QUEUED"),
        ("idea-beta", "queued", "data", "QUEUED"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path)
    outside = tmp_path / "outside.json"
    path.rename(outside)
    path.symlink_to(outside)
    assert reconcile_decision_batches(results_dir, cfg)["reason"] == (
        "decision_receipt_file_invalid")


def test_receipt_timestamps_must_be_real_ordered_utc(tmp_path):
    _create_lake(tmp_path, [
        ("idea-alpha", "queued", "architecture", "QUEUED"),
        ("idea-beta", "queued", "data", "QUEUED"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["admitted_at"] = "2000-01-01T00:00:00+00:00"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert reconcile_decision_batches(results_dir, cfg)["reason"] == (
        "decision_receipt_admission_invalid")


def test_admission_refuses_missing_staged_receipt(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cfg = _cfg(tmp_path)
    path, payload = stage_decision_contract(
        results_dir, cfg, 1, _contract(), _ideas())
    path.unlink()
    with pytest.raises(ValueError, match="stage_changed"):
        admit_decision_contract(path, payload, 2, cfg)


def test_policy_is_optional_and_admission_count_is_exact(tmp_path):
    results_dir = tmp_path / "results"
    assert reconcile_decision_batches(results_dir, {}) == {
        "allow_new_batch": True,
        "reason": "decision_contract_not_required",
        "blocked_families": (),
        "pending_receipts": 0,
        "resolved_receipts": 0,
    }
    results_dir.mkdir()
    cfg = _cfg(tmp_path)
    path, payload = stage_decision_contract(
        results_dir, cfg, 1, _contract(), _ideas())
    assert validate_idea_decision_admission(
        results_dir, cfg, "idea-alpha"
    ) == "decision_contract_launch_not_admitted"
    with pytest.raises(ValueError, match="admission_count_mismatch"):
        admit_decision_contract(path, payload, 1, cfg)
    admit_decision_contract(path, payload, 2, cfg)
    assert validate_idea_decision_admission(
        results_dir, cfg, "idea-alpha") is None
    assert validate_idea_decision_admission(
        results_dir, cfg, "idea-unbound"
    ) == "decision_contract_launch_admission_missing"
    with pytest.raises(ValueError, match="duplicate_receipt"):
        stage_decision_contract(
            results_dir, cfg, 1, _contract(), _ideas())


def test_concurrent_producers_can_stage_only_one_batch(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cfg = _cfg(tmp_path)
    barrier = threading.Barrier(2)
    outcomes = []

    def stage(threshold):
        barrier.wait()
        try:
            stage_decision_contract(
                results_dir, cfg, int(threshold * 10),
                _contract(threshold=threshold), _ideas())
        except (OSError, ValueError) as exc:
            outcomes.append(("rejected", str(exc)))
        else:
            outcomes.append(("staged", ""))

    threads = [
        threading.Thread(target=stage, args=(threshold,))
        for threshold in (0.7, 0.8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert [outcome[0] for outcome in outcomes].count("staged") == 1
    assert [outcome[0] for outcome in outcomes].count("rejected") == 1
