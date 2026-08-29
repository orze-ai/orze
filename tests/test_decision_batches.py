"""Prospective decision receipts must bound the autonomous outcome loop."""

import json
import sqlite3
import threading
import time

import pytest

import orze.core.decision_batches as decision_module
import orze.reporting.evidence as evidence_module
from orze.core.decision_batches import (
    admit_decision_contract,
    audit_campaign_decision_receipts,
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


def _contract(on_failure="redirect_family", threshold=0.7,
              required_successes=None):
    contract = {
        "uncertainty": "Whether this bounded family improves qualified score.",
        "metric": "score",
        "baseline": 0.5,
        "comparator": "gt",
        "threshold": threshold,
        "on_failure": on_failure,
        "max_experiments": 2,
    }
    if required_successes is not None:
        contract["required_successes"] = required_successes
    return contract


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


def test_campaign_decision_audit_verifies_exact_resolved_receipt(tmp_path):
    start = time.time() - 1
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "running", "data", "IN_PROGRESS"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path, threshold=0.7)
    identity = json.loads(path.read_text())["identity_sha256"]
    _replace_lifecycle(db_path, [
        ("idea-alpha", "completed", "architecture", "COMPLETE"),
        ("idea-beta", "failed", "data", "FAILED"),
    ])
    _write_score(results_dir, "idea-alpha", 0.8)
    assert reconcile_decision_batches(results_dir, cfg)["allow_new_batch"] is True

    audit = audit_campaign_decision_receipts(
        results_dir,
        cfg,
        expected_identity_sha256=[identity],
        start_epoch=start,
        end_epoch=time.time() + 1,
    )

    assert audit["status"] == "VERIFIED"
    assert audit["idea_ids"] == ["idea-alpha", "idea-beta"]
    assert audit["qualified_success_count"] == 1
    assert audit["qualified_success_idea_ids"] == ["idea-alpha"]
    assert audit["qualified_success_identity_complete"] is True
    assert audit["decision_input_evidence_complete"] is True
    assert audit["evidence_mismatch_idea_ids"] == []
    assert audit["terminal_count"] == 2
    assert audit["time_to_first_decision_seconds"] >= 0
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["schema"] == 3
    assert receipt["decision_evidence"] == [
        {
            "idea_id": "idea-alpha",
            "lifecycle_state": "COMPLETE",
            "qualification_reason": (
                "authoritative_local_evidence_verified"
            ),
            "primary_metric_value": 0.8,
            "evidence_sha256": receipt["decision_evidence"][0][
                "evidence_sha256"
            ],
        },
        {
            "idea_id": "idea-beta",
            "lifecycle_state": "FAILED",
            "qualification_reason": "lifecycle_not_complete",
            "primary_metric_value": None,
            "evidence_sha256": None,
        },
    ]
    assert len(receipt["decision_evidence"][0]["evidence_sha256"]) == 64


def test_missing_resolution_event_fails_closed_without_recreating_it(
        tmp_path):
    start = time.time() - 1
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "running", "data", "IN_PROGRESS"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path, threshold=0.7)
    identity = json.loads(path.read_text())[
        "identity_sha256"
    ]
    _replace_lifecycle(db_path, [
        ("idea-alpha", "completed", "architecture", "COMPLETE"),
        ("idea-beta", "failed", "data", "FAILED"),
    ])
    _write_score(results_dir, "idea-alpha", 0.8)
    assert reconcile_decision_batches(results_dir, cfg)[
        "allow_new_batch"
    ] is True
    event_path = (
        path.parent / "_resolution_events" / f"{identity}.json"
    )
    assert event_path.is_file()
    event_path.unlink()

    audit = audit_campaign_decision_receipts(
        results_dir,
        cfg,
        expected_identity_sha256=[identity],
        start_epoch=start,
        end_epoch=time.time() + 1,
    )

    assert audit == {
        "schema_version": 1,
        "status": "UNVERIFIED",
        "reason": "decision_resolution_event_invalid",
        "rank_claim_proven": False,
    }
    assert not event_path.exists()
    assert reconcile_decision_batches(results_dir, cfg, apply=False)[
        "reason"
    ] == "decision_resolution_event_invalid"
    assert not event_path.exists()


def test_resolution_event_recovers_exact_receipt_after_publish_failure(
        tmp_path, monkeypatch):
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "running", "data", "IN_PROGRESS"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path, threshold=0.7)
    identity = json.loads(path.read_text())[
        "identity_sha256"
    ]
    _replace_lifecycle(db_path, [
        ("idea-alpha", "completed", "architecture", "COMPLETE"),
        ("idea-beta", "failed", "data", "FAILED"),
    ])
    _write_score(results_dir, "idea-alpha", 0.8)
    original_write = decision_module._write_verified

    def fail_resolved_receipt(candidate_path, payload):
        if payload.get("status") in decision_module._RESOLVED_STATUSES:
            raise OSError("synthetic_publish_failure")
        original_write(candidate_path, payload)

    monkeypatch.setattr(
        decision_module, "_write_verified", fail_resolved_receipt
    )
    failed = reconcile_decision_batches(results_dir, cfg)

    assert failed["allow_new_batch"] is False
    assert failed["reason"] == "decision_contract_reconciliation_failed"
    assert json.loads(path.read_text())["status"] == "admitted"
    event_path = (
        path.parent / "_resolution_events" / f"{identity}.json"
    )
    event = json.loads(event_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        decision_module, "_write_verified", original_write
    )
    recovered = reconcile_decision_batches(results_dir, cfg)
    receipt = json.loads(path.read_text(encoding="utf-8"))

    assert recovered["allow_new_batch"] is True
    assert receipt["status"] == "succeeded"
    assert receipt["resolved_at"] == event["resolved_at"]
    assert event["resolved_receipt_sha256"] == (
        decision_module._resolved_receipt_sha256(receipt)
    )


def test_campaign_decision_audit_rejects_metric_rewrite_after_resolution(
        tmp_path):
    start = time.time() - 1
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "running", "data", "IN_PROGRESS"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path, threshold=0.7)
    identity = json.loads(path.read_text())["identity_sha256"]
    _replace_lifecycle(db_path, [
        ("idea-alpha", "completed", "architecture", "COMPLETE"),
        ("idea-beta", "failed", "data", "FAILED"),
    ])
    _write_score(results_dir, "idea-alpha", 0.8)
    assert reconcile_decision_batches(results_dir, cfg)[
        "allow_new_batch"
    ] is True
    _write_score(results_dir, "idea-alpha", 0.9)

    audit = audit_campaign_decision_receipts(
        results_dir,
        cfg,
        expected_identity_sha256=[identity],
        start_epoch=start,
        end_epoch=time.time() + 1,
    )

    assert audit["status"] == "UNVERIFIED"
    assert audit["decision_input_evidence_complete"] is False
    assert audit["evidence_mismatch_idea_ids"] == ["idea-alpha"]


def test_campaign_decision_audit_rejects_lifecycle_rewrite_after_resolution(
        tmp_path):
    start = time.time() - 1
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "running", "data", "IN_PROGRESS"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path, threshold=0.7)
    identity = json.loads(path.read_text())["identity_sha256"]
    _replace_lifecycle(db_path, [
        ("idea-alpha", "completed", "architecture", "COMPLETE"),
        ("idea-beta", "failed", "data", "FAILED"),
    ])
    _write_score(results_dir, "idea-alpha", 0.8)
    assert reconcile_decision_batches(results_dir, cfg)[
        "allow_new_batch"
    ] is True
    _replace_lifecycle(db_path, [
        ("idea-alpha", "completed", "architecture", "COMPLETE"),
        ("idea-beta", "skipped", "data", "SKIPPED"),
    ])

    audit = audit_campaign_decision_receipts(
        results_dir,
        cfg,
        expected_identity_sha256=[identity],
        start_epoch=start,
        end_epoch=time.time() + 1,
    )

    assert audit["status"] == "UNVERIFIED"
    assert audit["evidence_mismatch_idea_ids"] == ["idea-beta"]


def test_recomputed_receipt_hash_cannot_forge_success_evidence(tmp_path):
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "running", "data", "IN_PROGRESS"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path, threshold=0.7)
    _replace_lifecycle(db_path, [
        ("idea-alpha", "completed", "architecture", "COMPLETE"),
        ("idea-beta", "failed", "data", "FAILED"),
    ])
    _write_score(results_dir, "idea-alpha", 0.8)
    assert reconcile_decision_batches(results_dir, cfg)[
        "allow_new_batch"
    ] is True
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["decision_evidence"][0]["primary_metric_value"] = 0.1
    receipt["resolution_sha256"] = decision_module._resolution_hash(receipt)
    path.write_text(json.dumps(receipt), encoding="utf-8")

    gate = reconcile_decision_batches(results_dir, cfg)

    assert gate["allow_new_batch"] is False
    assert gate["reason"] == "decision_receipt_evidence_outcome_mismatch"


def test_resolution_waits_when_metric_evidence_identity_is_unavailable(
        tmp_path, monkeypatch):
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "running", "data", "IN_PROGRESS"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path, threshold=0.7)
    _replace_lifecycle(db_path, [
        ("idea-alpha", "completed", "architecture", "COMPLETE"),
        ("idea-beta", "failed", "data", "FAILED"),
    ])
    _write_score(results_dir, "idea-alpha", 0.8)
    monkeypatch.setattr(
        evidence_module,
        "qualify_authoritative_report_evidence_with_identity",
        lambda *_args, **_kwargs: (
            {}, {}, None, "report_evidence_changed_during_read", None,
        ),
    )

    gate = reconcile_decision_batches(results_dir, cfg)

    assert gate["allow_new_batch"] is False
    assert gate["reason"] == "decision_evidence_identity_unavailable"
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == (
        "admitted"
    )


def test_metric_change_during_qualification_has_no_usable_identity(
        tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    _write_score(results_dir, "idea-alpha", 0.8)
    cfg = _cfg(tmp_path)
    original = evidence_module.qualify_authoritative_report_evidence

    def mutate_after_read(*args, **kwargs):
        outcome = original(*args, **kwargs)
        _write_score(results_dir, "idea-alpha", 0.9)
        return outcome

    monkeypatch.setattr(
        evidence_module,
        "qualify_authoritative_report_evidence",
        mutate_after_read,
    )

    _, _, value, reason, digest = (
        evidence_module.qualify_authoritative_report_evidence_with_identity(
            "idea-alpha", results_dir, cfg, {"idea-alpha"},
        )
    )

    assert value is None
    assert reason == "report_evidence_changed_during_read"
    assert digest is None


def test_legacy_resolved_receipt_is_loadable_but_not_campaign_verified(
        tmp_path):
    start = time.time() - 1
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "running", "data", "IN_PROGRESS"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path, threshold=0.7)
    identity = json.loads(path.read_text())["identity_sha256"]
    _replace_lifecycle(db_path, [
        ("idea-alpha", "completed", "architecture", "COMPLETE"),
        ("idea-beta", "failed", "data", "FAILED"),
    ])
    _write_score(results_dir, "idea-alpha", 0.8)
    assert reconcile_decision_batches(results_dir, cfg)[
        "allow_new_batch"
    ] is True
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["schema"] = 1
    receipt.pop("decision_evidence")
    receipt["resolution_sha256"] = decision_module._resolution_hash(receipt)
    path.write_text(json.dumps(receipt), encoding="utf-8")

    audit = audit_campaign_decision_receipts(
        results_dir,
        cfg,
        expected_identity_sha256=[identity],
        start_epoch=start,
        end_epoch=time.time() + 1,
    )

    assert audit["status"] == "UNVERIFIED"
    assert audit["legacy_evidence_identity_sha256"] == [identity]


def test_schema_two_receipt_is_loadable_but_not_campaign_verified(tmp_path):
    start = time.time() - 1
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "running", "data", "IN_PROGRESS"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path, threshold=0.7)
    identity = json.loads(path.read_text())["identity_sha256"]
    _replace_lifecycle(db_path, [
        ("idea-alpha", "completed", "architecture", "COMPLETE"),
        ("idea-beta", "failed", "data", "FAILED"),
    ])
    _write_score(results_dir, "idea-alpha", 0.8)
    assert reconcile_decision_batches(results_dir, cfg)[
        "allow_new_batch"
    ] is True
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["schema"] = 2
    receipt["resolution_sha256"] = decision_module._resolution_hash(receipt)
    path.write_text(json.dumps(receipt), encoding="utf-8")

    audit = audit_campaign_decision_receipts(
        results_dir,
        cfg,
        expected_identity_sha256=[identity],
        start_epoch=start,
        end_epoch=time.time() + 1,
    )

    assert audit["status"] == "UNVERIFIED"
    assert audit["legacy_evidence_identity_sha256"] == [identity]


def test_campaign_decision_audit_fails_closed_while_batch_unresolved(tmp_path):
    start = time.time() - 1
    _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "queued", "data", "QUEUED"),
    ])
    results_dir, cfg, path = _stage_and_admit(tmp_path)
    identity = json.loads(path.read_text())["identity_sha256"]

    audit = audit_campaign_decision_receipts(
        results_dir,
        cfg,
        expected_identity_sha256=[identity],
        start_epoch=start,
        end_epoch=time.time() + 1,
    )

    assert audit["status"] == "UNVERIFIED"
    assert audit["unresolved_identity_sha256"] == [identity]
    assert audit["qualified_success_count"] == 0


def test_required_successes_prevents_best_of_n_batch_success(tmp_path):
    db_path = _create_lake(tmp_path, [
        ("idea-alpha", "running", "architecture", "IN_PROGRESS"),
        ("idea-beta", "running", "data", "IN_PROGRESS"),
    ])
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cfg = _cfg(tmp_path)
    contract = _contract(
        on_failure="stop_branch", threshold=0.7, required_successes=2)
    path, payload = stage_decision_contract(
        results_dir, cfg, 7, contract, _ideas())
    admit_decision_contract(path, payload, 2, cfg)
    _replace_lifecycle(db_path, [
        ("idea-alpha", "completed", "architecture", "COMPLETE"),
        ("idea-beta", "completed", "data", "COMPLETE"),
    ])
    _write_score(results_dir, "idea-alpha", 0.8)
    _write_score(results_dir, "idea-beta", 0.6)

    gate = reconcile_decision_batches(results_dir, cfg)

    assert gate["allow_new_batch"] is False
    assert gate["reason"] == "decision_contract_stop_active"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed_stopped"
    assert receipt["qualified_success_count"] == 1


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
