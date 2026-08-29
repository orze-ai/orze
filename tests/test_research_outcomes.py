import datetime
import hashlib
import hmac
import json
import os
import time
from types import SimpleNamespace

import orze.engine.campaign_efficiency as campaign_module
from orze.core.data_separation import ensure_data_separation
from orze.core.decision_batches import (
    admit_decision_contract,
    reconcile_decision_batches,
    stage_decision_contract,
)
from orze.core.model_lineage import (
    finalize_model_lineage,
    prepare_model_lineage_launch,
    receive_model_lineage_attestation,
)
from orze.engine.accounting import record_compute_start, record_compute_terminal
from orze.engine.campaign_efficiency import (
    DEFAULT_CAMPAIGN_TARGETS,
    DEFAULT_OUTCOME_TARGETS,
    preregister_campaign,
)
from orze.engine.research_outcomes import analyze_research_outcomes
from orze.idea_lake import IdeaLake


def _manifest_file(path, *, role, sample, key):
    namespace = "a" * 64
    normalization = "b" * 64
    fingerprint = hmac.new(key, sample.encode(), hashlib.sha256).hexdigest()
    rows = [
        {
            "schema_version": 1,
            "role": role,
            "fingerprint_algorithm": "hmac-sha256",
            "fingerprint_namespace_sha256": namespace,
            "normalization_contract_sha256": normalization,
            "fields": ["sample"],
        },
        {"sample": fingerprint},
    ]
    content = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rows
    ) + "\n"
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def _cfg(tmp_path):
    key = b"research-outcome-test"
    train_manifest = tmp_path / "train.jsonl"
    evaluation_manifest = tmp_path / "evaluation.jsonl"
    train_sha = _manifest_file(
        train_manifest, role="train", sample="train", key=key
    )
    evaluation_sha = _manifest_file(
        evaluation_manifest, role="evaluation", sample="eval", key=key
    )
    held_out = tmp_path / "held-out"
    held_out.mkdir()
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
            "max_decision_batch": 1,
            "min_decision_effect": 0.1,
        },
        "_project_root": str(tmp_path),
        "_orze_dir": str(tmp_path / ".orze"),
        "data_boundaries": {
            "forbidden_in_training": [str(held_out)],
            "watch_paths": [],
            "training_network": "deny",
        },
        "data_separation": {
            "enabled": True,
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": train_sha,
            "evaluation_manifest": str(evaluation_manifest),
            "evaluation_manifest_sha256": evaluation_sha,
            "fingerprint_namespace_sha256": "a" * 64,
            "normalization_contract_sha256": "b" * 64,
            "fields": ["sample"],
            "max_overlap": {"sample": 0},
            "max_records": 100,
            "max_bytes": 1024 * 1024,
            "max_line_bytes": 4096,
        },
        "model_lineage": {
            "enabled": True,
            "artifact": "model.bin",
            "max_files": 10,
            "max_bytes": 1024,
            "attestation_timeout": 1,
        },
    }


def _build_campaign(tmp_path, monkeypatch, *, qualified=True):
    now = time.time()
    start = now - 10
    end = now + 10
    results = tmp_path / "results"
    results.mkdir()
    cfg = _cfg(tmp_path)
    idea_id = "idea-outcome"
    contract = {
        "uncertainty": "Whether this exact bounded recipe improves the score.",
        "metric": "score",
        "baseline": 0.5,
        "comparator": "gt",
        "threshold": 0.7,
        "on_failure": "stop_branch",
        "max_experiments": 1,
        "required_successes": 1,
    }
    path, staged = stage_decision_contract(
        results,
        cfg,
        1,
        contract,
        [{"idea_id": idea_id, "approach_family": "architecture"}],
    )
    identity = staged["identity_sha256"]
    manifest = {
        "campaign_id": "research-outcome-test",
        "start_epoch": start,
        "end_epoch": end,
        "physical_scope": [4, 5, 6, 7],
        "poll_seconds": 10.0,
        "minimum_samples": 1,
        "minimum_claims": 1,
        "minimum_release_to_claim_pairs": 1,
        "targets": dict(DEFAULT_CAMPAIGN_TARGETS),
        "outcome_contract": {
            "expected_decision_identity_sha256": [identity],
            "artifact_relation": "any",
            "targets": dict(DEFAULT_OUTCOME_TARGETS),
        },
    }
    # Registration is prospectively enforced in production. Move only its
    # clock backward in this synthetic test so real filesystem receipts can use
    # the current wall clock while preserving registration-before-start order.
    with monkeypatch.context() as context:
        context.setattr(campaign_module.time, "time", lambda: start - 10)
        preregister_campaign(cfg["idea_lake_db"], manifest)
    admit_decision_contract(path, staged, 1, cfg)

    lake = IdeaLake(cfg["idea_lake_db"])
    lake.insert(idea_id, "outcome", "seed: 1\n", "", status="queued")
    assert lake.record_state_transition(idea_id, "QUEUED", "CLAIMED")
    assert lake.record_state_transition(idea_id, "CLAIMED", "IN_PROGRESS")

    idea_dir = results / idea_id
    idea_dir.mkdir(exist_ok=True)
    artifact = idea_dir / "model.bin"
    artifact.write_bytes(b"one standalone research model")
    process = SimpleNamespace(pid=os.getpid())
    tp = SimpleNamespace(
        idea_id=idea_id,
        attempt_id="a" * 32,
        execution_identity="c" * 64,
        process=process,
        gpu=4,
        start_time=now - 2,
    )
    separation = ensure_data_separation(cfg)
    boundary = prepare_model_lineage_launch(
        idea_id=idea_id,
        attempt_id=tp.attempt_id,
        execution_identity=tp.execution_identity,
        idea_dir=idea_dir,
        cfg=cfg,
        separation_receipt=separation,
    )
    os.write(
        boundary["write_fd"], (boundary["nonce"] + "\n").encode("ascii")
    )
    receive_model_lineage_attestation(boundary, process_pid=os.getpid())
    record_compute_start(tp, idea_dir, phase="training")
    finalize_model_lineage(tp, idea_dir, cfg)
    record_compute_terminal(
        tp, idea_dir, "completed", "trainer_completed",
        phase="training", return_code=0,
    )
    assert lake.record_stage_transition(
        idea_id, "training", "IN_PROGRESS", "COMPLETE", "trained"
    )
    assert lake.record_stage_transition(
        idea_id, "evaluation", "PENDING", "IN_PROGRESS", "evaluating"
    )
    assert lake.record_state_transition(
        idea_id,
        "IN_PROGRESS",
        "COMPLETE" if qualified else "FAILED",
        reason="synthetic_decision",
    )
    lake.close()
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED", "score": 0.8}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "orze.reporting.evidence.qualify_authoritative_report_evidence",
        lambda *args: (
            None,
            None,
            0.8 if qualified else 0.6,
            "authoritative_local_evidence_verified",
        ),
    )
    gate = reconcile_decision_batches(results, cfg)
    assert gate["reason"] in {
        "decision_contract_ready", "decision_contract_stop_active",
    }
    return cfg, results, manifest, end


def test_research_outcome_receipt_verifies_yield_compute_and_lineage(
        tmp_path, monkeypatch):
    cfg, results, manifest, end = _build_campaign(
        tmp_path, monkeypatch, qualified=True
    )

    receipt = analyze_research_outcomes(
        cfg["idea_lake_db"], results, cfg, manifest, now_epoch=end + 1
    )

    assert receipt["status"] == "VERIFIED"
    assert receipt["metrics"]["qualified_successes"] == 1
    assert receipt["metrics"]["qualified_success_rate"] == 1.0
    assert receipt["metrics"]["duplicate_training_attempts"] == 0
    assert receipt["metrics"]["zero_gpu_rejection_rate"] == 1.0
    assert receipt["lineage_evidence"]["status"] == "VERIFIED"
    assert receipt["lineage_evidence"]["rank_claim_proven"] is False


def test_research_outcome_receipt_reports_complete_zero_yield_as_failed(
        tmp_path, monkeypatch):
    cfg, results, manifest, end = _build_campaign(
        tmp_path, monkeypatch, qualified=False
    )

    receipt = analyze_research_outcomes(
        cfg["idea_lake_db"], results, cfg, manifest, now_epoch=end + 1
    )

    assert receipt["status"] == "FAILED"
    assert receipt["checks"]["qualified_success_rate"]["passed"] is False
    assert receipt["checks"]["gpu_hours_per_qualified_success"][
        "passed"
    ] is False
    assert receipt["lineage_evidence"]["status"] == "NOT_APPLICABLE"


def test_research_outcome_receipt_is_unverified_when_compute_is_incomplete(
        tmp_path, monkeypatch):
    cfg, results, manifest, end = _build_campaign(
        tmp_path, monkeypatch, qualified=True
    )
    terminal = next(results.glob(
        "idea-outcome/_compute_receipts/*/terminal.json"
    ))
    terminal.unlink()

    receipt = analyze_research_outcomes(
        cfg["idea_lake_db"], results, cfg, manifest, now_epoch=end + 1
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["compute_evidence_complete"]["passed"] is False
    assert receipt["compute_evidence"]["incomplete_started_attempts"] == 1
    assert receipt["lineage_evidence"]["status"] == "UNVERIFIED"
