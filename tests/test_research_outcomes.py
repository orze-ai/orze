import datetime
import hashlib
import hmac
import json
import os
import time
from types import SimpleNamespace

import orze.core.decision_batches as decision_module
import orze.engine.campaign_efficiency as campaign_module
import orze.engine.research_outcomes as outcome_module
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
from orze.engine.accounting import (
    record_compute_start,
    record_compute_terminal,
    record_zero_gpu_outcome,
)
from orze.engine.campaign_efficiency import (
    DEFAULT_CAMPAIGN_TARGETS,
    DEFAULT_OUTCOME_TARGETS,
    preregister_campaign,
)
from orze.engine.research_outcomes import analyze_research_outcomes
from orze.engine.reproducibility import config_identity_sha256
from orze.engine.scheduler import claim
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


def _build_campaign(
    tmp_path, monkeypatch, *, qualified=True, expected_idea_ids=None,
    observe_rejection=True, expected_rejections=None,
    outcome_target_overrides=None,
):
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
        "operator_progress_root": str(results / "_campaign_progress"),
        "expected_idea_ids": expected_idea_ids or [idea_id],
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
            "expected_rejections": (
                expected_rejections if expected_rejections is not None else [{
                    "idea_id": idea_id,
                    "phase": "admission",
                    "reason_code": "synthetic_preallocation_rejection",
                }]
            ),
            "artifact_relation": "any",
            "reproducibility_contract": {
                "mode": "not_applicable",
                "rationale": (
                    "This single-idea campaign has no replication question."
                ),
                "expected_config_identity_sha256": {
                    expected_id: config_identity_sha256({"seed": 1})
                    for expected_id in (expected_idea_ids or [idea_id])
                },
            },
            "targets": {
                **DEFAULT_OUTCOME_TARGETS,
                **(outcome_target_overrides or {}),
            },
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
    if observe_rejection:
        assert claim(idea_id, results, 4)
        record_zero_gpu_outcome(
            idea_id,
            idea_dir,
            4,
            "rejected",
            "synthetic_preallocation_rejection",
        )
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


def test_research_outcome_does_not_pass_unobserved_rejection_target(
        tmp_path, monkeypatch):
    cfg, results, manifest, end = _build_campaign(
        tmp_path, monkeypatch, qualified=True, observe_rejection=False
    )

    receipt = analyze_research_outcomes(
        cfg["idea_lake_db"], results, cfg, manifest, now_epoch=end + 1
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["metrics"]["zero_gpu_rejection_rate"] == 0.0
    assert receipt["checks"]["rejection_contract_complete"]["passed"] is False
    assert receipt["checks"]["zero_gpu_rejection_rate"]["passed"] is False


def test_late_policy_rejection_cannot_hide_behind_failed_outcome(
        tmp_path, monkeypatch):
    cfg, results, manifest, end = _build_campaign(
        tmp_path,
        monkeypatch,
        qualified=True,
        expected_rejections=[
            {
                "idea_id": "idea-outcome",
                "phase": "admission",
                "reason_code": "synthetic_preallocation_rejection",
            },
            {
                "idea_id": "idea-outcome",
                "phase": "evaluation",
                "reason_code": "late_policy_rejection",
            },
        ],
    )
    late = SimpleNamespace(
        idea_id="idea-outcome",
        attempt_id="d" * 32,
        process=SimpleNamespace(pid=os.getpid()),
        gpu=5,
        start_time=time.time() - 1,
    )
    idea_dir = results / late.idea_id
    record_compute_start(late, idea_dir, phase="evaluation")
    record_compute_terminal(
        late, idea_dir, "failed", "late_policy_rejection",
        phase="evaluation", return_code=1,
    )

    receipt = analyze_research_outcomes(
        cfg["idea_lake_db"], results, cfg, manifest, now_epoch=end + 1
    )

    assert receipt["status"] == "FAILED"
    assert receipt["metrics"]["zero_gpu_rejection_rate"] == 0.5
    assert receipt["checks"]["zero_gpu_rejection_rate"]["passed"] is False


def test_research_outcome_receipt_is_unverified_when_compute_is_incomplete(
        tmp_path, monkeypatch):
    cfg, results, manifest, end = _build_campaign(
        tmp_path, monkeypatch, qualified=True
    )
    terminal = (
        results / "idea-outcome" / "_compute_receipts" / ("a" * 32)
        / "terminal.json"
    )
    terminal.unlink()

    receipt = analyze_research_outcomes(
        cfg["idea_lake_db"], results, cfg, manifest, now_epoch=end + 1
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["compute_evidence_complete"]["passed"] is False
    assert receipt["compute_evidence"]["incomplete_started_attempts"] == 1
    assert receipt["lineage_evidence"]["status"] == "UNVERIFIED"


def test_research_outcome_rejects_underreported_gpu_hours(
        tmp_path, monkeypatch):
    cfg, results, manifest, end = _build_campaign(
        tmp_path, monkeypatch, qualified=True
    )
    terminal_path = (
        results / "idea-outcome" / "_compute_receipts" / ("a" * 32)
        / "terminal.json"
    )
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal["allocated_gpu_seconds"] > 1.0
    terminal["allocated_gpu_seconds"] = 0.0
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")

    receipt = analyze_research_outcomes(
        cfg["idea_lake_db"], results, cfg, manifest, now_epoch=end + 1
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["compute_evidence_complete"]["passed"] is False
    assert len(receipt["compute_evidence"][
        "allocation_duration_mismatch_attempts"
    ]) == 1
    assert receipt["metrics"]["allocated_gpu_seconds_total"] > 1.0


def test_post_decision_model_and_lineage_replacement_invalidates_outcome(
        tmp_path, monkeypatch):
    cfg, results, manifest, end = _build_campaign(
        tmp_path, monkeypatch, qualified=True
    )
    idea_dir = results / "idea-outcome"
    artifact = idea_dir / "model.bin"
    original = artifact.read_bytes()
    replacement = bytes([original[0] ^ 1]) + original[1:]
    artifact.write_bytes(replacement)
    lineage_path = idea_dir / "_model_lineage.json"
    envelope = json.loads(lineage_path.read_text(encoding="utf-8"))
    envelope["payload"]["artifact_sha256"] = hashlib.sha256(
        replacement
    ).hexdigest()
    canonical = json.dumps(
        envelope["payload"], sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    envelope["payload_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    lineage_path.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    receipt = analyze_research_outcomes(
        cfg["idea_lake_db"], results, cfg, manifest, now_epoch=end + 1
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["decision_input_evidence_complete"][
        "passed"
    ] is False
    assert receipt["decision_evidence"]["evidence_mismatch_idea_ids"] == [
        "idea-outcome"
    ]


def test_post_decision_resolution_time_rewrite_invalidates_outcome(
        tmp_path, monkeypatch):
    cfg, results, manifest, end = _build_campaign(
        tmp_path, monkeypatch, qualified=True
    )
    receipt_path = next(
        (tmp_path / ".orze" / "policy" / "decision_contracts").glob(
            "cycle-*.json"
        )
    )
    decision = json.loads(receipt_path.read_text(encoding="utf-8"))
    decision["resolved_at"] = decision["admitted_at"]
    decision["resolution_sha256"] = decision_module._resolution_hash(
        decision
    )
    receipt_path.write_text(json.dumps(decision), encoding="utf-8")

    receipt = analyze_research_outcomes(
        cfg["idea_lake_db"], results, cfg, manifest, now_epoch=end + 1
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["decision_evidence_complete"]["passed"] is False


def test_backdated_resolution_content_cannot_hide_late_decision(
        tmp_path, monkeypatch):
    cfg, results, manifest, end = _build_campaign(
        tmp_path,
        monkeypatch,
        qualified=True,
        outcome_target_overrides={
            "max_time_to_first_decision_seconds": 15.0,
            "max_time_to_all_decisions_seconds": 15.0,
        },
    )
    contracts = tmp_path / ".orze" / "policy" / "decision_contracts"
    receipt_path = next(contracts.glob("cycle-*.json"))
    identity = json.loads(receipt_path.read_text(encoding="utf-8"))[
        "identity_sha256"
    ]
    event_path = contracts / "_resolution_events" / f"{identity}.json"
    published_at = end + 30
    for path in (event_path, receipt_path):
        os.utime(path, (published_at, published_at))

    receipt = analyze_research_outcomes(
        cfg["idea_lake_db"], results, cfg, manifest, now_epoch=end + 60
    )

    assert receipt["status"] == "FAILED", receipt["checks"]
    assert receipt["metrics"][
        "reported_time_to_first_decision_seconds"
    ] < 15
    assert abs(
        receipt["metrics"]["time_to_first_decision_seconds"] - 50
    ) < 1e-5
    assert abs(
        receipt["metrics"]["time_to_all_decisions_seconds"] - 50
    ) < 1e-5
    assert receipt["checks"]["time_to_first_decision"]["passed"] is False
    assert receipt["checks"]["time_to_all_decisions"]["passed"] is False


def test_future_resolution_publication_is_unverified(tmp_path, monkeypatch):
    cfg, results, manifest, end = _build_campaign(
        tmp_path, monkeypatch, qualified=True
    )
    contracts = tmp_path / ".orze" / "policy" / "decision_contracts"
    receipt_path = next(contracts.glob("cycle-*.json"))
    identity = json.loads(receipt_path.read_text(encoding="utf-8"))[
        "identity_sha256"
    ]
    event_path = contracts / "_resolution_events" / f"{identity}.json"
    future = end + 30
    for path in (event_path, receipt_path):
        os.utime(path, (future, future))

    receipt = analyze_research_outcomes(
        cfg["idea_lake_db"], results, cfg, manifest, now_epoch=end + 1
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["decision_evidence"][
        "resolution_publication_timing_invalid_identity_sha256"
    ] == [identity]
    assert receipt["checks"]["decision_evidence_complete"]["passed"] is False


def test_research_outcome_rejects_decision_universe_mismatch(
        tmp_path, monkeypatch):
    cfg, results, manifest, end = _build_campaign(
        tmp_path,
        monkeypatch,
        qualified=True,
        expected_idea_ids=["idea-other"],
        expected_rejections=[],
    )

    receipt = analyze_research_outcomes(
        cfg["idea_lake_db"], results, cfg, manifest, now_epoch=end + 1
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["registration"]["valid"] is True
    assert receipt["checks"]["decision_evidence_complete"]["passed"] is True
    assert receipt["checks"]["exact_preregistered_idea_universe"][
        "passed"
    ] is False
    assert receipt["lineage_evidence"]["status"] == "UNVERIFIED"


def test_research_outcome_propagates_complete_reproduction_failure(
        tmp_path, monkeypatch):
    cfg, results, manifest, end = _build_campaign(
        tmp_path, monkeypatch, qualified=True
    )
    monkeypatch.setattr(
        outcome_module,
        "audit_campaign_reproducibility",
        lambda *_args, **_kwargs: {
            "status": "FAILED",
            "reason": "reproducibility_targets_failed",
            "groups": [],
            "rank_claim_proven": False,
        },
    )

    receipt = analyze_research_outcomes(
        cfg["idea_lake_db"], results, cfg, manifest, now_epoch=end + 1
    )

    assert receipt["status"] == "FAILED"
    assert receipt["checks"]["reproducibility_evidence_complete"][
        "passed"
    ] is True
    assert receipt["checks"]["reproducibility"]["passed"] is False


def test_research_outcome_propagates_incomplete_reproduction_evidence(
        tmp_path, monkeypatch):
    cfg, results, manifest, end = _build_campaign(
        tmp_path, monkeypatch, qualified=True
    )
    monkeypatch.setattr(
        outcome_module,
        "audit_campaign_reproducibility",
        lambda *_args, **_kwargs: {
            "status": "UNVERIFIED",
            "reason": "reproducibility_metric_audit_failed",
            "groups": [],
            "rank_claim_proven": False,
        },
    )

    receipt = analyze_research_outcomes(
        cfg["idea_lake_db"], results, cfg, manifest, now_epoch=end + 1
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["reproducibility_evidence_complete"][
        "passed"
    ] is False
