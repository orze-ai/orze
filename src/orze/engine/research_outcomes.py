"""Fail-closed research yield and decision-efficiency campaign analysis."""

from __future__ import annotations

import datetime
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from orze.core.decision_batches import audit_campaign_decision_receipts
from orze.core.model_lineage import audit_campaign_model_lineage
from orze.engine.accounting import audit_campaign_compute_receipts
from orze.engine.campaign_efficiency import (
    DEFAULT_OUTCOME_TARGETS,
    _manifest_error,
    verify_campaign_registration,
)
from orze.engine.reproducibility import audit_campaign_reproducibility


def analyze_research_outcomes(
    db_path: str | Path,
    results_dir: str | Path,
    cfg: Mapping,
    manifest: Dict[str, Any],
    *,
    now_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    """Combine authoritative decision, compute, and artifact evidence."""
    now = time.time() if now_epoch is None else float(now_epoch)
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.datetime.fromtimestamp(
            now, datetime.timezone.utc
        ).isoformat(),
        "campaign_id": (
            manifest.get("campaign_id") if isinstance(manifest, dict) else None
        ),
        "status": "UNVERIFIED",
        "checks": {},
        "metrics": {},
    }
    error = _manifest_error(manifest, now, require_ended=True)
    outcome = manifest.get("outcome_contract") if isinstance(manifest, dict) else None
    if error or not isinstance(outcome, dict):
        receipt["checks"]["manifest_valid"] = {
            "passed": False,
            "reason": error or "outcome_contract_missing",
        }
        return receipt
    receipt["checks"]["manifest_valid"] = {"passed": True}
    registration = verify_campaign_registration(db_path, manifest)
    receipt["registration"] = registration
    receipt["checks"]["preregistered_manifest_match"] = {
        "passed": registration["valid"]
    }

    try:
        decision = audit_campaign_decision_receipts(
            Path(results_dir),
            cfg,
            expected_identity_sha256=outcome[
                "expected_decision_identity_sha256"
            ],
            start_epoch=manifest["start_epoch"],
            end_epoch=manifest["end_epoch"],
        )
    except Exception as exc:
        decision = {
            "schema_version": 1,
            "status": "UNVERIFIED",
            "reason": f"decision_audit_error:{type(exc).__name__}",
            "idea_ids": [],
            "qualified_success_idea_ids": [],
            "qualified_success_identity_complete": False,
            "rank_claim_proven": False,
        }
    receipt["decision_evidence"] = decision
    decision_idea_ids = decision.get("idea_ids") or []
    expected_idea_ids = list(manifest["expected_idea_ids"])
    idea_universe_matches = (
        len(decision_idea_ids) == len(expected_idea_ids)
        and set(decision_idea_ids) == set(expected_idea_ids)
    )
    if expected_idea_ids:
        try:
            compute = audit_campaign_compute_receipts(
                Path(results_dir),
                idea_ids=expected_idea_ids,
                start_epoch=manifest["start_epoch"],
                end_epoch=manifest["end_epoch"],
                physical_scope=manifest["physical_scope"],
                expected_rejections=outcome["expected_rejections"],
            )
        except Exception as exc:
            compute = {
                "schema_version": 1,
                "status": "UNVERIFIED",
                "reason": f"compute_audit_error:{type(exc).__name__}",
            }
    else:
        compute = {
            "schema_version": 1,
            "status": "UNVERIFIED",
            "reason": "decision_idea_set_unavailable",
        }
    receipt["compute_evidence"] = compute

    success_ids = decision.get("qualified_success_idea_ids") or []
    success_identity_complete = (
        decision.get("qualified_success_identity_complete") is True
        and idea_universe_matches
        and set(success_ids).issubset(expected_idea_ids)
    )
    if success_ids and success_identity_complete:
        try:
            lineage = audit_campaign_model_lineage(
                Path(results_dir),
                cfg,
                idea_ids=success_ids,
                artifact_relation=outcome["artifact_relation"],
            )
        except Exception as exc:
            lineage = {
                "schema_version": 1,
                "status": "UNVERIFIED",
                "reason": f"lineage_audit_error:{type(exc).__name__}",
            }
    elif success_identity_complete:
        lineage = {
            "schema_version": 1,
            "status": "NOT_APPLICABLE",
            "reason": "no_qualified_successes",
            "rank_claim_proven": False,
        }
    else:
        lineage = {
            "schema_version": 1,
            "status": "UNVERIFIED",
            "reason": "qualified_success_identity_missing",
        }
    receipt["lineage_evidence"] = lineage

    try:
        reproduction = audit_campaign_reproducibility(
            db_path,
            results_dir,
            cfg,
            expected_idea_ids=expected_idea_ids,
            contract=outcome["reproducibility_contract"],
        )
    except Exception as exc:
        reproduction = {
            "schema_version": 1,
            "status": "UNVERIFIED",
            "reason": f"reproducibility_audit_error:{type(exc).__name__}",
            "rank_claim_proven": False,
        }
    receipt["reproducibility_evidence"] = reproduction

    successes = decision.get("qualified_success_count")
    admitted = decision.get("admitted_count")
    gpu_seconds = compute.get("allocated_gpu_seconds_total")
    success_rate = (
        successes / admitted
        if isinstance(successes, int) and isinstance(admitted, int) and admitted
        else None
    )
    gpu_hours_per_success = (
        float(gpu_seconds) / 3600.0 / successes
        if isinstance(gpu_seconds, (int, float))
        and not isinstance(gpu_seconds, bool)
        and isinstance(successes, int) and successes > 0
        else None
    )
    receipt["metrics"] = {
        "admitted_ideas": admitted,
        "qualified_successes": successes,
        "qualified_success_rate": success_rate,
        "allocated_gpu_seconds_total": gpu_seconds,
        "gpu_hours_per_qualified_success": gpu_hours_per_success,
        "time_to_first_decision_seconds": decision.get(
            "time_to_first_decision_seconds"
        ),
        "time_to_all_decisions_seconds": decision.get(
            "time_to_all_decisions_seconds"
        ),
        "duplicate_training_attempts": compute.get(
            "duplicate_training_attempts"
        ),
        "zero_gpu_rejection_rate": compute.get("zero_gpu_rejection_rate"),
        "expected_rejection_attempts": compute.get(
            "expected_rejection_attempts"
        ),
        "reproducibility_groups": len(reproduction.get("groups") or []),
    }

    evidence_checks = {
        "decision_evidence_complete": decision.get("status") == "VERIFIED",
        "decision_input_evidence_complete": (
            decision.get("decision_input_evidence_complete") is True
        ),
        "decision_rank_not_inferred": (
            decision.get("rank_claim_proven") is False
        ),
        "exact_preregistered_idea_universe": idea_universe_matches,
        "compute_evidence_complete": compute.get("status") == "VERIFIED",
        "rejection_contract_complete": (
            compute.get("rejection_contract_complete") is True
        ),
        "qualified_success_identity_complete": success_identity_complete,
        "lineage_evidence_complete": lineage.get("status") in {
            "VERIFIED", "NOT_APPLICABLE",
        },
        "official_rank_not_inferred": lineage.get("rank_claim_proven") is False,
        "reproducibility_evidence_complete": reproduction.get("status") in {
            "VERIFIED", "FAILED",
        },
        "reproducibility_rank_not_inferred": (
            reproduction.get("rank_claim_proven") is False
        ),
    }
    for name, passed in evidence_checks.items():
        receipt["checks"][name] = {"passed": passed}

    targets = outcome["targets"]
    target_checks = {
        "time_to_first_decision": (
            receipt["metrics"]["time_to_first_decision_seconds"] is not None
            and receipt["metrics"]["time_to_first_decision_seconds"]
            <= targets["max_time_to_first_decision_seconds"]
        ),
        "time_to_all_decisions": (
            receipt["metrics"]["time_to_all_decisions_seconds"] is not None
            and receipt["metrics"]["time_to_all_decisions_seconds"]
            <= targets["max_time_to_all_decisions_seconds"]
        ),
        "qualified_success_rate": (
            success_rate is not None
            and success_rate >= targets["min_qualified_success_rate"]
        ),
        "gpu_hours_per_qualified_success": (
            gpu_hours_per_success is not None
            and gpu_hours_per_success
            <= targets["max_gpu_hours_per_qualified_success"]
        ),
        "duplicate_training_attempts": (
            isinstance(compute.get("duplicate_training_attempts"), int)
            and compute["duplicate_training_attempts"]
            <= targets["max_duplicate_training_attempts"]
        ),
        "zero_gpu_rejection_rate": (
            isinstance(compute.get("zero_gpu_rejection_rate"), (int, float))
            and compute["zero_gpu_rejection_rate"]
            >= targets["min_zero_gpu_rejection_rate"]
        ),
        "reproducibility": reproduction.get("status") == "VERIFIED",
    }
    for name, passed in target_checks.items():
        receipt["checks"][name] = {"passed": passed}

    sources = {
        "analyzer": Path(__file__),
        "campaign": Path(__file__).with_name("campaign_efficiency.py"),
        "accounting": Path(__file__).with_name("accounting.py"),
        "decision_batches": Path(__file__).parents[1] / "core/decision_batches.py",
        "model_lineage": Path(__file__).parents[1] / "core/model_lineage.py",
        "reporting_evidence": (
            Path(__file__).parents[1] / "reporting/evidence.py"
        ),
        "reproducibility": Path(__file__).with_name("reproducibility.py"),
    }
    receipt["source_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sources.items()
    }
    evidence_complete = registration["valid"] and all(evidence_checks.values())
    if evidence_complete:
        receipt["status"] = (
            "VERIFIED" if all(target_checks.values()) else "FAILED"
        )
    return receipt


def write_research_outcome_receipt(
    db_path: str | Path,
    results_dir: str | Path,
    cfg: Mapping,
    manifest_path: str | Path,
    output_path: str | Path,
) -> Dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    receipt = analyze_research_outcomes(
        db_path, results_dir, cfg, manifest
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv=None) -> int:
    import argparse
    from orze.core.config import load_project_config

    parser = argparse.ArgumentParser(
        description="Verify preregistered research outcome efficiency"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cfg = load_project_config(args.config)
    receipt = write_research_outcome_receipt(
        args.db, args.results_dir, cfg, args.manifest, args.output
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
