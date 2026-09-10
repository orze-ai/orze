"""Current-attempt training completion with one owned publication boundary.

The generic attempt store owns acceptance; training interprets only its own
output contract. No hypothesis, metric improvement, or observation is invented
by an execution's zero exit code. Provider repair is a separate action.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time

from orze.core.execution_attempts import AttemptAuthorityError, StaleAttempt, finish_attempt
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.execution_authority import execution_transaction, lifecycle_fence
from orze.engine.native_evaluation import CompletionEvent, _identities, _verify_compute
from orze.engine.training_attempts import current, import_for_completion


def _artifact_binding(candidate, tp, idea_dir, cfg):
    from orze.core.artifact_contract import artifact_publication_binding
    from orze.engine.execution_authority import canonical_identity_equal
    try:
        requested = artifact_publication_binding(
            cfg, idea_dir, getattr(tp, "execution_identity", None))
    except (ValueError, TypeError) as exc:
        raise AttemptEffectBusy("training_artifact_contract_unbound_or_changed") from exc
    captured = candidate.get("binding", {}).get("artifact_publication")
    if requested is None and captured is None:
        return None
    if (candidate.get("legacy") or candidate.get("binding", {}).get("origin")
            != "native_training" or not canonical_identity_equal(requested, captured)):
        raise AttemptEffectBusy("training_artifact_contract_unbound_or_changed")
    return captured


def finish(lake, tp, slot, idea_dir, cfg, ret, failure_counts, *, forced=None, interruption=None):
    """Return only a newly accepted completion; stale/duplicate is a no-op."""
    from orze.engine import launcher
    from orze.engine.accounting import record_compute_terminal
    from orze.engine.failure_analysis import classify_failure, write_failure_analysis

    initial_candidate = current(lake, tp, idea_dir)
    if not initial_candidate:
        return None
    artifact_binding = _artifact_binding(initial_candidate, tp, idea_dir, cfg)
    if type(ret) is not int:
        raise AttemptEffectBusy("training_exit_unconfirmed")
    metrics_path = idea_dir / "metrics.json"
    identities = _identities([metrics_path])
    metrics = {}
    error = ""
    if forced is not None:
        error = forced[2]
    elif ret == 0 and metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if type(metrics) is not dict:
                raise ValueError("object required")
        except (ValueError, OSError, UnicodeError, RecursionError):
            metrics = {}
            error = "metrics.json is unreadable or not a JSON object"
        if metrics.get("status") not in ("COMPLETED", "FAILED"):
            error = error or "metrics.json must declare status COMPLETED or FAILED"
    else:
        error = f"Process exited with code {ret}" if ret else "metrics.json is missing"
    if identities != _identities([metrics_path]):
        return None
    success = ret == 0 and not error and metrics.get("status") == "COMPLETED"
    invalid = ret == 0 and metrics_path.exists() and bool(error)
    detail = error or str(metrics.get("error") or "Training script reported FAILED")
    outcome = "completed" if success else "failed"
    reason = ("trainer_completed" if success else "metrics_invalid" if invalid
              else "trainer_declared_failed" if ret == 0 and not error
              else "process_exit_nonzero" if ret else "metrics_missing")
    if forced is not None:
        outcome, reason, detail = forced
    elif (ret == 0 and metrics.get("status") == "FAILED"
          and str(metrics.get("error", "")).startswith("insufficient_vram:")):
        return requeue(lake, tp, slot, idea_dir, cfg, ret,
                       "trainer_vram_precheck", input_identities=identities)
    lineage = None
    if success:
        from orze.core.model_lineage import (
            ModelLineageError, ModelLineagePublicationUnsupported,
            prepare_model_lineage_finalization,
        )
        try:
            lineage = prepare_model_lineage_finalization(tp, idea_dir, cfg)
        except ModelLineagePublicationUnsupported as exc:
            raise AttemptEffectBusy("training_lineage_publication_unsupported") from exc
        except ModelLineageError:
            success, outcome, reason = False, "failed", "model_lineage_invalid"
            invalid, detail = True, "model_lineage_validation_failed"
    target_count = failure_counts.get(tp.idea_id, 0) + (0 if success else 1)
    artifacts = None
    if success and artifact_binding is not None:
        from orze.engine.artifact_publication import prepare_artifacts
        artifacts = prepare_artifacts(tp.attempt_ref, idea_dir, artifact_binding)
    try:
        with execution_transaction(lake, idea_dir) as tx:
            candidate = current(lake, tp, idea_dir)
            if not candidate or identities != _identities([metrics_path]):
                return None
            _artifact_binding(candidate, tp, idea_dir, cfg)
            ref = import_for_completion(tx, tp, candidate)
            plan = {
                "operation": "training_terminal", "outcome": outcome,
                "reason_code": reason, "return_code": ret,
            }
            artifact_records = None
            if artifacts is not None:
                from orze.engine.artifact_publication import verify_prepared_artifacts
                artifact_records = verify_prepared_artifacts(
                    artifacts, ref, idea_dir, artifact_binding)
                plan["artifact_ids"] = [record["artifact_id"] for record in artifact_records]
            digest = tx.prepare(ref, plan)
            if artifact_records is not None:
                # Recheck after the actual prepare seam before granting rows.
                verify_prepared_artifacts(artifacts, ref, idea_dir, artifact_binding)
                from orze.core.research_artifacts import register_artifacts
                artifact_ids = register_artifacts(tx.conn, ref, list(artifact_records))
            if invalid:
                prefix = "metrics.lineage_invalid" if reason == "model_lineage_invalid" else "metrics.invalid"
                destination = metrics_path.with_name(f"{prefix}.{time.time_ns()}.json")
                os.replace(metrics_path, destination)
                detail += f"; original preserved as {destination.name}"
            if not success and (forced is not None or invalid or not metrics_path.exists()):
                launcher._write_failure(idea_dir, detail, lake=None, cfg=cfg,
                                        effect_lease=tx.lease)
            if success:
                from orze.core.model_lineage import publish_model_lineage_finalization
                publish_model_lineage_finalization(lineage, tp, idea_dir, cfg)
            else:
                write_failure_analysis(idea_dir, classify_failure(detail, ret or -1, "training"), detail)
            if interruption is not None:
                from orze.engine.interruption_publication import publish_interruption
                publish_interruption(interruption, tp, idea_dir.parent, cfg)
                if not candidate.get("legacy"):
                    from orze.engine.training_attempts import _read
                    receipt, _ = _read(idea_dir / "_compute_receipts" / ref.attempt_id / "terminal.json")
                    _verify_compute(idea_dir, receipt, process=tp, phase="training",
                                    event="terminal", outcome=outcome,
                                    reason_code=reason, return_code=ret)
                if reason == "interruption_admin_kill":
                    (idea_dir / ".kill").unlink(missing_ok=True)
            else:
                receipt = record_compute_terminal(tp, idea_dir, outcome, reason, return_code=ret)
                if candidate.get("legacy"):
                    # Explicit pre-native import retains its historical clock
                    # compatibility; it is never advertised as a native launch.
                    _verify_compute(idea_dir, receipt)
                else:
                    _verify_compute(idea_dir, receipt, process=tp, phase="training",
                                    event="terminal", outcome=outcome,
                                    reason_code=reason, return_code=ret)
            if success and cfg.get("eval_script"):
                accepted = lake._record_stage_transition_in_tx(
                    tp.idea_id, "training", "IN_PROGRESS", "COMPLETE",
                    "training_completed_evaluation_pending", socket.gethostname(),
                    os.getpid(), lake._transition_time(lake.conn))
            else:
                accepted = lake._record_state_transition_in_tx(
                    tp.idea_id, "IN_PROGRESS", "COMPLETE" if success else "FAILED",
                    reason=reason, host=socket.gethostname(), pid=os.getpid(), sop_type="training")
            if not accepted:
                raise AttemptAuthorityError("training_terminal_lifecycle_rejected")
            terminal = {"outcome": outcome, "reason_code": reason,
                        "return_code": ret, "effect_receipt_sha256": digest,
                        "failure_count_after": target_count,
                        "lifecycle": lifecycle_fence(lake, tp.idea_id, "training")}
            if artifact_records is not None:
                verify_prepared_artifacts(artifacts, ref, idea_dir, artifact_binding)
            if artifact_binding is not None:
                terminal["artifact_ids"] = list(artifact_ids) if artifact_records is not None else []
            if not success:
                terminal["repair_status"] = (
                    "pending_explicit_action" if cfg.get("max_fix_attempts", 0) > 0
                    else "not_requested")
            if finish_attempt(tx.conn, ref, terminal) != "committed":
                raise AttemptAuthorityError("training_terminal_not_new")
            if artifact_records is not None:
                from orze.core.research_artifacts import artifacts_for_attempt
                actual = artifacts_for_attempt(tx.conn, ref)
                expected = sorted(artifact_records, key=lambda record: record["logical_name"])
                canonical = lambda value: json.dumps(
                    value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                if canonical(actual) != canonical(expected):
                    raise AttemptAuthorityError("training_artifact_final_readback_failed")
    except StaleAttempt:
        return None
    if not success:
        failure_counts[tp.idea_id] = max(failure_counts.get(tp.idea_id, 0), target_count)
    return CompletionEvent(tp.idea_id, slot, ref)


def requeue(lake, tp, slot, idea_dir, cfg, ret, reason, *, input_identities=None):
    """Close a stopped allocation and atomically readmit its task, not a result."""
    from orze.engine.accounting import record_compute_terminal
    from orze.engine.failure import _reset_idea_for_retry
    if type(ret) is not int:
        raise AttemptEffectBusy("training_requeue_exit_unconfirmed")
    with execution_transaction(lake, idea_dir) as tx:
        candidate = current(lake, tp, idea_dir)
        if not candidate:
            return None
        if input_identities is not None and input_identities != _identities([idea_dir / "metrics.json"]):
            return None
        ref = import_for_completion(tx, tp, candidate)
        digest = tx.prepare(ref, {"operation": "training_requeue", "reason_code": reason,
                                  "return_code": ret})
        receipt = record_compute_terminal(tp, idea_dir, "requeued", reason, return_code=ret)
        if candidate.get("legacy"):
            _verify_compute(idea_dir, receipt)
        else:
            _verify_compute(idea_dir, receipt, process=tp, phase="training",
                            event="terminal", outcome="requeued",
                            reason_code=reason, return_code=ret)
        if not lake._record_state_transition_in_tx(
                tp.idea_id, "IN_PROGRESS", "QUEUED", reason=reason,
                host=socket.gethostname(), pid=os.getpid(), sop_type="training"):
            raise AttemptAuthorityError("training_requeue_lifecycle_rejected")
        terminal = {"outcome": "requeued", "reason_code": reason, "return_code": ret,
                    "effect_receipt_sha256": digest,
                    "lifecycle": lifecycle_fence(lake, tp.idea_id, "training")}
        if finish_attempt(tx.conn, ref, terminal) != "committed":
            raise AttemptAuthorityError("training_requeue_not_new")
        _reset_idea_for_retry(idea_dir, release_claim=True, lake=lake, effect_lease=tx.lease)
    return CompletionEvent(tp.idea_id, slot, ref)
