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


def finish(lake, tp, slot, idea_dir, cfg, ret, failure_counts):
    """Return only a newly accepted completion; stale/duplicate is a no-op."""
    from orze.engine import launcher
    from orze.engine.accounting import record_compute_terminal
    from orze.engine.failure_analysis import classify_failure, write_failure_analysis

    if not current(lake, tp, idea_dir):
        return None
    if type(ret) is not int:
        raise AttemptEffectBusy("training_exit_unconfirmed")
    metrics_path = idea_dir / "metrics.json"
    identities = _identities([metrics_path])
    metrics = {}
    error = ""
    if ret == 0 and metrics_path.exists():
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
    try:
        with execution_transaction(lake, idea_dir) as tx:
            candidate = current(lake, tp, idea_dir)
            if not candidate or identities != _identities([metrics_path]):
                return None
            ref = import_for_completion(tx, tp, candidate)
            digest = tx.prepare(ref, {
                "operation": "training_terminal", "outcome": outcome,
                "reason_code": reason, "return_code": ret,
            })
            if invalid:
                prefix = "metrics.lineage_invalid" if reason == "model_lineage_invalid" else "metrics.invalid"
                destination = metrics_path.with_name(f"{prefix}.{time.time_ns()}.json")
                os.replace(metrics_path, destination)
                detail += f"; original preserved as {destination.name}"
            if not success and (invalid or not metrics_path.exists()):
                launcher._write_failure(idea_dir, detail, lake=None, cfg=cfg)
            if success:
                from orze.core.model_lineage import publish_model_lineage_finalization
                publish_model_lineage_finalization(lineage, tp, idea_dir, cfg)
            else:
                write_failure_analysis(idea_dir, classify_failure(detail, ret or -1, "training"), detail)
            receipt = record_compute_terminal(tp, idea_dir, outcome, reason, return_code=ret)
            _verify_compute(idea_dir, receipt)
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
            if finish_attempt(tx.conn, ref, terminal) != "committed":
                raise AttemptAuthorityError("training_terminal_not_new")
    except StaleAttempt:
        return None
    if not success:
        failure_counts[tp.idea_id] = max(failure_counts.get(tp.idea_id, 0), target_count)
    return CompletionEvent(tp.idea_id, slot, ref)
