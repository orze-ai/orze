"""Parent-only posthoc publication after exact current owned-tree closure.

Adapters return legacy metric dictionaries, not B2 observations. A normal
return without a status is operational completion only; zero/negative values
are preserved and no global domain metric or significance rule is applied.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import time

from orze.core.execution_attempts import AttemptAuthorityError, StaleAttempt, _json, finish_attempt
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.execution_authority import canonical_identity_equal, execution_transaction, lifecycle_fence
from orze.engine.native_evaluation import _identities
from orze.engine import posthoc_attempts as attempts, posthoc_supervision as proof


def _inputs(tp, row, folder, cfg):
    from orze.engine.native_posthoc import artifact_binding
    inputs = row["binding"].get("launch_inputs")
    work = folder / "_posthoc_attempts" / tp.attempt_ref.attempt_id / "work"
    if (type(inputs) is not dict or inputs.get("work_dir") != str(work)
            or inputs.get("execution_identity") != getattr(tp, "execution_identity", None)):
        raise AttemptEffectBusy("posthoc_execution_binding_changed")
    bound = row["binding"].get("artifact_publication")
    requested = artifact_binding(cfg, folder, tp.execution_identity)
    if not canonical_identity_equal({"value": bound}, {"value": requested}):
        raise AttemptEffectBusy("posthoc_artifact_contract_changed")
    return inputs, work, bound


def finish(lake, tp, slot, idea_dir, cfg, ret, failure_counts, *, forced=None):
    """Publish once in the caller-owned transaction, or ignore an old delivery."""
    folder = Path(idea_dir).absolute()
    row = attempts.current(lake, tp, folder)
    if not row:
        return None
    closure = proof.require_closed(tp, row, folder, ret)
    forced = proof.failure_override(closure, forced)
    inputs, work, bound = _inputs(tp, row, folder, cfg)
    result_path, metrics_path = work / "_posthoc_result.json", work / "metrics.json"
    identities = _identities([result_path, metrics_path])
    metrics, hashes, detail = {}, {}, ""
    outcome, reason = "failed", "posthoc_process_exit_nonzero"
    if forced is not None:
        outcome, reason, detail = forced
    elif ret == 0:
        try:
            receipt, hashes["result"] = attempts._read(result_path)
            metrics, hashes["metrics"] = attempts._read(metrics_path)
            _json(metrics)
            if (set(receipt) != {"schema", "attempt_ref", "payload_sha256", "outcome"}
                    or type(receipt["schema"]) is not int or receipt["schema"] != 1
                    or not canonical_identity_equal(receipt["attempt_ref"], asdict(tp.attempt_ref))
                    or receipt["payload_sha256"] != inputs["payload_sha256"]
                    or receipt["outcome"] not in ("completed", "failed")):
                raise ValueError("posthoc_result_binding_invalid")
            if metrics.get("status") not in (None, "COMPLETED", "FAILED"):
                raise ValueError("posthoc_metrics_status_invalid")
            if receipt["outcome"] == "failed" or metrics.get("status") == "FAILED":
                reason, detail = "posthoc_adapter_failed", str(metrics.get("error") or "Adapter failed")[:2000]
            else:
                outcome, reason = "completed", "posthoc_completed"
        except (OSError, ValueError, UnicodeError, RecursionError, AttemptEffectBusy, AttemptAuthorityError):
            metrics, detail, reason = {}, "Posthoc candidate missing, invalid or unbound", "posthoc_result_invalid"
    else:
        detail = f"Posthoc worker exited with code {ret}"
    if identities != _identities([result_path, metrics_path]):
        return None
    success = outcome == "completed"
    requeued = outcome == "requeued"
    if requeued and reason != "scheduler_slot_race":
        raise AttemptEffectBusy("posthoc_automatic_requeue_not_authorized")
    count = failure_counts.get(tp.idea_id, 0)
    if type(count) is not int or count < 0:
        raise AttemptEffectBusy("posthoc_failure_counter_invalid")
    target_count = count + (0 if success or requeued else 1)
    artifacts = None
    if success and bound is not None:
        from orze.engine.artifact_publication import prepare_artifacts
        artifacts = prepare_artifacts(tp.attempt_ref, folder, bound, source_dir=work)
    try:
        with execution_transaction(lake, folder) as tx:
            row = attempts.current(lake, tp, folder)
            if not row or identities != _identities([result_path, metrics_path]):
                return None
            if not canonical_identity_equal(closure, proof.require_closed(tp, row, folder, ret)):
                raise AttemptEffectBusy("posthoc_process_tree_receipt_changed")
            _inputs(tp, row, folder, cfg)
            ref = tp.attempt_ref
            plan = {"operation": "posthoc_requeue" if requeued else "posthoc_terminal",
                    "outcome": outcome, "reason_code": reason, "return_code": ret,
                    "process_tree": closure, "candidate_sha256": hashes}
            records = None
            if artifacts is not None:
                from orze.engine.artifact_publication import verify_prepared_artifacts
                records = verify_prepared_artifacts(artifacts, ref, folder, bound, source_dir=work)
                plan["artifact_ids"] = [record["artifact_id"] for record in records]
            digest = tx.prepare(ref, plan)
            artifact_ids = []
            if records is not None:
                verify_prepared_artifacts(artifacts, ref, folder, bound, source_dir=work)
                from orze.core.research_artifacts import register_artifacts
                artifact_ids = register_artifacts(tx.conn, ref, list(records))
            public_hash = None
            if not requeued:
                from orze.engine import launcher
                if success:
                    payload = dict(metrics, status="COMPLETED")
                    launcher.atomic_write(folder / "metrics.json", _json(payload))
                else:
                    payload = launcher._write_failure(folder, detail or reason, cfg=cfg, effect_lease=tx.lease)
                actual, public_hash = attempts._read(folder / "metrics.json")
                if not canonical_identity_equal(actual, payload):
                    raise AttemptAuthorityError("posthoc_metrics_readback_failed")
            from orze.engine.accounting import record_compute_terminal
            from orze.engine.compute_publication import verify_compute_receipt
            receipt = record_compute_terminal(tp, folder, outcome, reason, phase="posthoc", return_code=ret)
            verify_compute_receipt(folder, receipt, process=tp, phase="posthoc", event="terminal",
                                   outcome=outcome, reason_code=reason, return_code=ret)
            target = "QUEUED" if requeued else "COMPLETE" if success else "FAILED"
            if not lake._record_state_transition_in_tx(tp.idea_id, "IN_PROGRESS", target,
                    reason=reason, host=socket.gethostname(), pid=os.getpid(), sop_type="training"):
                raise AttemptAuthorityError("posthoc_terminal_lifecycle_rejected")
            terminal = {"outcome": outcome, "reason_code": reason, "return_code": ret,
                        "effect_receipt_sha256": digest, "process_tree": closure,
                        "failure_count_after": target_count, "lifecycle_phase": "training",
                        "lifecycle": lifecycle_fence(lake, tp.idea_id, "training"),
                        "candidate_sha256": hashes, "metrics_sha256": public_hash}
            if bound is not None:
                terminal["artifact_ids"] = list(artifact_ids)
            if not success and not requeued:
                terminal["repair_status"] = "pending_explicit_action" if cfg.get("max_fix_attempts", 0) > 0 else "not_requested"
            if finish_attempt(tx.conn, ref, terminal) != "committed":
                raise AttemptAuthorityError("posthoc_terminal_not_new")
            if records is not None:
                from orze.core.research_artifacts import artifacts_for_attempt
                verify_prepared_artifacts(artifacts, ref, folder, bound, source_dir=work)
                actual = artifacts_for_attempt(tx.conn, ref)
                expected = sorted(records, key=lambda record: record["logical_name"])
                canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                if canonical(actual) != canonical(expected):
                    raise AttemptAuthorityError("posthoc_artifact_readback_failed")
            if identities != _identities([result_path, metrics_path]):
                raise AttemptAuthorityError("posthoc_candidate_changed")
            if requeued:
                from orze.engine.failure import _reset_idea_for_retry
                _reset_idea_for_retry(folder, release_claim=True, lake=lake, effect_lease=tx.lease)
            elif reason == "posthoc_admin_kill":
                (folder / ".kill").unlink(missing_ok=True)
    except StaleAttempt:
        return None
    if not success and not requeued:
        failure_counts[tp.idea_id] = max(failure_counts.get(tp.idea_id, 0), target_count)
    from orze.engine.completion_events import CompletionEvent
    return CompletionEvent(tp.idea_id, slot, ref)


def requeue(lake, tp, slot, idea_dir, cfg, ret, reason):
    if reason != "scheduler_slot_race":
        raise AttemptEffectBusy("posthoc_automatic_requeue_not_authorized")
    return finish(lake, tp, slot, idea_dir, cfg, ret, {}, forced=("requeued", reason, reason))


def poll(lake, tp, slot, idea_dir, cfg, failure_counts, stall_minutes):
    """Configured stop conditions only; no root-only activity heuristics."""
    from orze.engine import launcher
    from orze.engine.health import check_stalled, detect_fatal_in_log
    from orze.engine.supervised_process import SupervisionUncertain
    from orze.engine.termination_hold import TerminationUnconfirmed, terminate_execution
    if not attempts.current(lake, tp, idea_dir):
        return None
    reason = None
    if time.time() - tp.start_time > tp.timeout:
        reason = "posthoc_timeout"
    elif check_stalled(tp, stall_minutes):
        reason = "posthoc_stall"
    elif detect_fatal_in_log(tp):
        reason = "posthoc_fatal_log"
    elif (idea_dir / ".kill").exists():
        reason = "posthoc_admin_kill"
    if reason is None:
        return None
    try:
        ret = terminate_execution(tp, idea_dir, phase="posthoc", reaper=launcher._terminate_and_reap)
    except (SupervisionUncertain, TerminationUnconfirmed):
        tp._termination_unconfirmed = True
        return None
    return finish(lake, tp, slot, idea_dir, cfg, ret, failure_counts,
                  forced=("interrupted", reason, reason))
