"""Deliver a confirmed native launch error as an exact, once-only action.

CALLING SPEC: launcher attaches its captured ref only after failed_launch
confirms closure. The phase delivers that exception here before any provider
repair. This controller action is not an OS launch or scientific observation.
It records repair as pending; no shared-code repair worker is authorized here.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket

from orze.core.execution_attempts import (
    AttemptAuthorityError, AttemptRef, StaleAttempt, create_attempt,
    current_attempt, finish_attempt, mark_running, require_current,
)
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.execution_authority import (
    canonical_identity_equal, execution_transaction, lifecycle_fence,
)
from orze.engine.training_attempts import _launch_state, _read, require_catalog


def bind_launch_error(error, ref):
    """Preserve the original exception class and the already captured ref."""
    if ref is not None:
        if not isinstance(ref, AttemptRef) or ref.phase != "training":
            raise AttemptEffectBusy("launch_failure_reference_invalid")
        error._orze_launch_attempt_ref = ref


def _source(tx, ref):
    row = require_current(tx.conn, ref, states=("TERMINAL", "NOT_STARTED"))
    if (row["binding"].get("origin") != "native_training"
            or (row["terminal"] or {}).get("outcome") not in ("failed", "not_started")):
        raise AttemptEffectBusy("launch_failure_source_not_failed_launch")
    tx.watch_dependency(ref)
    return row


def _claim_identity(lake, folder, ref, cfg):
    require_catalog(lake, folder, cfg)
    value, digest = _read(folder / "claim.json", 8192)
    if value.get("attempt_id") != ref.attempt_id:
        raise StaleAttempt("launch_failure_claim_replaced")
    return digest


def report_launch_failure(lake, idea_dir, error, failure_counts, cfg):
    """Return handled status, or None only for a genuinely legacy failure.

    A stale delivered error is handled without effects. Unknown/missing native
    authority raises HOLD instead of looking up a new attempt to appropriate.
    Duplicate delivery reprojects its durable counter, never adds one again.
    """
    folder = Path(idea_dir)
    ref = getattr(error, "_orze_launch_attempt_ref", None)
    require_catalog(lake, folder, cfg)
    if ref is None:
        from orze.engine.execution_catalog import declared_catalog
        if (declared_catalog(folder) is not None
                or (lake is not None and current_attempt(lake.conn, folder.name, "training") is not None)):
            raise AttemptEffectBusy("launch_failure_native_reference_required")
        return None
    if (lake is None or not isinstance(ref, AttemptRef) or ref.phase != "training"
            or ref.task_id != folder.name):
        raise AttemptEffectBusy("launch_failure_reference_invalid")
    source_identity = asdict(ref)
    action_id = "launch-failure-" + hashlib.sha256(json.dumps(
        source_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    message = f"Launch error: {error}"[:2000]
    try:
        with execution_transaction(lake, folder) as tx:
            source = _source(tx, ref)
            claim_sha = _claim_identity(lake, folder, ref, cfg)
            prior = current_attempt(tx.conn, ref.task_id, "launch_failure_report")
            if prior is not None and prior["attempt_id"] == action_id:
                tx.watch_dependency(AttemptRef(ref.task_id, "launch_failure_report",
                                               action_id, prior["generation"]))
                terminal = prior["terminal"] or {}
                count = terminal.get("failure_count_after")
                if (prior["state"] != "TERMINAL" or type(count) is not int or count < 1
                        or not canonical_identity_equal(terminal.get("source_attempt"), source_identity)):
                    raise AttemptEffectBusy("launch_failure_report_unconfirmed")
                status = "duplicate"
            else:
                binding = source["binding"]
                if "launch_lifecycle" in binding:
                    if (claim_sha != binding.get("claim_sha256")
                            or not canonical_identity_equal(_launch_state(lake, ref.task_id),
                                                            binding["launch_lifecycle"])):
                        raise StaleAttempt("launch_failure_source_lifecycle_changed")
                    from_state = "CLAIMED"
                else:
                    current = lifecycle_fence(lake, ref.task_id, "training")
                    if (current["global_state"] != "IN_PROGRESS"
                            or not canonical_identity_equal(current, binding.get("lifecycle"))):
                        raise StaleAttempt("launch_failure_source_lifecycle_changed")
                    from_state = "IN_PROGRESS"
                count = failure_counts.get(ref.task_id, 0)
                if type(count) is not int or count < 0:
                    raise AttemptEffectBusy("launch_failure_counter_invalid")
                previous_count = ((prior or {}).get("terminal") or {}).get("failure_count_after", 0)
                if type(previous_count) is not int or previous_count < 0:
                    raise AttemptEffectBusy("launch_failure_counter_receipt_invalid")
                count = max(count, previous_count) + 1
                action = create_attempt(tx.conn, ref.task_id, "launch_failure_report", action_id, {
                    "origin": "controller_action", "source_attempt": source_identity,
                    "operation": "launch_failure_report",
                })
                mark_running(tx.conn, action)
                digest = tx.prepare(action, {
                    "operation": "launch_failure_report", "source_attempt": source_identity,
                    "repair_status": "pending_explicit_action",
                })
                from orze.engine.launcher import _write_failure
                payload = _write_failure(folder, message, cfg=cfg, effect_lease=tx.lease)
                actual, _ = _read(folder / "metrics.json")
                if not canonical_identity_equal(actual, payload):
                    raise AttemptAuthorityError("launch_failure_metrics_not_confirmed")
                directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
                if not lake._record_state_transition_in_tx(
                        ref.task_id, from_state, "FAILED", reason="training_launch_failed",
                        host=socket.gethostname(), pid=os.getpid(), sop_type="training"):
                    raise AttemptAuthorityError("launch_failure_lifecycle_rejected")
                terminal = {
                    "outcome": "reported", "source_attempt": source_identity,
                    "failure_count_after": count, "repair_status": "pending_explicit_action",
                    "effect_receipt_sha256": digest, "lifecycle_phase": "training",
                    "lifecycle": lifecycle_fence(lake, ref.task_id, "training"),
                }
                if finish_attempt(tx.conn, action, terminal) != "committed":
                    raise AttemptAuthorityError("launch_failure_report_not_new")
                if _claim_identity(lake, folder, ref, cfg) != claim_sha:
                    raise StaleAttempt("launch_failure_claim_changed")
                status = "reported"
    except StaleAttempt:
        return {"status": "stale"}
    failure_counts[ref.task_id] = max(failure_counts.get(ref.task_id, 0), count)
    return {"status": status, "failure_count_after": count,
            "repair_status": "pending_explicit_action"}
