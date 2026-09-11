"""Native posthoc launch authority; no tokenless import or worker publication.

Execution phase is posthoc; the existing workload lifecycle stage is training.
All lifecycle writers own a short effect/SQLite transaction. Caller-prepared
launch_inputs are opaque bounded metadata, never executable configuration or
scientific permission. Large input reads and process waits belong outside.
"""
from __future__ import annotations

import json
from pathlib import Path
import socket

from orze.core.execution_attempts import (
    AttemptAuthorityError, AttemptRef, StaleAttempt, _json, create_attempt,
    current_attempt, finish_attempt, mark_running, require_current,
)
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.execution_authority import (
    canonical_identity_equal, execution_transaction, lifecycle_fence,
)
from orze.engine.training_attempts import _claim, _launch_state, _read
from orze.engine import posthoc_supervision as proof


def require_catalog(lake, idea_dir, cfg, *, handle=None):
    """Missing wiring or mutable flags cannot erase durable posthoc routing."""
    from orze.engine.training_attempts import require_catalog as structural_scope
    structural_scope(lake, idea_dir, cfg, handle=handle)
    folder = Path(idea_dir).absolute()
    if lake is None:
        from orze.reporting.evidence import report_lifecycle_db_path
        from orze.core.evaluation_retry_state import open_existing_lake
        path = report_lifecycle_db_path(folder.parent, cfg)
        if path.exists() or path.is_symlink():
            existing = None
            try:
                existing = open_existing_lake(path)
                if current_attempt(existing.conn, folder.name, "posthoc") is not None:
                    raise AttemptEffectBusy("posthoc_native_catalog_required")
            except AttemptEffectBusy:
                raise
            except Exception as exc:
                raise AttemptEffectBusy("posthoc_catalog_unverifiable") from exc
            finally:
                if existing is not None:
                    existing.close()
        return
    row = current_attempt(lake.conn, folder.name, "posthoc")
    if handle is not None:
        ref = getattr(handle, "attempt_ref", None)
        active = row is not None and row["state"] not in ("TERMINAL", "NOT_STARTED")
        if row is not None and ref is None:
            raise AttemptEffectBusy("posthoc_native_token_required")
        if active and (not isinstance(ref, AttemptRef) or ref.phase != "posthoc"
                       or ref.task_id != folder.name
                       or ref.attempt_id != getattr(handle, "attempt_id", None)):
            raise AttemptEffectBusy("posthoc_native_phase_mismatch")
        if (getattr(ref, "phase", None) == "posthoc" or active) and (
                getattr(handle, "is_posthoc", None) is not True):
            raise AttemptEffectBusy("posthoc_native_phase_mismatch")


def _ref(tp, idea_dir):
    ref = getattr(tp, "attempt_ref", None)
    if (not isinstance(ref, AttemptRef) or ref.phase != "posthoc"
            or ref.task_id != tp.idea_id or ref.task_id != Path(idea_dir).name
            or ref.attempt_id != tp.attempt_id):
        raise AttemptEffectBusy("posthoc_attempt_identity_invalid")
    return ref


def _row(lake, tp, idea_dir, states=("LAUNCHING", "RUNNING")):
    row = require_current(lake.conn, _ref(tp, idea_dir), states=states)
    binding = row["binding"]
    if (binding.get("origin") != "native_posthoc"
            or binding.get("process_supervision_protocol") != proof.PROTOCOL
            or binding.get("lifecycle_phase") != "training"):
        raise AttemptEffectBusy("posthoc_supervision_unbound")
    return row


def _launch_authority(lake, tp, idea_dir, row):
    claim, digest = _claim(tp, idea_dir, lake)
    if (digest != row["binding"].get("claim_sha256") or not canonical_identity_equal(
            _launch_state(lake, tp.idea_id), row["binding"].get("launch_lifecycle"))):
        raise StaleAttempt("posthoc_launch_authority_changed")
    return claim


def _other_phases_closed(lake, idea_id):
    for phase in ("training", "evaluation"):
        row = current_attempt(lake.conn, idea_id, phase)
        if row is not None and row["state"] not in ("TERMINAL", "NOT_STARTED"):
            raise AttemptEffectBusy("posthoc_other_phase_active")


def begin(lake, tp, idea_dir, *, launch_inputs, artifact_binding=None):
    """Pin intent before work-directory creation, READY allocation or GO."""
    from orze.core.artifact_contract import validate_artifact_publication_binding
    inputs = json.loads(_json(launch_inputs))
    folder = Path(idea_dir).absolute()
    if tp.idea_id != folder.name or getattr(tp, "process", None) is not None:
        raise AttemptEffectBusy("posthoc_launch_context_invalid")
    artifact = None
    if artifact_binding is not None:
        artifact = validate_artifact_publication_binding(artifact_binding)
        if artifact["scope"] != str(folder.parent):
            raise AttemptEffectBusy("posthoc_artifact_scope_mismatch")
    with execution_transaction(lake, folder) as tx:
        from orze.engine.native_pre_script import require_launch_ready
        require_launch_ready(lake, folder, {})
        _other_phases_closed(lake, tp.idea_id)
        _, claim_sha = _claim(tp, folder, lake)
        state = _launch_state(lake, tp.idea_id)
        binding = {"origin": "native_posthoc", "claim_sha256": claim_sha,
                   "launch_lifecycle": state, "lifecycle_phase": "training",
                   "process_supervision_protocol": proof.PROTOCOL, "launch_inputs": inputs}
        if artifact is not None:
            binding["artifact_publication"] = artifact
        from orze.engine.execution_catalog import bind_catalog
        bind_catalog(lake, folder, tx.lease)
        ref = create_attempt(tx.conn, tp.idea_id, "posthoc", tp.attempt_id, binding)
        if (_claim(tp, folder, lake)[1] != claim_sha or not canonical_identity_equal(
                _launch_state(lake, tp.idea_id), state)):
            raise AttemptAuthorityError("posthoc_launch_authority_changed")
        _other_phases_closed(lake, tp.idea_id)
        tx.watch_attempt(ref)
    return ref


def record_ready_start(lake, tp, idea_dir, record_start):
    """Record allocated blocked worker before lease exit; never authorize GO."""
    from orze.engine.compute_publication import verify_compute_receipt
    with execution_transaction(lake, idea_dir) as tx:
        row = _row(lake, tp, idea_dir, ("LAUNCHING",))
        proof.ready_binding(tp, idea_dir)
        _launch_authority(lake, tp, idea_dir, row)
        record_start(tp, idea_dir, phase="posthoc")
        start, _ = _read(idea_dir / "_compute_receipts" / tp.attempt_id / "start.json")
        verify_compute_receipt(idea_dir, start, process=tp, phase="posthoc",
                               event="start", outcome="started")
        _launch_authority(lake, tp, idea_dir, row)
        tx.watch_attempt(tp.attempt_ref)


def started(lake, tp, idea_dir, process_identity):
    """Bind exact READY/claim/start before RUNNING; the caller sends GO later."""
    from orze.engine import launcher
    from orze.engine.compute_publication import verify_compute_receipt
    with execution_transaction(lake, idea_dir) as tx:
        row = _row(lake, tp, idea_dir, ("LAUNCHING",))
        supervision = proof.ready_binding(tp, idea_dir)
        claim = _launch_authority(lake, tp, idea_dir, row)
        if (type(process_identity) is not dict
                or any(type(process_identity.get(key)) is not int for key in ("pid", "pgid", "start_ticks"))
                or process_identity["pgid"] <= 0
                or process_identity["pid"] != supervision["worker"]["pid"]
                or process_identity["start_ticks"] != supervision["worker"]["start_ticks"]):
            raise AttemptAuthorityError("posthoc_process_identity_mismatch")
        start, _ = _read(idea_dir / "_compute_receipts" / tp.attempt_id / "start.json")
        verify_compute_receipt(idea_dir, start, process=tp, phase="posthoc",
                               event="start", outcome="started")
        claim.update(trainer_pid=process_identity["pid"], trainer_pgid=process_identity["pgid"],
                     trainer_start_ticks=process_identity["start_ticks"], trainer_started_at=tp.start_time)
        try:
            launcher.atomic_write(idea_dir / "claim.json", json.dumps(claim, indent=2))
            actual, claim_sha = _claim(tp, idea_dir, lake)
            if not canonical_identity_equal(actual, claim):
                raise AttemptAuthorityError("posthoc_started_claim_readback_failed")
            if not lake._record_state_transition_in_tx(
                    tp.idea_id, "CLAIMED", "IN_PROGRESS", reason=f"posthoc_launched on gpu {tp.gpu}",
                    host=socket.gethostname(), pid=tp.process.pid, sop_type="training"):
                raise AttemptAuthorityError("posthoc_started_lifecycle_rejected")
            binding = dict(row["binding"])
            binding.update(process_pid=tp.process.pid, supervision=supervision,
                           started_claim_sha256=claim_sha,
                           lifecycle=lifecycle_fence(lake, tp.idea_id, "training"))
            mark_running(tx.conn, tp.attempt_ref, binding=binding)
            if _claim(tp, idea_dir, lake)[1] != claim_sha:
                raise AttemptAuthorityError("posthoc_started_claim_changed")
            tx.watch_attempt(tp.attempt_ref)
        except BaseException as exc:
            raise AttemptEffectInDoubt("posthoc_start_publication_unconfirmed") from exc


def current(lake, tp, idea_dir):
    """Only current RUNNING has publication permission; no legacy import."""
    try:
        row = current_attempt(lake.conn, tp.idea_id, "posthoc")
        if getattr(tp, "attempt_ref", None) is None:
            if row is not None:
                raise AttemptEffectBusy("posthoc_native_token_required")
            return False
        ref = _ref(tp, idea_dir)
        if row is None:
            raise AttemptEffectBusy("posthoc_attempt_record_missing")
        if ((row["attempt_id"], row["generation"]) != (ref.attempt_id, ref.generation)
                or row["state"] in ("TERMINAL", "NOT_STARTED")):
            return False
        if row["state"] != "RUNNING":
            raise AttemptEffectBusy("posthoc_attempt_not_running")
        row = _row(lake, tp, idea_dir, ("RUNNING",))
        _, claim_sha = _claim(tp, idea_dir, lake)
        binding = row["binding"]
        if claim_sha != binding.get("started_claim_sha256"):
            raise AttemptEffectBusy("posthoc_started_claim_changed")
        if (type(binding.get("process_pid")) is not int
                or type(tp.process.pid) is not int or binding["process_pid"] != tp.process.pid):
            raise AttemptEffectBusy("posthoc_process_identity_mismatch")
        if not canonical_identity_equal(lifecycle_fence(lake, tp.idea_id, "training"), binding.get("lifecycle")):
            return False
        proof.bound_binding(tp, row, idea_dir)
        return row
    except StaleAttempt:
        return False
    except OSError as exc:
        raise AttemptEffectBusy("posthoc_authority_unreadable") from exc


def failed_launch(lake, tp, idea_dir, ret, *, not_started=False):
    """Close only known uncreated/stopped execution; do not complete its task."""
    from orze.engine.accounting import record_compute_terminal
    from orze.engine.compute_publication import verify_compute_receipt
    if type(not_started) is not bool:
        raise AttemptEffectBusy("posthoc_not_started_flag_invalid")
    if not_started and (getattr(tp, "process", None) is not None
                        or getattr(tp, "_termination_unconfirmed", False) is True):
        raise AttemptEffectBusy("posthoc_created_process_cannot_be_not_started")
    ref = _ref(tp, idea_dir)
    row = _row(lake, tp, idea_dir)
    closure = None if not_started else proof.require_closed(
        tp, row, idea_dir, ret, allow_launch_cleanup=True)
    with execution_transaction(lake, idea_dir) as tx:
        row = _row(lake, tp, idea_dir)
        if row["state"] == "LAUNCHING":
            _launch_authority(lake, tp, idea_dir, row)
        elif not current(lake, tp, idea_dir):
            raise StaleAttempt("posthoc_failed_launch_stale")
        if closure is not None and not canonical_identity_equal(closure, proof.require_closed(
                tp, row, idea_dir, ret, allow_launch_cleanup=True)):
            raise AttemptEffectBusy("posthoc_process_tree_receipt_changed")
        plan = {"operation": "posthoc_failed_launch", "return_code": ret, "not_started": not_started}
        if closure is not None:
            plan["process_tree"] = closure
        digest = tx.prepare(ref, plan)
        if row["state"] == "LAUNCHING" and not not_started:
            mark_running(tx.conn, ref, binding=proof.bind_launch_cleanup(row, closure))
        if not not_started:
            payload = record_compute_terminal(tp, idea_dir, "failed", "posthoc_launch_initialization_failed",
                                              phase="posthoc", return_code=ret)
            verify_compute_receipt(idea_dir, payload, process=tp, phase="posthoc", event="terminal",
                                   outcome="failed", reason_code="posthoc_launch_initialization_failed",
                                   return_code=ret, require_start=row["state"] == "RUNNING")
        terminal = {"outcome": "not_started" if not_started else "failed",
                    "return_code": ret, "effect_receipt_sha256": digest,
                    "lifecycle_phase": "training"}
        if "lifecycle" in row["binding"]:
            terminal["lifecycle"] = row["binding"]["lifecycle"]
        if closure is not None:
            terminal["process_tree"] = closure
        if finish_attempt(tx.conn, ref, terminal, not_started=not_started) != "committed":
            raise AttemptAuthorityError("posthoc_failed_launch_not_new")
        if "lifecycle" not in row["binding"]:
            _launch_authority(lake, tp, idea_dir, row)
        tx.watch_attempt(ref)
