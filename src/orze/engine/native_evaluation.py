"""Evaluation adapter for generic, durable execution-attempt authority.

Domain validation is outside the short publication transaction. Its file
identities and the launch's lifecycle revision are rechecked before effects.
This fences controller callbacks; it does not sandbox external worker writes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import stat

from orze.core.execution_attempts import (
    AttemptAuthorityError, AttemptRef, StaleAttempt, create_attempt,
    current_attempt, finish_attempt, mark_running, require_current,
)
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.execution_authority import (
    canonical_identity_equal, execution_transaction, lifecycle_fence,
)


from orze.engine.completion_events import CompletionEvent


def require_catalog(lake, idea_id, results_dir, cfg, *, handle=None):
    """Omitting a Python Lake object never opts a native task into legacy IO."""
    from orze.engine.execution_catalog import declared_catalog
    declared = declared_catalog(Path(results_dir) / idea_id)
    if lake is not None:
        if declared is not None:
            paths = [row[2] for row in lake.conn.execute("PRAGMA database_list") if row[1] == "main"]
            if len(paths) != 1 or str(Path(paths[0]).absolute()) != declared:
                raise AttemptEffectBusy("evaluation_catalog_scope_mismatch")
        return
    if declared is not None:
        raise AttemptEffectBusy("evaluation_native_catalog_required")
    if handle is not None and getattr(handle, "attempt_ref", None) is not None:
        raise AttemptEffectBusy("evaluation_native_catalog_required")
    from orze.core.evaluation_retry_state import open_existing_lake
    from orze.reporting.evidence import report_lifecycle_db_path
    path = report_lifecycle_db_path(Path(results_dir), cfg)
    if not path.exists() and not path.is_symlink():
        return
    existing = None
    try:
        existing = open_existing_lake(path)
        if current_attempt(existing.conn, idea_id, "evaluation") is not None:
            raise AttemptEffectBusy("evaluation_native_catalog_required")
    except AttemptEffectBusy:
        raise
    except Exception as exc:
        raise AttemptEffectBusy("evaluation_catalog_unverifiable") from exc
    finally:
        if existing is not None:
            existing.close()


def _ref(ep):
    ref = getattr(ep, "attempt_ref", None)
    if (not isinstance(ref, AttemptRef) or ref.task_id != ep.idea_id
            or ref.phase != "evaluation" or ref.attempt_id != ep.attempt_id):
        raise AttemptEffectBusy("evaluation_attempt_identity_missing")
    return ref


def _owned(lake, ep, *, states=("RUNNING",)):
    row = require_current(lake.conn, _ref(ep), states=states)
    binding = row["binding"]
    if not canonical_identity_equal(
            {"physical_gpu": getattr(ep, "gpu", None)},
            {"physical_gpu": binding.get("physical_gpu")}):
        raise AttemptEffectBusy("evaluation_resource_identity_changed")
    if row["state"] == "RUNNING" and not canonical_identity_equal(
            {"process_pid": getattr(getattr(ep, "process", None), "pid", None)},
            {"process_pid": binding.get("process_pid")}):
        raise AttemptEffectBusy("evaluation_process_identity_changed")
    if not canonical_identity_equal(
            lifecycle_fence(lake, ep.idea_id, "evaluation"), row["binding"].get("lifecycle")):
        raise StaleAttempt("evaluation_lifecycle_revision_changed")
    return row


def is_current(lake, ep) -> bool:
    """No tokenless enrollment: legacy compatibility requires no native row."""
    row = current_attempt(lake.conn, ep.idea_id, "evaluation")
    if row is None:
        if getattr(ep, "attempt_ref", None) is not None:
            raise AttemptEffectBusy("evaluation_attempt_record_missing")
        return True  # Explicit pre-native compatibility; never for a native row.
    ref = _ref(ep)
    if (row["attempt_id"], row["generation"]) != (ref.attempt_id, ref.generation):
        return False
    if row["state"] in ("TERMINAL", "NOT_STARTED"):
        return False
    _owned(lake, ep, states=("LAUNCHING", "RUNNING"))
    return True


def begin(lake, idea_dir: Path, attempt_id: str, gpu, *, source_event=None) -> AttemptRef:
    """Commit launch intent and stage revision before any Popen."""
    with execution_transaction(lake, idea_dir) as tx:
        from orze.engine.completion_events import require_completion, training_source
        source = source_event or training_source(idea_dir.name, gpu, lake, idea_dir.parent)
        if source is not None:
            row = require_completion(source, lake, idea_dir.parent, phase="training")
            if row is not None and row["terminal"].get("outcome") != "completed":
                raise StaleAttempt("evaluation_training_source_not_completed")
            if getattr(source, "attempt_ref", None) is not None:
                tx.watch_dependency(source.attempt_ref)
        from orze.engine.execution_catalog import bind_catalog
        bind_catalog(lake, idea_dir, tx.lease)
        idea_id = idea_dir.name
        if lake.get_fsm_state(idea_id) != "IN_PROGRESS":
            raise AttemptAuthorityError("evaluation_global_not_running")
        at = lake._transition_time(lake.conn)
        training = lake.get_stage_state(idea_id, "training")
        if training != "COMPLETE" and not lake._record_stage_transition_in_tx(
                idea_id, "training", training, "COMPLETE",
                "reconcile_validated_training_output", socket.gethostname(), os.getpid(), at):
            raise AttemptAuthorityError("training_stage_not_ready_for_evaluation")
        evaluation = lake.get_stage_state(idea_id, "evaluation")
        if evaluation not in ("NOT_STARTED", "PENDING") or not lake._record_stage_transition_in_tx(
                idea_id, "evaluation", evaluation, "IN_PROGRESS",
                f"evaluation_launched on gpu {gpu}", socket.gethostname(), os.getpid(), at):
            raise AttemptAuthorityError("evaluation_stage_transition_rejected")
        fence = lifecycle_fence(lake, idea_id, "evaluation")
        binding = {"origin": "native_evaluation", "lifecycle": fence,
                   "physical_gpu": gpu}
        if source is not None and getattr(source, "attempt_ref", None) is not None:
            from dataclasses import asdict
            binding["source_ref"] = asdict(source.attempt_ref)
        ref = create_attempt(lake.conn, idea_id, "evaluation", attempt_id, binding)
        tx.watch_attempt(ref)
        return ref


def _verify_compute(idea_dir, payload, *, process=None, phase=None, event=None,
                    outcome=None, reason_code=None, return_code=None, require_start=True):
    """Explicit native context is strict; two-argument compatibility remains."""
    if process is not None:
        from orze.engine.compute_publication import verify_compute_receipt
        return verify_compute_receipt(
            idea_dir, payload, process=process, phase=phase, event=event,
            outcome=outcome, reason_code=reason_code, return_code=return_code,
            require_start=require_start)
    path = idea_dir / "_compute_receipts" / payload["attempt_id"] / (payload["event"] + ".json")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    identity = lambda info: (info.st_dev, info.st_ino, info.st_mode,
                             info.st_nlink, info.st_size,
                             info.st_mtime_ns, info.st_ctime_ns)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 65536:
            raise AttemptEffectInDoubt("evaluation_compute_receipt_invalid")
        raw = os.read(fd, 65537)
        expected = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if (raw != expected or identity(os.fstat(fd)) != identity(before)
                or identity(path.lstat()) != identity(before)):
            raise AttemptEffectInDoubt("evaluation_compute_receipt_readback_failed")
        os.fsync(fd)
    finally:
        os.close(fd)
    for folder in (path.parent, path.parent.parent, idea_dir):
        directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def started(lake, ep, idea_dir, record_start):
    """Record observed process creation without holding a lock across Popen."""
    with execution_transaction(lake, idea_dir) as tx:
        row = _owned(lake, ep, states=("LAUNCHING",))
        # Failure leaves the already committed LAUNCHING intent. The launcher
        # must stop the known process; confirmed stop permits failed closure.
        receipt = record_start(ep, idea_dir, phase="evaluation")
        _verify_compute(idea_dir, receipt, process=ep, phase="evaluation",
                        event="start", outcome="started")
        binding = dict(row["binding"])
        binding["process_pid"] = getattr(getattr(ep, "process", None), "pid", None)
        mark_running(lake.conn, _ref(ep), binding=binding)
        tx.watch_attempt(_ref(ep))


def _inputs(idea_dir, cfg):
    from orze.reporting.evidence import report_evidence_paths
    from orze.reporting.evaluation_output import evaluation_output_path
    evidence_cfg = dict(cfg)
    evidence_cfg["report"] = cfg.get("report") or {}
    paths = report_evidence_paths(idea_dir.name, idea_dir.parent, evidence_cfg)
    output = evaluation_output_path(idea_dir, cfg)
    if output is not None:
        paths.append(output)
    paths.extend(Path(path) for path in (cfg.get("sealed_files") or []))
    paths = sorted(set(paths), key=str)
    if len(paths) > 128:
        raise AttemptEffectBusy("evaluation_evidence_path_limit")
    return paths


def _identities(paths):
    result = []
    for path in paths:
        try:
            info = path.lstat()
        except FileNotFoundError:
            result.append((str(path), None))
        except NotADirectoryError:
            # A failed evaluator's diagnostic destination can be blocked by
            # a file. This is absence of output, not a valid result and not
            # permission to overwrite the blocking path.
            result.append((str(path), "parent_not_directory"))
        else:
            result.append((str(path), (info.st_dev, info.st_ino, info.st_mode,
                                      info.st_nlink, info.st_size,
                                      info.st_mtime_ns, info.st_ctime_ns)))
    return result


def not_started(lake, ep, idea_dir, reason):
    """Close admission only; a preflight rejection is not a failed evaluation."""
    ref = _ref(ep)
    with execution_transaction(lake, idea_dir) as tx:
        _owned(lake, ep, states=("LAUNCHING",))
        digest = tx.prepare(ref, {"operation": "evaluation_not_started",
                                  "reason_code": reason})
        if not lake._record_stage_transition_in_tx(
                ep.idea_id, "evaluation", "IN_PROGRESS", "PENDING",
                reason, socket.gethostname(), os.getpid(), lake._transition_time(lake.conn)):
            raise AttemptAuthorityError("evaluation_not_started_stage_rejected")
        terminal = {"outcome": "not_started", "reason_code": reason,
                    "return_code": None, "effect_receipt_sha256": digest,
                    "lifecycle": lifecycle_fence(lake, ep.idea_id, "evaluation")}
        if finish_attempt(tx.conn, ref, terminal, not_started=True) != "committed":
            raise AttemptAuthorityError("evaluation_not_started_not_new")


def finish(lake, ep, idea_dir, cfg, ret, *, forced=None, not_started=False):
    """Accept one current terminal, then deliver once in this live controller.

    A crash after SQL commit but before delivery does not automatically replay
    downstream work. Durable downstream action acknowledgement is separate.
    ``forced`` is (outcome, reason_code, diagnostic) after confirmed cleanup.
    """
    from orze.engine import evaluator
    from orze.engine.accounting import record_compute_terminal

    if not is_current(lake, ep):
        return None
    if not not_started and type(ret) is not int:
        raise AttemptEffectBusy("evaluation_exit_unconfirmed")
    paths = _inputs(idea_dir, cfg)
    identities = _identities(paths)
    success = False
    reason = "evaluation_failed_validation_or_process"
    detail = f"Evaluation exited with code {ret}"
    outcome = "failed"
    if forced is not None:
        outcome, reason, detail = forced
    elif ret == 0:
        success, detail = evaluator.validate_evaluation_result(idea_dir, cfg)
        if success:
            outcome, reason = "completed", "evaluation_validated"
        else:
            detail = f"Evaluation evidence validation failed: {detail}"
    if identities != _identities(paths):
        raise AttemptEffectBusy("evaluation_evidence_changed_during_validation")
    ref = _ref(ep)
    try:
        with execution_transaction(lake, idea_dir) as tx:
            row = _owned(lake, ep, states=("LAUNCHING", "RUNNING"))
            if identities != _identities(paths):
                raise StaleAttempt("evaluation_evidence_changed_before_publication")
            digest = tx.prepare(ref, {
                "operation": "evaluation_terminal", "outcome": outcome,
                "reason_code": reason, "return_code": ret,
                "input_identity_sha256": hashlib.sha256(
                    json.dumps(identities, separators=(",", ":")).encode()).hexdigest(),
            })
            if row["state"] == "LAUNCHING" and not not_started:
                # Only a launcher with its actual Popen handle and confirmed
                # cleanup takes this branch after initialization failed.
                mark_running(lake.conn, ref)
            if not success:
                evaluator._write_eval_failure_marker(
                    idea_dir.parent, ep.idea_id,
                    cfg.get("eval_output") or "eval_report.json", detail, lake=None,
                    effect_lease=tx.lease)
                if ret == 0 and forced is None:
                    from orze.engine.failure_analysis import write_failure_analysis
                    write_failure_analysis(idea_dir, "eval_failure", detail)
                evaluator._record_eval_audit(idea_dir, "reject", reason, detail=detail[:500])
            if not not_started:
                receipt = record_compute_terminal(
                    ep, idea_dir, outcome, reason, phase="evaluation", return_code=ret)
                _verify_compute(idea_dir, receipt, process=ep, phase="evaluation",
                                event="terminal", outcome=outcome, reason_code=reason,
                                return_code=ret, require_start=row["state"] == "RUNNING")
            if not lake._record_state_transition_in_tx(
                    ep.idea_id, "IN_PROGRESS", "COMPLETE" if success else "FAILED",
                    reason=reason, host=socket.gethostname(), pid=os.getpid(), sop_type="training"):
                raise AttemptAuthorityError("evaluation_terminal_lifecycle_rejected")
            terminal = {"outcome": outcome, "reason_code": reason, "return_code": ret,
                        "effect_receipt_sha256": digest,
                        "lifecycle": lifecycle_fence(lake, ep.idea_id, "evaluation")}
            if finish_attempt(lake.conn, ref, terminal, not_started=not_started) != "committed":
                raise AttemptAuthorityError("evaluation_terminal_not_new")
    except StaleAttempt:
        return None
    return CompletionEvent(ep.idea_id, ep.gpu, ref)
