"""Once-only controller reporting for an explicitly delivered failed CPU hook.

Only a current, confirmed native pre-script terminal may authorize this action.
The pre-script occurrence is NOT the claim ID: zero-allocation receipts belong
to the captured claim. A short effect transaction binds metrics, that receipt,
the CLAIMED -> FAILED edge and a durable failure counter. No fixer is invoked.
Duplicate delivery checks the original source and claim but the action's new
FAILED fence, never revives the source's historical CLAIMED lifecycle.

File effects are not rolled back. A failure after prepare retains the existing
coordinator HOLD; neither counters nor successful reporting are projected from
uncertain writes. This is local publication, not automatic repair or recovery.
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
from orze.engine import accounting
from orze.engine.attempt_effect_receipts import (
    _decode, _read as _read_effect, _scan, _sync,
)
from orze.engine.compute_publication import _read as _read_compute
from orze.engine.execution_authority import (
    canonical_identity_equal as same, execution_transaction, lifecycle_fence,
)
from orze.engine.native_pre_script import PreScriptHOLD, _cached, _scope
from orze.engine.training_attempts import _launch_state, _read
from orze.engine.termination_hold import TerminationUnconfirmed


PHASE = "pre_script_failure_report"
_REPAIR = "pending_explicit_action"


def _source(tx, folder, ref, cfg):
    _scope(tx.lake, folder, cfg)
    row = require_current(tx.conn, ref, states=("TERMINAL",))
    pending = ((tx.prepared_ref, tx.prepared_sha256)
               if tx.prepare_started else None)
    if (row["binding"].get("origin") != "native_pre_script"
            or row["binding"].get("scope") != str(folder)
            or _cached(row, folder, pending=pending) is not False):
        raise PreScriptHOLD("pre_script_failure_source_invalid")
    return row


def _claim(folder, binding):
    claim, digest = _read(folder / "claim.json", 8192)
    claim_id = binding.get("claim_attempt_id")
    if (type(claim_id) is not str or not claim_id
            or type(claim.get("gpu")) is not int or claim["gpu"] < 0
            or claim.get("attempt_id") != claim_id
            or digest != binding.get("claim_sha256")):
        raise PreScriptHOLD("pre_script_failure_claim_changed")
    return claim_id, digest, claim["gpu"]


def _zero_expected(ref, claim_id, gpu):
    return {"schema_version": 1, "idea_id": ref.task_id,
            "attempt_id": claim_id, "phase": "admission", "event": "terminal",
            "outcome": "rejected", "physical_gpu": gpu,
            "allocated_gpu_seconds": 0.0, "return_code": None,
            "reason_code": "pre_script_failed", "process_pid": None}


def _no_start(folder, claim_id):
    path = folder / "_compute_receipts" / claim_id / "start.json"
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise PreScriptHOLD("pre_script_failure_claim_already_allocated")


def _zero_read(folder, expected, payload=None):
    _no_start(folder, expected["attempt_id"])
    directory = folder / "_compute_receipts" / expected["attempt_id"]
    actual, raw, _ = _read_compute(directory / "terminal.json")
    if (set(actual) != set(expected) | {"finished_at"}
            or type(actual.get("finished_at")) is not str or not actual["finished_at"]
            or not same({key: actual.get(key) for key in expected}, expected)
            or payload is not None and not same(actual, payload)):
        raise PreScriptHOLD("pre_script_failure_zero_receipt_invalid")
    for path in (directory, directory.parent, folder):
        _sync(path)
    return hashlib.sha256(raw).hexdigest()


def _plan(source, claim_id, claim_sha, expected, count):
    return {"operation": PHASE, "source_attempt": source,
            "claim_attempt_id": claim_id, "claim_sha256": claim_sha,
            "zero_gpu_receipt": expected, "failure_count_after": count,
            "repair_status": _REPAIR}


def _confirmed_action(folder, row):
    terminal = row["terminal"] or {}
    count = terminal.get("failure_count_after")
    if (row["state"] != "TERMINAL" or terminal.get("outcome") != "reported"
            or type(count) is not int or count < 1
            or terminal.get("repair_status") != _REPAIR
            or _scan(folder).get(row["attempt_id"]) !=
               (terminal.get("effect_receipt_sha256"), True)):
        raise PreScriptHOLD("pre_script_failure_report_unconfirmed")
    prepared = _decode(_read_effect(
        folder / "_execution_effects" / row["attempt_id"] / "prepared.json"))
    if not same(prepared["plan"], _plan(
            terminal.get("source_attempt"), terminal.get("claim_attempt_id"),
            terminal.get("claim_sha256"), terminal.get("zero_gpu_receipt"), count)):
        raise PreScriptHOLD("pre_script_failure_report_receipt_changed")
    return terminal, count


def _reported_files(folder, terminal, expected):
    metrics, digest = _read(folder / "metrics.json")
    if (metrics.get("status") != "FAILED"
            or digest != terminal.get("metrics_sha256")
            or _zero_read(folder, expected) != terminal.get("zero_gpu_receipt_sha256")):
        raise PreScriptHOLD("pre_script_failure_report_files_changed")


def report_pre_script_failure(lake, idea_dir, ref, failure_counts, cfg):
    """Report/duplicate an exact failed source, or handle a stale ref inertly.

Missing tokens and unconfirmed/changed claims raise HOLD. The caller must stop
its launch path for every return status, including ``stale``. Counters change
only after the transaction and its filesystem confirmation both return.
"""
    folder = Path(idea_dir).absolute()
    if (lake is None or not isinstance(ref, AttemptRef)
            or ref.phase != "pre_script" or ref.task_id != folder.name):
        raise PreScriptHOLD("pre_script_failure_reference_required")
    memory_count = failure_counts.get(ref.task_id, 0)
    if type(memory_count) is not int or memory_count < 0:
        raise PreScriptHOLD("pre_script_failure_counter_invalid")
    source_identity = asdict(ref)
    action_id = "pre-script-failure-" + hashlib.sha256(json.dumps(
        source_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    try:
        with execution_transaction(lake, folder) as tx:
            source = _source(tx, folder, ref, cfg)
            tx.watch_dependency(ref)
            claim_id, claim_sha, gpu = _claim(folder, source["binding"])
            if claim_id == ref.attempt_id:
                raise PreScriptHOLD("pre_script_failure_claim_identity_invalid")
            expected = _zero_expected(ref, claim_id, gpu)
            _no_start(folder, claim_id)
            binding = {"origin": "controller_action", "operation": PHASE,
                       "source_attempt": source_identity,
                       "claim_attempt_id": claim_id, "claim_sha256": claim_sha}
            prior = current_attempt(tx.conn, ref.task_id, PHASE)
            if prior is not None and prior["attempt_id"] == action_id:
                terminal, count = _confirmed_action(folder, prior)
                fence = lifecycle_fence(lake, ref.task_id, "training")
                if (not same(prior["binding"], binding)
                        or not same(terminal.get("source_attempt"), source_identity)
                        or terminal.get("claim_attempt_id") != claim_id
                        or terminal.get("claim_sha256") != claim_sha
                        or not same(terminal.get("zero_gpu_receipt"), expected)
                        or fence["global_state"] != "FAILED"
                        or not same(terminal.get("lifecycle"), fence)):
                    raise PreScriptHOLD("pre_script_failure_duplicate_changed")
                _reported_files(folder, terminal, expected)
                tx.watch_attempt(AttemptRef(ref.task_id, PHASE, action_id, prior["generation"]))
                status = "duplicate"
            else:
                if not same(_launch_state(lake, ref.task_id),
                            source["binding"].get("launch_lifecycle")):
                    raise PreScriptHOLD("pre_script_failure_lifecycle_changed")
                previous = 0 if prior is None else _confirmed_action(folder, prior)[1]
                count = max(memory_count, previous) + 1
                action = create_attempt(tx.conn, ref.task_id, PHASE, action_id, binding)
                mark_running(tx.conn, action)
                digest = tx.prepare(action, _plan(
                    source_identity, claim_id, claim_sha, expected, count))
                from orze.engine.launcher import _write_failure
                payload = _write_failure(folder, "Pre-script failed", cfg=cfg, effect_lease=tx.lease)
                metrics, metrics_sha = _read(folder / "metrics.json")
                if not same(metrics, payload):
                    raise AttemptAuthorityError("pre_script_failure_metrics_unconfirmed")
                _sync(folder)
                receipt = accounting.record_zero_gpu_outcome(
                    ref.task_id, folder, gpu, "rejected", "pre_script_failed", phase="admission")
                zero_sha = _zero_read(folder, expected, receipt)
                if not lake._record_state_transition_in_tx(
                        ref.task_id, "CLAIMED", "FAILED", reason="pre_script_failed",
                        host=socket.gethostname(), pid=os.getpid(), sop_type="training"):
                    raise AttemptAuthorityError("pre_script_failure_lifecycle_rejected")
                # The legal FAILED edge creates the stage; pre-script itself
                # never fabricates a training-start or prior stage receipt.
                terminal = {"outcome": "reported", "source_attempt": source_identity,
                            "claim_attempt_id": claim_id, "claim_sha256": claim_sha,
                            "failure_count_after": count, "repair_status": _REPAIR,
                            "effect_receipt_sha256": digest, "zero_gpu_receipt": expected,
                            "zero_gpu_receipt_sha256": zero_sha, "metrics_sha256": metrics_sha,
                            "lifecycle_phase": "training",
                            "lifecycle": lifecycle_fence(lake, ref.task_id, "training")}
                if finish_attempt(tx.conn, action, terminal) != "committed":
                    raise AttemptAuthorityError("pre_script_failure_report_not_new")
                _reported_files(folder, terminal, expected)
                status = "reported"
            if _claim(folder, source["binding"]) != (claim_id, claim_sha, gpu):
                raise PreScriptHOLD("pre_script_failure_claim_changed")
            _source(tx, folder, ref, cfg)
    except StaleAttempt:
        return {"status": "stale"}
    except TerminationUnconfirmed:
        raise
    except Exception as exc:
        raise PreScriptHOLD("pre_script_failure_publication_unconfirmed") from exc
    failure_counts[ref.task_id] = max(memory_count, count)
    return {"status": status, "failure_count_after": count, "repair_status": _REPAIR}
