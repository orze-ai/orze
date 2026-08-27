"""Immutable, non-sensitive compute allocation receipts.

Trainer metrics are experiment output and may be missing, malformed, or
incorrect. These receipts are written by Orze from its own process clock and
record only allocation facts needed for efficiency analysis.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_PHASES = {"training", "posthoc", "evaluation", "pre_script", "admission"}
_OUTCOMES = {
    "started", "completed", "failed", "interrupted", "rejected", "requeued",
}


class ComputeAccountingError(RuntimeError):
    """Raised when an allocation receipt is invalid or contradictory."""


def _token(value: object, label: str) -> str:
    text = str(value)
    if not _TOKEN_RE.fullmatch(text):
        raise ComputeAccountingError(f"{label}_invalid")
    return text


def ensure_attempt_id(tp) -> str:
    """Return a stable attempt ID, deriving one for legacy process handles."""
    existing = getattr(tp, "attempt_id", None)
    if existing:
        return _token(existing, "attempt_id")
    material = "|".join([
        str(getattr(tp, "idea_id", "")),
        str(getattr(tp, "gpu", "")),
        str(getattr(getattr(tp, "process", None), "pid", "")),
        f"{float(getattr(tp, 'start_time', 0.0)):.9f}",
    ])
    attempt_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    tp.attempt_id = attempt_id
    return attempt_id


def _receipt_path(idea_dir: Path, attempt_id: str, event: str) -> Path:
    return idea_dir / "_compute_receipts" / attempt_id / f"{event}.json"


def _write_once(path: Path, payload: dict) -> dict:
    """Create one receipt without overwriting a concurrent/earlier writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"))
               + "\n").encode("utf-8")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ComputeAccountingError("existing_receipt_unreadable") from exc
        identity_fields = (
            "schema_version", "idea_id", "attempt_id", "phase", "event",
            "outcome", "physical_gpu",
        )
        if any(existing.get(key) != payload.get(key) for key in identity_fields):
            raise ComputeAccountingError("conflicting_receipt")
        return existing
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    return payload


def _base(tp, phase: str, event: str, outcome: str) -> dict:
    idea_id = _token(getattr(tp, "idea_id", ""), "idea_id")
    attempt_id = ensure_attempt_id(tp)
    phase = _token(phase, "phase")
    outcome = _token(outcome, "outcome")
    if phase not in _PHASES:
        raise ComputeAccountingError("phase_invalid")
    if outcome not in _OUTCOMES:
        raise ComputeAccountingError("outcome_invalid")
    gpu = getattr(tp, "gpu", None)
    if (gpu is not None
            and (isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0)):
        raise ComputeAccountingError("physical_gpu_invalid")
    payload = {
        "schema_version": 1,
        "idea_id": idea_id,
        "attempt_id": attempt_id,
        "phase": phase,
        "event": event,
        "outcome": outcome,
        "physical_gpu": gpu,
    }
    return payload


def record_compute_start(tp, idea_dir: Path, phase: str = "training") -> dict:
    """Persist the allocation start before a process is advertised active."""
    payload = _base(tp, phase, "start", "started")
    started = float(getattr(tp, "start_time", time.time()))
    if not math.isfinite(started) or started < 0:
        raise ComputeAccountingError("start_time_invalid")
    payload.update({
        "started_at_epoch": round(started, 6),
        "recorded_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "allocated_gpu_seconds": 0.0,
        "process_pid": getattr(getattr(tp, "process", None), "pid", None),
    })
    return _write_once(
        _receipt_path(Path(idea_dir), payload["attempt_id"], "start"), payload)


def record_compute_terminal(
    tp,
    idea_dir: Path,
    outcome: str,
    reason_code: str,
    *,
    phase: str = "training",
    return_code: Optional[int] = None,
) -> dict:
    """Persist the first terminal allocation outcome for one attempt."""
    if outcome == "started":
        raise ComputeAccountingError("terminal_outcome_invalid")
    payload = _base(tp, phase, "terminal", outcome)
    reason_code = _token(reason_code, "reason_code")
    started = float(getattr(tp, "start_time", time.time()))
    if not math.isfinite(started) or started < 0:
        raise ComputeAccountingError("start_time_invalid")
    elapsed = max(0.0, time.time() - started)
    if not math.isfinite(elapsed):
        raise ComputeAccountingError("allocated_gpu_seconds_invalid")
    if return_code is not None and (
        isinstance(return_code, bool) or not isinstance(return_code, int)
    ):
        raise ComputeAccountingError("return_code_invalid")
    payload.update({
        "started_at_epoch": round(started, 6),
        "finished_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        # Allocated-slot wall time; no utilization or trainer self-report.
        "allocated_gpu_seconds": round(elapsed, 3),
        "return_code": return_code,
        "reason_code": reason_code,
        "process_pid": getattr(getattr(tp, "process", None), "pid", None),
    })
    return _write_once(
        _receipt_path(Path(idea_dir), payload["attempt_id"], "terminal"),
        payload,
    )


def record_zero_gpu_outcome(
    idea_id: str,
    idea_dir: Path,
    physical_gpu: int,
    outcome: str,
    reason_code: str,
    *,
    phase: str = "admission",
) -> dict:
    """Record a claimed attempt rejected/requeued before GPU allocation."""
    idea_id = _token(idea_id, "idea_id")
    idea_dir = Path(idea_dir)
    try:
        claim = json.loads(
            (idea_dir / "claim.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComputeAccountingError("claim_receipt_missing_or_invalid") from exc
    attempt_id = _token(claim.get("attempt_id", ""), "attempt_id")
    if _receipt_path(idea_dir, attempt_id, "start").exists():
        raise ComputeAccountingError("attempt_already_allocated")
    phase = _token(phase, "phase")
    outcome = _token(outcome, "outcome")
    reason_code = _token(reason_code, "reason_code")
    if phase not in {"admission", "pre_script"}:
        raise ComputeAccountingError("zero_gpu_phase_invalid")
    if outcome not in {"rejected", "requeued"}:
        raise ComputeAccountingError("zero_gpu_outcome_invalid")
    if (isinstance(physical_gpu, bool) or not isinstance(physical_gpu, int)
            or physical_gpu < 0):
        raise ComputeAccountingError("physical_gpu_invalid")
    claimed_gpu = claim.get("gpu")
    if claimed_gpu != physical_gpu:
        raise ComputeAccountingError("physical_gpu_claim_mismatch")
    payload = {
        "schema_version": 1,
        "idea_id": idea_id,
        "attempt_id": attempt_id,
        "phase": phase,
        "event": "terminal",
        "outcome": outcome,
        "physical_gpu": physical_gpu,
        "finished_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "allocated_gpu_seconds": 0.0,
        "return_code": None,
        "reason_code": reason_code,
        "process_pid": None,
    }
    return _write_once(
        _receipt_path(idea_dir, attempt_id, "terminal"), payload)


def record_recovered_compute_terminal(
    idea_dir: Path,
    claim: dict,
    *,
    outcome: str = "interrupted",
    reason_code: str = "startup_recovery",
) -> dict:
    """Close the allocation ledger after recovery proves a trainer stopped.

    The lifecycle recovery path, not this helper, is responsible for proving
    the recorded process group is empty.  This function converts that verdict
    into the same immutable terminal receipt used during normal shutdown, so a
    repaired/resumed attempt can be admitted without discarding crash evidence.
    """
    idea_dir = Path(idea_dir)
    if not isinstance(claim, dict):
        raise ComputeAccountingError("recovery_claim_invalid")
    attempt_id = _token(claim.get("attempt_id", ""), "attempt_id")
    idea_id = _token(idea_dir.name, "idea_id")
    gpu = claim.get("gpu")
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ComputeAccountingError("physical_gpu_invalid")

    start_path = _receipt_path(idea_dir, attempt_id, "start")
    started_at = claim.get("trainer_started_at")
    if start_path.exists():
        try:
            start = json.loads(start_path.read_text(encoding="utf-8"))
            if (not isinstance(start, dict)
                    or start.get("idea_id") != idea_id
                    or start.get("attempt_id") != attempt_id
                    or start.get("event") != "start"):
                raise ValueError("start receipt identity mismatch")
            started_at = start.get("started_at_epoch")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError,
                TypeError, ValueError) as exc:
            raise ComputeAccountingError("recovery_start_invalid") from exc
    try:
        started_at = float(started_at)
    except (TypeError, ValueError) as exc:
        raise ComputeAccountingError("recovery_start_missing") from exc
    if not math.isfinite(started_at) or started_at < 0:
        raise ComputeAccountingError("recovery_start_invalid")

    tp = SimpleNamespace(
        idea_id=idea_id,
        attempt_id=attempt_id,
        gpu=gpu,
        start_time=started_at,
        process=SimpleNamespace(pid=claim.get("trainer_pid")),
    )
    return record_compute_terminal(
        tp, idea_dir, outcome, reason_code,
        phase="training", return_code=None,
    )


def summarize_compute_receipts(results_dir: Path) -> dict:
    """Aggregate framework receipts without trusting trainer metrics."""
    results_dir = Path(results_dir)
    starts = set()
    terminals = set()
    invalid = 0
    total_seconds = 0.0
    by_phase = {}

    for path in results_dir.glob("*/_compute_receipts/*/*.json"):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(receipt, dict):
                raise ValueError("receipt must be a mapping")
            if receipt.get("schema_version") != 1:
                raise ValueError("unsupported schema")
            idea_id = _token(receipt.get("idea_id", ""), "idea_id")
            attempt_id = _token(
                receipt.get("attempt_id", ""), "attempt_id")
            phase = _token(receipt.get("phase", ""), "phase")
            event = _token(receipt.get("event", ""), "event")
            outcome = _token(receipt.get("outcome", ""), "outcome")
            if phase not in _PHASES or outcome not in _OUTCOMES:
                raise ValueError("invalid phase or outcome")
            if event not in {"start", "terminal"}:
                raise ValueError("invalid event")
            if path.name != f"{event}.json" or path.parent.name != attempt_id:
                raise ValueError("receipt path does not match identity")
            key = (idea_id, attempt_id)
            if event == "start":
                if outcome != "started":
                    raise ValueError("start outcome invalid")
                starts.add(key)
                continue
            if outcome == "started":
                raise ValueError("terminal outcome invalid")
            seconds = receipt.get("allocated_gpu_seconds")
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(float(seconds)) or seconds < 0):
                raise ValueError("allocated seconds invalid")
            terminals.add(key)
            total_seconds += float(seconds)
            phase_summary = by_phase.setdefault(phase, {
                "attempts": 0,
                "allocated_gpu_seconds": 0.0,
                "outcomes": {},
            })
            phase_summary["attempts"] += 1
            phase_summary["allocated_gpu_seconds"] += float(seconds)
            phase_summary["outcomes"][outcome] = (
                phase_summary["outcomes"].get(outcome, 0) + 1)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError,
                ComputeAccountingError, ValueError, TypeError):
            invalid += 1

    for phase_summary in by_phase.values():
        phase_summary["allocated_gpu_seconds"] = round(
            phase_summary["allocated_gpu_seconds"], 3)
        phase_summary["outcomes"] = dict(sorted(
            phase_summary["outcomes"].items()))
    return {
        "schema_version": 1,
        "attempts_started": len(starts),
        "attempts_terminal": len(terminals),
        "incomplete_started_attempts": len(starts - terminals),
        "zero_gpu_terminal_attempts": len(terminals - starts),
        "allocated_gpu_seconds_total": round(total_seconds, 3),
        "invalid_receipts": invalid,
        "by_phase": dict(sorted(by_phase.items())),
    }
