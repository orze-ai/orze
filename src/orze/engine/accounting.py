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
from typing import Mapping, Optional


_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_PHASES = {
    "training", "posthoc", "evaluation", "post_script", "pre_script",
    "admission",
}
_OUTCOMES = {
    "started", "completed", "failed", "interrupted", "rejected", "requeued",
}
_ALLOCATION_DURATION_TOLERANCE_SECONDS = 0.01


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
            "outcome", "physical_gpu", "execution_identity_sha256",
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
    execution_identity = getattr(tp, "execution_identity", None)
    if execution_identity is not None:
        if (not isinstance(execution_identity, str)
                or _HEX64_RE.fullmatch(execution_identity) is None):
            raise ComputeAccountingError("execution_identity_invalid")
        payload["execution_identity_sha256"] = execution_identity
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


def finalize_failed_launch_accounting(
    idea_id: str,
    idea_dir: Path,
    physical_gpu: int,
    reason_code: str,
) -> dict:
    """Close a final launch failure without misclassifying GPU allocation.

    ``launch()`` can fail either before Popen or during initialization after a
    child has started.  The former needs a zero-GPU admission terminal; the
    latter must already have a paired allocation receipt. Any half-written or
    contradictory state fails closed instead of being relabelled zero-GPU.
    """
    idea_id = _token(idea_id, "idea_id")
    idea_dir = Path(idea_dir)
    try:
        claim = json.loads(
            (idea_dir / "claim.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComputeAccountingError("claim_receipt_missing_or_invalid") from exc
    if not isinstance(claim, dict):
        raise ComputeAccountingError("claim_receipt_missing_or_invalid")
    attempt_id = _token(claim.get("attempt_id", ""), "attempt_id")
    if claim.get("gpu") != physical_gpu:
        raise ComputeAccountingError("physical_gpu_claim_mismatch")
    start_path = _receipt_path(idea_dir, attempt_id, "start")
    terminal_path = _receipt_path(idea_dir, attempt_id, "terminal")
    start_exists = start_path.exists()
    terminal_exists = terminal_path.exists()
    if start_exists != terminal_exists:
        raise ComputeAccountingError("failed_launch_receipt_pair_incomplete")
    if not start_exists:
        return record_zero_gpu_outcome(
            idea_id, idea_dir, physical_gpu, "rejected", reason_code,
            phase="admission",
        )
    try:
        start = json.loads(start_path.read_text(encoding="utf-8"))
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComputeAccountingError("failed_launch_receipt_unreadable") from exc
    if (not isinstance(start, dict) or not isinstance(terminal, dict)
            or start.get("idea_id") != idea_id
            or terminal.get("idea_id") != idea_id
            or start.get("attempt_id") != attempt_id
            or terminal.get("attempt_id") != attempt_id
            or start.get("event") != "start"
            or terminal.get("event") != "terminal"
            or start.get("physical_gpu") != physical_gpu
            or terminal.get("physical_gpu") != physical_gpu
            or start.get("phase") not in {"training", "posthoc"}
            or terminal.get("phase") != start.get("phase")):
        raise ComputeAccountingError("failed_launch_receipt_contradictory")
    return terminal


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
        execution_identity=(
            start.get("execution_identity_sha256")
            if start_path.exists() else None
        ),
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

    receipt_dirs = sorted(results_dir.glob("*/_compute_receipts/*"))
    paths = []
    for receipt_dir in receipt_dirs:
        if not receipt_dir.is_dir():
            invalid += 1
            continue
        try:
            json_files = sorted(receipt_dir.glob("*.json"))
        except OSError:
            invalid += 1
            continue
        # ``boundary.json`` is the separately validated model-lineage launch
        # envelope, not a compute event.  Counting it as malformed made every
        # managed training attempt poison the accounting summary. Unknown JSON
        # sidecars still fail closed.
        for sidecar in json_files:
            if sidecar.name not in {"start.json", "terminal.json", "boundary.json"}:
                invalid += 1
        paths.extend(
            receipt_dir / name
            for name in ("start.json", "terminal.json")
            if (receipt_dir / name).exists()
        )

    for path in paths:
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


def audit_campaign_compute_receipts(
    results_dir: Path,
    *,
    idea_ids: list[str],
    start_epoch: float,
    end_epoch: float,
    physical_scope: list[int],
    expected_rejections: list[Mapping],
) -> dict:
    """Audit exact campaign compute evidence without trusting trainer output.

    Unlike the repository-wide operational summary, this verifier requires a
    closed idea set, time/GPU scope, and exact preregistered rejection
    descriptors. Every expected idea must have terminal accounting, every
    allocated start must close, paired receipts must agree, and zero-GPU
    outcomes must be genuine pre-allocation rejections/requeues.
    """
    if (not isinstance(idea_ids, list) or not idea_ids
            or len(idea_ids) != len(set(idea_ids))):
        raise ComputeAccountingError("campaign_idea_ids_invalid")
    expected = {_token(idea_id, "idea_id") for idea_id in idea_ids}
    if (isinstance(start_epoch, bool) or isinstance(end_epoch, bool)
            or not isinstance(start_epoch, (int, float))
            or not isinstance(end_epoch, (int, float))
            or not math.isfinite(float(start_epoch))
            or not math.isfinite(float(end_epoch))
            or not 0 <= start_epoch < end_epoch):
        raise ComputeAccountingError("campaign_window_invalid")
    if (not isinstance(physical_scope, list) or not physical_scope
            or len(physical_scope) != len(set(physical_scope))
            or any(isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0
                   for gpu in physical_scope)):
        raise ComputeAccountingError("campaign_physical_scope_invalid")
    allowed = set(physical_scope)
    if not isinstance(expected_rejections, list):
        raise ComputeAccountingError("expected_rejections_invalid")
    expected_rejection_keys = []
    for item in expected_rejections:
        if (not isinstance(item, Mapping)
                or set(item) != {"idea_id", "phase", "reason_code"}):
            raise ComputeAccountingError("expected_rejections_invalid")
        idea_id = _token(item["idea_id"], "idea_id")
        phase = _token(item["phase"], "phase")
        reason_code = _token(item["reason_code"], "reason_code")
        if idea_id not in expected or phase not in _PHASES:
            raise ComputeAccountingError("expected_rejections_invalid")
        expected_rejection_keys.append((idea_id, phase, reason_code))
    if len(expected_rejection_keys) != len(set(expected_rejection_keys)):
        raise ComputeAccountingError("expected_rejections_duplicate")
    expected_rejection_set = set(expected_rejection_keys)

    invalid = 0
    out_of_scope = 0
    events = {}
    unexpected_sidecars = 0
    unexpected_ideas = set()
    # The preregistered decision set is an exact compute universe, not a filter
    # that may hide losing experiments. Any other framework allocation whose
    # start/terminal falls inside the campaign window invalidates the receipt.
    for path in Path(results_dir).glob("*/_compute_receipts/*/*.json"):
        idea_id = path.parents[2].name
        if idea_id in expected or path.name not in {"start.json", "terminal.json"}:
            continue
        at = None
        try:
            info = path.lstat()
            if (path.is_symlink() or not path.is_file()
                    or info.st_nlink != 1
                    or not 1 <= info.st_size <= 1024 * 1024):
                raise OSError("unsafe compute receipt")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if path.name == "start.json":
                at = payload.get("started_at_epoch")
            else:
                at = _parse_receipt_time(payload.get("finished_at"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError,
                AttributeError, TypeError, ValueError):
            # Modification time is only a conservative inclusion fallback; it
            # can invalidate evidence but can never make it pass.
            try:
                at = path.lstat().st_mtime
            except OSError:
                at = start_epoch
        if (isinstance(at, (int, float)) and not isinstance(at, bool)
                and math.isfinite(float(at))
                and start_epoch <= float(at) <= end_epoch):
            unexpected_ideas.add(idea_id)

    for idea_id in sorted(expected):
        idea_dir = Path(results_dir) / idea_id
        receipt_root = idea_dir / "_compute_receipts"
        if not receipt_root.exists():
            continue
        if idea_dir.is_symlink() or receipt_root.is_symlink():
            invalid += 1
            continue
        try:
            receipt_dirs = sorted(receipt_root.iterdir())
        except OSError:
            invalid += 1
            continue
        for receipt_dir in receipt_dirs:
            if not receipt_dir.is_dir() or receipt_dir.is_symlink():
                invalid += 1
                continue
            attempt_id = receipt_dir.name
            try:
                _token(attempt_id, "attempt_id")
                json_files = sorted(receipt_dir.glob("*.json"))
            except (ComputeAccountingError, OSError):
                invalid += 1
                continue
            for sidecar in json_files:
                if sidecar.name not in {
                    "start.json", "terminal.json", "boundary.json",
                }:
                    unexpected_sidecars += 1
                    invalid += 1
            for event in ("start", "terminal"):
                path = receipt_dir / f"{event}.json"
                if not path.exists():
                    continue
                try:
                    info = path.lstat()
                    if (path.is_symlink() or not path.is_file()
                            or info.st_nlink != 1
                            or not 1 <= info.st_size <= 1024 * 1024):
                        raise OSError("unsafe compute receipt")
                    receipt = json.loads(path.read_text(encoding="utf-8"))
                    if (not isinstance(receipt, dict)
                            or receipt.get("schema_version") != 1
                            or receipt.get("idea_id") != idea_id
                            or receipt.get("attempt_id") != attempt_id
                            or receipt.get("event") != event):
                        raise ValueError
                    phase = _token(receipt.get("phase", ""), "phase")
                    outcome = _token(receipt.get("outcome", ""), "outcome")
                    if event == "terminal":
                        _token(receipt.get("reason_code", ""), "reason_code")
                    if phase not in _PHASES or outcome not in _OUTCOMES:
                        raise ValueError
                    execution_identity = receipt.get(
                        "execution_identity_sha256"
                    )
                    if (phase == "training"
                            and (not isinstance(execution_identity, str)
                                 or _HEX64_RE.fullmatch(
                                     execution_identity
                                 ) is None)):
                        raise ValueError
                    gpu = receipt.get("physical_gpu")
                    if (isinstance(gpu, bool) or not isinstance(gpu, int)
                            or gpu < 0):
                        raise ValueError
                    if gpu not in allowed:
                        out_of_scope += 1
                    if event == "start":
                        if outcome != "started":
                            raise ValueError
                        at = receipt.get("started_at_epoch")
                    else:
                        if outcome == "started":
                            raise ValueError
                        seconds = receipt.get("allocated_gpu_seconds")
                        if (isinstance(seconds, bool)
                                or not isinstance(seconds, (int, float))
                                or not math.isfinite(float(seconds))
                                or seconds < 0):
                            raise ValueError
                        at = _parse_receipt_time(receipt.get("finished_at"))
                    if (isinstance(at, bool)
                            or not isinstance(at, (int, float))
                            or not math.isfinite(float(at))
                            or not start_epoch <= float(at) <= end_epoch):
                        raise ValueError
                    key = (idea_id, attempt_id)
                    bucket = events.setdefault(key, {})
                    if event in bucket:
                        raise ValueError
                    bucket[event] = receipt
                except (OSError, UnicodeDecodeError, json.JSONDecodeError,
                        ComputeAccountingError, TypeError, ValueError):
                    invalid += 1

    incomplete = 0
    zero_gpu = 0
    zero_gpu_valid = 0
    rejection_attempts = 0
    zero_gpu_rejections = 0
    valid_zero_gpu_rejection_keys = []
    allocated_seconds = 0.0
    recorded_allocated_seconds = 0.0
    allocation_duration_mismatches = []
    terminal_ideas = set()
    training_starts = {}
    training_execution_starts = {}
    by_phase = {}
    for (idea_id, _attempt_id), pair in events.items():
        start = pair.get("start")
        terminal = pair.get("terminal")
        if start is not None:
            if start["phase"] == "training":
                training_starts[idea_id] = training_starts.get(idea_id, 0) + 1
                execution_identity = start["execution_identity_sha256"]
                training_execution_starts[execution_identity] = (
                    training_execution_starts.get(execution_identity, 0) + 1
                )
            if terminal is None:
                incomplete += 1
                continue
        if terminal is None:
            continue
        terminal_ideas.add(idea_id)
        seconds = float(terminal["allocated_gpu_seconds"])
        allocated_seconds += seconds
        recorded_allocated_seconds += seconds
        phase = terminal["phase"]
        phase_summary = by_phase.setdefault(phase, {
            "attempts": 0,
            "allocated_gpu_seconds": 0.0,
            "outcomes": {},
        })
        phase_summary["attempts"] += 1
        phase_summary["allocated_gpu_seconds"] += seconds
        outcome = terminal["outcome"]
        if outcome == "rejected":
            rejection_attempts += 1
        phase_summary["outcomes"][outcome] = (
            phase_summary["outcomes"].get(outcome, 0) + 1
        )
        if start is None:
            zero_gpu += 1
            if (phase in {"admission", "pre_script"}
                    and outcome in {"rejected", "requeued"}
                    and seconds == 0.0
                    and terminal.get("process_pid") is None
                    and "started_at_epoch" not in terminal):
                zero_gpu_valid += 1
                if outcome == "rejected":
                    zero_gpu_rejections += 1
                    valid_zero_gpu_rejection_keys.append((
                        idea_id, phase, terminal["reason_code"],
                    ))
            else:
                invalid += 1
            continue
        if (start["idea_id"] != terminal["idea_id"]
                or start["attempt_id"] != terminal["attempt_id"]
                or start["phase"] != terminal["phase"]
                or start["physical_gpu"] != terminal["physical_gpu"]
                or start.get("execution_identity_sha256")
                != terminal.get("execution_identity_sha256")
                or terminal.get("started_at_epoch")
                != start.get("started_at_epoch")):
            invalid += 1
            continue
        finished_at_epoch = _parse_receipt_time(terminal.get("finished_at"))
        timestamp_seconds = (
            finished_at_epoch - float(start["started_at_epoch"])
            if finished_at_epoch is not None else float("nan")
        )
        accounted_seconds = max(0.0, timestamp_seconds)
        allocated_seconds += accounted_seconds - seconds
        phase_summary["allocated_gpu_seconds"] += (
            accounted_seconds - seconds
        )
        if (not math.isfinite(timestamp_seconds)
                or timestamp_seconds < 0
                or abs(seconds - timestamp_seconds)
                > _ALLOCATION_DURATION_TOLERANCE_SECONDS):
            invalid += 1
            allocation_duration_mismatches.append({
                "idea_id": idea_id,
                "attempt_id": _attempt_id,
                "recorded_seconds": seconds,
                "timestamp_seconds": (
                    round(timestamp_seconds, 3)
                    if math.isfinite(timestamp_seconds) else None
                ),
            })

    for phase_summary in by_phase.values():
        phase_summary["allocated_gpu_seconds"] = round(
            phase_summary["allocated_gpu_seconds"], 3
        )
        phase_summary["outcomes"] = dict(sorted(
            phase_summary["outcomes"].items()
        ))
    duplicate_training_attempts = sum(
        max(0, count - 1) for count in training_execution_starts.values()
    )
    observed_expected_rejections = {}
    unexpected_rejection_attempts = []
    for (idea_id, attempt_id), pair in events.items():
        terminal = pair.get("terminal")
        if terminal is None:
            continue
        key = (idea_id, terminal["phase"], terminal["reason_code"])
        if key in expected_rejection_set:
            observed_expected_rejections.setdefault(key, []).append({
                "idea_id": idea_id,
                "attempt_id": attempt_id,
                "phase": terminal["phase"],
                "reason_code": terminal["reason_code"],
                "outcome": terminal["outcome"],
                "allocated": pair.get("start") is not None,
            })
        elif terminal["outcome"] == "rejected":
            unexpected_rejection_attempts.append({
                "idea_id": idea_id,
                "attempt_id": attempt_id,
                "phase": terminal["phase"],
                "reason_code": terminal["reason_code"],
                "allocated": pair.get("start") is not None,
            })
    missing_expected_rejections = [
        {"idea_id": idea_id, "phase": phase, "reason_code": reason_code}
        for idea_id, phase, reason_code in expected_rejection_keys
        if (idea_id, phase, reason_code) not in observed_expected_rejections
    ]
    duplicate_expected_rejections = [
        event
        for key in expected_rejection_keys
        for event in observed_expected_rejections.get(key, [])[1:]
    ]
    expected_rejection_observations = [
        event
        for key in expected_rejection_keys
        for event in observed_expected_rejections.get(key, [])
    ]
    valid_zero_gpu_rejection_set = set(valid_zero_gpu_rejection_keys)
    matched_zero_gpu_rejections = sum(
        key in valid_zero_gpu_rejection_set for key in expected_rejection_keys
    )
    rejection_contract_complete = (
        not missing_expected_rejections
        and not duplicate_expected_rejections
        and not unexpected_rejection_attempts
    )
    missing_terminal_ideas = sorted(expected - terminal_ideas)
    evidence_complete = (
        invalid == 0
        and out_of_scope == 0
        and incomplete == 0
        and not missing_terminal_ideas
        and not unexpected_ideas
        and zero_gpu == zero_gpu_valid
        and rejection_contract_complete
    )
    return {
        "schema_version": 1,
        "status": "VERIFIED" if evidence_complete else "UNVERIFIED",
        "expected_idea_count": len(expected),
        "ideas_with_terminal_evidence": len(terminal_ideas),
        "missing_terminal_ideas": missing_terminal_ideas,
        "unexpected_campaign_idea_ids": sorted(unexpected_ideas),
        "attempts_started": sum(1 for pair in events.values() if "start" in pair),
        "attempts_terminal": sum(
            1 for pair in events.values() if "terminal" in pair
        ),
        "incomplete_started_attempts": incomplete,
        "zero_gpu_terminal_attempts": zero_gpu,
        "valid_zero_gpu_terminal_attempts": zero_gpu_valid,
        "expected_rejection_attempts": len(expected_rejection_keys),
        "observed_rejection_outcome_attempts": rejection_attempts,
        "observed_zero_gpu_rejection_outcome_attempts": zero_gpu_rejections,
        "rejection_attempts": len(expected_rejection_keys),
        "zero_gpu_rejection_attempts": matched_zero_gpu_rejections,
        "zero_gpu_rejection_rate": (
            matched_zero_gpu_rejections / len(expected_rejection_keys)
            if expected_rejection_keys else None
        ),
        "rejection_contract_complete": rejection_contract_complete,
        "missing_expected_rejections": missing_expected_rejections,
        "duplicate_expected_rejections": duplicate_expected_rejections,
        "unexpected_rejection_attempts": unexpected_rejection_attempts,
        "expected_rejection_observations": expected_rejection_observations,
        "allocated_expected_rejection_attempts": sum(
            event["allocated"] for event in expected_rejection_observations
        ),
        "out_of_scope_receipts": out_of_scope,
        "invalid_receipts": invalid,
        "unexpected_sidecars": unexpected_sidecars,
        "allocated_gpu_seconds_total": round(allocated_seconds, 3),
        "recorded_allocated_gpu_seconds_total": round(
            recorded_allocated_seconds, 3
        ),
        "allocation_duration_mismatch_attempts": (
            allocation_duration_mismatches
        ),
        "training_start_counts_by_idea": dict(sorted(training_starts.items())),
        "training_start_counts_by_execution_identity": dict(sorted(
            training_execution_starts.items()
        )),
        "duplicate_training_attempts": duplicate_training_attempts,
        "by_phase": dict(sorted(by_phase.items())),
    }


def _parse_receipt_time(value) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()
