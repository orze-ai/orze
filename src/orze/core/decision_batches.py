"""Durable outcome gates for prospective autonomous experiment batches.

The producer stages a content-bound contract before queue append and marks it
admitted afterward. A later producer cycle must reconcile that receipt against
the authoritative Idea Lake and current qualified result artifacts before it
may propose another batch. This module never infers an official rank.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

from orze.core.fs import _fs_lock, _fs_unlock, atomic_write
from orze.core.ideas import IDEA_ID_PATTERN
from orze.core.research_policy import (
    AUTONOMOUS_APPROACH_FAMILIES,
    batch_decision_contract_required,
    validate_batch_decision_contract,
    validate_research_policy_config,
)


_MAX_RECEIPTS = 4096
_MAX_RECEIPT_BYTES = 65536
_TERMINAL_STATES = frozenset({"COMPLETE", "FAILED", "SKIPPED", "ARCHIVED"})
_RESOLVED_STATUSES = frozenset({
    "succeeded", "failed_redirect", "failed_stopped",
})
_RECEIPT_STATUSES = _RESOLVED_STATUSES | frozenset({"staged", "admitted"})
_IDEA_RE = re.compile(IDEA_ID_PATTERN)
_FAMILY_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_COMMON_FIELDS = frozenset({
    "schema", "status", "cycle", "created_at", "identity_sha256",
    "contract", "idea_ids",
})
_ADMISSION_FIELDS = frozenset({"admitted_count", "admitted_at"})
_RESOLUTION_FIELDS = frozenset({
    "resolved_at", "terminal_count", "qualified_success_count",
    "blocked_families", "resolution_sha256",
})


def _now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _control_dir(results_dir: Path) -> Path:
    return (
        Path(results_dir).absolute().parent / ".orze" / "policy"
        / "decision_contracts"
    )


def _identity(contract: dict, idea_ids: list[str]) -> tuple[dict, str]:
    identity = {"contract": contract, "idea_ids": idea_ids}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return identity, digest


def _resolution_hash(payload: Mapping) -> str:
    resolution = {
        "identity_sha256": payload.get("identity_sha256"),
        "status": payload.get("status"),
        "terminal_count": payload.get("terminal_count"),
        "qualified_success_count": payload.get("qualified_success_count"),
        "blocked_families": payload.get("blocked_families"),
    }
    canonical = json.dumps(resolution, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_time(value):
    if (not isinstance(value, str) or not 10 <= len(value) <= 64
            or any(ord(character) < 32 for character in value)):
        return None
    try:
        parsed = _datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != _datetime.timedelta(0):
        return None
    return parsed


def _safe_idea_ids(value) -> list[str] | None:
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        return None
    if any(
        not isinstance(idea_id, str)
        or len(idea_id) > 128
        or _IDEA_RE.fullmatch(idea_id) is None
        for idea_id in value
    ):
        return None
    if len(set(value)) != len(value):
        return None
    return list(value)


def _safe_families(value) -> list[str] | None:
    if not isinstance(value, list) or len(value) > 64:
        return None
    if any(
        not isinstance(family, str) or _FAMILY_RE.fullmatch(family) is None
        for family in value
    ):
        return None
    if value != sorted(set(value)):
        return None
    return list(value)


def _validate_receipt(payload, path: Path, cfg: Mapping) -> str | None:
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        return "decision_receipt_schema_invalid"
    status = payload.get("status")
    if status not in _RECEIPT_STATUSES:
        return "decision_receipt_status_invalid"
    expected_fields = set(_COMMON_FIELDS)
    if status != "staged":
        expected_fields.update(_ADMISSION_FIELDS)
    if status in _RESOLVED_STATUSES:
        expected_fields.update(_RESOLUTION_FIELDS)
    if set(payload) != expected_fields:
        return "decision_receipt_fields_invalid"

    cycle = payload.get("cycle")
    if (isinstance(cycle, bool) or not isinstance(cycle, int)
            or not 0 <= cycle <= 99_999_999):
        return "decision_receipt_cycle_invalid"
    created_at = _parse_time(payload.get("created_at"))
    if created_at is None:
        return "decision_receipt_created_at_invalid"
    idea_ids = _safe_idea_ids(payload.get("idea_ids"))
    if idea_ids is None:
        return "decision_receipt_idea_ids_invalid"
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        return "decision_receipt_contract_invalid"
    experiments = contract.get("max_experiments")
    if (isinstance(experiments, bool) or not isinstance(experiments, int)
            or len(idea_ids) > experiments):
        return "decision_receipt_contract_count_invalid"
    decision_error = validate_batch_decision_contract(
        contract,
        cfg,
        idea_count=experiments,
        qualified_best=contract.get("baseline"),
    )
    if decision_error:
        return "decision_receipt_contract_invalid"
    _, digest = _identity(contract, idea_ids)
    if payload.get("identity_sha256") != digest:
        return "decision_receipt_identity_invalid"
    if path.name != f"cycle-{cycle:08d}-{digest[:16]}.json":
        return "decision_receipt_filename_invalid"

    if status != "staged":
        count = payload.get("admitted_count")
        admitted_at = _parse_time(payload.get("admitted_at"))
        if (isinstance(count, bool) or count != len(idea_ids)
                or admitted_at is None or admitted_at < created_at):
            return "decision_receipt_admission_invalid"
    if status in _RESOLVED_STATUSES:
        terminal = payload.get("terminal_count")
        successes = payload.get("qualified_success_count")
        families = _safe_families(payload.get("blocked_families"))
        if (isinstance(terminal, bool) or terminal != len(idea_ids)
                or isinstance(successes, bool)
                or not isinstance(successes, int)
                or not 0 <= successes <= len(idea_ids)
                or families is None):
            return "decision_receipt_resolution_invalid"
        resolved_at = _parse_time(payload.get("resolved_at"))
        if resolved_at is None or resolved_at < admitted_at:
            return "decision_receipt_resolution_invalid"
        if status == "succeeded" and (successes < 1 or families):
            return "decision_receipt_resolution_invalid"
        if status == "failed_redirect" and (successes != 0 or not families):
            return "decision_receipt_resolution_invalid"
        if status == "failed_stopped" and (successes != 0 or families):
            return "decision_receipt_resolution_invalid"
        if payload.get("resolution_sha256") != _resolution_hash(payload):
            return "decision_receipt_resolution_hash_invalid"
    return None


def _redirected(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
    return False


def _write_verified(path: Path, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write(path, rendered)
    try:
        if path.is_symlink() or path.stat().st_nlink != 1:
            raise OSError("decision_receipt_redirected")
        observed = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OSError("decision_receipt_write_unverified") from exc
    if observed != rendered:
        raise OSError("decision_receipt_write_unverified")


def _load_receipts_locked(
    directory: Path,
    cfg: Mapping,
) -> tuple[list[tuple[Path, dict]], str | None]:
    try:
        paths = sorted(directory.glob("cycle-*.json"))
    except OSError:
        return [], "decision_receipt_directory_unreadable"
    if len(paths) > _MAX_RECEIPTS:
        return [], "decision_receipt_count_exceeded"
    receipts = []
    for path in paths:
        try:
            stat = path.lstat()
            if (path.is_symlink() or not path.is_file() or stat.st_nlink != 1
                    or not 1 <= stat.st_size <= _MAX_RECEIPT_BYTES):
                return [], "decision_receipt_file_invalid"
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return [], "decision_receipt_file_invalid"
        error = _validate_receipt(payload, path, cfg)
        if error:
            return [], error
        receipts.append((path, payload))
    return receipts, None


def _acquire(directory: Path) -> tuple[Path | None, str | None]:
    if _redirected(directory):
        return None, "decision_receipt_directory_redirected"
    lock = directory / ".reconcile.lock"
    try:
        locked = _fs_lock(lock, stale_seconds=300)
    except OSError:
        locked = False
    if not locked:
        return None, "decision_receipt_lock_unavailable"
    return lock, None


def _gate(allowed: bool, reason: str, *, blocked=(), pending=0,
          resolved=0) -> dict:
    return {
        "allow_new_batch": bool(allowed),
        "reason": reason,
        "blocked_families": tuple(sorted(set(blocked))),
        "pending_receipts": int(pending),
        "resolved_receipts": int(resolved),
    }


def _database_path(results_dir: Path, cfg: Mapping) -> Path:
    root = Path(results_dir).absolute().parent
    configured = cfg.get("idea_lake_db")
    path = Path(configured) if isinstance(configured, str) and configured else (
        root / ".orze" / "idea_lake.db"
    )
    return path if path.is_absolute() else root / path


def _passes(value: float, comparator: str, threshold: float) -> bool:
    if comparator == "lt":
        return value < threshold
    if comparator == "lte":
        return value <= threshold
    if comparator == "gt":
        return value > threshold
    return value >= threshold


def reconcile_decision_batches(
    results_dir: Path,
    cfg: Mapping,
    *,
    apply: bool = True,
) -> dict:
    """Resolve admitted batches and decide whether another may be proposed."""
    if not batch_decision_contract_required(cfg):
        return _gate(True, "decision_contract_not_required")
    if validate_research_policy_config(cfg):
        return _gate(False, "decision_contract_policy_invalid")
    directory = _control_dir(results_dir)
    try:
        exists = directory.exists()
    except OSError:
        return _gate(False, "decision_receipt_directory_unreadable")
    if not exists:
        return _gate(True, "decision_contract_ready")

    lock = None
    if apply:
        lock, error = _acquire(directory)
        if lock is None:
            return _gate(False, error or "decision_receipt_lock_unavailable")
    elif _redirected(directory):
        return _gate(False, "decision_receipt_directory_redirected")
    try:
        receipts, error = _load_receipts_locked(directory, cfg)
        if error:
            return _gate(False, error)
        blocked = set()
        admitted = []
        resolved_count = 0
        for path, payload in receipts:
            status = payload["status"]
            if status == "failed_stopped":
                return _gate(
                    False, "decision_contract_stop_active",
                    blocked=blocked, resolved=resolved_count + 1)
            if status == "failed_redirect":
                blocked.update(payload["blocked_families"])
                resolved_count += 1
            elif status == "succeeded":
                resolved_count += 1
            elif status == "staged":
                return _gate(
                    False, "decision_contract_staged_unresolved",
                    blocked=blocked, pending=1, resolved=resolved_count)
            elif status == "admitted":
                admitted.append((path, payload))

        if not admitted:
            return _gate(
                True, "decision_contract_ready", blocked=blocked,
                resolved=resolved_count)
        unresolved_ids = [
            idea_id
            for _, payload in admitted
            for idea_id in payload["idea_ids"]
        ]
        if len(unresolved_ids) > 64 or len(set(unresolved_ids)) != len(
                unresolved_ids):
            return _gate(
                False, "decision_contract_unresolved_set_invalid",
                blocked=blocked, pending=len(admitted),
                resolved=resolved_count)

        from orze.reporting.evidence import (
            authoritative_completed_idea_ids,
            authoritative_idea_lifecycle,
            qualify_authoritative_report_evidence,
        )
        db_path = _database_path(results_dir, cfg)
        lifecycle, lifecycle_reason = authoritative_idea_lifecycle(
            db_path, unresolved_ids)
        if lifecycle_reason != "authoritative_lifecycle_loaded":
            return _gate(
                False, lifecycle_reason, blocked=blocked,
                pending=len(admitted), resolved=resolved_count)
        completed_ids, complete_reason = authoritative_completed_idea_ids(
            db_path)
        if complete_reason != "authoritative_lifecycle_loaded":
            return _gate(
                False, complete_reason, blocked=blocked,
                pending=len(admitted), resolved=resolved_count)

        qualified_values = {}
        for idea_id in unresolved_ids:
            if lifecycle[idea_id]["state"] != "COMPLETE":
                continue
            try:
                _, _, value, evidence_reason = (
                    qualify_authoritative_report_evidence(
                        idea_id, Path(results_dir), cfg, completed_ids)
                )
            except Exception:
                continue
            if evidence_reason in {
                    "authoritative_local_evidence_verified",
                    "benchmark_evidence_verified"}:
                qualified_values[idea_id] = float(value)

        pending_count = 0
        stop_active = False
        for path, payload in admitted:
            states = [
                lifecycle[idea_id]["state"] for idea_id in payload["idea_ids"]
            ]
            if any(state not in _TERMINAL_STATES for state in states):
                pending_count += 1
                continue
            contract = payload["contract"]
            successes = [
                idea_id for idea_id in payload["idea_ids"]
                if idea_id in qualified_values and _passes(
                    qualified_values[idea_id], contract["comparator"],
                    float(contract["threshold"]))
            ]
            resolved = dict(payload)
            resolved.update({
                "resolved_at": _now(),
                "terminal_count": len(states),
                "qualified_success_count": len(successes),
                "blocked_families": [],
            })
            if successes:
                resolved["status"] = "succeeded"
            elif contract["on_failure"] == "redirect_family":
                families = sorted({
                    (
                        lifecycle[idea_id]["family"]
                        if lifecycle[idea_id]["family"]
                        in AUTONOMOUS_APPROACH_FAMILIES
                        else "other"
                    )
                    for idea_id in payload["idea_ids"]
                })
                resolved["status"] = "failed_redirect"
                resolved["blocked_families"] = families
                blocked.update(families)
            else:
                resolved["status"] = "failed_stopped"
                stop_active = True
            resolved["resolution_sha256"] = _resolution_hash(resolved)
            if apply:
                _write_verified(path, resolved)
            resolved_count += 1

        if stop_active:
            return _gate(
                False, "decision_contract_stop_active", blocked=blocked,
                pending=pending_count, resolved=resolved_count)
        if pending_count:
            return _gate(
                False, "decision_contract_batch_pending", blocked=blocked,
                pending=pending_count, resolved=resolved_count)
        return _gate(
            True, "decision_contract_ready", blocked=blocked,
            resolved=resolved_count)
    except (OSError, TypeError, ValueError):
        return _gate(False, "decision_contract_reconciliation_failed")
    finally:
        if lock is not None:
            _fs_unlock(lock)


def stage_decision_contract(
    results_dir: Path,
    cfg: Mapping,
    cycle: int,
    contract: dict,
    ideas: list,
) -> tuple[Path, dict]:
    """Stage one contract only when no earlier batch is unresolved/stopped."""
    if not batch_decision_contract_required(cfg):
        raise ValueError("decision_contract_not_required")
    if (isinstance(cycle, bool) or not isinstance(cycle, int)
            or not 0 <= cycle <= 99_999_999):
        raise ValueError("decision_contract_cycle_invalid")
    if not isinstance(ideas, list) or not ideas:
        raise ValueError("decision_contract_ideas_invalid")
    idea_ids = _safe_idea_ids([
        idea.get("idea_id") if isinstance(idea, dict) else None
        for idea in ideas
    ])
    if idea_ids is None:
        raise ValueError("decision_contract_idea_ids_invalid")
    experiments = contract.get("max_experiments") if isinstance(
        contract, dict) else None
    if (isinstance(experiments, bool) or not isinstance(experiments, int)
            or len(idea_ids) > experiments):
        raise ValueError("decision_contract_count_invalid")
    if validate_batch_decision_contract(
            contract, cfg, idea_count=experiments,
            qualified_best=contract.get("baseline")):
        raise ValueError("decision_contract_invalid")

    directory = _control_dir(results_dir)
    lock, error = _acquire(directory)
    if lock is None:
        raise OSError(error or "decision_receipt_lock_unavailable")
    try:
        receipts, error = _load_receipts_locked(directory, cfg)
        if error:
            raise ValueError(error)
        identity, digest = _identity(contract, idea_ids)
        if any(receipt["identity_sha256"] == digest
               for _, receipt in receipts):
            raise ValueError("decision_contract_duplicate_receipt")
        prior_ids = {
            idea_id
            for _, receipt in receipts
            for idea_id in receipt["idea_ids"]
        }
        if prior_ids.intersection(idea_ids):
            raise ValueError("decision_contract_idea_reused")
        blocked = {
            family
            for _, receipt in receipts
            if receipt["status"] == "failed_redirect"
            for family in receipt["blocked_families"]
        }
        if any(receipt["status"] in {"staged", "admitted"}
               for _, receipt in receipts):
            raise ValueError("decision_contract_previous_batch_unresolved")
        if any(receipt["status"] == "failed_stopped"
               for _, receipt in receipts):
            raise ValueError("decision_contract_stop_active")
        proposed_families = set()
        for idea in ideas:
            family = str(
                (idea.get("approach_family") or "other")
            ).strip().lower()
            proposed_families.add(
                family if family in AUTONOMOUS_APPROACH_FAMILIES else "other")
        if proposed_families & blocked:
            raise ValueError("decision_contract_redirect_not_applied")

        path = directory / f"cycle-{cycle:08d}-{digest[:16]}.json"
        if path.exists() or path.is_symlink():
            raise ValueError("decision_contract_receipt_exists")
        payload = {
            "schema": 1,
            "status": "staged",
            "cycle": cycle,
            "created_at": _now(),
            "identity_sha256": digest,
            **identity,
        }
        error = _validate_receipt(payload, path, cfg)
        if error:
            raise ValueError(error)
        _write_verified(path, payload)
        return path, payload
    finally:
        _fs_unlock(lock)


def admit_decision_contract(
    path: Path,
    payload: dict,
    count: int,
    cfg: Mapping,
) -> None:
    """Atomically mark exactly one staged, content-identical batch admitted."""
    path = Path(path)
    directory = path.parent
    lock, error = _acquire(directory)
    if lock is None:
        raise OSError(error or "decision_receipt_lock_unavailable")
    try:
        receipts, error = _load_receipts_locked(directory, cfg)
        if error:
            raise ValueError(error)
        current = next(
            (receipt for candidate, receipt in receipts if candidate == path),
            None,
        )
        if (not isinstance(current, dict) or current != payload
                or current.get("status") != "staged"):
            raise ValueError("decision_contract_stage_changed")
        expected = payload.get("idea_ids")
        if (not isinstance(expected, list) or not expected
                or isinstance(count, bool) or count != len(expected)):
            raise ValueError("decision_contract_admission_count_mismatch")
        admitted = dict(payload)
        admitted.update({
            "status": "admitted",
            "admitted_count": count,
            "admitted_at": _now(),
        })
        error = _validate_receipt(admitted, path, cfg)
        if error:
            raise ValueError(error)
        _write_verified(path, admitted)
    finally:
        _fs_unlock(lock)


def validate_idea_decision_admission(
    results_dir: Path,
    cfg: Mapping,
    idea_id: str,
) -> str | None:
    """Require one current admitted receipt before any experiment launch."""
    if not batch_decision_contract_required(cfg):
        return None
    if _safe_idea_ids([idea_id]) is None:
        return "decision_contract_launch_idea_id_invalid"
    directory = _control_dir(results_dir)
    try:
        if not directory.exists():
            return "decision_contract_launch_admission_missing"
    except OSError:
        return "decision_contract_launch_receipts_unavailable"
    lock, error = _acquire(directory)
    if lock is None:
        return error or "decision_receipt_lock_unavailable"
    try:
        receipts, error = _load_receipts_locked(directory, cfg)
        if error:
            return error
        matches = [
            payload for _, payload in receipts
            if idea_id in payload["idea_ids"]
        ]
        if not matches:
            return "decision_contract_launch_admission_missing"
        if len(matches) != 1:
            return "decision_contract_launch_admission_ambiguous"
        if matches[0]["status"] != "admitted":
            return "decision_contract_launch_not_admitted"
        return None
    finally:
        _fs_unlock(lock)
