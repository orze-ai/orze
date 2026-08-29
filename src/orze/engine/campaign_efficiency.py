"""Local, fail-closed evidence for scheduler and GPU campaign efficiency.

This module never launches work.  Sampling queries only the caller-provided
physical GPU scope and stores no idea IDs, configs, model outputs, or metrics.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from orze.core.fs import atomic_write
from orze.core.ideas import IDEA_ID_PATTERN
from orze.engine.reproducibility import validate_reproducibility_contract
from orze.hardware.gpu import _query_gpu_details
from orze.idea_lake import IdeaLake


DEFAULT_CAMPAIGN_TARGETS = {
    "max_sample_gap_poll_intervals": 2.0,
    "min_allocation_duty_cycle": 0.90,
    "max_queue_to_claim_poll_intervals": 2.0,
    "max_terminal_to_next_claim_poll_intervals": 1.0,
    "max_operator_update_gap_seconds": 600.0,
}

DEFAULT_OUTCOME_TARGETS = {
    "max_time_to_first_decision_seconds": 14_400.0,
    "max_time_to_all_decisions_seconds": 86_400.0,
    "min_qualified_success_rate": 0.25,
    "max_gpu_hours_per_qualified_success": 8.0,
    "max_duplicate_training_attempts": 0,
    "min_zero_gpu_rejection_rate": 1.0,
}

_IDEA_RE = re.compile(IDEA_ID_PATTERN)
_CAMPAIGN_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_CONTROLLER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_MAX_CAMPAIGN_IDEAS = 4096
_PROGRESS_BLOCKERS = frozenset({
    "disk_unavailable",
    "eligible_queue_waiting",
    "evaluation_active",
    "evaluation_queued",
    "launcher_paused",
    "no_eligible_work",
    "training_active",
})


def capture_campaign_efficiency_sample(
    lake: IdeaLake,
    *,
    campaign_id: Optional[str],
    controller_id: str,
    host: str,
    iteration: int,
    poll_seconds: float,
    physical_scope: List[int],
    active_training_gpus: Iterable[int],
    active_evaluation_gpus: Iterable[int],
    remaining_training: int,
    remaining_evaluation: int,
    launcher_paused: bool,
    disk_ok: bool,
    observed_at_epoch: Optional[float] = None,
    telemetry_query: Callable[[Optional[List[int]]], List[dict]] = (
        _query_gpu_details
    ),
) -> bool:
    """Query exactly ``physical_scope`` and persist one scheduler sample.

    Query failures become incomplete rows.  They are not omitted, retried with
    an unscoped query, or converted into zero utilization.
    """
    scope = list(physical_scope)
    try:
        telemetry = telemetry_query(scope)
    except Exception:
        telemetry = []
    if not isinstance(telemetry, list):
        telemetry = []
    return lake.record_harness_efficiency_sample(
        campaign_id=campaign_id,
        controller_id=controller_id,
        host=host,
        iteration=iteration,
        observed_at_epoch=(
            time.time() if observed_at_epoch is None else observed_at_epoch
        ),
        poll_seconds=poll_seconds,
        physical_scope=scope,
        gpu_telemetry=telemetry,
        active_training_gpus=sorted(active_training_gpus),
        active_evaluation_gpus=sorted(active_evaluation_gpus),
        remaining_training=remaining_training,
        remaining_evaluation=remaining_evaluation,
        launcher_paused=launcher_paused,
        disk_ok=disk_ok,
    )


def _qualified_artifact_identity(
    completed_rows: Iterable[dict], primary_metric: str
) -> tuple[Optional[str], Optional[str]]:
    """Return a content identity for the current best qualified result row."""
    if not isinstance(primary_metric, str) or not primary_metric:
        return None, None
    for row in completed_rows:
        if not isinstance(row, dict) or row.get("evidence_qualified") is not True:
            continue
        idea_id = row.get("id")
        value = row.get("primary_val")
        reason = row.get("evidence_reason")
        evidence_sha = row.get("evidence_sha256")
        if (_IDEA_RE.fullmatch(idea_id) is None
                if isinstance(idea_id, str) else True):
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not isinstance(reason, str) or not reason
                or not isinstance(evidence_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", evidence_sha) is None):
            continue
        return idea_id, evidence_sha
    return None, None


def _write_progress_file(path: Path, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        try:
            stat = path.lstat()
            if path.is_symlink() or stat.st_nlink != 1:
                raise OSError("campaign_progress_file_redirected")
            if path.read_text(encoding="utf-8") != rendered:
                raise OSError("campaign_progress_identity_conflict")
            return
        except UnicodeDecodeError as exc:
            raise OSError("campaign_progress_file_invalid") from exc
    atomic_write(path, rendered)
    stat = path.lstat()
    if path.is_symlink() or stat.st_nlink != 1:
        raise OSError("campaign_progress_file_redirected")
    if path.read_text(encoding="utf-8") != rendered:
        raise OSError("campaign_progress_write_unverified")


def _progress_path_redirected(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            if current.exists() and current.is_symlink():
                return True
        except OSError:
            return True
    return False


def capture_campaign_progress_update(
    lake: IdeaLake,
    *,
    results_dir: str | Path,
    campaign_id: str,
    controller_id: str,
    host: str,
    iteration: int,
    completed_rows: Iterable[dict],
    primary_metric: str,
    blocker_code: str,
    observed_at_epoch: Optional[float] = None,
) -> Optional[dict]:
    """Publish one operator-visible update and its immutable DB evidence."""
    if (_CAMPAIGN_RE.fullmatch(campaign_id) is None
            if isinstance(campaign_id, str) else True):
        raise ValueError("campaign progress campaign_id is invalid")
    if (_CONTROLLER_RE.fullmatch(controller_id) is None
            if isinstance(controller_id, str) else True):
        raise ValueError("campaign progress controller_id is invalid")
    if (not isinstance(host, str) or not host.strip()
            or any(ord(char) < 32 for char in host)):
        raise ValueError("campaign progress host is invalid")
    if (isinstance(iteration, bool) or not isinstance(iteration, int)
            or iteration < 0):
        raise ValueError("campaign progress iteration is invalid")
    if blocker_code not in _PROGRESS_BLOCKERS:
        raise ValueError("campaign progress blocker_code is invalid")
    now = time.time() if observed_at_epoch is None else observed_at_epoch
    if (isinstance(now, bool) or not isinstance(now, (int, float))
            or not math.isfinite(float(now)) or now <= 0):
        raise ValueError("campaign progress observed_at_epoch is invalid")
    row = lake.conn.execute(
        "SELECT manifest_json FROM harness_campaign_registrations "
        "WHERE campaign_id = ?", (campaign_id,),
    ).fetchone()
    if row is None:
        raise ValueError("campaign progress registration is missing")
    try:
        manifest = json.loads(row["manifest_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("campaign progress registration is invalid") from exc
    manifest_validation_epoch = float(manifest.get("start_epoch", 0)) - 1.0
    if _manifest_error(
            manifest, manifest_validation_epoch, require_ended=False) is not None:
        raise ValueError("campaign progress manifest is invalid")
    start = float(manifest["start_epoch"])
    end = float(manifest["end_epoch"])
    if not start <= float(now) <= end:
        return None
    max_gap = float(manifest["targets"]["max_operator_update_gap_seconds"])
    next_deadline = min(end, float(now) + max_gap)
    artifact_id, artifact_sha = _qualified_artifact_identity(
        completed_rows, primary_metric
    )
    artifact_at = None
    if artifact_sha is not None:
        prior = lake.conn.execute(
            "SELECT last_valid_artifact_sha256, "
            "last_valid_artifact_at_epoch FROM harness_campaign_progress "
            "WHERE campaign_id = ? ORDER BY observed_at_epoch DESC, id DESC "
            "LIMIT 1", (campaign_id,),
        ).fetchone()
        artifact_at = (
            float(prior["last_valid_artifact_at_epoch"])
            if prior is not None
            and prior["last_valid_artifact_sha256"] == artifact_sha
            and prior["last_valid_artifact_at_epoch"] is not None
            else float(now)
        )
    core = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "controller_id": controller_id,
        "host": host,
        "iteration": iteration,
        "observed_at_epoch": float(now),
        "observed_at": datetime.datetime.fromtimestamp(
            float(now), datetime.timezone.utc
        ).isoformat(),
        "last_valid_artifact_sha256": artifact_sha,
        "last_valid_artifact_idea_id": artifact_id,
        "last_valid_artifact_at_epoch": artifact_at,
        "blocker_code": blocker_code,
        "next_deadline_epoch": next_deadline,
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"))
    payload = {
        **core,
        "update_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    results_path = Path(results_dir).absolute()
    if _progress_path_redirected(results_path):
        raise OSError("campaign_progress_directory_redirected")
    progress_root = results_path / "_campaign_progress"
    if _progress_path_redirected(progress_root):
        raise OSError("campaign_progress_directory_redirected")
    progress_dir = progress_root / campaign_id
    progress_dir.mkdir(parents=True, exist_ok=True)
    if _progress_path_redirected(progress_dir):
        raise OSError("campaign_progress_directory_redirected")
    update_path = progress_dir / f"{controller_id}-{iteration:08d}.json"
    _write_progress_file(update_path, payload)
    latest = progress_dir / "latest.json"
    latest_rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write(latest, latest_rendered)
    latest_stat = latest.lstat()
    if latest.is_symlink() or latest_stat.st_nlink != 1:
        raise OSError("campaign_progress_latest_redirected")
    if latest.read_text(encoding="utf-8") != latest_rendered:
        raise OSError("campaign_progress_latest_write_unverified")
    inserted = lake.record_harness_campaign_progress(
        campaign_id=campaign_id,
        controller_id=controller_id,
        host=host,
        iteration=iteration,
        observed_at_epoch=float(now),
        last_valid_artifact_sha256=artifact_sha,
        last_valid_artifact_idea_id=artifact_id,
        last_valid_artifact_at_epoch=artifact_at,
        blocker_code=blocker_code,
        next_deadline_epoch=next_deadline,
    )
    if not inserted:
        existing = lake.conn.execute(
            "SELECT observed_at_epoch, last_valid_artifact_sha256, "
            "last_valid_artifact_idea_id, last_valid_artifact_at_epoch, "
            "blocker_code, next_deadline_epoch FROM harness_campaign_progress "
            "WHERE campaign_id = ? AND controller_id = ? AND iteration = ?",
            (campaign_id, controller_id, iteration),
        ).fetchone()
        expected = (
            float(now), artifact_sha, artifact_id, artifact_at,
            blocker_code, next_deadline,
        )
        observed = tuple(existing) if existing is not None else None
        if observed != expected:
            raise OSError("campaign_progress_database_identity_conflict")
    return payload


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _epoch(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError):
        return None


def _manifest_error(
    manifest: dict,
    now_epoch: float,
    *,
    require_ended: bool,
) -> Optional[str]:
    required = {
        "campaign_id", "start_epoch", "end_epoch",
        "physical_scope", "poll_seconds", "minimum_samples",
        "minimum_claims", "minimum_release_to_claim_pairs", "targets",
        "expected_idea_ids",
    }
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        return "manifest is missing required fields"
    if (not isinstance(manifest["campaign_id"], str)
            or _CAMPAIGN_RE.fullmatch(manifest["campaign_id"]) is None):
        return "campaign_id must be a safe non-empty identifier"
    idea_ids = manifest["expected_idea_ids"]
    if (not isinstance(idea_ids, list)
            or not 1 <= len(idea_ids) <= _MAX_CAMPAIGN_IDEAS
            or len(idea_ids) != len(set(idea_ids))
            or any(not isinstance(idea_id, str)
                   or _IDEA_RE.fullmatch(idea_id) is None
                   for idea_id in idea_ids)):
        return "expected_idea_ids must contain unique valid idea IDs"
    numeric = ("start_epoch", "end_epoch", "poll_seconds")
    if any(isinstance(manifest[key], bool)
           or not isinstance(manifest[key], (int, float))
           or not math.isfinite(float(manifest[key])) for key in numeric):
        return "manifest timestamps and poll_seconds must be finite numbers"
    if manifest["poll_seconds"] <= 0:
        return "poll_seconds must be positive"
    if not manifest["start_epoch"] < manifest["end_epoch"]:
        return "campaign window is invalid"
    if require_ended and manifest["end_epoch"] > now_epoch:
        return "campaign has not ended"
    if not require_ended and manifest["start_epoch"] <= now_epoch:
        return "campaign registration must occur before its start"
    scope = manifest["physical_scope"]
    if (not isinstance(scope, list) or not scope
            or any(isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0
                   for gpu in scope)
            or len(scope) != len(set(scope))):
        return "physical_scope must contain unique non-negative GPU IDs"
    for key in ("minimum_samples", "minimum_claims", "minimum_release_to_claim_pairs"):
        value = manifest[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return f"{key} must be a positive integer"
    targets = manifest["targets"]
    if not isinstance(targets, dict) or set(targets) != set(DEFAULT_CAMPAIGN_TARGETS):
        return "targets must exactly match the supported target names"
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) or value < 0
           for value in targets.values()):
        return "targets must be finite non-negative numbers"
    if not 0 <= targets["min_allocation_duty_cycle"] <= 1:
        return "min_allocation_duty_cycle must be between zero and one"
    for key in (
        "max_sample_gap_poll_intervals",
        "max_queue_to_claim_poll_intervals",
        "max_terminal_to_next_claim_poll_intervals",
        "max_operator_update_gap_seconds",
    ):
        if targets[key] > DEFAULT_CAMPAIGN_TARGETS[key]:
            return f"{key} cannot be weaker than the default target"
    if (targets["min_allocation_duty_cycle"]
            < DEFAULT_CAMPAIGN_TARGETS["min_allocation_duty_cycle"]):
        return "min_allocation_duty_cycle cannot be weaker than the default target"
    outcome = manifest.get("outcome_contract")
    if outcome is not None:
        if (not isinstance(outcome, dict)
                or set(outcome) != {
                    "expected_decision_identity_sha256",
                    "artifact_relation",
                    "reproducibility_contract",
                    "targets",
                }):
            return "outcome_contract fields are invalid"
        identities = outcome.get("expected_decision_identity_sha256")
        if (not isinstance(identities, list) or not identities
                or len(identities) != len(set(identities))
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in "0123456789abcdef" for char in value)
                       for value in identities)):
            return "outcome_contract decision identities are invalid"
        if outcome.get("artifact_relation") not in {
                "identical", "distinct", "any"}:
            return "outcome_contract artifact_relation is invalid"
        reproduction_error = validate_reproducibility_contract(
            outcome.get("reproducibility_contract"),
            manifest["expected_idea_ids"],
        )
        if reproduction_error:
            return reproduction_error
        outcome_targets = outcome.get("targets")
        if (not isinstance(outcome_targets, dict)
                or set(outcome_targets) != set(DEFAULT_OUTCOME_TARGETS)
                or any(isinstance(value, bool)
                       or not isinstance(value, (int, float))
                       or not math.isfinite(float(value)) or value < 0
                       for value in outcome_targets.values())):
            return "outcome_contract targets are invalid"
        duplicates = outcome_targets["max_duplicate_training_attempts"]
        if isinstance(duplicates, bool) or not isinstance(duplicates, int):
            return "outcome_contract max_duplicate_training_attempts must be integer"
        for key in (
            "max_time_to_first_decision_seconds",
            "max_time_to_all_decisions_seconds",
            "max_gpu_hours_per_qualified_success",
            "max_duplicate_training_attempts",
        ):
            if outcome_targets[key] > DEFAULT_OUTCOME_TARGETS[key]:
                return f"outcome_contract {key} cannot be weaker than default"
        for key in (
            "min_qualified_success_rate",
            "min_zero_gpu_rejection_rate",
        ):
            if (not 0 <= outcome_targets[key] <= 1
                    or outcome_targets[key] < DEFAULT_OUTCOME_TARGETS[key]):
                return f"outcome_contract {key} cannot be weaker than default"
    return None


def _canonical_manifest(manifest: Dict[str, Any]) -> str:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def preregister_campaign(
    db_path: str | Path,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Register a canonical manifest once through the public API before start."""
    registered_at_epoch = time.time()
    error = _manifest_error(
        manifest, registered_at_epoch, require_ended=False
    )
    if error:
        raise ValueError(error)
    canonical = _canonical_manifest(manifest)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    registered_at = datetime.datetime.fromtimestamp(
        registered_at_epoch, datetime.timezone.utc
    ).isoformat()
    lake = IdeaLake(str(db_path))
    try:
        try:
            lake.conn.execute(
                "INSERT INTO harness_campaign_registrations "
                "(campaign_id, manifest_sha256, manifest_json, "
                "registered_at_epoch, registered_at) VALUES (?, ?, ?, ?, ?)",
                (
                    manifest["campaign_id"], digest, canonical,
                    registered_at_epoch, registered_at,
                ),
            )
            lake.conn.commit()
        except Exception:
            lake.conn.rollback()
            raise
    finally:
        lake.close()
    return {
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": digest,
        "registered_at_epoch": registered_at_epoch,
        "registered_at": registered_at,
    }


def verify_campaign_registration(
    db_path: str | Path,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify the exact canonical manifest against its write-once DB row."""
    canonical = _canonical_manifest(manifest)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    lake = IdeaLake(str(db_path))
    try:
        row = lake.conn.execute(
            "SELECT manifest_sha256, manifest_json, registered_at_epoch, "
            "registered_at FROM harness_campaign_registrations "
            "WHERE campaign_id = ?",
            (manifest.get("campaign_id"),),
        ).fetchone()
    finally:
        lake.close()
    valid = bool(
        row
        and row["manifest_sha256"] == digest
        and row["manifest_json"] == canonical
        and row["registered_at_epoch"] <= manifest.get("start_epoch", -1)
    )
    return {
        "valid": valid,
        "manifest_sha256": digest,
        "registered_at_epoch": row["registered_at_epoch"] if row else None,
        "registered_at": row["registered_at"] if row else None,
    }


def require_active_campaign_registration(
    lake: IdeaLake,
    *,
    campaign_id: str,
    physical_scope: List[int],
    poll_seconds: float,
    now_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    """Require an exact preregistered active campaign before any new launch."""
    now = time.time() if now_epoch is None else float(now_epoch)
    row = lake.conn.execute(
        "SELECT manifest_sha256, manifest_json, registered_at_epoch "
        "FROM harness_campaign_registrations WHERE campaign_id = ?",
        (campaign_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("campaign_evidence_registration_missing")
    try:
        manifest = json.loads(row["manifest_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("campaign_evidence_registration_invalid") from exc
    validation_epoch = float(manifest.get("start_epoch", 0)) - 1.0
    error = _manifest_error(
        manifest, validation_epoch, require_ended=False,
    )
    canonical = _canonical_manifest(manifest)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if (error is not None
            or manifest.get("campaign_id") != campaign_id
            or row["manifest_json"] != canonical
            or row["manifest_sha256"] != digest
            or float(row["registered_at_epoch"])
            > float(manifest.get("start_epoch", -1))):
        raise RuntimeError("campaign_evidence_registration_invalid")
    if not float(manifest["start_epoch"]) <= now <= float(manifest["end_epoch"]):
        raise RuntimeError("campaign_evidence_window_inactive")
    if manifest["physical_scope"] != list(physical_scope):
        raise RuntimeError("campaign_evidence_physical_scope_mismatch")
    if float(manifest["poll_seconds"]) != float(poll_seconds):
        raise RuntimeError("campaign_evidence_poll_seconds_mismatch")
    return {
        "campaign_id": campaign_id,
        "manifest_sha256": digest,
        "expected_idea_count": len(manifest["expected_idea_ids"]),
        "start_epoch": float(manifest["start_epoch"]),
        "end_epoch": float(manifest["end_epoch"]),
        "physical_scope": list(physical_scope),
        "poll_seconds": float(poll_seconds),
    }


def analyze_campaign(
    db_path: str | Path,
    manifest: Dict[str, Any],
    *,
    now_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    """Analyze a preregistered completed campaign and return a JSON receipt."""
    now = time.time() if now_epoch is None else float(now_epoch)
    error = _manifest_error(manifest, now, require_ended=True)
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.datetime.fromtimestamp(
            now, datetime.timezone.utc
        ).isoformat(),
        "status": "UNVERIFIED",
        "campaign_id": manifest.get("campaign_id") if isinstance(manifest, dict) else None,
        "manifest": manifest,
        "checks": {},
        "metrics": {},
    }
    if error:
        receipt["checks"]["manifest_valid"] = {
            "passed": False, "reason": error,
        }
        return receipt
    receipt["checks"]["manifest_valid"] = {"passed": True}

    lake = IdeaLake(str(db_path))
    try:
        rows = lake.conn.execute(
            "SELECT * FROM harness_efficiency_samples "
            "WHERE campaign_id = ? AND observed_at_epoch >= ? "
            "AND observed_at_epoch <= ? "
            "ORDER BY observed_at_epoch, id",
            (
                manifest["campaign_id"], manifest["start_epoch"],
                manifest["end_epoch"],
            ),
        ).fetchall()
        progress_rows = lake.conn.execute(
            "SELECT * FROM harness_campaign_progress "
            "WHERE campaign_id = ? AND observed_at_epoch >= ? "
            "AND observed_at_epoch <= ? "
            "ORDER BY observed_at_epoch, id",
            (
                manifest["campaign_id"], manifest["start_epoch"],
                manifest["end_epoch"],
            ),
        ).fetchall()
        transitions = lake.conn.execute(
            "SELECT idea_id, to_state, ts FROM idea_transitions "
            "WHERE ts IS NOT NULL ORDER BY ts, id"
        ).fetchall()
        states = lake.conn.execute(
            "SELECT idea_id, queued_at, claimed_at FROM idea_state"
        ).fetchall()
        registration = lake.conn.execute(
            "SELECT * FROM harness_campaign_registrations "
            "WHERE campaign_id = ?",
            (manifest["campaign_id"],),
        ).fetchone()
    finally:
        lake.close()

    scope = sorted(manifest["physical_scope"])
    expected_idea_ids = list(manifest["expected_idea_ids"])
    expected_idea_set = set(expected_idea_ids)
    poll = float(manifest["poll_seconds"])
    canonical = _canonical_manifest(manifest)
    manifest_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    registration_valid = bool(
        registration
        and registration["manifest_sha256"] == manifest_sha256
        and registration["manifest_json"] == canonical
        and registration["registered_at_epoch"] <= manifest["start_epoch"]
    )
    receipt["registration"] = (
        {
            "manifest_sha256": registration["manifest_sha256"],
            "registered_at_epoch": registration["registered_at_epoch"],
            "registered_at": registration["registered_at"],
        }
        if registration else None
    )
    parsed_rows = []
    malformed_rows = 0
    for row in rows:
        try:
            parsed = {
                **dict(row),
                "scope": json.loads(row["physical_scope_json"]),
                "telemetry": json.loads(row["gpu_telemetry_json"]),
                "training": json.loads(row["active_training_gpus_json"]),
                "evaluation": json.loads(row["active_evaluation_gpus_json"]),
            }
            gpu_lists = (
                parsed["scope"], parsed["training"], parsed["evaluation"]
            )
            if any(not isinstance(values, list) for values in gpu_lists):
                raise ValueError
            if any(
                isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0
                for values in gpu_lists for gpu in values
            ):
                raise ValueError
            if any(len(values) != len(set(values)) for values in gpu_lists):
                raise ValueError
            if (not set(parsed["training"] + parsed["evaluation"]).issubset(
                    parsed["scope"])
                    or set(parsed["training"]) & set(parsed["evaluation"])):
                raise ValueError
            if not isinstance(parsed["telemetry"], list):
                raise ValueError
            for item in parsed["telemetry"]:
                if (not isinstance(item, dict)
                        or not isinstance(item.get("index"), int)
                        or item["index"] not in parsed["scope"]
                        or not isinstance(item.get("utilization_pct"), (int, float))
                        or isinstance(item.get("utilization_pct"), bool)
                        or not math.isfinite(float(item["utilization_pct"]))
                        or not 0 <= item["utilization_pct"] <= 100):
                    raise ValueError
            for key in ("remaining_training", "remaining_evaluation"):
                if (isinstance(parsed[key], bool)
                        or not isinstance(parsed[key], int)
                        or parsed[key] < 0):
                    raise ValueError
            if (not math.isfinite(float(parsed["observed_at_epoch"]))
                    or not math.isfinite(float(parsed["poll_seconds"]))
                    or parsed["poll_seconds"] <= 0):
                raise ValueError
            parsed_rows.append(parsed)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            malformed_rows += 1

    parsed_progress = []
    malformed_progress_rows = 0
    progress_deadlines_valid = True
    for row in progress_rows:
        try:
            parsed = dict(row)
            observed = float(parsed["observed_at_epoch"])
            deadline = float(parsed["next_deadline_epoch"])
            recorded_epoch = _epoch(parsed["observed_at"])
            if (not math.isfinite(observed) or not math.isfinite(deadline)
                    or recorded_epoch is None
                    or abs(recorded_epoch - observed) > 1e-6
                    or isinstance(parsed["iteration"], bool)
                    or not isinstance(parsed["iteration"], int)
                    or parsed["iteration"] < 0
                    or _CONTROLLER_RE.fullmatch(parsed["controller_id"]) is None
                    or not isinstance(parsed["host"], str)
                    or not parsed["host"].strip()
                    or parsed["blocker_code"] not in _PROGRESS_BLOCKERS):
                raise ValueError
            artifact = (
                parsed["last_valid_artifact_sha256"],
                parsed["last_valid_artifact_idea_id"],
                parsed["last_valid_artifact_at_epoch"],
            )
            if any(value is None for value in artifact):
                if not all(value is None for value in artifact):
                    raise ValueError
            else:
                artifact_at = float(parsed["last_valid_artifact_at_epoch"])
                if (not isinstance(parsed["last_valid_artifact_sha256"], str)
                        or re.fullmatch(
                            r"[0-9a-f]{64}",
                            parsed["last_valid_artifact_sha256"],
                        ) is None
                        or parsed["last_valid_artifact_idea_id"]
                        not in expected_idea_set
                        or not math.isfinite(artifact_at)
                        or not 0 < artifact_at <= observed):
                    raise ValueError
                parsed["last_valid_artifact_at_epoch"] = artifact_at
            if (deadline < observed
                    or deadline > min(
                        float(manifest["end_epoch"]),
                        observed + float(manifest["targets"][
                            "max_operator_update_gap_seconds"
                        ]),
                    ) + 1e-9):
                progress_deadlines_valid = False
            parsed_progress.append(parsed)
        except (KeyError, TypeError, ValueError):
            malformed_progress_rows += 1

    timestamps = [float(row["observed_at_epoch"]) for row in parsed_rows]
    boundary_timestamps = [manifest["start_epoch"], *timestamps, manifest["end_epoch"]]
    gaps = [right - left for left, right in zip(
        boundary_timestamps, boundary_timestamps[1:]
    )]
    max_gap = max(gaps) if gaps else manifest["end_epoch"] - manifest["start_epoch"]
    progress_timestamps = [
        float(row["observed_at_epoch"]) for row in parsed_progress
    ]
    progress_boundaries = [
        manifest["start_epoch"], *progress_timestamps, manifest["end_epoch"]
    ]
    progress_gaps = [
        right - left for left, right in zip(
            progress_boundaries, progress_boundaries[1:]
        )
    ]
    max_progress_gap = (
        max(progress_gaps) if progress_gaps
        else manifest["end_epoch"] - manifest["start_epoch"]
    )
    samples_by_iteration = {
        (row["controller_id"], row["iteration"]): row
        for row in parsed_rows
    }
    progress_by_iteration = {
        (row["controller_id"], row["iteration"]): row
        for row in parsed_progress
    }
    operator_updates_match_samples = (
        len(samples_by_iteration) == len(parsed_rows)
        and len(progress_by_iteration) == len(parsed_progress)
        and set(samples_by_iteration) == set(progress_by_iteration)
        and all(
            progress_by_iteration[key]["host"] == sample["host"]
            and abs(
                float(progress_by_iteration[key]["observed_at_epoch"])
                - float(sample["observed_at_epoch"])
            ) <= 1e-6
            for key, sample in samples_by_iteration.items()
        )
    )
    exact_scope = all(row["scope"] == scope for row in parsed_rows)
    exact_poll = all(abs(float(row["poll_seconds"]) - poll) <= 1e-9
                     for row in parsed_rows)
    telemetry_complete = all(
        bool(row["telemetry_complete"])
        and sorted(item.get("index") for item in row["telemetry"]) == scope
        for row in parsed_rows
    )

    allocation_ratios = []
    allocated_utilization = []
    demand_samples = 0
    for row in parsed_rows:
        active = set(row["training"]) | set(row["evaluation"])
        remaining = int(row["remaining_training"]) + int(row["remaining_evaluation"])
        desired = min(len(scope), len(active) + remaining)
        if desired:
            demand_samples += 1
            allocation_ratios.append(len(active) / desired)
        telemetry_by_gpu = {
            item["index"]: item["utilization_pct"] for item in row["telemetry"]
            if isinstance(item, dict) and "index" in item
            and "utilization_pct" in item
        }
        allocated_utilization.extend(
            telemetry_by_gpu[gpu] for gpu in active if gpu in telemetry_by_gpu
        )

    state_ids = {str(state["idea_id"]) for state in states}
    missing_lifecycle_idea_ids = sorted(expected_idea_set - state_ids)
    unexpected_lifecycle_idea_ids = set()
    for state in states:
        claimed = _epoch(state["claimed_at"])
        if (claimed is not None
                and manifest["start_epoch"] <= claimed <= manifest["end_epoch"]
                and state["idea_id"] not in expected_idea_set):
            unexpected_lifecycle_idea_ids.add(str(state["idea_id"]))
    for transition in transitions:
        at = _epoch(transition["ts"])
        if (at is not None
                and manifest["start_epoch"] <= at <= manifest["end_epoch"]
                and transition["to_state"] in {
                    "CLAIMED", "IN_PROGRESS", "COMPLETE", "FAILED", "SKIPPED",
                }
                and transition["idea_id"] not in expected_idea_set):
            unexpected_lifecycle_idea_ids.add(str(transition["idea_id"]))

    queue_to_claim = []
    for state in states:
        if state["idea_id"] not in expected_idea_set:
            continue
        queued = _epoch(state["queued_at"])
        claimed = _epoch(state["claimed_at"])
        if (queued is not None and claimed is not None
                and manifest["start_epoch"] <= claimed <= manifest["end_epoch"]
                and claimed >= queued):
            queue_to_claim.append(claimed - queued)

    timeline = []
    for transition in transitions:
        if transition["idea_id"] not in expected_idea_set:
            continue
        at = _epoch(transition["ts"])
        if at is not None:
            timeline.append((at, transition["to_state"], transition["idea_id"]))
    release_to_claim = []
    for index, (released_at, state, _) in enumerate(timeline):
        if (state not in {"COMPLETE", "FAILED", "SKIPPED"}
                or not manifest["start_epoch"] <= released_at <= manifest["end_epoch"]):
            continue
        for claimed_at, later_state, _ in timeline[index + 1:]:
            if claimed_at > manifest["end_epoch"]:
                break
            if later_state == "CLAIMED":
                release_to_claim.append(claimed_at - released_at)
                break

    allocation_duty = (
        sum(allocation_ratios) / len(allocation_ratios)
        if allocation_ratios else None
    )
    queue_p95 = _percentile(queue_to_claim, 0.95)
    release_p95 = _percentile(release_to_claim, 0.95)
    receipt["metrics"] = {
        "sample_count": len(parsed_rows),
        "malformed_sample_count": malformed_rows,
        "operator_update_count": len(parsed_progress),
        "malformed_operator_update_count": malformed_progress_rows,
        "max_operator_update_gap_seconds": max_progress_gap,
        "operator_blocker_counts": dict(sorted({
            code: sum(1 for row in parsed_progress
                      if row["blocker_code"] == code)
            for code in {row["blocker_code"] for row in parsed_progress}
        }.items())),
        "last_valid_artifact_sha256": (
            parsed_progress[-1]["last_valid_artifact_sha256"]
            if parsed_progress else None
        ),
        "last_valid_artifact_idea_id": (
            parsed_progress[-1]["last_valid_artifact_idea_id"]
            if parsed_progress else None
        ),
        "controller_count": len({row["controller_id"] for row in parsed_rows}),
        "host_count": len({row["host"] for row in parsed_rows}),
        "expected_idea_count": len(expected_idea_ids),
        "missing_lifecycle_idea_ids": missing_lifecycle_idea_ids,
        "unexpected_lifecycle_idea_ids": sorted(
            unexpected_lifecycle_idea_ids
        ),
        "demand_sample_count": demand_samples,
        "max_sample_gap_seconds": max_gap,
        "max_sample_gap_poll_intervals": max_gap / poll,
        "allocation_duty_cycle": allocation_duty,
        "allocated_gpu_utilization_mean_pct": (
            sum(allocated_utilization) / len(allocated_utilization)
            if allocated_utilization else None
        ),
        "queue_to_claim_count": len(queue_to_claim),
        "queue_to_claim_p95_seconds": queue_p95,
        "terminal_to_next_claim_count": len(release_to_claim),
        "terminal_to_next_claim_p95_seconds": release_p95,
    }

    evidence_checks = {
        "preregistered_manifest_match": registration_valid,
        "minimum_samples": len(parsed_rows) >= manifest["minimum_samples"],
        "no_malformed_samples": malformed_rows == 0,
        "exact_physical_scope": bool(parsed_rows) and exact_scope,
        "exact_poll_seconds": bool(parsed_rows) and exact_poll,
        "telemetry_complete": bool(parsed_rows) and telemetry_complete,
        "operator_updates_complete": (
            bool(parsed_progress)
            and malformed_progress_rows == 0
            and operator_updates_match_samples
        ),
        "operator_update_deadlines_valid": progress_deadlines_valid,
        "single_physical_host": (
            len({row["host"] for row in parsed_rows}) == 1
        ),
        "exact_campaign_idea_universe": (
            not missing_lifecycle_idea_ids
            and not unexpected_lifecycle_idea_ids
        ),
        "minimum_claims": len(queue_to_claim) >= manifest["minimum_claims"],
        "minimum_release_to_claim_pairs": (
            len(release_to_claim) >= manifest["minimum_release_to_claim_pairs"]
        ),
        "demand_observed": demand_samples > 0,
    }
    for name, passed in evidence_checks.items():
        receipt["checks"][name] = {"passed": passed}

    target_checks = {
        "sample_gap": (
            max_gap / poll
            <= manifest["targets"]["max_sample_gap_poll_intervals"]
        ),
        "allocation_duty_cycle": (
            allocation_duty is not None
            and allocation_duty
            >= manifest["targets"]["min_allocation_duty_cycle"]
        ),
        "queue_to_claim": (
            queue_p95 is not None
            and queue_p95 / poll
            <= manifest["targets"]["max_queue_to_claim_poll_intervals"]
        ),
        "terminal_to_next_claim": (
            release_p95 is not None
            and release_p95 / poll
            <= manifest["targets"]["max_terminal_to_next_claim_poll_intervals"]
        ),
        "operator_update_gap": (
            max_progress_gap
            <= manifest["targets"]["max_operator_update_gap_seconds"]
        ),
    }
    for name, passed in target_checks.items():
        receipt["checks"][name] = {"passed": passed}

    source_path = Path(__file__)
    receipt["analyzer_source"] = {
        "path": str(source_path),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    evidence_complete = all(evidence_checks.values())
    if evidence_complete:
        receipt["status"] = "VERIFIED" if all(target_checks.values()) else "FAILED"
    return receipt


def write_campaign_receipt(
    db_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
) -> Dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    receipt = analyze_campaign(db_path, manifest)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Register or analyze fail-closed campaign evidence"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--db", required=True)
    register.add_argument("--manifest", required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--db", required=True)
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "register":
        manifest = json.loads(
            Path(args.manifest).read_text(encoding="utf-8")
        )
        result = preregister_campaign(args.db, manifest)
    else:
        result = write_campaign_receipt(
            args.db, args.manifest, args.output
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status", "VERIFIED") == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
