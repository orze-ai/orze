"""Local, fail-closed evidence for scheduler and GPU campaign efficiency.

This module never launches work.  Sampling queries only the caller-provided
physical GPU scope and stores no idea IDs, configs, model outputs, or metrics.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from orze.hardware.gpu import _query_gpu_details
from orze.idea_lake import IdeaLake


DEFAULT_CAMPAIGN_TARGETS = {
    "max_sample_gap_poll_intervals": 2.0,
    "min_allocation_duty_cycle": 0.90,
    "max_queue_to_claim_poll_intervals": 2.0,
    "max_terminal_to_next_claim_poll_intervals": 1.0,
}


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
    }
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        return "manifest is missing required fields"
    if not isinstance(manifest["campaign_id"], str) or not manifest["campaign_id"]:
        return "campaign_id must be a non-empty string"
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
    ):
        if targets[key] > DEFAULT_CAMPAIGN_TARGETS[key]:
            return f"{key} cannot be weaker than the default target"
    if (targets["min_allocation_duty_cycle"]
            < DEFAULT_CAMPAIGN_TARGETS["min_allocation_duty_cycle"]):
        return "min_allocation_duty_cycle cannot be weaker than the default target"
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
        transitions = lake.conn.execute(
            "SELECT idea_id, to_state, ts FROM idea_transitions "
            "WHERE ts IS NOT NULL ORDER BY ts, id"
        ).fetchall()
        states = lake.conn.execute(
            "SELECT queued_at, claimed_at FROM idea_state "
            "WHERE queued_at IS NOT NULL AND claimed_at IS NOT NULL"
        ).fetchall()
        registration = lake.conn.execute(
            "SELECT * FROM harness_campaign_registrations "
            "WHERE campaign_id = ?",
            (manifest["campaign_id"],),
        ).fetchone()
    finally:
        lake.close()

    scope = sorted(manifest["physical_scope"])
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

    timestamps = [float(row["observed_at_epoch"]) for row in parsed_rows]
    boundary_timestamps = [manifest["start_epoch"], *timestamps, manifest["end_epoch"]]
    gaps = [right - left for left, right in zip(
        boundary_timestamps, boundary_timestamps[1:]
    )]
    max_gap = max(gaps) if gaps else manifest["end_epoch"] - manifest["start_epoch"]
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

    queue_to_claim = []
    for state in states:
        queued = _epoch(state["queued_at"])
        claimed = _epoch(state["claimed_at"])
        if (queued is not None and claimed is not None
                and manifest["start_epoch"] <= claimed <= manifest["end_epoch"]
                and claimed >= queued):
            queue_to_claim.append(claimed - queued)

    timeline = []
    for transition in transitions:
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
        "controller_count": len({row["controller_id"] for row in parsed_rows}),
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
