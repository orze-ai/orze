"""Mandatory promotion qualification and optional operational anomaly policy.

CALLING SPEC:
    check_promotion(results_dir, idea_id, new_metric, cfg, *, idea_cfg=None,
                    notify_fn=None, create_audit_idea_fn=None, lake=None)
        Return (allowed, info). Qualification is always required, including when
        champion_guard.enabled is false (the default). Explicit enabled=true
        applies an absolute primary-metric z-score over currently qualified,
        distinct prior ideas under the same objective/contract. action='hold'
        preserves explicit opt-in blocking; action='warn' is advisory.

The legacy info['verified'] field means current qualified artifact value, not
independent replication. This heuristic is not a significance test, scientific
progress judgment, or guarantee of an atomic filesystem/database snapshot.
Reproducer shell commands are unsupported here: schedule an explicit evaluation
or replication task with its own execution and output contract.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from orze.engine.champion_history import (
    objective_scope, read_history, record_history,
)
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence_with_identity,
    report_lifecycle_db_path,
)

logger = logging.getLogger("champion_guard")


def _finite(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


@dataclass
class GuardConfig:
    enabled: bool = False
    z_threshold: float = 4.0
    min_history: int = 10
    history_size: int = 50
    action: str = "hold"

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "GuardConfig":
        raw = cfg.get("champion_guard") or {}
        if not isinstance(raw, dict):
            raise ValueError("guard_config_invalid")
        guard = cls(**{key: raw[key] for key in (
            "enabled", "z_threshold", "min_history", "history_size", "action",
        ) if key in raw})
        if (not isinstance(guard.enabled, bool)
                or not _finite(guard.z_threshold) or guard.z_threshold <= 0
                or isinstance(guard.min_history, bool)
                or not isinstance(guard.min_history, int)
                or isinstance(guard.history_size, bool)
                or not isinstance(guard.history_size, int)
                or not 2 <= guard.min_history <= guard.history_size <= 1000
                or guard.action not in ("hold", "warn")):
            raise ValueError("guard_config_invalid")
        return guard


def _zscore(history: list[float], value: float) -> Optional[float]:
    if len(history) < 2:
        return None
    scale = max(abs(item) for item in history)
    if scale == 0:
        return None
    normalized = [item / scale for item in history]
    std = statistics.pstdev(normalized)
    if std == 0:
        return None
    z = (value / scale - statistics.fmean(normalized)) / std
    # Finite inputs can still overflow a ratio. Keep the diagnostic JSON finite.
    return z if math.isfinite(z) else math.copysign(sys.float_info.max, z)


def check_promotion(
    results_dir: Path,
    idea_id: str,
    new_metric: float,
    cfg: Dict[str, Any],
    *,
    idea_cfg: Optional[Dict[str, Any]] = None,
    notify_fn=None,
    create_audit_idea_fn=None,
    lake=None,
) -> Tuple[bool, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "enabled": False, "claimed": new_metric, "verified": None,
        "z": None, "blocked": False, "audit_idea_id": None,
        "history_size": 0, "claim_scope": "operational_anomaly",
    }

    def reject(reason):
        info.update(blocked=True, reason=reason)
        return False, info

    try:
        guard = GuardConfig.from_cfg(cfg)
        info.update(enabled=guard.enabled, z_threshold=guard.z_threshold,
                    action=guard.action)
        report = cfg.get("report") or {}
        if (not isinstance(report, dict)
                or not isinstance(report.get("primary_metric"), str)
                or not report["primary_metric"].strip()
                or report.get("sort", "descending") not in
                ("ascending", "descending")):
            return reject("objective_declaration_invalid")
        if not _finite(new_metric):
            return reject("claimed_metric_invalid")
        if (idea_cfg or {}).get("reproducer"):
            return reject("reproducer_requires_explicit_evaluation_task")
        scoped_cfg = dict(cfg)
        scoped_cfg["_env_ORZE_RESULTS_DIR"] = str(Path(results_dir).resolve())
        db_path = report_lifecycle_db_path(
            results_dir, cfg, getattr(lake, "db_path", None))
        completed, reason = authoritative_completed_idea_ids(db_path)
        if reason != "authoritative_lifecycle_loaded":
            return reject(reason)
        metrics, _, value, reason, identity = (
            qualify_authoritative_report_evidence_with_identity(
                idea_id, results_dir, scoped_cfg, completed))
        info["honest"] = metrics.get("honest")
        info["verified"] = value
        info["evidence_identity"] = identity
        if value is None or identity is None:
            return reject(reason)
        if value != new_metric:
            return reject("claimed_metric_mismatch")
        if not guard.enabled:
            info["reason"] = "qualified_anomaly_policy_disabled"
            return True, info

        scope = objective_scope(scoped_cfg)
        info["objective_scope"] = scope
        history = []
        for prior in read_history(db_path, scope, guard.history_size):
            if prior["idea_id"] == idea_id:
                continue
            _, _, prior_value, _, _ = (
                qualify_authoritative_report_evidence_with_identity(
                    prior["idea_id"], results_dir, scoped_cfg, completed))
            # A shared exposure ledger can change a full snapshot identity
            # without producing another observation. Bind values and unique IDs;
            # changed source values require a newly accepted revision.
            if prior_value is not None and prior_value == prior["metric"]:
                history.append(prior_value)
        info["history_size"] = len(history)
        z = (_zscore(history, value)
             if len(history) >= guard.min_history else None)
        info["z"] = z
        anomalous = z is not None and abs(z) > guard.z_threshold
        info["anomalous"] = anomalous
        if anomalous:
            info["reason"] = "operational_outlier"
            payload = {
                "idea_id": idea_id, "claimed": new_metric, "verified": value,
                "z": z, "threshold": guard.z_threshold, "action": guard.action,
                "claim_scope": "operational_anomaly",
                "objective_scope": scope, "evidence_identity": identity,
            }
            if notify_fn is not None:
                try:
                    notify_fn("audit", payload, cfg)
                except Exception:
                    logger.warning("Operational anomaly notification unavailable")
            if guard.action == "hold":
                # An explicitly supplied callback retains its legacy API. The
                # controller does not supply one or implicitly create tasks.
                if create_audit_idea_fn is not None:
                    audit_id = "audit-" + hashlib.sha256(
                        f"{scope}:{idea_id}:{identity}".encode()).hexdigest()[:24]
                    info["audit_idea_id"] = audit_id
                    payload["audit_idea_id"] = audit_id
                    create_audit_idea_fn(audit_id, idea_id, payload)
                return reject("operational_outlier")
        else:
            info["reason"] = ("insufficient_history" if z is None
                              else "operational_check_passed")
        record_history(db_path, scope, idea_id, value, identity, guard.history_size)
        return True, info
    except Exception as exc:
        logger.warning("Promotion check unavailable: %s", type(exc).__name__)
        return reject("promotion_check_unavailable")


def create_audit_idea(lake, audit_id: str, suspect_idea_id: str,
                      payload: Dict[str, Any]) -> None:
    """Explicit legacy audit action; never invoked implicitly by the controller."""
    cfg_yaml = json.dumps({
        "suspect_idea_id": suspect_idea_id,
        "claimed": payload.get("claimed"), "verified": payload.get("verified"),
        "z": payload.get("z"), "action": "audit",
    }, indent=2)
    lake.insert(
        audit_id, f"Audit suspicious promotion: {suspect_idea_id}",
        cfg_yaml, raw_markdown=cfg_yaml, status="pending", kind="audit",
        parent=suspect_idea_id,
    )
