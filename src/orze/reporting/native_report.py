"""Native report qualification; a display cache cannot grant authority.

CALLING SPEC:
    report_authority(results_dir, cfg, lake=None) -> (requested, ids, reason)
        Resolve the explicit/open project database read-only once per report.
        No configured database and no native lake retains the legacy offline
        display API; a configured but unavailable database never falls back.
    native_report_evidence(idea_id, results_dir, cfg, completed_ids, reason)
        Return the shared (metrics, values, value, reason, identity) tuple.
        Use the declared full policy, not inferred display/harvest columns.

These functions establish per-row current evidence, not a global atomic
filesystem/lifecycle snapshot. Pipeline counts remain a separate FSM view.
"""
from __future__ import annotations

from pathlib import Path

from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence_with_identity,
    report_lifecycle_db_path,
)


def report_authority(results_dir: Path, cfg: dict, lake=None):
    override = getattr(lake, "db_path", None)
    requested = override is not None or "idea_lake_db" in cfg
    if not requested:
        return False, set(), "unverified_local_artifact"
    from orze.reporting.catalog import CatalogSnapshot
    if isinstance(lake, CatalogSnapshot) and not lake.available:
        return True, set(), lake.reason
    try:
        db_path = report_lifecycle_db_path(results_dir, cfg, override)
        completed, reason = authoritative_completed_idea_ids(db_path)
        return True, completed, reason
    except (OSError, TypeError, ValueError):
        return True, set(), "authoritative_lifecycle_database_invalid"


def native_report_evidence(idea_id: str, results_dir: Path, cfg: dict,
                           completed_ids: set[str], authority_reason: str):
    if authority_reason != "authoritative_lifecycle_loaded":
        return {}, {}, None, authority_reason, None
    report = cfg.get("report") or {}
    if (not isinstance(report, dict)
            or not isinstance(report.get("primary_metric"), str)
            or not report["primary_metric"].strip()
            or report.get("sort", "descending") not in
            ("ascending", "descending")):
        return {}, {}, None, "objective_declaration_invalid", None
    scoped_cfg = dict(cfg)
    scoped_cfg["_env_ORZE_RESULTS_DIR"] = str(Path(results_dir).resolve())
    return qualify_authoritative_report_evidence_with_identity(
        idea_id, results_dir, scoped_cfg, completed_ids)
