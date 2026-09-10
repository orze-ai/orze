"""Current evidence for event-driven bookkeeping; caches cannot grant authority.

CALLING SPEC:
    qualified_notification_rows(results_dir, cfg, candidates, lake=None) -> list
        Requalify candidate IDs read-only and return explicitly ordered rows.
    refresh_metric_snapshot(lake, row) -> None
        Best-effort update of an existing completed row's diagnostic metric
        mirror. Never inserts an idea or changes lifecycle or other metadata.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence_with_identity,
    report_lifecycle_db_path,
)
from orze.reporting.objective import objective_sort_key

logger = logging.getLogger("orze")


def qualified_notification_rows(results_dir: Path, cfg: dict, candidates: list,
                                lake=None) -> list:
    """Candidate rows nominate IDs/titles, never scores or eligibility."""
    report = cfg.get("report") or {}
    if (not isinstance(report, dict)
            or not isinstance(report.get("primary_metric"), str)
            or not report["primary_metric"].strip()
            or report.get("sort", "descending") not in
            ("ascending", "descending")):
        return []
    db_path = report_lifecycle_db_path(
        results_dir, cfg, getattr(lake, "db_path", None))
    completed, reason = authoritative_completed_idea_ids(db_path)
    if reason != "authoritative_lifecycle_loaded":
        logger.warning("Notification evidence unavailable: %s", reason)
        return []
    scoped_cfg = dict(cfg)
    scoped_cfg["_env_ORZE_RESULTS_DIR"] = str(Path(results_dir).resolve())
    rows, seen = [], set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        idea_id = candidate.get("id")
        if not isinstance(idea_id, str) or idea_id in seen:
            continue
        seen.add(idea_id)
        metrics, values, value, reason, identity = (
            qualify_authoritative_report_evidence_with_identity(
                idea_id, results_dir, scoped_cfg, completed))
        if value is None or identity is None:
            logger.debug("Notification candidate %s rejected: %s", idea_id, reason)
            continue
        rows.append({
            "id": idea_id, "title": candidate.get("title") or idea_id,
            "primary_val": value, "values": values, "metrics": metrics,
            "evidence_identity": identity,
        })
    rows.sort(key=lambda row: objective_sort_key(
        row["primary_val"], row["values"], report, row["id"]))
    return rows


def refresh_metric_snapshot(lake, row: dict) -> None:
    """Preserve legacy archive displays without granting lifecycle authority.

    The SQL predicate rechecks completion at the write boundary, including a
    requeue after qualification. This is a mutable diagnostic mirror, not an
    immutable observation receipt or proof of an atomic filesystem/DB view.
    """
    if lake is None or not row.get("evidence_identity"):
        return
    try:
        lake.conn.execute(
            "UPDATE ideas SET eval_metrics = ? WHERE idea_id = ? "
            "AND lower(status) = 'completed' "
            "AND EXISTS (SELECT 1 FROM idea_state AS s "
            "WHERE s.idea_id = ideas.idea_id AND s.current_state = 'COMPLETE')",
            (json.dumps(row["values"]), row["id"]),
        )
        lake.conn.commit()
    except Exception as exc:
        logger.warning("Metric snapshot not refreshed for %s: %s",
                       row.get("id"), type(exc).__name__)
