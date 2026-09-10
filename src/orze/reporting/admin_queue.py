"""Derived admin task state, separate from result evidence and ranking.

CALLING SPEC:
    build_admin_queue(results_dir, ideas, cfg) -> dict
        Merge expanded inbox metadata with one closed read-only catalog
        snapshot. Native task status uses agreed lifecycle state only; raw
        training metrics never close a task. Persistent-only configurations
        are loaded through the catalog's bounded optional config reader.

Without a declared idea_lake_db key, retain legacy artifact-only display and
label it unverified. The result is an observer snapshot, not launch authority,
qualified research evidence, or a globally atomic filesystem/database view.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

from orze.core.ideas import expand_sweeps
from orze.reporting.catalog import load_catalog_snapshot
from orze.reporting.evidence import report_lifecycle_db_path


_DISPLAY_STATUS = {
    "QUEUED": "pending", "CLAIMED": "running", "IN_PROGRESS": "running",
    "COMPLETE": "completed", "FAILED": "failed", "SKIPPED": "skipped",
    "ARCHIVED": "archived", "UNKNOWN": "unknown",
}


def _offline_status(idea_dir: Path) -> str:
    """Preserve legacy display conventions without granting lifecycle truth."""
    if not idea_dir.exists():
        return "pending"
    path = idea_dir / "metrics.json"
    if not path.exists():
        return "running"
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "running"
    if not isinstance(metrics, dict):
        return "unknown"
    status = metrics.get("status", "COMPLETED")
    return status.lower() if isinstance(status, str) else "unknown"


def _metadata(idea_id: str, inbox: dict, record: dict) -> dict:
    """Presentation may use hot overrides; metadata never changes task state."""
    raw = inbox.get("raw", "")
    raw = raw if isinstance(raw, str) else ""

    def field(name, default):
        match = re.search(r"\*\*" + name.title() + r"\*\*:\s*(.+)", raw)
        return match.group(1).strip() if match else str(record.get(name) or default)

    if "config" in inbox:
        config = inbox["config"]
        config_available = isinstance(config, dict)
    else:
        config = record.get("config")
        config_available = record.get("config_available") is True
    if not isinstance(config, dict):
        config, config_available = {}, False
    return {
        "idea_id": idea_id,
        "title": str(inbox.get("title") or record.get("title") or idea_id),
        "priority": str(inbox.get("priority") or record.get("priority") or "medium"),
        "config": config, "config_available": config_available,
        "sweep_parent": inbox.get("_sweep_parent"),
        "category": field("category", "architecture"),
        "parent": field("parent", "none"),
        "hypothesis": field("hypothesis", ""),
    }


def build_admin_queue(results_dir: Path, ideas: dict, cfg: dict) -> dict:
    """Build compatible queue rows with explicitly scoped lifecycle authority."""
    native = "idea_lake_db" in cfg
    records = {}
    reason = "unverified_local_artifact"
    authority = "unverified_local_artifact"
    if native:
        authority = "unavailable_idea_lake"
        try:
            snapshot = load_catalog_snapshot(
                report_lifecycle_db_path(results_dir, cfg), include_configs=True)
            reason = snapshot.reason
            if snapshot.available:
                records = snapshot.records
                authority = "agreed_idea_lake"
        except (OSError, TypeError, ValueError):
            reason = "authoritative_catalog_invalid"

    sweep = cfg.get("sweep") or {}
    sweep_max = sweep.get("max_combos", 20) if isinstance(sweep, dict) else 20
    expanded = expand_sweeps(dict(ideas), max_combos=sweep_max)
    items = []
    for idea_id in sorted(set(records) | set(expanded)):
        if (not isinstance(idea_id, str) or idea_id in ("", ".", "..")
                or Path(idea_id).parts != (idea_id,)):
            continue
        inbox = expanded.get(idea_id) or {}
        inbox = inbox if isinstance(inbox, dict) else {}
        record = records.get(idea_id) or {}
        item = _metadata(idea_id, inbox, record)
        if native:
            state = record.get("lifecycle_state", "UNKNOWN")
            item.update({
                "status": _DISPLAY_STATUS.get(state, "unknown"),
                "lifecycle_state": state,
                "lifecycle_reason": record.get("lifecycle_reason") or (
                    "lifecycle_catalog_row_missing" if authority == "agreed_idea_lake"
                    else reason),
                "training_state": record.get("training_state"),
                "evaluation_state": record.get("evaluation_state"),
            })
        else:
            item.update({
                "status": _offline_status(Path(results_dir) / idea_id),
                "lifecycle_state": "UNKNOWN", "lifecycle_reason": reason,
                "training_state": None, "evaluation_state": None,
            })
        item["lifecycle_authority"] = authority
        items.append(item)
    return {
        "items": items, "counts": dict(Counter(item["status"] for item in items)),
        "total_all": len(items), "lifecycle_authority": authority,
        "lifecycle_authority_reason": reason,
    }
