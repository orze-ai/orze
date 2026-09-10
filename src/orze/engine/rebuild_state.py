"""Rebuild per-host orze state from idea_lake.db.

Why this exists
---------------
The plateau-breaking skill ``axiom_removal`` is gated on
``on_plateau(N)``. The plateau-detector uses ``best_idea_id`` and
``completions_since_best`` from the per-host state file to count
"completed ideas since the champion was set". If those fields are ever
``null`` (e.g., state file deleted, upgrade reset, first boot after a
long run), the plateau counter never advances and the breaker never
fires.

Production recovery rebuilds both fields from lifecycle-complete, qualified
result evidence. Cached lake metrics are never a fallback for rejected or
missing artifacts. The legacy lake-only helper remains an archive query, not a
source of steering authority:

    best_idea_id = best eligible completed idea under report.sort
    completions_since_best = count(completed ideas with archived_at >= best.archived_at) - 1

The primary metric is taken from ``cfg.report.primary_metric`` and
resolved the same way ``orze.reporting.leaderboard`` does. Callers:
* ``orze rebuild-state`` CLI (one-shot)
* orchestrator startup (idempotent; no-op if fields are already set)

CALLING SPEC
------------
    rebuild_best_from_evidence(results_dir, cfg, lake=None) -> (best_id, since)
        Read-only shared recovery path for CLI and orchestrator startup.
        Uses the complete report, lifecycle, and benchmark qualification policy.

    rebuild_state_file(results_dir, cfg, overwrite=False) -> dict
        Applies rebuild_best_from_evidence to the host's state file on disk.
        Returns a summary dict.
"""
from __future__ import annotations

import json
import logging
import math
import socket
from pathlib import Path
from typing import Optional, Tuple

from orze.reporting.state import load_state, save_state

logger = logging.getLogger("orze")


def _report_dataset_keys(report_cfg: dict) -> list[str]:
    keys = [
        col["key"] for col in (report_cfg.get("columns") or [])
        if isinstance(col, dict) and col.get("key")
    ]
    primary = report_cfg.get("primary_metric")
    wer_keys = [
        key for key in keys
        if key.startswith("wer_") and key != primary
    ]
    return wer_keys or keys


def _eligible_metric(metrics: dict, primary_metric: str,
                     min_datasets: int, dataset_keys: list[str]) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(primary_metric)
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        return None
    if min_datasets > 0:
        count = sum(
            1 for key in dataset_keys
            if isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics.get(key), bool)
            and math.isfinite(float(metrics[key]))
        )
        if count == 0:
            count = sum(
                1 for key, item in metrics.items()
                if key.startswith("wer_")
                and isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
            )
        if count < min_datasets:
            return None
    return float(value)


def rebuild_best_from_lake(lake, primary_metric: str,
                           sort_order: str = "descending",
                           min_datasets: int = 0,
                           dataset_keys: Optional[list[str]] = None,
                           ) -> Tuple[Optional[str], int]:
    """Return (best_idea_id, completions_since_best) from the lake.

    Queries ``eval_metrics`` JSON column. Returns (None, 0) if no
    completed idea has the metric recorded — in that case the caller
    should try ``rebuild_best_from_results_dir``.
    """
    if lake is None or getattr(lake, "conn", None) is None:
        return None, 0
    rows = lake.conn.execute(
        "SELECT idea_id, archived_at, eval_metrics FROM ideas "
        "WHERE status = 'completed' AND eval_metrics IS NOT NULL"
    ).fetchall()
    candidates = []
    for row in rows:
        raw = row[2] if isinstance(row, tuple) else row["eval_metrics"]
        try:
            metrics = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            continue
        value = _eligible_metric(
            metrics, primary_metric, min_datasets, dataset_keys or [])
        if value is not None:
            idea_id = row[0] if isinstance(row, tuple) else row["idea_id"]
            archived = row[1] if isinstance(row, tuple) else row["archived_at"]
            candidates.append((idea_id, archived, value))
    if not candidates:
        return None, 0
    reverse = sort_order == "descending"
    candidates.sort(key=lambda item: item[2], reverse=reverse)
    best_id, best_archived, _ = candidates[0]
    if best_archived is None:
        since = 0
    else:
        row = lake.conn.execute(
            "SELECT COUNT(*) FROM ideas "
            "WHERE status = 'completed' AND archived_at > ?",
            (best_archived,),
        ).fetchone()
        since = int(row[0]) if row and row[0] else 0
    return best_id, since


def rebuild_best_from_results_dir(results_dir: Path | str,
                                  primary_metric: str,
                                  sort_order: str = "descending",
                                  min_datasets: int = 0,
                                  dataset_keys: Optional[list[str]] = None,
                                  ) -> Tuple[Optional[str], int]:
    """Scan authoritative ``<results>/idea-*/metrics.json`` artifacts.

    Returns (best_id, completions_since_best). ``since_best`` counts
    completed ideas newer than ``best`` (by metrics.json mtime).
    """
    import json as _json
    rd = Path(results_dir)
    best_id: Optional[str] = None
    best_val: Optional[float] = None
    best_mtime: Optional[float] = None
    newer_completed = 0
    completed: list = []

    for idea_dir in rd.glob("idea-*"):
        if not idea_dir.is_dir():
            continue
        mpath = idea_dir / "metrics.json"
        if not mpath.exists():
            continue
        try:
            data = _json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        # Accept either {"status":"COMPLETED","metrics":{...}} or flat
        # {<metric>: <val>, ...}. Reject explicit non-completed states.
        status = data.get("status")
        if status and status != "COMPLETED":
            continue
        nested = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        metrics = dict(nested)
        metrics.update(data)
        val = _eligible_metric(
            metrics, primary_metric, min_datasets, dataset_keys or [])
        if val is None:
            continue
        mtime = mpath.stat().st_mtime
        completed.append((idea_dir.name, val, mtime))
        is_better = (
            best_val is None
            or (sort_order == "ascending" and val < best_val)
            or (sort_order != "ascending" and val > best_val)
        )
        if is_better:
            best_val = val
            best_id = idea_dir.name
            best_mtime = mtime

    if best_id is None:
        return None, 0
    for _id, _val, mtime in completed:
        if best_mtime is not None and mtime > best_mtime:
            newer_completed += 1
    return best_id, newer_completed


def rebuild_best_from_evidence(results_dir: Path, cfg: dict,
                               lake=None) -> Tuple[Optional[str], int]:
    """Rebuild steering state without weakening the current evidence policy.

    Missing lifecycle authority or zero eligible observations yields no best
    and zero plateau budget. Never create a database or promote cached metrics
    in order to make recovery appear successful. Only qualified observations
    count toward the existing completion/mtime-based plateau counter.
    """
    from orze.reporting.evidence import (
        authoritative_completed_idea_ids,
        qualify_authoritative_report_evidence_with_identity,
    )

    results_dir = Path(results_dir)
    scoped_cfg = dict(cfg)
    report = dict(cfg.get("report") or {})
    report.setdefault("primary_metric", "test_accuracy")
    scoped_cfg["report"] = report
    scoped_cfg["_env_ORZE_RESULTS_DIR"] = str(results_dir.resolve())
    db_path = (getattr(lake, "db_path", None) or cfg.get("idea_lake_db")
               or results_dir / "idea_lake.db")
    completed, reason = authoritative_completed_idea_ids(Path(db_path))
    if reason != "authoritative_lifecycle_loaded":
        logger.warning("Champion recovery unavailable: %s", reason)
        return None, 0

    candidates = []
    for idea_id in sorted(completed):
        _, _, value, _, _ = qualify_authoritative_report_evidence_with_identity(
            idea_id, results_dir, scoped_cfg, completed)
        if value is None:
            continue
        try:
            mtime = (results_dir / idea_id / "metrics.json").stat().st_mtime
        except OSError:
            continue
        candidates.append((idea_id, value, mtime))
    if not candidates:
        return None, 0
    lower_is_better = str(report.get("sort", "descending")).lower().startswith("asc")
    candidates.sort(key=lambda item: item[1], reverse=not lower_is_better)
    best_id, _, best_mtime = candidates[0]
    return best_id, sum(mtime > best_mtime for _, _, mtime in candidates)


def restore_reporter_from_evidence(reporter, results_dir: Path, cfg: dict,
                                   lake=None) -> None:
    """Restore or revoke persisted champion state on controller startup."""
    best_id, since = rebuild_best_from_evidence(results_dir, cfg, lake=lake)
    previous = reporter._best_idea_id
    if best_id != previous or since < reporter._completions_since_best:
        reporter._plateau_notified = False
    reporter._best_idea_id = best_id
    reporter._completions_since_best = since
    logger.info("Reconciled best_idea_id=%s (previous=%s) "
                "completions_since_best=%d from qualified evidence",
                best_id, previous, since)


def rebuild_state_file(results_dir: Path, cfg: dict,
                       overwrite: bool = False,
                       lake=None,
                       all_hosts: bool = False) -> dict:
    """Rebuild best_idea_id + completions_since_best in the state file.

    If ``overwrite`` is False, we only fill in nulls (idempotent safe
    startup call). If True, we always rewrite.

    If ``all_hosts`` is True, the same rebuilt values are written to
    every ``.orze_state_<host>.json`` file in the results dir (multi-
    daemon shared FSx case).
    """
    report_cfg = cfg.get("report", {})
    primary = report_cfg.get("primary_metric", "test_accuracy")
    best_id, since = rebuild_best_from_evidence(results_dir, cfg, lake=lake)

    state = load_state(Path(results_dir))
    existing_best = state.get("best_idea_id")
    existing_since = state.get("completions_since_best", 0)

    will_write = overwrite or existing_best is None
    if will_write and best_id is None and existing_best is not None and not overwrite:
        will_write = False

    summary = {
        "primary_metric": primary,
        "best_idea_id": best_id,
        "completions_since_best": since,
        "previous_best_idea_id": existing_best,
        "previous_completions_since_best": existing_since,
        "wrote_state_file": False,
        "state_file": None,
        "updated_hosts": [],
    }
    if not will_write:
        return summary

    if best_id != existing_best or since < existing_since:
        state["plateau_notified"] = False
    state["best_idea_id"] = best_id
    state["completions_since_best"] = since
    save_state(Path(results_dir), state)
    summary["wrote_state_file"] = True
    summary["state_file"] = str(
        Path(results_dir) / f".orze_state_{socket.gethostname()}.json")
    summary["updated_hosts"].append(socket.gethostname())

    if all_hosts:
        import json as _json
        for p in Path(results_dir).glob(".orze_state_*.json"):
            if p.name == Path(summary["state_file"]).name:
                continue
            try:
                d = _json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (best_id != d.get("best_idea_id")
                    or since < (d.get("completions_since_best") or 0)):
                d["plateau_notified"] = False
            d["best_idea_id"] = best_id
            d["completions_since_best"] = since
            try:
                p.write_text(_json.dumps(d, indent=2), encoding="utf-8")
                # Strip off prefix/suffix: .orze_state_<host>.json
                stem = p.name[len(".orze_state_"):-len(".json")]
                summary["updated_hosts"].append(stem)
            except OSError:
                continue
    return summary
