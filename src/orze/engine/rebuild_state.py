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

This module rebuilds both fields from authoritative terminal result artifacts,
falling back to ``idea_lake.db`` when artifacts are unavailable:

    best_idea_id = best eligible completed idea under report.sort
    completions_since_best = count(completed ideas with archived_at >= best.archived_at) - 1

The primary metric is taken from ``cfg.report.primary_metric`` and
resolved the same way ``orze.reporting.leaderboard`` does. Callers:
* ``orze rebuild-state`` CLI (one-shot)
* orchestrator startup (idempotent; no-op if fields are already set)

CALLING SPEC
------------
    rebuild_best_from_lake(lake, primary_metric) -> (best_id, since_best)
        Pure-function core: inspects the lake, returns the two fields.
        ``best_id`` is None iff no completed idea has the metric.

    rebuild_state_file(results_dir, cfg, overwrite=False) -> dict
        Applies rebuild_best_from_lake to the host's state file on disk.
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
    from orze.idea_lake import IdeaLake

    report_cfg = cfg.get("report", {})
    primary = report_cfg.get("primary_metric", "test_accuracy")
    sort_order = report_cfg.get("sort", "descending")
    min_datasets = int(report_cfg.get("min_datasets", 0) or 0)
    dataset_keys = _report_dataset_keys(report_cfg)

    # Prefer terminal artifacts. Historical lake rows can contain normalized
    # fractions while metrics.json/report columns use percentages; ranking the
    # mixed units can manufacture a false champion.
    best_id, since = rebuild_best_from_results_dir(
        results_dir, primary, sort_order, min_datasets, dataset_keys)

    # Artifact scan had no eligible metric — use lake-only archives as a
    # compatibility fallback.
    if best_id is None:
        own_lake = False
        if lake is None:
            db_path = (cfg.get("idea_lake_db")
                       or str(Path(results_dir) / "idea_lake.db"))
            lake = IdeaLake(db_path)
            own_lake = True
        try:
            best_id, since = rebuild_best_from_lake(
                lake, primary, sort_order, min_datasets, dataset_keys)
        finally:
            if own_lake:
                try:
                    lake.close()
                except Exception:
                    pass

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
