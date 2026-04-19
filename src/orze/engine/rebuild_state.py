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

This module rebuilds both fields from the authoritative
``idea_lake.db``:

    best_idea_id = argmax_metric(completed ideas)
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
import socket
from pathlib import Path
from typing import Optional, Tuple

from orze.reporting.state import load_state, save_state

logger = logging.getLogger("orze")


def rebuild_best_from_lake(lake, primary_metric: str
                           ) -> Tuple[Optional[str], int]:
    """Return (best_idea_id, completions_since_best) from the lake.

    Queries ``eval_metrics`` JSON column. Returns (None, 0) if no
    completed idea has the metric recorded — in that case the caller
    should try ``rebuild_best_from_results_dir``.
    """
    if lake is None or getattr(lake, "conn", None) is None:
        return None, 0
    # Highest primary metric among completed ideas.
    cur = lake.conn.execute(
        "SELECT idea_id, archived_at, "
        "json_extract(eval_metrics, ?) AS val "
        "FROM ideas "
        "WHERE status = 'completed' "
        "AND json_extract(eval_metrics, ?) IS NOT NULL "
        "ORDER BY val DESC, archived_at ASC LIMIT 1",
        (f"$.{primary_metric}", f"$.{primary_metric}"),
    ).fetchone()
    if cur is None:
        return None, 0
    best_id = cur[0] if isinstance(cur, tuple) else cur["idea_id"]
    best_archived = cur[1] if isinstance(cur, tuple) else cur["archived_at"]
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
                                  primary_metric: str
                                  ) -> Tuple[Optional[str], int]:
    """Fallback: scan ``<results>/idea-*/metrics.json`` when the lake
    has no ``eval_metrics`` populated.

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
        val = None
        for candidate in (data.get("metrics", {}).get(primary_metric)
                          if isinstance(data.get("metrics"), dict) else None,
                          data.get(primary_metric)):
            if isinstance(candidate, (int, float)):
                val = float(candidate)
                break
        if val is None:
            continue
        mtime = mpath.stat().st_mtime
        completed.append((idea_dir.name, val, mtime))
        if best_val is None or val > best_val:
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

    primary = cfg.get("report", {}).get("primary_metric", "test_accuracy")

    own_lake = False
    if lake is None:
        db_path = cfg.get("idea_lake_db") or str(Path(results_dir) / "idea_lake.db")
        lake = IdeaLake(db_path)
        own_lake = True
    try:
        best_id, since = rebuild_best_from_lake(lake, primary)
    finally:
        if own_lake:
            try:
                lake.close()
            except Exception:
                pass

    # Lake had no eval_metrics populated — fall back to per-idea
    # metrics.json scan (authoritative source, but slower).
    if best_id is None:
        best_id, since = rebuild_best_from_results_dir(results_dir, primary)

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
