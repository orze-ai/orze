"""Admin cache writer and report formatting for the admin panel.

CALLING SPEC:
    from orze.reporting.leaderboard_admin import write_admin_cache, format_report_text

    write_admin_cache(results_dir, ideas, cfg)
        -> None (writes _admin_cache.json)

    format_report_text(data) -> str
        data: dict with keys title, completed, failed, active_count, queued,
              leaderboard, metric_name, machines
"""

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict

from orze.core.fs import atomic_write
from orze.core.ideas import expand_sweeps


def _read_all_heartbeats(results_dir: Path, stale_seconds: int = 600):
    """Read heartbeat JSON files from results dir."""
    heartbeats = []
    try:
        for f in results_dir.glob("_heartbeat_*.json"):
            try:
                hb = json.loads(f.read_text(encoding="utf-8"))
                heartbeats.append(hb)
            except (json.JSONDecodeError, OSError):
                pass
    except OSError:
        pass
    return heartbeats


def write_admin_cache(results_dir: Path, ideas: dict, cfg: dict):
    """Write pre-aggregated _admin_cache.json for instant admin panel access."""
    now = time.time()

    # Nodes
    raw_hb = _read_all_heartbeats(results_dir, stale_seconds=600)
    heartbeats = []
    for hb in raw_hb:
        age = now - hb.get("epoch", 0)
        status = "online" if age <= 120 else ("degraded" if age <= 300 else "offline")
        heartbeats.append({**hb, "status": status, "heartbeat_age_sec": round(age, 1)})

    # Queue
    sweep_max = cfg.get("sweep", {}).get("max_combos", 20)
    expanded = expand_sweeps(dict(ideas), max_combos=sweep_max)
    all_statuses: dict = {}
    queue_items = []
    for idea_id, idea in expanded.items():
        idea_dir = results_dir / idea_id
        idea_status = "pending"
        if idea_dir.exists():
            mpath = idea_dir / "metrics.json"
            if mpath.exists():
                try:
                    m = json.loads(mpath.read_text(encoding="utf-8"))
                    idea_status = m.get("status", "COMPLETED").lower()
                except (json.JSONDecodeError, OSError):
                    idea_status = "running"
            else:
                idea_status = "running"
        all_statuses[idea_status] = all_statuses.get(idea_status, 0) + 1
        raw = idea.get("raw", "")
        _cat_m = re.search(r"\*\*Category\*\*:\s*(.+)", raw)
        _par_m = re.search(r"\*\*Parent\*\*:\s*(.+)", raw)
        _hyp_m = re.search(r"\*\*Hypothesis\*\*:\s*(.+)", raw)
        queue_items.append({
            "idea_id": idea_id,
            "title": idea.get("title", ""),
            "priority": idea.get("priority", "medium"),
            "status": idea_status,
            "config": idea.get("config", {}),
            "sweep_parent": idea.get("_sweep_parent"),
            "category": _cat_m.group(1).strip() if _cat_m else "architecture",
            "parent": _par_m.group(1).strip() if _par_m else "none",
            "hypothesis": _hyp_m.group(1).strip() if _hyp_m else "",
        })

    # Alerts
    alerts = []
    two_hours_ago = now - 7200
    try:
        with os.scandir(results_dir) as it:
            for entry in it:
                if not entry.is_dir() or not entry.name.startswith("idea-"):
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                if mtime < two_hours_ago:
                    continue
                mpath = Path(entry.path) / "metrics.json"
                if not mpath.exists():
                    continue
                try:
                    m = json.loads(mpath.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if m.get("status") not in ("FAILED", "ERROR"):
                    continue
                alerts.append({
                    "type": "failure", "idea_id": entry.name,
                    "error": str(m.get("error", m.get("status", "")))[:200],
                    "minutes_ago": round((now - mtime) / 60, 1),
                })
    except OSError:
        pass

    for hb in heartbeats:
        if hb.get("status") == "offline":
            alerts.append({
                "type": "stale_host",
                "host": hb.get("host", "unknown"),
                "minutes_ago": round(hb.get("heartbeat_age_sec", 0) / 60, 1),
            })

    try:
        usage = shutil.disk_usage(results_dir)
        if round(usage.free / (1024 ** 3), 1) < 50:
            alerts.append({"type": "low_disk",
                           "disk_free_gb": round(usage.free / (1024 ** 3), 1)})
    except Exception:
        pass

    cache = {
        "nodes": {"heartbeats": heartbeats, "local_gpus": []},
        "queue": {"items": queue_items, "counts": all_statuses,
                  "total_all": sum(all_statuses.values())},
        "alerts": {"alerts": alerts, "count": len(alerts)},
        "epoch": now,
    }
    atomic_write(results_dir / "_admin_cache.json",
                 json.dumps(cache, default=str))


def format_report_text(data: dict) -> str:
    """Format a periodic report summary for notifications."""
    c, f, a, q = (data.get(k, 0) for k in
                   ("completed", "failed", "active_count", "queued"))
    title = data.get("title", "Report")
    metric = data.get("metric_name", "score")
    board = data.get("leaderboard", [])
    machines = data.get("machines", [])

    lines = [title, f"{c} completed | {f} failed | {a} active | {q} queued", ""]

    if machines:
        lines.append("Machines:")
        for m in machines:
            lines.append(f"  {m.get('host','?')}: "
                         f"{m.get('gpus_busy',0)}/{m.get('gpus_total',0)} GPUs, "
                         f"{m.get('utilization','?')}% util")
        lines.append("")

    if board:
        lines.append(f"Top {len(board)} ({metric}):")
        for i, entry in enumerate(board, 1):
            val = entry.get("value")
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            lines.append(f"  #{i} {entry.get('id','?')}: {val_str} "
                         f"{entry.get('title','')[:25]}")

    return "\n".join(lines)
