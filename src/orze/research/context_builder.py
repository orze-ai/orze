"""Bounded research-context digest (≤16 KB).

Produces a compact text summary of recent experiments for the research
role: top-N metrics, recent failures (classified), last-3-cycle deltas,
and approach-family counts inferred on-the-fly from config keys.

Replaces the prior ``engine/experiment_analysis.py`` +
``engine/retrospection.py`` prose generators. No taxonomy file; families
are inferred from the config keys that differ across ideas.

CALLING SPEC:
    build_digest(results_dir, cfg, top_n=10, max_bytes=16384) -> str
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("orze")

MAX_BYTES_DEFAULT = 16 * 1024


def _load_metrics(idea_dir: Path) -> Optional[dict]:
    mf = idea_dir / "metrics.json"
    if not mf.exists():
        return None
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_config(idea_dir: Path) -> Dict[str, Any]:
    p = idea_dir / "idea_config.yaml"
    if not p.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (str, int, float, bool)):
            out[key] = v
    return out


def _infer_family(flat_cfg: Dict[str, Any]) -> str:
    """Infer approach family from salient config keys (on-the-fly)."""
    for hint_key in ("model", "backbone", "arch", "architecture", "optimizer",
                     "train_script"):
        for k, v in flat_cfg.items():
            if k.endswith(hint_key) and isinstance(v, str):
                return f"{hint_key}={v}"
    return "other"


def _collect(results_dir: Path, limit: int = 200) -> List[dict]:
    rows: List[dict] = []
    if not results_dir.exists():
        return rows
    for d in results_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        m = _load_metrics(d)
        if not m:
            continue
        cfg = _load_config(d)
        rows.append({
            "id": d.name,
            "status": m.get("status"),
            "metrics": m,
            "config_flat": _flatten(cfg),
            "mtime": (d / "metrics.json").stat().st_mtime,
        })
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows[:limit]


def build_digest(results_dir: Path,
                 cfg: dict,
                 top_n: int = 10,
                 max_bytes: int = MAX_BYTES_DEFAULT) -> str:
    """Build a bounded text digest for the research role.

    Returns a plain-text string ≤ ``max_bytes`` bytes.
    """
    results_dir = Path(results_dir)
    report_cfg = cfg.get("report", {})
    primary = report_cfg.get("primary_metric", "")
    sort_desc = report_cfg.get("sort", "descending") == "descending"

    rows = _collect(results_dir)
    completed = [r for r in rows if r["status"] == "COMPLETED"
                 and isinstance(r["metrics"].get(primary), (int, float))]
    failed = [r for r in rows if r["status"] and r["status"] != "COMPLETED"]

    completed.sort(key=lambda r: r["metrics"].get(primary, 0),
                   reverse=sort_desc)

    lines: List[str] = []
    lines.append(f"# Research context digest ({len(completed)} completed, "
                 f"{len(failed)} failed)")
    lines.append(f"primary_metric: {primary}  sort: "
                 f"{'desc' if sort_desc else 'asc'}")
    lines.append("")

    # Top-N
    lines.append(f"## Top-{top_n} by {primary}")
    for r in completed[:top_n]:
        v = r["metrics"].get(primary)
        fam = _infer_family(r["config_flat"])
        lines.append(f"  {r['id']:30s}  {primary}={v:<8}  [{fam}]")
    lines.append("")

    # Last-3-cycle deltas (most recent 3 completions)
    recent3 = sorted(completed, key=lambda r: r["mtime"], reverse=True)[:3]
    if len(recent3) >= 2:
        lines.append("## Last-3 deltas")
        for a, b in zip(recent3, recent3[1:]):
            va, vb = a["metrics"].get(primary), b["metrics"].get(primary)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                lines.append(f"  {a['id']} vs {b['id']}: Δ={va - vb:+.4f}")
        lines.append("")

    # Recent failures classified
    if failed:
        try:
            from orze.engine.failure import classify_failure
        except Exception:
            classify_failure = None
        lines.append(f"## Recent failures (top {min(10, len(failed))})")
        for r in failed[:10]:
            err = r["metrics"].get("error_message") or r["metrics"].get("error") or ""
            cat = "?"
            if classify_failure:
                try:
                    cat = classify_failure(str(err))
                except Exception:
                    cat = "?"
            msg = (str(err)[:80]).replace("\n", " ")
            lines.append(f"  {r['id']:30s}  [{cat}]  {msg}")
        lines.append("")

    # Approach-family counts
    families = Counter(_infer_family(r["config_flat"]) for r in completed)
    if families:
        lines.append("## Approach-family counts")
        for fam, n in families.most_common(10):
            lines.append(f"  {n:4d}  {fam}")

    text = "\n".join(lines)
    if len(text.encode("utf-8")) > max_bytes:
        # Truncate to fit budget
        text = text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
    return text
