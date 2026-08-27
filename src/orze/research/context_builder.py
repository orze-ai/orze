"""Bounded research-context digest (≤16 KB).

Produces a compact text summary of recent experiments for the research
role: evidence-qualified top-N metrics, recent failures (classified),
last-3-cycle deltas, and approach-family counts inferred on-the-fly from
config keys.

Replaces the prior ``engine/experiment_analysis.py`` +
``engine/retrospection.py`` prose generators. No taxonomy file; families
are inferred from the config keys that differ across ideas.

CALLING SPEC:
    build_digest(results_dir, cfg, top_n=10, max_bytes=16384) -> str
"""

from __future__ import annotations

import heapq
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("orze")

MAX_BYTES_DEFAULT = 16 * 1024
RECENT_RESULT_LIMIT = 200
LEADER_CANDIDATE_LIMIT = 50


def _load_metrics(idea_dir: Path) -> Optional[dict]:
    mf = idea_dir / "metrics.json"
    if mf.is_symlink() or not mf.is_file():
        return None
    try:
        if mf.stat().st_size > 4 * 1024 * 1024:
            return None
        data = json.loads(mf.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _load_config(idea_dir: Path) -> Dict[str, Any]:
    p = idea_dir / "idea_config.yaml"
    if p.is_symlink() or not p.is_file():
        return {}
    try:
        if p.stat().st_size > 1024 * 1024:
            return {}
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
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
    """Infer a coarse family without copying arbitrary config values.

    The digest is sent to an LLM-backed role. A model path, URL, optimizer
    string, or other user-controlled value can contain credentials or private
    identifiers, so only the presence of a known structural key is exported.
    """
    for hint_key in ("model", "backbone", "arch", "architecture", "optimizer",
                     "train_script"):
        for k, v in flat_cfg.items():
            if k.endswith(hint_key) and isinstance(v, str):
                return f"{hint_key}-configured"
    return "other"


def _row_from_dir(idea_dir: Path) -> Optional[dict]:
    if idea_dir.is_symlink() or not idea_dir.is_dir():
        return None
    metrics = _load_metrics(idea_dir)
    if metrics is None:
        return None
    cfg = _load_config(idea_dir)
    try:
        mtime = (idea_dir / "metrics.json").stat().st_mtime
    except OSError:
        return None
    return {
        "id": idea_dir.name,
        "status": metrics.get("status"),
        "metrics": metrics,
        "config_flat": _flatten(cfg),
        "mtime": mtime,
    }


def _collect_recent(results_dir: Path,
                    limit: int = RECENT_RESULT_LIMIT) -> List[dict]:
    if not results_dir.exists() or limit <= 0:
        return []
    newest: List[tuple[float, str, Path]] = []
    for d in results_dir.iterdir():
        if d.is_symlink() or not d.is_dir() or not d.name.startswith("idea-"):
            continue
        metrics_path = d / "metrics.json"
        if metrics_path.is_symlink() or not metrics_path.is_file():
            continue
        try:
            candidate = (metrics_path.stat().st_mtime, d.name, d)
        except OSError:
            continue
        if len(newest) < limit:
            heapq.heappush(newest, candidate)
        elif candidate[:2] > newest[0][:2]:
            heapq.heapreplace(newest, candidate)
    rows = []
    for _, _, idea_dir in sorted(newest, reverse=True):
        row = _row_from_dir(idea_dir)
        if row is not None:
            rows.append(row)
    return rows


def _cached_leader_ids(results_dir: Path, primary: str,
                       limit: int = LEADER_CANDIDATE_LIMIT) -> List[str]:
    """Return bounded candidate IDs from the report cache, never its scores.

    The cache can be stale or malformed, so it is only an index into current
    artifacts. Every returned candidate is independently re-qualified below.
    """
    path = results_dir / "_leaderboard.json"
    if path.is_symlink() or not path.is_file():
        return []
    try:
        if path.stat().st_size > 4 * 1024 * 1024:
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict) or data.get("metric") != primary:
        return []
    top = data.get("top")
    if not isinstance(top, list):
        return []
    out: List[str] = []
    for entry in top:
        if not isinstance(entry, dict):
            continue
        idea_id = entry.get("idea_id")
        if (not isinstance(idea_id, str) or not idea_id.startswith("idea-")
                or Path(idea_id).parts != (idea_id,)
                or idea_id in (".", "..")):
            continue
        if idea_id not in out:
            out.append(idea_id)
        if len(out) >= limit:
            break
    return out


def _candidate_rows(results_dir: Path, primary: str) -> tuple[List[dict], int]:
    """Combine recent outcomes with cached historic leaders by safe identity."""
    rows = _collect_recent(results_dir)
    by_id = {row["id"]: row for row in rows}
    historic_added = 0
    for idea_id in _cached_leader_ids(results_dir, primary):
        if idea_id in by_id:
            continue
        row = _row_from_dir(results_dir / idea_id)
        if row is not None:
            by_id[idea_id] = row
            historic_added += 1
    return list(by_id.values()), historic_added


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

    rows, historic_added = _candidate_rows(results_dir, primary)
    # Bind qualification to the exact directory passed by the caller. The
    # report resolver re-reads current artifacts, enforces configured source
    # mappings/coverage/validation, and verifies benchmark receipts when a
    # contract is configured. Raw values collected above are never ranked.
    scoped_cfg = dict(cfg)
    scoped_cfg["_env_ORZE_RESULTS_DIR"] = str(results_dir.resolve())
    from orze.reporting.search_path import make_evidence_metric_resolver
    metric_of, lower_is_better, resolved_primary, qualification = (
        make_evidence_metric_resolver("unused.db", scoped_cfg)
    )
    primary = resolved_primary
    completed = []
    for row in rows:
        value = metric_of({"idea_id": row["id"]})
        if value is None:
            continue
        qualified = dict(row)
        qualified["qualified_metric"] = value
        completed.append(qualified)
    failed = [r for r in rows if r["status"] and r["status"] != "COMPLETED"]

    completed.sort(key=lambda r: r["qualified_metric"],
                   reverse=not lower_is_better)

    lines: List[str] = []
    lines.append(f"# Research context digest ({len(completed)} qualified "
                 f"candidates, {len(failed)} recent failures)")
    lines.append(f"primary_metric: {primary}  sort: "
                 f"{'asc' if lower_is_better else 'desc'}")
    lines.append(
        f"evidence_mode: {qualification.get('mode', 'unavailable')}  "
        "leaderboard_rank_comparable: false"
    )
    lines.append(
        f"candidate_scope: {RECENT_RESULT_LIMIT} most recent artifacts + "
        f"{historic_added} independently re-qualified cached leaders"
    )
    rejected_count = sum((qualification.get("rejected") or {}).values())
    lines.append(
        f"qualification: {qualification.get('accepted', 0)} accepted, "
        f"{rejected_count} rejected"
    )
    if qualification.get("mode") == "benchmark_contract":
        lines.append(
            "comparison_identity: benchmark contract verified; local ordering "
            "only, never an official rank"
        )
    else:
        lines.append(
            "comparison_identity: local artifact policy only; evaluator/dataset "
            "identity is not cryptographically proven"
        )
    lines.append("")

    # Top-N
    lines.append(
        f"## Top-{top_n} evidence-qualified local candidates by {primary}"
    )
    for r in completed[:top_n]:
        v = r["qualified_metric"]
        fam = _infer_family(r["config_flat"])
        lines.append(f"  {r['id']:30s}  {primary}={v:<8}  [{fam}]")
    if not completed:
        lines.append("  none")
    lines.append("")

    # Last-3-cycle deltas (most recent 3 completions)
    recent3 = sorted(completed, key=lambda r: r["mtime"], reverse=True)[:3]
    if len(recent3) >= 2:
        lines.append("## Last-3 deltas")
        for a, b in zip(recent3, recent3[1:]):
            va, vb = a["qualified_metric"], b["qualified_metric"]
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
            # Raw errors can contain tokens, URLs, paths, dataset excerpts, or
            # other operator-controlled content. Only the stable local
            # classifier result crosses the research-prompt boundary.
            lines.append(f"  {r['id']:30s}  [{cat}]")
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
