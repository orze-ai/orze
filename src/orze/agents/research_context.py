"""Context gathering for the orze research agent.

CALLING SPEC:
    load_status(results_dir) -> dict
    load_full_leaderboard(results_dir) -> (entries, metric_name)
    analyze_failures(results_dir, max_scan=2000) -> {total, patterns, recent}
    load_lake_summary(lake_db_path, primary_metric) -> dict | None
    build_context(results_dir, ideas_path, report_cfg, lake_db_path=None) -> str
    get_existing_idea_ids(ideas_path) -> list[str]
    parse_idea_configs(ideas_path) -> dict[str, dict]
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("orze.research_agent")


# ---------------------------------------------------------------------------
#  Context gathering (generic — reads orze results)
# ---------------------------------------------------------------------------

def load_status(results_dir: Path) -> dict:
    """Load status.json written by the orchestrator."""
    status_path = results_dir / "status.json"
    if status_path.exists():
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def load_full_leaderboard(results_dir: Path) -> Tuple[list, str]:
    """Load full leaderboard from _leaderboard.json (richer than status.json).

    Returns (entries, metric_name). Falls back to status.json top_results.
    """
    lb_path = results_dir / "_leaderboard.json"
    if lb_path.exists():
        try:
            data = json.loads(lb_path.read_text(encoding="utf-8"))
            return data.get("top", []), data.get("metric", "")
        except (json.JSONDecodeError, OSError):
            pass
    # Fallback to status.json
    status = load_status(results_dir)
    return status.get("top_results", []), ""


def analyze_failures(results_dir: Path,
                     max_scan: int = 2000) -> Dict[str, Any]:
    """Scan results for failures, categorize by error pattern.

    Only scans the last `max_scan` idea dirs (by name) to avoid slow
    filesystem walks on large result sets. This captures recent failure
    patterns which are most relevant for idea generation.

    Returns dict with:
        total: int
        patterns: list of {pattern, count, example_ids}
        recent: list of {idea_id, error} (last 5)
    """
    errors: List[Tuple[str, str]] = []  # (idea_id, error_msg)
    if not results_dir.exists():
        return {"total": 0, "patterns": [], "recent": []}

    # Collect idea dirs without stat-ing each entry (os.scandir is faster)
    idea_names = []
    try:
        with os.scandir(results_dir) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False) and entry.name.startswith("idea-"):
                    idea_names.append(entry.name)
    except OSError:
        return {"total": 0, "patterns": [], "recent": []}
    idea_names.sort()
    # Only scan the tail for efficiency on large result sets
    scan_names = idea_names[-max_scan:] if len(idea_names) > max_scan else idea_names

    for name in scan_names:
        metrics_path = results_dir / name / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("status") not in ("FAILED", "ERROR"):
            continue
        raw = data.get("error") or data.get("traceback") or data.get("status", "unknown")
        errors.append((name, str(raw).split("\n")[0][:200]))

    if not errors:
        return {"total": 0, "patterns": [], "recent": []}

    # Normalize errors into patterns by stripping variable parts
    def _normalize(msg: str) -> str:
        msg = re.sub(r"idea-[a-z0-9]+", "idea-XXX", msg)
        msg = re.sub(r"Unknown \w+: \S+", "Unknown <type>: <name>", msg)
        msg = re.sub(r"Stalled \(no output for \d+m\)", "Stalled (hung process)", msg)
        msg = re.sub(r"\d+\.\d+", "N", msg)
        return msg.strip()

    pattern_counter: Counter = Counter()
    pattern_examples: Dict[str, List[str]] = defaultdict(list)
    for idea_id, msg in errors:
        pat = _normalize(msg)
        pattern_counter[pat] += 1
        if len(pattern_examples[pat]) < 3:
            pattern_examples[pat].append(idea_id)

    patterns = [
        {"pattern": pat, "count": cnt, "example_ids": pattern_examples[pat]}
        for pat, cnt in pattern_counter.most_common(10)
    ]

    recent = [{"idea_id": iid, "error": msg} for iid, msg in errors[-5:]]
    return {"total": len(errors), "patterns": patterns, "recent": recent}


def _find_best_lake_metric(conn: sqlite3.Connection,
                           primary_metric: str) -> str:
    """Find the best available metric in the lake.

    Tries the primary_metric first, then strips prefixes to find alternatives.
    E.g., if 'my_adjusted_auc' isn't in the lake, tries 'my_auc'.
    """
    if primary_metric:
        count = conn.execute(
            "SELECT COUNT(*) FROM ideas "
            "WHERE json_extract(eval_metrics, ?) IS NOT NULL",
            (f"$.{primary_metric}",),
        ).fetchone()[0]
        if count > 0:
            return primary_metric

    # Try common suffixes/variants
    candidates = []
    if primary_metric:
        # Strip "adjusted_" prefix: e.g. foo_adjusted_auc -> foo_auc
        candidates.append(re.sub(r"_adjusted_", "_", primary_metric))
        # Just "auc" or the last part
        parts = primary_metric.split("_")
        if len(parts) > 1:
            candidates.append("_".join(parts[1:]))  # drop first prefix

    # Find which has most data
    best, best_count = "", 0
    for cand in candidates:
        if not cand:
            continue
        count = conn.execute(
            "SELECT COUNT(*) FROM ideas "
            "WHERE json_extract(eval_metrics, ?) IS NOT NULL",
            (f"$.{cand}",),
        ).fetchone()[0]
        if count > best_count:
            best, best_count = cand, count

    if best:
        logger.info("Lake metric fallback: %s -> %s (%d ideas)",
                     primary_metric, best, best_count)
    return best


def load_lake_summary(lake_db_path: Path, primary_metric: str) -> Optional[dict]:
    """Query idea_lake.db for historical stats and performance patterns.

    Returns None if lake doesn't exist. Otherwise returns:
        status_counts: {status: count}
        top_from_lake: list of {idea_id, title, metric_value}
        config_patterns: {dimension: [{value, mean, count}]}
        lake_metric: str (the metric actually used)
    """
    if not lake_db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(lake_db_path), timeout=10)
        conn.row_factory = sqlite3.Row
    except Exception:
        return None

    result: Dict[str, Any] = {}
    try:
        # Status breakdown
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM ideas GROUP BY status"
        ).fetchall()
        result["status_counts"] = {r["status"]: r["cnt"] for r in rows}

        # Find best available metric
        lake_metric = _find_best_lake_metric(conn, primary_metric)
        result["lake_metric"] = lake_metric

        if lake_metric:
            metric_path = f"$.{lake_metric}"
            top_rows = conn.execute(
                "SELECT idea_id, title, "
                "json_extract(eval_metrics, ?) as metric_val "
                "FROM ideas "
                "WHERE json_extract(eval_metrics, ?) IS NOT NULL "
                "ORDER BY json_extract(eval_metrics, ?) DESC LIMIT 10",
                (metric_path, metric_path, metric_path),
            ).fetchall()
            result["top_from_lake"] = [
                {"idea_id": r["idea_id"], "title": r["title"],
                 "metric_value": r["metric_val"]}
                for r in top_rows
            ]

            result["config_patterns"] = _analyze_config_dimensions(
                conn, lake_metric
            )

    except Exception as e:
        logger.warning("Lake query failed: %s", e)
    finally:
        conn.close()

    return result


def _extract_flat_keys(config: dict, prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested config dict into dot-separated keys.

    Only keeps leaf scalar values (str, int, float, bool).
    Limits depth to 2 levels to avoid noise.
    """
    result = {}
    for key, val in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict) and not prefix:
            result.update(_extract_flat_keys(val, full_key))
        elif not isinstance(val, (dict, list)) and val is not None:
            result[full_key] = val
    return result


def _analyze_config_dimensions(
    conn: sqlite3.Connection, primary_metric: str, max_samples: int = 5000
) -> Dict[str, list]:
    """Group completed ideas by config keys, compute per-value stats.

    Uses pre-computed config_summary column for O(1) key access.
    Limits analysis to top max_samples ideas for scalability.
    """
    metric_path = f"$.{primary_metric}"
    # Sample the best ideas to find winning patterns
    rows = conn.execute(
        "SELECT config_summary, "
        "json_extract(eval_metrics, ?) as metric_val "
        "FROM ideas "
        "WHERE json_extract(eval_metrics, ?) IS NOT NULL "
        "ORDER BY metric_val DESC LIMIT ?",
        (metric_path, metric_path, max_samples),
    ).fetchall()

    if not rows:
        return {}

    # Collect all config dimensions and their values with metrics
    dim_data: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        metric_val = r["metric_val"]
        if metric_val is None:
            continue

        cs = None
        if r["config_summary"]:
            try:
                cs = json.loads(r["config_summary"])
            except (json.JSONDecodeError, TypeError):
                pass

        if not cs:
            continue

        for key, val in cs.items():
            if val is None or isinstance(val, (dict, list)):
                continue
            dim_data[key][str(val)].append(float(metric_val))

    # For each dimension, compute mean + count per value, sorted by mean desc
    # Only keep dimensions where at least 2 values have n >= 3
    patterns: Dict[str, list] = {}
    for dim, val_map in dim_data.items():
        if len(val_map) < 2:
            continue
        entries = []
        for val, metrics in val_map.items():
            if len(metrics) < 3:
                continue  # skip values with too few samples
            entries.append({
                "value": val,
                "mean": round(sum(metrics) / len(metrics), 4),
                "count": len(metrics),
            })
        if len(entries) < 2:
            continue  # need at least 2 values to compare
        entries.sort(key=lambda x: -x["mean"])
        patterns[dim] = entries[:5]

    return patterns


def get_existing_idea_ids(ideas_path: Path) -> List[str]:
    """Parse ideas.md and return list of existing idea IDs."""
    if not ideas_path.exists():
        return []
    text = ideas_path.read_text(encoding="utf-8")
    return [m.group(1) for m in re.finditer(r"^## (idea-[a-z0-9]+):", text, re.MULTILINE)]


def parse_idea_configs(ideas_path: Path) -> Dict[str, dict]:
    """Parse ideas.md and return {idea_id: config_dict} for ideas with YAML."""
    if not ideas_path.exists():
        return {}
    text = ideas_path.read_text(encoding="utf-8")
    ideas = {}
    pattern = re.compile(r"^## (idea-[a-z0-9]+):\s*(.+?)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        idea_id = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        yaml_match = re.search(r"```ya?ml\s*\n(.*?)```", block, re.DOTALL)
        if yaml_match:
            try:
                config = yaml.safe_load(yaml_match.group(1))
                if isinstance(config, dict):
                    ideas[idea_id] = config
            except yaml.YAMLError:
                pass
    return ideas


def build_context(results_dir: Path, ideas_path: Path, report_cfg: dict,
                  lake_db_path: Optional[Path] = None) -> str:
    """Build a comprehensive context string for the LLM.

    Includes: stats, full leaderboard with all metrics, categorized failure
    analysis, historical performance patterns from idea_lake.db, and example
    configs from top performers.
    """
    status = load_status(results_dir)
    existing_ids = get_existing_idea_ids(ideas_path)
    primary_metric = report_cfg.get("primary_metric", "")
    columns = report_cfg.get("columns", [])

    lines = ["# Current Research State\n"]
    lines.append(f"- Total ideas in queue: {len(existing_ids)}")
    lines.append(f"- Completed: {status.get('completed', 0)}")
    lines.append(f"- Failed: {status.get('failed', 0)}")
    lines.append(f"- In queue: {status.get('queue_depth', 0)}")
    if primary_metric:
        lines.append(f"- Primary metric: {primary_metric}")
    lines.append("")

    # --- Full leaderboard from _leaderboard.json ---
    lb_entries, lb_metric = load_full_leaderboard(results_dir)
    if lb_entries:
        lines.append("## Leaderboard (top and bottom focus)\n")
        col_labels = [c.get("label", c["key"]) for c in columns] if columns else []

        def _format_row(i, entry):
            idea_id = entry.get("idea_id", "?")
            title = entry.get("title", "")[:40]
            em = entry.get("eval_metrics", {})
            if col_labels and columns:
                vals = []
                for c in columns:
                    key = c["key"]
                    fmt = c.get("fmt", "")
                    val = em.get(key, entry.get(key, ""))
                    if val != "" and val is not None and fmt:
                        try:
                            val = f"{float(val):{fmt}}"
                        except (ValueError, TypeError):
                            val = str(val)
                    vals.append(str(val) if val is not None else "")
                return f"| {i} | {idea_id} | {title} | " + " | ".join(vals) + " |"
            else:
                score = em.get(primary_metric, entry.get("metric_value", "?"))
                return f"| {i} | {idea_id} | {title} | {score} |"

        if col_labels:
            header = "| # | Idea | Title | " + " | ".join(col_labels) + " |"
            sep = "|---|------|-------|" + "|".join(["---"] * len(col_labels)) + "|"
        else:
            header = "| # | Idea | Title | Score |"
            sep = "|---|------|-------|-------|"

        lines.append(header)
        lines.append(sep)

        # Token Optimization: Only show top 10 and bottom 5 in full detail
        # Show middle as summary to save repetitive tokens
        n_total = len(lb_entries)
        for i, entry in enumerate(lb_entries, 1):
            if i <= 10 or i > (n_total - 5):
                lines.append(_format_row(i, entry))
            elif i == 11:
                lines.append(f"| ... | ... | [{n_total-15} intermediate models omitted] | ... |")
        lines.append("")

    # --- Failure analysis (structured + legacy) ---
    from orze.engine.failure_analysis import load_recent_failures
    structured = load_recent_failures(results_dir)
    if structured:
        total = sum(len(v) for v in structured.values())
        lines.append(f"## Failure Analysis ({total} recent failures, by category)\n")
        for cat, entries in sorted(structured.items(), key=lambda x: -len(x[1])):
            lines.append(f"### {cat} ({len(entries)} failures)")
            lines.append(f"- **Why**: {entries[0]['why']}")
            lines.append(f"- **Lesson**: {entries[0]['lesson']}")
            examples = [e["idea_id"] for e in entries[:3]]
            lines.append(f"- **Examples**: {', '.join(examples)}")
            lines.append("")
    else:
        # Fallback to legacy pattern-based analysis
        failures = analyze_failures(results_dir)
        if failures["total"] > 0:
            lines.append(f"## Failure Analysis ({failures['total']} total failures)\n")
            lines.append("**Top failure patterns** (avoid generating ideas that would hit these):\n")
            for p in failures["patterns"]:
                examples = ", ".join(p["example_ids"][:2])
                lines.append(f"- **{p['count']}x**: {p['pattern']} (e.g. {examples})")
            if failures["recent"]:
                lines.append("\n**Most recent failures:**\n")
                for f in failures["recent"]:
                    lines.append(f"- {f['idea_id']}: {f['error']}")
            lines.append("")

    # --- Historical stats from idea_lake.db ---
    if lake_db_path is None:
        lake_db_path = ideas_path.parent / "idea_lake.db"
    lake = load_lake_summary(lake_db_path, primary_metric)
    if lake:
        sc = lake.get("status_counts", {})
        total_historical = sum(sc.values())
        if total_historical > 0:
            lines.append(f"## Historical Stats (from {total_historical} total experiments)\n")
            parts = [f"{s}: {c}" for s, c in sorted(sc.items(), key=lambda x: -x[1])]
            lines.append("- " + ", ".join(parts))
            lines.append("")

        lake_metric = lake.get("lake_metric", primary_metric)
        top_lake = lake.get("top_from_lake", [])
        if top_lake:
            lines.append(f"## All-Time Best ({lake_metric})\n")
            for i, t in enumerate(top_lake[:5], 1):
                mv = t["metric_value"]
                if isinstance(mv, float):
                    mv = f"{mv:.4f}"
                lines.append(f"{i}. {t['idea_id']}: {mv} — {t['title'][:50]}")
            lines.append("")

        config_patterns = lake.get("config_patterns", {})
        if config_patterns:
            lines.append("## Performance by Config Dimension\n")
            lines.append("Shows which config values correlate with better results.\n")
            for dim, entries in sorted(config_patterns.items()):
                if not entries:
                    continue
                best = entries[0]
                worst = entries[-1] if len(entries) > 1 else None
                line = f"- **{dim}**: best={best['value']} (mean {lake_metric}={best['mean']}, n={best['count']})"
                if worst and worst["value"] != best["value"]:
                    line += f", worst={worst['value']} ({worst['mean']}, n={worst['count']})"
                lines.append(line)
            lines.append("")

    # --- Example configs from top leaderboard ideas ---
    if lb_entries and ideas_path.exists():
        idea_configs = parse_idea_configs(ideas_path)
        top_ids = [e.get("idea_id") for e in lb_entries[:5] if e.get("idea_id")]
        shown = 0
        for tid in top_ids:
            if tid in idea_configs and shown < 2:
                lines.append(f"## Example Config ({tid} — top performer)\n")
                lines.append("```yaml")
                config_yaml = yaml.dump(idea_configs[tid],
                                        default_flow_style=False, sort_keys=False)
                lines.append(config_yaml.rstrip())
                lines.append("```")
                lines.append("")
                shown += 1
        if shown == 0:
            for iid, cfg in list(idea_configs.items())[-2:]:
                lines.append(f"## Example Config ({iid})\n")
                lines.append("```yaml")
                config_yaml = yaml.dump(cfg, default_flow_style=False,
                                        sort_keys=False)
                lines.append(config_yaml.rstrip())
                lines.append("```")
                lines.append("")

    # --- Approach family distribution ---
    try:
        from orze.engine.family_guard import APPROACH_FAMILIES
        if lake_db_path and lake_db_path.exists():
            import sqlite3 as _sql
            _conn = _sql.connect(str(lake_db_path), timeout=5)
            _rows = _conn.execute(
                "SELECT approach_family, COUNT(*) as cnt FROM ideas "
                "WHERE status = 'completed' GROUP BY approach_family"
            ).fetchall()
            _conn.close()
            if _rows:
                lines.append("## Approach Family Distribution (completed ideas)\n")
                for family, count in sorted(_rows, key=lambda x: -x[1]):
                    lines.append(f"- {family or 'other'}: {count}")
                lines.append("")
                lines.append("Ensure diversity: explore under-represented families.\n")
    except Exception:
        pass

    # --- Existing idea IDs (dedup hint) ---
    if existing_ids:
        tail = existing_ids[-50:]
        lines.append(f"## Existing Ideas (last {len(tail)} of {len(existing_ids)} — avoid duplicates)\n")
        lines.append(", ".join(tail))
        lines.append("")

    return "\n".join(lines)
