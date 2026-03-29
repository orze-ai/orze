#!/usr/bin/env python3
"""Meta-research agent: adjusts research strategy based on performance patterns.

CALLING SPEC:
    run_meta_research(cfg, results_dir, signal) -> bool
        Analyzes approach family performance from idea_lake.
        Updates RESEARCH_RULES.md with adjusted exploration strategy.
        Returns True if rules were updated.

    analyze_family_performance(lake_db_path, primary_metric) -> dict[str, dict]
        Per-family stats: count, mean, best, win_rate.

    build_strategy_prompt(family_stats, current_rules, signal) -> str
        Prompt LLM to update research rules based on family analysis.

CLI:
    python -m orze.agents.meta_research \\
        -c orze.yaml --results-dir results --signal "family_imbalance"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orze.meta_research")


def analyze_family_performance(lake_db_path: Path,
                               primary_metric: str) -> Dict[str, Dict[str, Any]]:
    """Compute per-family performance stats from idea_lake.

    Returns {family: {count, mean, best, worst, win_rate}}.
    """
    if not lake_db_path.exists():
        return {}

    try:
        conn = sqlite3.connect(str(lake_db_path), timeout=10)
    except Exception:
        return {}

    stats: Dict[str, Dict[str, Any]] = {}
    try:
        metric_path = f"$.{primary_metric}" if primary_metric else None

        # Count per family
        rows = conn.execute(
            "SELECT approach_family, COUNT(*) as cnt FROM ideas "
            "WHERE status = 'completed' GROUP BY approach_family"
        ).fetchall()
        total = sum(r[1] for r in rows)

        for family, count in rows:
            family = family or "other"
            stats[family] = {"count": count, "share": round(count / total, 2) if total else 0}

        # Metric stats per family (if metric available)
        if metric_path:
            for family in list(stats.keys()):
                rows = conn.execute(
                    "SELECT json_extract(eval_metrics, ?) as val FROM ideas "
                    "WHERE status = 'completed' AND approach_family = ? "
                    "AND json_extract(eval_metrics, ?) IS NOT NULL",
                    (metric_path, family, metric_path),
                ).fetchall()
                vals = [float(r[0]) for r in rows if r[0] is not None]
                if vals:
                    stats[family]["mean"] = round(sum(vals) / len(vals), 4)
                    stats[family]["best"] = round(max(vals), 4)
                    stats[family]["worst"] = round(min(vals), 4)

            # Win rate: how often a family produces top-10% results
            if total > 10:
                top_threshold = conn.execute(
                    "SELECT json_extract(eval_metrics, ?) as val FROM ideas "
                    "WHERE status = 'completed' AND json_extract(eval_metrics, ?) IS NOT NULL "
                    "ORDER BY val DESC LIMIT 1 OFFSET ?",
                    (metric_path, metric_path, max(1, total // 10)),
                ).fetchone()
                if top_threshold and top_threshold[0] is not None:
                    threshold_val = float(top_threshold[0])
                    for family in list(stats.keys()):
                        top_count = conn.execute(
                            "SELECT COUNT(*) FROM ideas "
                            "WHERE status = 'completed' AND approach_family = ? "
                            "AND json_extract(eval_metrics, ?) >= ?",
                            (family, metric_path, threshold_val),
                        ).fetchone()[0]
                        fam_count = stats[family]["count"]
                        stats[family]["win_rate"] = (
                            round(top_count / fam_count, 2) if fam_count else 0
                        )

    except Exception as e:
        logger.warning("Family performance analysis failed: %s", e)
    finally:
        conn.close()

    return stats


def build_strategy_prompt(family_stats: Dict[str, Dict[str, Any]],
                          current_rules: str,
                          signal: str) -> str:
    """Build prompt for the meta-research LLM."""
    stats_text = ""
    for family, s in sorted(family_stats.items(),
                             key=lambda x: x[1].get("count", 0), reverse=True):
        line = f"- **{family}**: {s.get('count', 0)} ideas ({s.get('share', 0):.0%})"
        if "mean" in s:
            line += f", mean={s['mean']}, best={s['best']}"
        if "win_rate" in s:
            line += f", win_rate={s['win_rate']:.0%}"
        stats_text += line + "\n"

    return f"""\
You are the meta-research agent for orze, an automated ML experiment system.
The system has detected: **{signal}**.

Your job: analyze which approach families are working and adjust the research
strategy to improve exploration efficiency.

## Current Performance by Approach Family
{stats_text}

## Current Research Rules
```
{current_rules[:4000] if current_rules else "(no rules file found)"}
```

## Your Task
Based on the performance data above:

1. Identify which families are over-represented but under-performing
2. Identify which families are under-explored but show promise
3. Write an updated strategy section that should be APPENDED to the
   research rules file

Output ONLY the text to append to the rules file. Format as markdown.
Example:
```
## Auto-Generated Strategy (updated by meta-research agent)

- Allocate ~40% of ideas to architecture exploration (currently under-represented)
- Reduce training_config variations (saturated at {{n}} ideas, diminishing returns)
- Try at least 1 ensemble idea per research cycle
- Focus data augmentation on techniques not yet tried: [specific suggestions]
```

Be specific and actionable. Reference the performance data to justify each directive.
"""


def run_meta_research(cfg: dict, results_dir: Path,
                      signal: str = "family_imbalance") -> bool:
    """Run meta-research to adjust research strategy. Returns True if rules updated."""
    logger.info("=" * 60)
    logger.info("META-RESEARCH — signal: %s", signal)
    logger.info("=" * 60)

    # 1. Analyze family performance
    lake_path = results_dir.parent / "idea_lake.db"
    if not lake_path.exists():
        lake_path = results_dir / ".." / "idea_lake.db"
    primary = cfg.get("report", {}).get("primary_metric", "")
    stats = analyze_family_performance(lake_path, primary)
    if not stats:
        logger.warning("No family performance data available")
        return False

    logger.info("Family stats: %s", {k: v.get("count", 0) for k, v in stats.items()})

    # 2. Read current rules
    rules_file = None
    for role_cfg in (cfg.get("roles") or {}).values():
        if isinstance(role_cfg, dict):
            rf = role_cfg.get("rules_file")
            if rf and Path(rf).exists():
                rules_file = Path(rf)
                break
    current_rules = ""
    if rules_file:
        try:
            current_rules = rules_file.read_text(encoding="utf-8")
        except OSError:
            pass

    # 3. Build prompt and call LLM
    prompt = build_strategy_prompt(stats, current_rules, signal)

    meta_cfg = cfg.get("meta_research", cfg.get("evolution", {}))
    model = meta_cfg.get("model", "sonnet")
    timeout = meta_cfg.get("timeout", 300)

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)

    claude_bin = meta_cfg.get("claude_bin") or "claude"
    cmd = [
        claude_bin, "-p", prompt,
        "--dangerously-skip-permissions",
        "--output-format", "text",
        "--model", model,
    ]

    logger.info("Calling Claude CLI (model=%s)...", model)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, env=env,
            cwd=str(results_dir.parent),
        )
        response = result.stdout.strip() if result.stdout else ""
    except subprocess.TimeoutExpired:
        logger.warning("Meta-research timed out")
        return False
    except FileNotFoundError:
        logger.warning("Claude CLI not found — skipping meta-research")
        return False
    except Exception as e:
        logger.error("Meta-research error: %s", e)
        return False

    if not response:
        logger.warning("Meta-research returned empty response")
        return False

    # 4. Append strategy to rules file
    if rules_file:
        try:
            # Remove any prior auto-generated section
            content = rules_file.read_text(encoding="utf-8")
            marker = "## Auto-Generated Strategy"
            if marker in content:
                content = content[:content.index(marker)].rstrip() + "\n\n"
            else:
                content = content.rstrip() + "\n\n"

            content += response.strip() + "\n"
            rules_file.write_text(content, encoding="utf-8")
            logger.info("Updated research rules: %s", rules_file)
            return True
        except OSError as e:
            logger.error("Could not update rules file: %s", e)
            return False
    else:
        # No rules file — write to results dir
        output_path = results_dir / "_meta_research_strategy.md"
        try:
            output_path.write_text(response, encoding="utf-8")
            logger.info("Strategy written to %s (no rules_file configured)",
                        output_path)
            return True
        except OSError:
            return False


def main():
    parser = argparse.ArgumentParser(
        description="orze meta-research agent — adjust research strategy",
    )
    parser.add_argument("-c", "--config", default="orze.yaml",
                        help="Path to orze.yaml")
    parser.add_argument("--results-dir", default="",
                        help="Path to results dir")
    parser.add_argument("--signal", default="family_imbalance",
                        help="What triggered this meta-research cycle")

    args = parser.parse_args()

    cfg = {}
    config_path = Path(args.config)
    if config_path.exists():
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning("Could not load %s: %s", args.config, e)

    results_dir = Path(args.results_dir or cfg.get("results_dir", "results"))
    success = run_meta_research(cfg, results_dir, args.signal)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
