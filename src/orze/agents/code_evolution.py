#!/usr/bin/env python3
"""Code evolution agent: proactive backward-compatible code changes.

CALLING SPEC:
    run_evolution_cycle(cfg, results_dir, ideas_path, trigger) -> int
        Reads leaderboard, failure analyses, and training script source.
        Proposes backward-compatible code changes via Claude CLI.
        Generates matching ideas that exercise new code paths.
        Returns number of ideas generated.

    build_evolution_context(results_dir, ideas_path, cfg) -> str
        Build context for the evolution LLM: leaderboard, failures,
        training script source, sealed file list.

    build_evolution_prompt(context, trigger, train_script_content,
                           sealed_files) -> str
        Build prompt instructing the LLM to propose code changes.

CLI:
    python -m orze.agents.code_evolution \\
        -c orze.yaml --results-dir results --ideas-md ideas.md \\
        --trigger "plateau"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orze.code_evolution")


def build_evolution_context(results_dir: Path, ideas_path: Path,
                            cfg: dict) -> str:
    """Build context for code evolution: leaderboard, failures, family stats."""
    from orze.agents.research_context import (
        load_full_leaderboard, load_status,
    )
    from orze.engine.failure_analysis import load_recent_failures

    lines = ["# Code Evolution Context\n"]

    # Status
    status = load_status(results_dir)
    lines.append(f"- Completed: {status.get('completed', 0)}")
    lines.append(f"- Failed: {status.get('failed', 0)}")
    lines.append("")

    # Top leaderboard
    report_cfg = cfg.get("report", {})
    primary = report_cfg.get("primary_metric", "")
    lb_entries, _ = load_full_leaderboard(results_dir)
    if lb_entries:
        lines.append("## Top 5 Results\n")
        for i, entry in enumerate(lb_entries[:5], 1):
            iid = entry.get("idea_id", "?")
            em = entry.get("eval_metrics", {})
            score = em.get(primary, "?") if primary else "?"
            lines.append(f"{i}. {iid}: {score}")
        lines.append("")

    # Failure analysis
    failures = load_recent_failures(results_dir)
    if failures:
        total = sum(len(v) for v in failures.values())
        lines.append(f"## Recent Failures ({total} total)\n")
        for cat, entries in sorted(failures.items(), key=lambda x: -len(x[1])):
            lines.append(f"- **{cat}** ({len(entries)}): {entries[0]['lesson']}")
        lines.append("")

    # Family distribution from lake
    lake_path = ideas_path.parent / "idea_lake.db"
    if lake_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(lake_path), timeout=5)
            rows = conn.execute(
                "SELECT approach_family, COUNT(*) FROM ideas "
                "WHERE status = 'completed' GROUP BY approach_family"
            ).fetchall()
            conn.close()
            if rows:
                lines.append("## Approach Family Distribution\n")
                for family, count in sorted(rows, key=lambda x: -x[1]):
                    lines.append(f"- {family or 'other'}: {count}")
                lines.append("")
        except Exception:
            pass

    return "\n".join(lines)


def build_evolution_prompt(context: str, trigger: str,
                           train_script_content: str,
                           sealed_files: list) -> str:
    """Build the prompt for the code evolution LLM."""
    sealed_section = ""
    if sealed_files:
        sealed_section = (
            "\n## Sealed Files (DO NOT MODIFY)\n"
            + "\n".join(f"- {f}" for f in sealed_files)
            + "\n"
        )

    return f"""\
You are the code evolution agent for orze, an automated ML experiment system.
The system has detected: **{trigger}**.

Your job: make backward-compatible code changes to the training pipeline that
unlock new experiment possibilities, then generate ideas that use those changes.

{context}

## Current Training Script
```python
{train_script_content[:8000]}
```
{sealed_section}
## Rules
1. **Backward compatible**: All existing configs must still work unchanged.
   Use `if config.get("new_key"):` branches, never replace existing behavior.
2. **Additive only**: Add new functions, classes, or config branches. Do not
   remove or rename existing code.
3. **Sealed files**: Do NOT modify any sealed files listed above.
4. **Framework off-limits**: Do NOT modify files under `orze/` directory.
5. **Generate ideas**: After making code changes, generate 2-3 new ideas
   (as JSON) that exercise the new code paths.
6. **Verify syntax**: After editing, run `python -c "import ast; ast.parse(open('FILE').read())"`
   on each modified Python file.

## What to Change
Based on the failure patterns and leaderboard plateau:
- Add new model architectures or layers
- Add new loss functions or training strategies
- Add new data augmentation or preprocessing options
- Add new regularization techniques
- Optimize bottlenecks (memory, speed)

## Output Format
After making code changes, output ideas as a JSON array:
```json
[
  {{
    "title": "Try new feature X",
    "hypothesis": "The new code path adds X which should improve Y because Z",
    "config": {{"new_key": "value", "existing_key": "existing_value"}},
    "approach_family": "architecture",
    "priority": "high"
  }}
]
```
"""


def _parse_ideas_from_output(output: str, results_dir: Path,
                             cycle: int = 0) -> list:
    """Extract ideas JSON from Claude CLI output."""
    from orze.agents.research import parse_llm_ideas
    return parse_llm_ideas(output, results_dir, cycle)


def run_evolution_cycle(cfg: dict, results_dir: Path, ideas_path: Path,
                        trigger: str = "plateau") -> int:
    """Run one code evolution cycle. Returns number of ideas generated."""
    logger.info("=" * 60)
    logger.info("CODE EVOLUTION — trigger: %s", trigger)
    logger.info("=" * 60)

    # 1. Build context
    context = build_evolution_context(results_dir, ideas_path, cfg)

    # 2. Read training script
    train_script = cfg.get("train_script", "train.py")
    train_content = ""
    try:
        train_content = Path(train_script).read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Could not read training script %s: %s", train_script, e)

    # 3. Build prompt
    sealed_files = cfg.get("sealed_files", [])
    prompt = build_evolution_prompt(context, trigger, train_content, sealed_files)

    # 4. Call Claude CLI
    evo_cfg = cfg.get("evolution", {})
    model = evo_cfg.get("model", "opus")
    timeout = evo_cfg.get("timeout", 900)

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)

    claude_bin = evo_cfg.get("claude_bin") or "claude"
    cmd = [
        claude_bin, "-p", prompt,
        "--dangerously-skip-permissions",
        "--output-format", "text",
        "--model", model,
    ]

    logger.info("Calling Claude CLI (model=%s, timeout=%ds)...", model, timeout)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, env=env,
            cwd=str(results_dir.parent),
        )
        response = result.stdout[-10000:] if result.stdout else ""
    except subprocess.TimeoutExpired:
        logger.warning("Code evolution timed out after %ds", timeout)
        return 0
    except FileNotFoundError:
        logger.warning("Claude CLI not found — skipping code evolution")
        return 0
    except Exception as e:
        logger.error("Code evolution error: %s", e)
        return 0

    if not response:
        logger.warning("Code evolution returned empty response")
        return 0

    # 5. Verify sealed files weren't touched
    if sealed_files:
        from orze.engine.sealed import load_sealed_manifest, verify_sealed_files
        manifest = load_sealed_manifest(results_dir)
        changed = verify_sealed_files(sealed_files, manifest)
        if changed:
            logger.error("Code evolution modified sealed files: %s — aborting",
                         changed)
            return 0

    # 6. Parse ideas from response
    ideas = _parse_ideas_from_output(response, results_dir)
    if not ideas:
        logger.info("Code evolution made changes but generated no new ideas")
        return 0

    # 7. Append ideas to ideas.md
    from orze.agents.research import format_idea_markdown, append_ideas_to_md
    ideas_md = []
    for idea in ideas:
        md = format_idea_markdown(
            idea_id=idea["idea_id"],
            title=idea["title"],
            hypothesis=idea["hypothesis"],
            config=idea["config"],
            priority=idea.get("priority", "high"),
            category=idea.get("category", "architecture"),
            parent=idea.get("parent", "none"),
            approach_family=idea.get("approach_family", "other"),
        )
        ideas_md.append(md)
        logger.info("  %s: %s", idea["idea_id"], idea["title"][:60])

    count = append_ideas_to_md(ideas_md, ideas_path, results_dir=results_dir)
    logger.info("Code evolution complete: %d new ideas", count)

    # Save evolution log
    log_dir = results_dir / "_evolution_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    import time
    log_file = log_dir / f"evolution_{int(time.time())}.log"
    try:
        log_file.write_text(
            f"Trigger: {trigger}\n"
            f"Ideas generated: {count}\n\n"
            f"Response:\n{response[:5000]}\n",
            encoding="utf-8",
        )
    except OSError:
        pass

    return count


def main():
    parser = argparse.ArgumentParser(
        description="orze code evolution agent — proactive code changes",
    )
    parser.add_argument("-c", "--config", default="orze.yaml",
                        help="Path to orze.yaml")
    parser.add_argument("--results-dir", default="",
                        help="Path to results dir")
    parser.add_argument("--ideas-md", default="",
                        help="Path to ideas.md")
    parser.add_argument("--trigger", default="plateau",
                        help="What triggered this evolution cycle")

    args = parser.parse_args()

    cfg = {}
    config_path = Path(args.config)
    if config_path.exists():
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning("Could not load %s: %s", args.config, e)

    ideas_path = Path(args.ideas_md or cfg.get("ideas_file", "ideas.md"))
    results_dir = Path(args.results_dir or cfg.get("results_dir", "results"))

    count = run_evolution_cycle(cfg, results_dir, ideas_path, args.trigger)
    sys.exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
