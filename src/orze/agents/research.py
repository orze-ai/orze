#!/usr/bin/env python3
"""orze research agent: generic LLM-powered idea generator.

CALLING SPEC:
    format_idea_markdown(idea_id, title, hypothesis, config, ...) -> str
    append_ideas_to_md(ideas_md, ideas_path, results_dir=None) -> int
    generate_idea_id(config, results_dir) -> str
    parse_llm_ideas(response, results_dir, cycle) -> list[dict]
    build_prompt(context, rules_content, num_ideas, retrospection="") -> str
    run_research_cycle(backend, cycle, ideas_path, results_dir, ...) -> int
    main()  # CLI entry point

Ships with orze. Works with any LLM backend (Gemini, OpenAI, Anthropic, local).
Context gathering is in research_context.py; LLM backends in research_llm.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time as _time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from orze.agents.research_context import (
    build_context,
    get_existing_idea_ids,
)
from orze.agents.research_llm import (
    DEFAULT_SYSTEM_PROMPT,
    call_llm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orze.research_agent")


# ---------------------------------------------------------------------------
#  Idea formatting and appending (generic — matches orze format)
# ---------------------------------------------------------------------------

def generate_idea_id(config: dict, results_dir: Path) -> str:
    """Generate a 6-char content-hash idea ID, collision-free.

    Hash = sha256(yaml.dump(config, sort_keys=True) + nonce)[:6].
    Checks results/ to avoid collisions with existing experiments.
    """
    raw = yaml.dump(config, sort_keys=True)
    for nonce in range(100):
        h = hashlib.sha256(f"{raw}:{nonce}".encode()).hexdigest()[:6]
        idea_id = f"idea-{h}"
        if not (results_dir / idea_id).exists():
            return idea_id
    # Fallback: timestamp-based
    return f"idea-{hashlib.sha256(f'{raw}:{_time.time()}'.encode()).hexdigest()[:6]}"


def format_idea_markdown(idea_id: str, title: str, hypothesis: str,
                         config: dict, priority: str = "high",
                         category: str = "architecture",
                         parent: str = "none",
                         cycle: int = 0,
                         approach_family: str = "other") -> str:
    """Format a single idea as markdown for appending to ideas.md."""
    lines = [
        f"\n## {idea_id}: {title}",
        f"- **Priority**: {priority}",
        f"- **Category**: {category}",
        f"- **Approach Family**: {approach_family}",
        f"- **Parent**: {parent}",
        f"- **Research Cycle**: {cycle}",
        f"- **Hypothesis**: {hypothesis}",
        "- **Config overrides**:",
        "  ```yaml",
    ]
    config_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)
    for line in config_yaml.splitlines():
        lines.append(f"  {line}")
    lines.append("  ```")
    lines.append("")
    return "\n".join(lines)


def append_ideas_to_md(ideas_md: list, ideas_path: Path,
                       results_dir: Optional[Path] = None) -> int:
    """Append formatted idea markdown strings to ideas.md. Returns count.

    Acquires filesystem lock to prevent race with orchestrator's
    ideas.md consumption (which wipes the file after ingesting to SQLite).
    """
    if not ideas_md:
        return 0
    from orze.core.fs import _fs_lock, _fs_unlock
    # Use same lock path as orchestrator: results_dir / ".ideas_md.lock"
    if results_dir:
        lock_dir = results_dir / ".ideas_md.lock"
    else:
        lock_dir = ideas_path.parent / ".ideas_md.lock"
    locked = _fs_lock(lock_dir, stale_seconds=60)
    try:
        with open(ideas_path, "a", encoding="utf-8") as f:
            f.write("\n")
            for md in ideas_md:
                f.write(md)
                f.write("\n")
    finally:
        if locked:
            _fs_unlock(lock_dir)
    return len(ideas_md)


# ---------------------------------------------------------------------------
#  LLM response parsing (generic — extracts ideas from LLM output)
# ---------------------------------------------------------------------------

def parse_llm_ideas(response: str, results_dir: Path, cycle: int) -> list:
    """Parse LLM response into structured ideas.

    Expects the LLM to return ideas as JSON array or markdown.
    Tries JSON first, falls back to markdown parsing.
    IDs are 6-char content hashes of the config YAML.

    Each idea needs: title, hypothesis, config (dict).
    Optional: priority, category, parent.
    """
    ideas = []

    # Try JSON array first
    json_ideas = _try_parse_json(response)
    if json_ideas:
        for i, item in enumerate(json_ideas):
            if not isinstance(item, dict):
                continue
            if not item.get("title") or not item.get("config"):
                continue
            idea_id = generate_idea_id(item["config"], results_dir)
            # Validate approach_family against known set
            from orze.engine.family_guard import APPROACH_FAMILIES, infer_approach_family
            raw_family = item.get("approach_family", "").lower()
            if raw_family not in APPROACH_FAMILIES:
                raw_family = infer_approach_family(
                    item["config"], item.get("category", ""))
            ideas.append({
                "idea_id": idea_id,
                "title": item["title"],
                "hypothesis": item.get("hypothesis", item.get("title")),
                "config": item["config"],
                "priority": item.get("priority", "high"),
                "category": item.get("category", "architecture"),
                "approach_family": raw_family,
                "parent": item.get("parent", "none"),
                "cycle": cycle,
            })
        return ideas

    # Fallback: try to find YAML blocks in markdown
    pattern = re.compile(
        r"##\s*(?:idea-[a-z0-9]+:\s*)?(.+?)$\s*"
        r"(?:.*?hypothesis[:\s]*(.+?)$)?\s*"
        r"```ya?ml\s*\n(.*?)```",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    for i, m in enumerate(pattern.finditer(response)):
        title = m.group(1).strip()
        hypothesis = (m.group(2) or title).strip()
        try:
            config = yaml.safe_load(m.group(3))
        except yaml.YAMLError:
            continue
        if not isinstance(config, dict):
            continue
        idea_id = generate_idea_id(config, results_dir)
        ideas.append({
            "idea_id": idea_id,
            "title": title,
            "hypothesis": hypothesis,
            "config": config,
            "priority": "high",
            "category": "architecture",
            "parent": "none",
            "cycle": cycle,
        })

    return ideas


def _try_parse_json(text: str) -> Optional[list]:
    """Try to extract a JSON array from text, handling brackets inside strings."""
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    end = start
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            if in_string:
                escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if depth != 0:
        return None
    try:
        result = json.loads(text[start:end])
        return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
#  Prompt building
# ---------------------------------------------------------------------------

def build_prompt(context: str, rules_content: str, num_ideas: int,
                 retrospection: str = "") -> str:
    """Build the full prompt for the LLM."""
    parts = [DEFAULT_SYSTEM_PROMPT]

    if retrospection:
        parts.append("## Retrospection Analysis\n")
        parts.append("The following is an automated analysis of recent experiment trends "
                     "and patterns. Use these insights to guide your idea generation.\n")
        parts.append(retrospection)
        parts.append("")

    if rules_content:
        parts.append("## Project-Specific Rules\n")
        parts.append(rules_content)
        parts.append("")

    parts.append(context)

    parts.append(f"\n## Your Task\n")
    parts.append(f"Generate exactly {num_ideas} new experiment ideas as a JSON array.")
    parts.append("Use the leaderboard, failure analysis, and performance patterns above to inform your choices.")
    parts.append("Each idea must have a unique, testable hypothesis grounded in evidence from the context.\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
#  Main research cycle
# ---------------------------------------------------------------------------

def run_research_cycle(
    backend: str,
    cycle: int,
    ideas_path: Path,
    results_dir: Path,
    report_cfg: dict,
    num_ideas: int = 5,
    api_key: str = "",
    model: str = "",
    endpoint: str = "",
    rules_file: str = "",
    lake_db_path: Optional[Path] = None,
    dry_run: bool = False,
    retrospection_file: str = "",
) -> int:
    """Run one research cycle. Returns number of ideas generated."""
    logger.info("=" * 60)
    logger.info("RESEARCH CYCLE %d (%s)", cycle, backend)
    logger.info("=" * 60)

    # 1. Build context from results
    logger.info("Step 1: Building research context...")
    context = build_context(results_dir, ideas_path, report_cfg,
                            lake_db_path=lake_db_path)
    existing_ids = get_existing_idea_ids(ideas_path)
    logger.info("  %d existing ideas in queue, using content-hash IDs", len(existing_ids))

    # 2. Load project-specific rules if provided
    rules_content = ""
    if rules_file:
        rules_path = Path(rules_file)
        if rules_path.exists():
            rules_content = rules_path.read_text(encoding="utf-8")
            logger.info("  Loaded rules from %s (%d chars)", rules_file, len(rules_content))

    # 2b. Load retrospection output if provided
    retro_content = ""
    if retrospection_file:
        retro_path = Path(retrospection_file)
        if retro_path.exists():
            try:
                retro_content = retro_path.read_text(encoding="utf-8").strip()
                if retro_content:
                    logger.info("  Loaded retrospection from %s (%d chars)",
                                retrospection_file, len(retro_content))
            except OSError as e:
                logger.warning("  Could not read retrospection file: %s", e)

    # 3. Build prompt
    prompt = build_prompt(context, rules_content, num_ideas,
                          retrospection=retro_content)

    if dry_run:
        print("=" * 60)
        print("DRY RUN — prompt that would be sent to LLM:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
        print(f"Prompt length: {len(prompt)} chars")
        return 0

    # 4. Call LLM
    logger.info("Step 2: Calling %s (prompt: %d chars)...", backend, len(prompt))
    response = call_llm(prompt, backend, api_key=api_key, model=model,
                        endpoint=endpoint)
    if not response:
        logger.error("LLM returned empty response — aborting cycle")
        return 0

    # 5. Parse ideas from response
    logger.info("Step 3: Parsing ideas from response...")
    ideas = parse_llm_ideas(response, results_dir, cycle)
    if not ideas:
        logger.warning("Could not parse any ideas from LLM response")
        logger.debug("Response was: %s", response[:2000])
        return 0
    logger.info("  Parsed %d ideas", len(ideas))

    # 6. Format and append to ideas.md
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
            cycle=cycle,
            approach_family=idea.get("approach_family", "other"),
        )
        ideas_md.append(md)
        logger.info("  %s: %s", idea["idea_id"], idea["title"][:60])

    count = append_ideas_to_md(ideas_md, ideas_path, results_dir=results_dir)
    logger.info("Appended %d ideas to %s", count, ideas_path)

    logger.info("Research cycle %d complete: %d new ideas", cycle, count)
    return count


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="orze research agent — generate experiment ideas with any LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Gemini
  GEMINI_API_KEY=... python orze/research_agent.py -c orze.yaml --backend gemini

  # OpenAI
  OPENAI_API_KEY=... python orze/research_agent.py -c orze.yaml --backend openai

  # Local ollama
  python orze/research_agent.py -c orze.yaml --backend ollama --model llama3

  # Custom endpoint
  python orze/research_agent.py -c orze.yaml --backend custom --endpoint http://localhost:8080/v1
""",
    )

    parser.add_argument("-c", "--config", default="orze.yaml",
                        help="Path to orze.yaml (for results_dir, ideas_file, report config)")
    parser.add_argument("--backend", default="gemini",
                        choices=["gemini", "openai", "anthropic", "ollama", "custom"],
                        help="LLM backend to use (default: gemini)")
    parser.add_argument("--model", default="",
                        help="Model name (default: backend-specific)")
    parser.add_argument("--api-key", default="",
                        help="API key (default: from environment)")
    parser.add_argument("--endpoint", default="",
                        help="API endpoint URL (for custom/ollama backends)")
    parser.add_argument("--cycle", type=int, default=1,
                        help="Research cycle number")
    parser.add_argument("--num-ideas", type=int, default=5,
                        help="Number of ideas to generate (default: 5)")
    parser.add_argument("--ideas-md", default="",
                        help="Path to ideas.md (overrides orze.yaml)")
    parser.add_argument("--results-dir", default="",
                        help="Path to results dir (overrides orze.yaml)")
    parser.add_argument("--rules-file", default="",
                        help="Path to project-specific rules file")
    parser.add_argument("--lake-db", default="",
                        help="Path to idea_lake.db (default: alongside ideas.md)")
    parser.add_argument("--retrospection-file", default="",
                        help="Path to retrospection output file (auto-generated by orze)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the prompt and exit without calling the LLM")

    args = parser.parse_args()

    # Load orze.yaml for defaults
    cfg = {}
    config_path = Path(args.config)
    if config_path.exists():
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning("Could not load %s: %s", args.config, e)

    ideas_path = Path(args.ideas_md or cfg.get("ideas_file", "ideas.md"))
    results_dir = Path(args.results_dir or cfg.get("results_dir", "results"))
    report_cfg = cfg.get("report", {})
    lake_db_path = Path(args.lake_db) if args.lake_db else None

    count = run_research_cycle(
        backend=args.backend,
        cycle=args.cycle,
        ideas_path=ideas_path,
        results_dir=results_dir,
        report_cfg=report_cfg,
        num_ideas=args.num_ideas,
        api_key=args.api_key,
        model=args.model,
        endpoint=args.endpoint,
        rules_file=args.rules_file,
        lake_db_path=lake_db_path,
        dry_run=args.dry_run,
        retrospection_file=args.retrospection_file,
    )

    sys.exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
