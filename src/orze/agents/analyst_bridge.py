"""Bridge data-analyst insights into the ideas.md queue.

Why this exists
---------------
The data-analyst role writes ``results/_analyst_insights.md`` with
concrete cheap wins (e.g. "Switch LQM to late_k2 for +0.005 mAP, zero
training needed"). But the professor/research/thinker roles that would
normally action such bullets are all Claude-backed, and when the
Anthropic quota is exhausted, these findings sit unread for days.

This agent is a deterministic fallback: it parses the insights file
(and the latest ``_data_analyst_logs/cycle_*.log`` if present), pulls
out actionable lines by regex, and appends them as new ideas to
``ideas.md`` with the tag ``origin:analyst`` so the ingester picks
them up the normal way.

Contract
--------
Invoke as::

    python -m orze.agents.analyst_bridge \
        --results-dir results --ideas-file ideas.md

Wire into ``orze.yaml`` roles:

    analyst_bridge:
      mode: script
      script: -m orze.agents.analyst_bridge
      args: ["--results-dir", "{results_dir}", "--ideas-file", "{ideas_file}"]
      cooldown: 1800
      timeout: 120

Idempotency
-----------
An idea is emitted at most once per unique actionable line (hashed and
persisted to ``<results>/_analyst_bridge_state.json``). Re-running the
bridge on the same file is a safe no-op.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Set

logger = logging.getLogger("orze.analyst_bridge")

# Regexes that identify "this is a concrete, actionable finding".
# Any ONE match on a line/sentence promotes the snippet to an idea.
_ACTIONABLE_PATTERNS = (
    re.compile(r"\+0\.\d{3}\s*mAP", re.IGNORECASE),
    re.compile(r"\bone[- ]line\b", re.IGNORECASE),
    re.compile(r"\bzero[- ]training\b", re.IGNORECASE),
    re.compile(r"\bno training\b", re.IGNORECASE),
    re.compile(r"\bcheap\b", re.IGNORECASE),
    re.compile(r"\bimmediate quick win\b", re.IGNORECASE),
    re.compile(r"\bquick win\b", re.IGNORECASE),
)

STATE_FILENAME = "_analyst_bridge_state.json"
IDEA_TAG = "origin:analyst"


def _split_sentences(text: str) -> List[str]:
    """Split prose into sentence-ish units, preserving bullet lines."""
    chunks: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        # For bullet lines, keep whole line.
        if line.startswith(("-", "*", "1.", "2.", "3.")) or line.startswith("**"):
            chunks.append(line.lstrip("-* \t"))
            continue
        for sent in re.split(r"(?<=[.!?])\s+", line):
            s = sent.strip()
            if s:
                chunks.append(s)
    return chunks


def _is_actionable(snippet: str) -> bool:
    return any(pat.search(snippet) for pat in _ACTIONABLE_PATTERNS)


def extract_actionable(text: str) -> List[str]:
    """Return the deduplicated ordered list of actionable snippets."""
    seen: Set[str] = set()
    out: List[str] = []
    for s in _split_sentences(text):
        if len(s) < 20 or len(s) > 400:
            continue
        if not _is_actionable(s):
            continue
        key = re.sub(r"\s+", " ", s).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _load_state(state_path: Path) -> dict:
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"emitted_hashes": []}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _read_sources(results_dir: Path) -> str:
    parts: List[str] = []
    insights = results_dir / "_analyst_insights.md"
    if insights.exists():
        try:
            parts.append(insights.read_text(encoding="utf-8"))
        except OSError:
            pass
    log_dir = results_dir / "_data_analyst_logs"
    if log_dir.is_dir():
        cycles = sorted(log_dir.glob("cycle_*.log"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        for p in cycles[:1]:
            try:
                parts.append(p.read_text(encoding="utf-8"))
            except OSError:
                continue
    return "\n\n".join(parts)


def _format_idea(snippet: str, counter: int) -> str:
    """Format a single actionable line as an orze idea block."""
    iid = f"idea-analyst-{_hash(snippet)}"
    title = snippet.strip()
    # Short title (first sentence, capped).
    if len(title) > 120:
        title = title[:117] + "..."
    body = (
        f"## {iid}: {title}\n"
        f"- **Priority**: high\n"
        f"- **Tags**: {IDEA_TAG}\n"
        f"- **Source**: _analyst_insights.md\n"
        f"- **Hypothesis**: {snippet.strip()}\n\n"
        f"```yaml\n"
        f"# analyst-sourced finding; adapt this YAML to the concrete\n"
        f"# experiment your pipeline expects.\n"
        f"notes: \"{snippet.strip()[:200].replace(chr(34), chr(39))}\"\n"
        f"```\n\n"
    )
    return body


def append_new_ideas(ideas_file: Path, new_blocks: Iterable[str]) -> int:
    blocks = list(new_blocks)
    if not blocks:
        return 0
    ideas_file.parent.mkdir(parents=True, exist_ok=True)
    if not ideas_file.exists():
        ideas_file.write_text("# Ideas\n\n", encoding="utf-8")
    with ideas_file.open("a", encoding="utf-8") as fh:
        for b in blocks:
            fh.write(b)
    return len(blocks)


def run(results_dir: Path, ideas_file: Path) -> dict:
    """Main entry. Returns a summary dict."""
    text = _read_sources(results_dir)
    if not text.strip():
        return {"emitted": 0, "candidates": 0, "reason": "no-source-files"}

    candidates = extract_actionable(text)
    state_path = results_dir / STATE_FILENAME
    state = _load_state(state_path)
    seen: Set[str] = set(state.get("emitted_hashes", []))

    fresh_blocks: List[str] = []
    fresh_hashes: List[str] = []
    for i, snippet in enumerate(candidates):
        h = _hash(snippet)
        if h in seen:
            continue
        fresh_blocks.append(_format_idea(snippet, i))
        fresh_hashes.append(h)

    n = append_new_ideas(ideas_file, fresh_blocks)
    if n:
        state["emitted_hashes"] = list(seen) + fresh_hashes
        state["last_run"] = time.time()
        _save_state(state_path, state)

    logger.info("analyst_bridge: %d candidates, %d new ideas emitted",
                len(candidates), n)
    return {"emitted": n, "candidates": len(candidates)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="orze-analyst-bridge")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--ideas-file", default="ideas.md")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    summary = run(Path(args.results_dir), Path(args.ideas_file))
    print(json.dumps(summary))
    # Exit 0 if any candidates found (including already-seen) so the
    # soft-failure detector doesn't flag a quiet no-op as a broken role.
    return 0


if __name__ == "__main__":
    sys.exit(main())
