"""Rolling-window compaction for append-only LLM-context files.

Why this exists
---------------
``results/_retrospection.txt`` and ``results/_skill_composed_research*.md``
are append-only: every research cycle tacks on a new block. Over days
the files cross 600 KB and when orze stuffs them into the research
prompt the total prompt length exceeds the Anthropic context budget
(observed 338 KB prompt → HTTP 400).

Compaction strategy
-------------------
Each file is treated as a series of *cycle blocks* separated by one of
a small set of regex-matched delimiters (``## cycle N``, ``=== cycle N
===``, ``[CYCLE N]`` etc.). We keep the **last N cycle blocks verbatim**
and replace everything older with a one-line summary header:

    <SUMMARY: 87 earlier cycles compacted 2026-04-19T... — N lines, M bytes>

The result is byte-capped at ``hard_max_bytes`` (default 150 KB) and
written atomically back to the same path. Idempotent: running the
compactor on an already-compacted file is a no-op.

Files without any cycle delimiter fall back to keeping the trailing
``hard_max_bytes`` bytes verbatim (truncating the head), because a
heuristic that would drop content by line count would be unpredictable.

Public API
----------
    compact_file(path, *, keep_last=50, hard_max_bytes=150_000) -> dict
        Rewrites ``path`` in place (atomic). Returns a summary dict.

    compact_many(paths, **kw) -> list[dict]
        Convenience: compact each path; swallow per-file errors.
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger("orze")

# Regexes that identify the *start* of a cycle block. Order matters:
# we try them in order and use the first that matches at least 2 times
# in the file (so one spurious "## cycle N" in prose doesn't carve the
# whole file into one block).
_CYCLE_HEADER_PATTERNS = (
    re.compile(r"^##+\s*cycle\s+\d+.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^=+\s*cycle\s+\d+\s*=+\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\[CYCLE\s+\d+\].*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^-{3,}\s*cycle\s+\d+.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^##+\s*iteration\s+\d+.*$", re.IGNORECASE | re.MULTILINE),
)

DEFAULT_KEEP_LAST = 50
DEFAULT_HARD_MAX_BYTES = 150_000


def _choose_pattern(text: str) -> Optional[re.Pattern]:
    for pat in _CYCLE_HEADER_PATTERNS:
        if len(pat.findall(text)) >= 2:
            return pat
    return None


def _split_by_header(text: str, pat: re.Pattern) -> List[str]:
    """Split into blocks where each block starts with the header line.

    The prelude (text before the first header) is the first element.
    """
    positions = [m.start() for m in pat.finditer(text)]
    if not positions:
        return [text]
    blocks = []
    if positions[0] > 0:
        blocks.append(text[:positions[0]])  # prelude
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        blocks.append(text[start:end])
    return blocks


def _atomic_write(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + f".compact.{os.getpid()}.tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def compact_file(path: Path | str,
                 keep_last: int = DEFAULT_KEEP_LAST,
                 hard_max_bytes: int = DEFAULT_HARD_MAX_BYTES) -> dict:
    """Compact ``path`` in place. Returns summary dict.

    Keys: path, bytes_before, bytes_after, cycles_before, cycles_kept,
    cycles_summarized, mode ("cycle"/"tail"/"noop").
    """
    p = Path(path)
    summary = {
        "path": str(p),
        "bytes_before": 0,
        "bytes_after": 0,
        "cycles_before": 0,
        "cycles_kept": 0,
        "cycles_summarized": 0,
        "mode": "noop",
    }
    if not p.exists():
        return summary
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("compact_file: read %s failed: %s", p, e)
        return summary
    summary["bytes_before"] = len(raw.encode("utf-8"))

    if summary["bytes_before"] <= hard_max_bytes:
        return summary  # no-op: already small enough

    pat = _choose_pattern(raw)
    if pat is not None:
        blocks = _split_by_header(raw, pat)
        prelude = ""
        if blocks and not pat.match(blocks[0]):
            prelude = blocks[0]
            cycles = blocks[1:]
        else:
            cycles = blocks
        summary["cycles_before"] = len(cycles)
        if len(cycles) > keep_last:
            kept = cycles[-keep_last:]
            older = cycles[:-keep_last]
            older_bytes = sum(len(b.encode("utf-8")) for b in older)
            older_lines = sum(b.count("\n") for b in older)
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            header = (
                f"<SUMMARY: {len(older)} earlier cycles compacted {ts} "
                f"— {older_lines} lines, {older_bytes} bytes>\n\n"
            )
            new_text = prelude + header + "".join(kept)
            summary["cycles_kept"] = len(kept)
            summary["cycles_summarized"] = len(older)
            summary["mode"] = "cycle"
        else:
            new_text = raw
            summary["cycles_kept"] = len(cycles)
            summary["mode"] = "cycle"
    else:
        # No cycle delimiter — tail-truncate to hard_max_bytes.
        encoded = raw.encode("utf-8")
        tail = encoded[-hard_max_bytes:].decode("utf-8", errors="ignore")
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        header = (
            f"<SUMMARY: head truncated at {ts} "
            f"— dropped {len(encoded) - len(tail.encode('utf-8'))} "
            f"bytes of older content>\n\n"
        )
        new_text = header + tail
        summary["mode"] = "tail"

    # Hard cap: if still over budget AND we did NOT do cycle-preserving
    # compaction, tail-truncate. In cycle mode we honor the last_N
    # verbatim guarantee even if the caller's hard_max_bytes is too low.
    if (summary["mode"] != "cycle"
            and len(new_text.encode("utf-8")) > hard_max_bytes):
        enc = new_text.encode("utf-8")[-hard_max_bytes:]
        new_text = enc.decode("utf-8", errors="ignore")
        summary["mode"] = summary["mode"] + "+tailcap"

    try:
        _atomic_write(p, new_text)
    except OSError as e:
        logger.warning("compact_file: write %s failed: %s", p, e)
        return summary
    summary["bytes_after"] = len(new_text.encode("utf-8"))
    logger.info(
        "compact_file(%s): %s %d→%d bytes (kept %d/%d cycles)",
        p.name, summary["mode"], summary["bytes_before"],
        summary["bytes_after"], summary["cycles_kept"],
        summary["cycles_before"])
    return summary


def compact_many(paths: Iterable[Path | str], **kw) -> List[dict]:
    out = []
    for p in paths:
        try:
            out.append(compact_file(p, **kw))
        except Exception as e:
            logger.warning("compact_many(%s): %s", p, e)
    return out


def compact_standard_paths(results_dir: Path | str,
                           **kw) -> List[dict]:
    """Compact the canonical oversized prompt files in a results dir."""
    rd = Path(results_dir)
    paths = [rd / "_retrospection.txt"]
    paths.extend(sorted(rd.glob("_skill_composed_*.md")))
    paths.extend(sorted(rd.glob("_skill_composed_*.txt")))
    return compact_many(paths, **kw)
