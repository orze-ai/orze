"""Periodic retrospection — lite.

Schedules `research.context_builder.build_digest()` and writes the result
to `<results>/_retrospection.txt` (tail-compacted to 64 KB).

Preserves the public API used by the orchestrator:
    run_retrospection(results_dir, cfg, completed_count, last_count,
                      retro_state=None) -> int   # returns new last_count
    is_research_paused(results_dir) -> bool
    resume_research(results_dir) -> bool
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orze")

PAUSE_SENTINEL = ".pause_research"
_DIGEST_FILE = "_retrospection.txt"
_COMPACT_BYTES = 64 * 1024


def is_research_paused(results_dir: Path) -> bool:
    return (Path(results_dir) / PAUSE_SENTINEL).exists()


def resume_research(results_dir: Path) -> bool:
    sentinel = Path(results_dir) / PAUSE_SENTINEL
    if sentinel.exists():
        sentinel.unlink()
        return True
    return False


def _write_pause(results_dir: Path, reason: str) -> None:
    sentinel = Path(results_dir) / PAUSE_SENTINEL
    sentinel.write_text(
        json.dumps({
            "paused_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reason": reason,
        }, indent=2),
        encoding="utf-8",
    )
    logger.warning("Research PAUSED by retrospection: %s", reason)


def _compact_tail(path: Path, max_bytes: int = _COMPACT_BYTES) -> None:
    try:
        if not path.exists():
            return
        data = path.read_bytes()
        if len(data) <= max_bytes:
            return
        path.write_bytes(data[-max_bytes:])
    except OSError as e:
        logger.debug("compact_tail(%s): %s", path, e)


def run_retrospection(results_dir: Path, cfg: dict,
                      completed_count: int, last_count: int,
                      retro_state: Optional[dict] = None) -> int:
    """Run retrospection if interval threshold crossed; return new last_count."""
    retro_cfg = cfg.get("retrospection", {})
    if not retro_cfg.get("enabled"):
        return last_count

    interval = retro_cfg.get("interval", 50)
    if completed_count < last_count + interval:
        return last_count

    results_dir = Path(results_dir)
    logger.info("Retrospection triggered: %d completed (last %d, interval %d)",
                completed_count, last_count, interval)

    try:
        from orze.research.context_builder import build_digest
        digest = build_digest(results_dir, cfg)
        out = results_dir / _DIGEST_FILE
        out.write_text(digest, encoding="utf-8")
        _compact_tail(out)
    except Exception as e:
        logger.warning("Retrospection digest failed: %s", e)

    # Run competition diagnosis if not cached
    _run_diagnosis_if_needed(results_dir, cfg)

    return completed_count


def _run_diagnosis_if_needed(results_dir: Path, cfg: dict) -> None:
    """Run competition diagnosis once per competition, cache the result."""
    try:
        from orze.engine.competition_diagnosis import (
            diagnose_dataset, save_diagnosis, load_cached_diagnosis,
        )
    except ImportError:
        return

    competition_id = results_dir.name
    if load_cached_diagnosis(results_dir, competition_id):
        return

    train_path = cfg.get("train_csv", "")
    if not train_path:
        for candidate in ("train.csv", "data/train.csv"):
            p = Path(cfg.get("_project_root", ".")) / candidate
            if p.exists():
                train_path = str(p)
                break
    if not train_path:
        return

    try:
        diag = diagnose_dataset(
            train_path,
            competition_id=competition_id,
        )
        save_diagnosis(results_dir, competition_id, diag)
        logger.info("Competition diagnosis saved for %s: target=%s, imbalanced=%s, groups=%s",
                    competition_id, diag.target_type, diag.is_imbalanced,
                    diag.group_columns)
    except Exception as e:
        logger.warning("Competition diagnosis failed: %s", e)
