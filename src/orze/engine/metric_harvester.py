"""Harvest per-epoch metrics from running train_output.log files.

Many training scripts emit per-epoch metrics to stdout but do not write
an incremental metrics.json. Without metrics.json, orze's telemetry
layer cannot report new_best, refresh the leaderboard, or feed live
frontier data to the research roles — professor keeps steering against
a stale champion.

This module scans `results/idea-*/train_output.log` once per N
iterations, extracts best-so-far for the configured `primary_metric`,
and writes a harvested `metrics.json`. It only rewrites files it
authored (sentinel `_source: "harvested_from_log"`); genuine metrics
written by the training script itself are left untouched.

CALLING SPEC:
    harvest_running_ideas(results_dir, primary_metric, extra_patterns) -> int
        Scan idea-* subdirs, update harvested metrics.json files, return count.

    extract_best_metric(log_text, metric_name, extra_patterns) -> (best, last_epoch) | None
        Pull best numeric score for metric_name from log text.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HARVEST_SOURCE = "harvested_from_log"

# Per-metric regex patterns. Each pattern captures the score as group 1.
# First pattern that yields any match wins — we take the max over all
# matches of that pattern. Patterns are tried in listed order so
# eval-time metrics (test_*) beat train-time metrics (train_*).
_DEFAULT_PATTERNS: Dict[str, List[str]] = {
    "map": [
        r"test_mAP\s*=\s*([0-9]*\.?[0-9]+)",
        r"val_mAP\s*=\s*([0-9]*\.?[0-9]+)",
        r"\btest_map\s*=\s*([0-9]*\.?[0-9]+)",
        r"\bval_map\s*=\s*([0-9]*\.?[0-9]+)",
        r"\bmAP\s*=\s*([0-9]*\.?[0-9]+)",
        r"\bmap\s*=\s*([0-9]*\.?[0-9]+)",
    ],
    "accuracy": [
        r"test_acc(?:uracy)?\s*=\s*([0-9]*\.?[0-9]+)",
        r"val_acc(?:uracy)?\s*=\s*([0-9]*\.?[0-9]+)",
        r"\bacc(?:uracy)?\s*=\s*([0-9]*\.?[0-9]+)",
    ],
    "auc": [
        r"test_auc\s*=\s*([0-9]*\.?[0-9]+)",
        r"val_auc\s*=\s*([0-9]*\.?[0-9]+)",
        r"\bauc\s*=\s*([0-9]*\.?[0-9]+)",
    ],
    "f1": [
        r"test_f1\s*=\s*([0-9]*\.?[0-9]+)",
        r"val_f1\s*=\s*([0-9]*\.?[0-9]+)",
        r"\bf1\s*=\s*([0-9]*\.?[0-9]+)",
    ],
    "loss": [
        r"val_loss\s*=\s*([0-9]*\.?[0-9]+)",
        r"test_loss\s*=\s*([0-9]*\.?[0-9]+)",
    ],
}

_EPOCH_PATTERN = re.compile(r"Epoch\s+(\d+)\s*/")


def extract_best_metric(log_text: str,
                        metric_name: str,
                        extra_patterns: Optional[List[str]] = None,
                        maximize: bool = True) -> Optional[Tuple[float, int]]:
    """Return (best_score, last_epoch_seen) from log text.

    Tries extra_patterns first, then defaults for metric_name, then a
    generic `<name>=X` / `<name>: X` fallback. Returns None if no match.
    """
    metric_key = metric_name.lower()
    patterns: List[str] = list(extra_patterns or [])
    patterns.extend(_DEFAULT_PATTERNS.get(metric_key, []))
    if not patterns:
        patterns = [
            rf"\b{re.escape(metric_name)}\s*=\s*([0-9]*\.?[0-9]+)",
            rf"\b{re.escape(metric_name)}\s*:\s*([0-9]*\.?[0-9]+)",
        ]

    best: Optional[float] = None
    for pat in patterns:
        try:
            regex = re.compile(pat, re.IGNORECASE)
        except re.error:
            continue
        any_match = False
        for m in regex.finditer(log_text):
            try:
                v = float(m.group(1))
            except (ValueError, IndexError):
                continue
            any_match = True
            if best is None or (maximize and v > best) or (not maximize and v < best):
                best = v
        if any_match:
            break  # stop at first pattern that yielded anything

    if best is None:
        return None

    epochs = _EPOCH_PATTERN.findall(log_text)
    last_epoch = int(epochs[-1]) if epochs else 0
    return (best, last_epoch)


def harvest_running_ideas(results_dir: Path,
                          primary_metric: str = "map",
                          extra_patterns: Optional[List[str]] = None,
                          maximize: bool = True) -> int:
    """Scan results/idea-*/train_output.log and write harvested metrics.json.

    Skips ideas whose metrics.json was authored by the training script
    itself (any file without our `_source` sentinel is left untouched).

    Returns number of metrics.json files written or updated.
    """
    if not results_dir.is_dir():
        return 0

    written = 0
    for idea_dir in results_dir.glob("idea-*"):
        if not idea_dir.is_dir():
            continue
        log_file = idea_dir / "train_output.log"
        if not log_file.is_file():
            continue

        metrics_file = idea_dir / "metrics.json"
        if metrics_file.exists():
            try:
                existing = json.loads(metrics_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if existing.get("_source") != HARVEST_SOURCE:
                # Genuine metrics.json from training script — do not touch.
                continue

        try:
            log_text = log_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        result = extract_best_metric(
            log_text, primary_metric, extra_patterns, maximize=maximize)
        if result is None:
            continue
        best, last_epoch = result

        metrics = {
            primary_metric: best,
            f"best_{primary_metric}": best,
            "last_epoch": last_epoch,
            "_source": HARVEST_SOURCE,
        }
        try:
            metrics_file.write_text(
                json.dumps(metrics, indent=2), encoding="utf-8")
            written += 1
        except OSError as e:
            logger.debug("harvest write failed for %s: %s", idea_dir.name, e)

    return written
