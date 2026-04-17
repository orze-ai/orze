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

Pattern resolution order (first hit wins):

    1. Cached patterns learned previously for this (train_script, metric)
       combination — stored in `results/_metric_patterns_cache.json`,
       keyed by train_script mtime so edits invalidate the cache.
    2. User-supplied `metric_harvest.patterns` from orze.yaml.
    3. Built-in defaults for common metrics (map/accuracy/auc/f1/loss).
    4. Generic fallback `<metric>\\s*[=:]\\s*NUMBER`.
    5. If all of the above miss and a `pattern_inferrer` callable is
       passed in, invoke it with (train_script_path, log_sample,
       metric_name). Inferrer returns a list of regex strings; we cache
       them keyed by mtime and retry extraction.

This keeps orze deterministic by default — the LLM-backed inferrer
(which lives in orze-pro) is an optional upgrade, not a hard
dependency.

CALLING SPEC:
    harvest_running_ideas(
        results_dir, primary_metric="map", extra_patterns=None,
        maximize=True, pattern_inferrer=None, train_script=None) -> int
        Scan idea-* subdirs, update harvested metrics.json files, return count.

    extract_best_metric(log_text, metric_name, extra_patterns, maximize)
        -> (best, last_epoch) | None
        Pull best numeric score for metric_name from log text.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HARVEST_SOURCE = "harvested_from_log"
PATTERN_CACHE_FILENAME = "_metric_patterns_cache.json"

# Signature for an optional LLM-backed pattern inferrer. Given a path
# to the train script, a sample of its log output, and the metric we
# want to capture, it should return a list of Python regex strings
# whose group(1) is the numeric value. Empty list = "couldn't figure
# it out, don't retry soon". The harvester caches whatever is returned.
PatternInferrer = Callable[[Path, str, str], List[str]]

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
                # Strip trailing punctuation (commas, periods from sentence
                # ends) that may have been greedily captured.
                v = float(m.group(1).rstrip(".,;:"))
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


def _load_pattern_cache(results_dir: Path) -> Dict:
    """Load per-train-script learned patterns. Returns {} on any error."""
    path = results_dir / PATTERN_CACHE_FILENAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_pattern_cache(results_dir: Path, cache: Dict) -> None:
    path = results_dir / PATTERN_CACHE_FILENAME
    try:
        path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError as e:
        logger.debug("pattern cache write failed: %s", e)


# Empty cache entries (inferrer returned no patterns) expire after this
# many seconds so that a log which simply didn't have metrics yet can
# trigger a fresh inference once it has grown. Non-empty (successful)
# entries stick around until the train script is edited.
_EMPTY_TTL_SECONDS = 1800  # 30 min

# Inference is an LLM call, so we only invoke it on logs that actually
# look like evaluation has produced numbers. A warmup-only log with no
# eval output would cause the inferrer to correctly return nothing,
# and that "correctly nothing" would poison the cache until TTL.
_HAS_EVAL_SIGNAL = re.compile(
    r"\b(epoch|eval|step|iter|iteration|valid|test)\s*[\s:=/]+\s*\d",
    re.IGNORECASE)
_HAS_NUMERIC = re.compile(r"\b0\.\d{2,}|\b\d+\.\d{3,}\b")


def _log_has_training_signal(log_text: str) -> bool:
    """Heuristic: is there enough in this log to warrant LLM inference?

    Require: (a) non-trivial log size, (b) at least one eval/epoch
    marker, (c) at least one numeric value that could be a metric.
    Warmup-only logs with just "loading checkpoint" lines get skipped
    so we don't waste LLM calls and poison the cache with empties.
    """
    if len(log_text) < 50:
        return False
    if not _HAS_EVAL_SIGNAL.search(log_text):
        return False
    if not _HAS_NUMERIC.search(log_text):
        return False
    return True


def _cached_patterns_for(cache: Dict, train_script: Path,
                         metric_name: str,
                         now: Optional[float] = None) -> Tuple[bool, List[str]]:
    """Return (is_cached, patterns).

    `is_cached=True` means we have a still-valid previous result.
    A cached empty list ("the inferrer tried and couldn't figure
    it out") counts as cached only for `_EMPTY_TTL_SECONDS` — after
    that it's treated as stale so the next harvest cycle gets to
    retry on a (hopefully longer) log.
    """
    key = train_script.name
    entry = cache.get(key)
    if not entry:
        return False, []
    try:
        current_mtime = train_script.stat().st_mtime
    except OSError:
        return False, []
    if abs(entry.get("mtime", 0) - current_mtime) > 0.5:
        return False, []  # script was edited — cache is stale

    pbm = entry.get("patterns_by_metric", {})
    if metric_name not in pbm:
        return False, []
    patterns = list(pbm[metric_name])
    if not patterns:
        learned_at_all = entry.get("learned_at", {}) or {}
        learned_at = learned_at_all.get(metric_name, 0)
        current = now if now is not None else time.time()
        if current - learned_at > _EMPTY_TTL_SECONDS:
            return False, []  # stale empty — allow retry
    return True, patterns


def _store_patterns(cache: Dict, train_script: Path, metric_name: str,
                    patterns: List[str],
                    now: Optional[float] = None) -> None:
    try:
        mtime = train_script.stat().st_mtime
    except OSError:
        return
    key = train_script.name
    entry = cache.setdefault(key, {"patterns_by_metric": {}})
    entry["mtime"] = mtime
    entry["patterns_by_metric"][metric_name] = patterns
    # Track when we learned each metric so TTL-on-empty works.
    learned_at = entry.setdefault("learned_at", {})
    learned_at[metric_name] = now if now is not None else time.time()


def _resolve_train_script(idea_dir: Path,
                          default_script: Optional[Path]) -> Optional[Path]:
    """Find the train script path for a given idea dir."""
    cfg_path = idea_dir / "idea_config.yaml"
    if cfg_path.is_file():
        # Simple regex scan is enough — we only need one key.
        try:
            for line in cfg_path.read_text(encoding="utf-8").splitlines():
                m = re.match(r"\s*train_script\s*:\s*(.+?)\s*$", line)
                if m:
                    cand = Path(m.group(1).strip().strip('"').strip("'"))
                    if not cand.is_absolute():
                        cand = idea_dir.parent.parent / cand
                    if cand.is_file():
                        return cand
        except OSError:
            pass
    return default_script if default_script and default_script.is_file() else None


def harvest_running_ideas(results_dir: Path,
                          primary_metric: str = "map",
                          extra_patterns: Optional[List[str]] = None,
                          maximize: bool = True,
                          pattern_inferrer: Optional[PatternInferrer] = None,
                          train_script: Optional[Path] = None) -> int:
    """Scan results/idea-*/train_output.log and write harvested metrics.json.

    Skips ideas whose metrics.json was authored by the training script
    itself (any file without our `_source` sentinel is left untouched).

    If `pattern_inferrer` is provided, it's invoked ONCE per
    (train_script, metric) tuple when every built-in and config
    pattern fails. Its returned regex patterns are cached until the
    train script is edited.

    Returns number of metrics.json files written or updated.
    """
    if not results_dir.is_dir():
        return 0

    cache = _load_pattern_cache(results_dir) if pattern_inferrer else {}
    cache_dirty = False
    written = 0

    for idea_dir in sorted(results_dir.glob("idea-*")):
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
                continue

        try:
            log_text = log_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not log_text.strip():
            continue

        # Pattern stack: cached (if any) > user extras > defaults/fallback.
        idea_script = _resolve_train_script(idea_dir, train_script)
        is_cached, cached = (
            _cached_patterns_for(cache, idea_script, primary_metric)
            if idea_script else (False, []))
        patterns = cached + list(extra_patterns or [])

        result = extract_best_metric(
            log_text, primary_metric, patterns, maximize=maximize)

        # Extract miss + inferrer available + script known + never
        # tried-or-cached for this (script, metric) + log looks like
        # training has actually produced numbers → call inferrer.
        if (result is None
                and pattern_inferrer is not None
                and idea_script is not None
                and not is_cached
                and _log_has_training_signal(log_text)):
            try:
                proposed = pattern_inferrer(
                    idea_script, log_text, primary_metric) or []
            except Exception as e:
                logger.debug("pattern inferrer crashed: %s", e)
                proposed = []
            # Cache whatever came back — empty list is a valid "I tried"
            # signal that prevents re-inference until mtime changes.
            _store_patterns(cache, idea_script, primary_metric, proposed)
            cache_dirty = True
            if proposed:
                result = extract_best_metric(
                    log_text, primary_metric, proposed + list(extra_patterns or []),
                    maximize=maximize)
                if result is not None:
                    logger.info(
                        "metric_harvester: learned %d pattern(s) for "
                        "%s/%s from %s",
                        len(proposed), idea_script.name,
                        primary_metric, idea_dir.name)

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

    if cache_dirty:
        _save_pattern_cache(results_dir, cache)

    return written
