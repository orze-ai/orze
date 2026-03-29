"""Structured failure classification, analysis, and persistence.

CALLING SPEC:
    FAILURE_CATEGORIES: frozenset of valid category strings

    classify_failure(error_text: str, exit_code: int, source: str) -> str
        Classify an error into a category via regex matching.
        source: "training" | "eval" | "pre_script"
        Returns one of FAILURE_CATEGORIES.

    build_failure_analysis(category: str, error_text: str) -> dict
        Build a structured analysis dict:
        {category, what, why, lesson, timestamp}

    write_failure_analysis(idea_dir: Path, category: str, error_text: str) -> None
        Write failure_analysis.json to idea_dir.

    load_recent_failures(results_dir: Path, limit: int = 200) -> dict
        Scan recent idea dirs for failure_analysis.json.
        Returns {category: [{idea_id, what, why, lesson}]} grouped by category.
        Falls back to metrics.json error field for dirs without analysis files.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("orze")

FAILURE_CATEGORIES = frozenset({
    "oom", "timeout", "stall", "crash", "pre_script_error",
    "eval_failure", "config_error", "sealed_violation",
})

# Regex patterns for classification (order matters — first match wins)
_OOM_PATTERNS = re.compile(
    r"CUDA out of memory|OutOfMemoryError|OOM|"
    r"out of memory|Cannot allocate memory|"
    r"CUDNN_STATUS_NOT_SUPPORTED.*memory",
    re.IGNORECASE,
)
_CONFIG_PATTERNS = re.compile(
    r"KeyError|ValueError.*config|Unknown \w+:|"
    r"missing.*required|invalid.*config|"
    r"yaml\.scanner\.ScannerError|"
    r"expected.*got|shape mismatch",
    re.IGNORECASE,
)

# Root-cause and lesson lookup per category
_CATEGORY_INFO: Dict[str, Dict[str, str]] = {
    "oom": {
        "why": "Model or batch size exceeds GPU VRAM capacity",
        "lesson": "Reduce batch_size, use gradient accumulation, reduce model dimensions, or enable mixed precision",
    },
    "timeout": {
        "why": "Training exceeded the configured time limit",
        "lesson": "Reduce epochs, use early stopping, or increase timeout for this config profile",
    },
    "stall": {
        "why": "Process hung — no log output for extended period",
        "lesson": "Check for deadlocks, data loader issues, or I/O bottlenecks",
    },
    "crash": {
        "why": "Process exited with non-zero code due to unhandled exception",
        "lesson": "Review traceback for bug in training script; may need code fix",
    },
    "pre_script_error": {
        "why": "Pre-training script failed before training could start",
        "lesson": "Check data preparation, feature extraction, or environment setup",
    },
    "eval_failure": {
        "why": "Post-training evaluation script failed",
        "lesson": "Check eval script compatibility with model output format",
    },
    "config_error": {
        "why": "Training script could not parse or use the provided config",
        "lesson": "Verify config key names, value types, and required fields",
    },
    "sealed_violation": {
        "why": "Protected evaluation files were modified during training or fixing",
        "lesson": "Do not modify sealed files; adjust training code or config instead",
    },
}


def classify_failure(error_text: str, exit_code: int = -1,
                     source: str = "training") -> str:
    """Classify an error into a failure category.

    Args:
        error_text: Error message or log tail.
        exit_code: Process exit code (-1 if unknown).
        source: Where the failure occurred — "training", "eval", "pre_script".
    """
    if source == "pre_script":
        return "pre_script_error"
    if source == "eval":
        return "eval_failure"

    text = error_text[:2000]  # limit scan length

    if _OOM_PATTERNS.search(text):
        return "oom"
    if "Timed out" in text or "Timeout" in text:
        return "timeout"
    if "Stalled" in text or "no output for" in text:
        return "stall"
    if _CONFIG_PATTERNS.search(text):
        return "config_error"

    # Default: generic crash
    return "crash"


def build_failure_analysis(category: str, error_text: str) -> dict:
    """Build a structured failure analysis dict."""
    info = _CATEGORY_INFO.get(category, _CATEGORY_INFO["crash"])
    return {
        "category": category,
        "what": error_text.split("\n")[0][:300],
        "why": info["why"],
        "lesson": info["lesson"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def write_failure_analysis(idea_dir: Path, category: str,
                           error_text: str) -> None:
    """Write failure_analysis.json to an idea's result directory."""
    analysis = build_failure_analysis(category, error_text)
    try:
        idea_dir.mkdir(parents=True, exist_ok=True)
        path = idea_dir / "failure_analysis.json"
        path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    except OSError as e:
        logger.warning("Could not write failure analysis for %s: %s",
                       idea_dir.name, e)


def load_recent_failures(results_dir: Path,
                         limit: int = 200) -> Dict[str, List[Dict[str, Any]]]:
    """Load recent structured failure analyses, grouped by category.

    Scans the last `limit` idea dirs (by name, newest last).
    Falls back to metrics.json error field for dirs without analysis files.

    Returns {category: [{idea_id, what, why, lesson}]}.
    """
    if not results_dir.exists():
        return {}

    # Collect idea dirs
    idea_names = []
    try:
        with os.scandir(results_dir) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False) and entry.name.startswith("idea-"):
                    idea_names.append(entry.name)
    except OSError:
        return {}

    idea_names.sort()
    scan_names = idea_names[-limit:] if len(idea_names) > limit else idea_names

    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for name in scan_names:
        idea_dir = results_dir / name

        # Try structured analysis first
        fa_path = idea_dir / "failure_analysis.json"
        if fa_path.exists():
            try:
                fa = json.loads(fa_path.read_text(encoding="utf-8"))
                cat = fa.get("category", "crash")
                grouped.setdefault(cat, []).append({
                    "idea_id": name,
                    "what": fa.get("what", ""),
                    "why": fa.get("why", ""),
                    "lesson": fa.get("lesson", ""),
                })
                continue
            except (json.JSONDecodeError, OSError):
                pass

        # Fallback: check metrics.json for FAILED status
        mf = idea_dir / "metrics.json"
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if m.get("status") not in ("FAILED", "ERROR"):
            continue
        error = m.get("error") or m.get("traceback") or "unknown"
        cat = classify_failure(str(error))
        grouped.setdefault(cat, []).append({
            "idea_id": name,
            "what": str(error).split("\n")[0][:300],
            "why": _CATEGORY_INFO.get(cat, _CATEGORY_INFO["crash"])["why"],
            "lesson": _CATEGORY_INFO.get(cat, _CATEGORY_INFO["crash"])["lesson"],
        })

    return grouped
