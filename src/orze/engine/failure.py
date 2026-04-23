"""Failure tracking, skip-listing, and LLM-based auto-fix for failed ideas.

CALLING SPEC:
    _record_failure(failure_counts: dict, idea_id: str)
        Increment failure_counts[idea_id] by 1.

    get_skipped_ideas(failure_counts: dict, max_failures: int) -> set[str]
        Return idea IDs that have failed >= max_failures times.
        Returns empty set if max_failures <= 0 (skip-listing disabled).

    _reset_idea_for_retry(idea_dir: Path)
        Delete metrics.json and rotate the log file so the idea can be
        re-launched. Preserves claim.json.

    _try_executor_fix(idea_id: str, error_text: str, results_dir: Path,
                      cfg: dict, fix_counts: dict) -> bool
        Spawn a Claude CLI process to diagnose and patch the failing idea's
        scripts/configs. Returns True if a fix was applied (idea should be
        retried). cfg keys used: max_fix_attempts, ideas_file, train_script,
        executor_fix.{claude_bin, model, timeout}.
"""
import logging
import os
import subprocess
from pathlib import Path

from orze.core.fs import tail_file
from orze.core.ideas import parse_ideas

logger = logging.getLogger("orze")


def _record_failure(failure_counts: dict, idea_id: str):
    """Increment failure count for an idea."""
    failure_counts[idea_id] = failure_counts.get(idea_id, 0) + 1


def get_skipped_ideas(failure_counts: dict, max_failures: int) -> set:
    """Return set of idea IDs that have exceeded max failure count."""
    if max_failures <= 0:
        return set()
    return {iid for iid, count in failure_counts.items()
            if count >= max_failures}


def _reset_idea_for_retry(idea_dir: Path):
    """Clean up an idea's result dir so it can be re-launched.

    Removes metrics.json and renames the old log for reference,
    but preserves claim.json (idea stays claimed by us).
    """
    metrics = idea_dir / "metrics.json"
    if metrics.exists():
        metrics.unlink(missing_ok=True)

    log = idea_dir / "train_output.log"
    if log.exists():
        attempt = 1
        while (idea_dir / f"train_output.attempt{attempt}.log").exists():
            attempt += 1
        log.rename(idea_dir / f"train_output.attempt{attempt}.log")


_ARGPARSE_UNRECOGNIZED_RE = __import__("re").compile(
    r"error:\s*unrecognized arguments:"
)


def _is_argparse_schema_invalid(error_text: str, exit_code: int,
                                log_tail_text: str = "") -> bool:
    """True if the failure is an argparse ``unrecognized arguments`` error.

    F2: such failures are NOT LLM-fixable in this direct-patch path — the
    engineer-triggered ``missing_key_set`` implementation path in
    ``phases.py`` handles schema gaps by adding args to train scripts.
    Running the LLM fix loop here just wastes tokens + time and loops
    forever (the schema problem isn't in the idea's own code).
    """
    if exit_code != 2:
        return False
    blob = f"{error_text}\n{log_tail_text}"
    return bool(_ARGPARSE_UNRECOGNIZED_RE.search(blob))


def _mark_lake_failure(idea_id: str, cfg: dict,
                       results_dir: Path, reason: str) -> None:
    """Best-effort: mark idea ``failed`` in idea_lake with a failure reason
    stored under ``eval_metrics.failure_reason``. Silent on any error —
    the filesystem ``metrics.json`` is the authoritative failure record.
    """
    import sqlite3
    import json as _json_mod
    try:
        db_path = cfg.get("idea_lake_db") or str(results_dir / "idea_lake.db")
        if not Path(db_path).exists():
            return
        conn = sqlite3.connect(db_path, timeout=5)
        try:
            cur = conn.execute(
                "SELECT eval_metrics FROM ideas WHERE idea_id = ?",
                (idea_id,))
            row = cur.fetchone()
            if row is None:
                return
            em_raw = row[0]
            try:
                em = _json_mod.loads(em_raw) if em_raw else {}
                if not isinstance(em, dict):
                    em = {"_prev": em}
            except (ValueError, TypeError):
                em = {}
            em["failure_reason"] = reason
            conn.execute(
                "UPDATE ideas SET status = 'failed', eval_metrics = ? "
                "WHERE idea_id = ?",
                (_json_mod.dumps(em), idea_id))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def _try_executor_fix(idea_id: str, error_text: str, results_dir: Path,
                      cfg: dict, fix_counts: dict,
                      exit_code: int = -1) -> bool:
    """Spawn an LLM to diagnose and fix a failed idea.

    The LLM can modify the project's scripts, configs, or any other project
    files needed to make the idea succeed. It does NOT touch orze framework
    code.

    Returns True if the LLM reports a fix was applied (idea should be retried).
    """
    # F2: short-circuit argparse schema errors. These are never fixable by
    # patching the idea's own files — the engineer SOP handles schema gaps.
    log_tail_text = ""
    try:
        log_tail_text = tail_file(results_dir / idea_id / "train_output.log", 8192)
    except Exception:
        pass
    if _is_argparse_schema_invalid(error_text, exit_code, log_tail_text):
        logger.info(
            "[SKIP-FIX] %s — schema_invalid: unrecognized arguments "
            "(argparse exit 2)", idea_id)
        _mark_lake_failure(idea_id, cfg, results_dir,
                           "schema_invalid")
        return False

    max_fix = cfg.get("max_fix_attempts", 0)
    if max_fix <= 0:
        return False

    attempts = fix_counts.get(idea_id, 0)
    if attempts >= max_fix:
        logger.info("[FIX] %s — exhausted %d fix attempts, giving up",
                     idea_id, attempts)
        return False

    attempt_num = attempts + 1

    idea_dir = results_dir / idea_id
    log_tail = tail_file(idea_dir / "train_output.log", 16384)

    # Read the idea config from ideas.md
    ideas_file = cfg.get("ideas_file", "ideas.md")
    idea_raw = ""
    try:
        ideas = parse_ideas(ideas_file)
        if idea_id in ideas:
            idea_raw = ideas[idea_id].get("raw", "")
    except Exception:
        pass

    train_script = cfg.get("train_script", "train.py")

    # Collect previous fix attempt logs so the LLM doesn't repeat itself
    prev_attempts_text = ""
    if attempt_num > 1:
        fix_log_dir = results_dir / "_fix_logs"
        parts = []
        for prev in range(1, attempt_num):
            prev_log = fix_log_dir / f"{idea_id}_attempt{prev}.log"
            if prev_log.exists():
                content = prev_log.read_text(encoding="utf-8")
                parts.append(f"### Attempt {prev}\n```\n{content[-2000:]}\n```")
        if parts:
            prev_attempts_text = (
                "\n\n## Previous Fix Attempts (ALREADY TRIED — do NOT repeat)\n"
                + "\n".join(parts)
            )

    prompt = f"""You are the executor for orze, an automated experiment orchestrator.
An experiment just failed. Your job: diagnose the error and fix whatever is needed
so the experiment can succeed on retry.

## Failed Experiment
- **Idea ID**: {idea_id}
- **Script**: {train_script}
- **Fix attempt**: {attempt_num} of {max_fix}

## Idea Config
```
{idea_raw[:3000]}
```

## Error
```
{error_text[:2000]}
```

## Full Log Tail (last 16KB)
```
{log_tail[-8000:]}
```
{prev_attempts_text}

## Your Task
1. Read the experiment script and any relevant project files to understand the error.
2. Diagnose the root cause.
3. Apply a minimal fix. Common patterns:
   - Wrong paths, missing files, empty inputs \u2192 fix paths or input handling
   - Resource exhaustion (memory, disk, etc.) \u2192 reduce resource usage in config
   - Dimension or type mismatches \u2192 fix the code to handle the config correctly
   - Missing dependencies \u2192 install or work around
   - Config handling errors \u2192 fix the script's config parsing/validation
4. After fixing, verify syntax: `python -c "import ast; ast.parse(open('FILE').read())"`
5. Report what you did.

## CRITICAL RULES
- You may modify ANY project file (scripts, configs, utilities, etc.)
- You may NOT modify files under `orze/` \u2014 that is the framework, not your scope
- You may NOT modify `ideas.md` \u2014 the idea config is fixed, adapt the code to handle it
- **CONCURRENT SAFETY**: Other experiments may be running right now using the same scripts.
  Your fix must not break them. Prefer fixes that are conditional on the specific config
  values (e.g., `if cfg.get("x") ...`) rather than changing default behavior.
  If you must change shared code, ensure the change is backward-compatible.
- Keep fixes minimal and targeted \u2014 don't refactor, just fix the specific error
- If the error is unfixable (e.g., corrupt data, impossible config), say "UNFIXABLE: <reason>"
  and exit without changes
"""

    logger.info("[FIX] %s — spawning LLM fix (attempt %d/%d): %s",
                 idea_id, attempt_num, max_fix, error_text[:100])

    fix_log_dir = results_dir / "_fix_logs"
    fix_log_dir.mkdir(parents=True, exist_ok=True)
    fix_log = fix_log_dir / f"{idea_id}_attempt{attempt_num}.log"

    try:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)

        fix_cfg = cfg.get("executor_fix", {})
        claude_bin = fix_cfg.get("claude_bin") or "claude"
        model = fix_cfg.get("model") or "sonnet"
        fix_timeout = fix_cfg.get("timeout", 300)

        cmd = [
            claude_bin, "-p", prompt,
            "--dangerously-skip-permissions",
            "--output-format", "text",
            "--model", model,
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=fix_timeout, env=env,
            cwd=str(results_dir.parent),
        )

        # LLM actually ran — count this attempt
        fix_counts[idea_id] = attempts + 1

        response = result.stdout[-5000:] if result.stdout else ""
        fix_log.write_text(
            f"=== Attempt {attempt_num} for {idea_id} ===\n"
            f"Error: {error_text[:500]}\n\n"
            f"=== LLM Response ===\n{response}\n",
            encoding="utf-8",
        )

        if "UNFIXABLE" in response.upper():
            logger.info("[FIX] %s — LLM says unfixable: %s",
                         idea_id, response[:200])
            return False

        # Verify sealed files weren't touched by the fix
        sealed_files = cfg.get("sealed_files", [])
        if sealed_files:
            from orze.engine.sealed import load_sealed_manifest, verify_sealed_files
            manifest = load_sealed_manifest(results_dir)
            changed = verify_sealed_files(sealed_files, manifest)
            if changed:
                logger.error("[FIX] %s — LLM modified sealed files: %s — rejecting fix",
                             idea_id, changed)
                return False

        logger.info("[FIX] %s — LLM applied fix (attempt %d), will retry",
                     idea_id, attempt_num)
        return True

    except subprocess.TimeoutExpired:
        # Timed out — still count it (LLM may have made partial changes)
        fix_counts[idea_id] = attempts + 1
        logger.warning("[FIX] %s — LLM fix timed out (attempt %d)",
                        idea_id, attempt_num)
        return False
    except FileNotFoundError:
        # Don't count — Claude CLI not installed, no attempt was made
        logger.warning("[FIX] Claude CLI not found — skipping executor fix")
        return False
    except Exception as e:
        logger.error("[FIX] %s — LLM fix error: %s", idea_id, e)
        return False


# ---------------------------------------------------------------------------
# Structured failure classification, analysis, and persistence
# (merged from failure_analysis.py in v4.0)
# ---------------------------------------------------------------------------

import json as _json
import re as _re
import time as _time
from typing import Any as _Any, Dict as _Dict, List as _List

FAILURE_CATEGORIES = frozenset({
    "oom", "timeout", "stall", "crash", "pre_script_error",
    "eval_failure", "config_error", "sealed_violation",
})

_OOM_PATTERNS = _re.compile(
    r"CUDA out of memory|OutOfMemoryError|OOM|"
    r"out of memory|Cannot allocate memory|"
    r"CUDNN_STATUS_NOT_SUPPORTED.*memory",
    _re.IGNORECASE,
)
_CONFIG_PATTERNS = _re.compile(
    r"KeyError|ValueError.*config|Unknown \w+:|"
    r"missing.*required|invalid.*config|"
    r"yaml\.scanner\.ScannerError|"
    r"expected.*got|shape mismatch",
    _re.IGNORECASE,
)

_CATEGORY_INFO: _Dict[str, _Dict[str, str]] = {
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
    if source == "pre_script":
        return "pre_script_error"
    if source == "eval":
        return "eval_failure"
    text = error_text[:2000]
    if _OOM_PATTERNS.search(text):
        return "oom"
    if "Timed out" in text or "Timeout" in text:
        return "timeout"
    if "Stalled" in text or "no output for" in text:
        return "stall"
    if _CONFIG_PATTERNS.search(text):
        return "config_error"
    return "crash"


def build_failure_analysis(category: str, error_text: str) -> dict:
    info = _CATEGORY_INFO.get(category, _CATEGORY_INFO["crash"])
    return {
        "category": category,
        "what": error_text.split("\n")[0][:300],
        "why": info["why"],
        "lesson": info["lesson"],
        "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def write_failure_analysis(idea_dir: Path, category: str,
                           error_text: str) -> None:
    analysis = build_failure_analysis(category, error_text)
    try:
        idea_dir.mkdir(parents=True, exist_ok=True)
        path = idea_dir / "failure_analysis.json"
        path.write_text(_json.dumps(analysis, indent=2), encoding="utf-8")
    except OSError as e:
        logger.warning("Could not write failure analysis for %s: %s",
                       idea_dir.name, e)


def load_recent_failures(results_dir: Path,
                         limit: int = 200) -> _Dict[str, _List[_Dict[str, _Any]]]:
    if not results_dir.exists():
        return {}
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
    grouped: _Dict[str, _List[_Dict[str, _Any]]] = {}
    for name in scan_names:
        idea_dir = results_dir / name
        fa_path = idea_dir / "failure_analysis.json"
        if fa_path.exists():
            try:
                fa = _json.loads(fa_path.read_text(encoding="utf-8"))
                cat = fa.get("category", "crash")
                grouped.setdefault(cat, []).append({
                    "idea_id": name,
                    "what": fa.get("what", ""),
                    "why": fa.get("why", ""),
                    "lesson": fa.get("lesson", ""),
                })
                continue
            except (_json.JSONDecodeError, OSError):
                pass
        mf = idea_dir / "metrics.json"
        if not mf.exists():
            continue
        try:
            m = _json.loads(mf.read_text(encoding="utf-8"))
        except (_json.JSONDecodeError, OSError):
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
