"""Periodic retrospection with signal dispatch for auto-evolution.

CALLING SPEC:
    run_retrospection(results_dir, cfg, completed_count, last_count,
                      retro_state=None) -> int
        results_dir:      Path to results directory
        cfg:              orze.yaml config dict (needs 'retrospection' key)
        completed_count:  current number of completed experiments
        last_count:       last completed count when retrospection ran
        retro_state:      mutable dict for tracking evolution attempts
                          (None = legacy pause-only mode)
        returns:          updated last_count (same as input if not triggered)

    Triggers when completed_count >= last_count + interval.
    Detects: plateau, high failure rate, family concentration.
    Dispatches signals to evolution roles via trigger files:
        plateau → code_evolution role
        family_imbalance → meta_research role
        high_failure_rate → meta_research role
    Falls back to pause after evolution attempts are exhausted.

    is_research_paused(results_dir) -> bool
        Returns True if .pause_research sentinel exists.

    resume_research(results_dir) -> bool
        Removes .pause_research sentinel. Returns True if it existed.
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orze")

PAUSE_SENTINEL = ".pause_research"


def is_research_paused(results_dir: Path) -> bool:
    """Check if research is paused by retrospection."""
    return (results_dir / PAUSE_SENTINEL).exists()


def resume_research(results_dir: Path) -> bool:
    """Remove the pause sentinel to resume research. Returns True if it existed."""
    sentinel = results_dir / PAUSE_SENTINEL
    if sentinel.exists():
        sentinel.unlink()
        logger.info("Research resumed (pause sentinel removed)")
        return True
    return False


def _detect_plateau(results_dir: Path, window: int = 200,
                    min_metric_keys: int = 4,
                    primary_metric: Optional[str] = None,
                    benchmark_keys: Optional[set] = None) -> tuple:
    """Check if the best primary metric has improved in the last N completions.

    Only counts experiments that evaluated on the full benchmark set.
    Coverage is measured authoritatively from the project's report.columns
    in orze.yaml — those are the per-dataset metric keys the project
    itself declares as "the benchmark." An experiment counts if at least
    ``min_metric_keys`` of the declared benchmark_keys are present and
    numeric in its metrics.json, filtering out single-dataset shards
    and eval-only sub-runs whose tiny-scope values would otherwise
    poison best_recent vs best_older comparison.

    If benchmark_keys is None (caller didn't declare report.columns),
    falls back to a conservative heuristic: count flat numeric keys
    that aren't the primary_metric and aren't common meta-fields.

    Returns (is_plateau: bool, message: str).
    """
    entries = []
    for d in results_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            if m.get("status") != "COMPLETED":
                continue
            # Coverage: count presence of the project-declared benchmark
            # keys. Fall back to "numeric non-primary, non-meta, flat"
            # heuristic only when caller didn't pass benchmark_keys.
            if benchmark_keys:
                present = sum(
                    1 for k in benchmark_keys
                    if isinstance(m.get(k), (int, float))
                    and not isinstance(m.get(k), bool))
            else:
                _FALLBACK_META = {"avg_wer", "test_accuracy", "score",
                                  "status", "training_time", "error",
                                  "num_eval_tasks", "timestamp",
                                  "eval_time", "avg"}
                skip = set(_FALLBACK_META)
                if primary_metric:
                    skip.add(primary_metric)
                present = sum(
                    1 for k, v in m.items()
                    if isinstance(k, str) and k not in skip
                    and not k.startswith("_")
                    and isinstance(v, (int, float))
                    and not isinstance(v, bool))
            if present < min_metric_keys:
                continue
            # Use the project-declared primary_metric; fall back to
            # common aggregate names.
            val = None
            if primary_metric:
                val = m.get(primary_metric)
            if val is None:
                val = (m.get("avg_wer") or m.get("test_accuracy")
                       or m.get("score"))
            if val is not None and isinstance(val, (int, float)) and val > 0:
                entries.append((mf.stat().st_mtime, float(val)))
        except Exception:
            continue

    if len(entries) < window:
        return False, f"Not enough data ({len(entries)}/{window})"

    entries.sort(reverse=True)  # newest first
    recent = entries[:window]
    older = entries[window:]

    best_recent = min(v for _, v in recent)
    best_older = min(v for _, v in older) if older else float("inf")

    improved = best_recent < best_older
    return (
        not improved,
        f"best_recent={best_recent:.4f} best_older={best_older:.4f} improved={improved}",
    )


def _detect_high_failure_rate(results_dir: Path, window: int = 100,
                               threshold: float = 0.5) -> tuple:
    """Check if recent failure rate exceeds threshold.

    Returns (is_high: bool, message: str).
    """
    entries = []
    for d in results_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            entries.append((mf.stat().st_mtime, m.get("status", "UNKNOWN")))
        except Exception:
            continue

    entries.sort(reverse=True)
    recent = entries[:window]
    if len(recent) < window // 2:
        return False, f"Not enough data ({len(recent)}/{window})"

    failures = sum(1 for _, s in recent if s == "FAILED")
    rate = failures / len(recent)
    return rate > threshold, f"recent_fail_rate={rate:.1%} ({failures}/{len(recent)})"


def _run_builtin_checks(results_dir: Path, cfg: dict,
                        completed_count: int) -> Optional[str]:
    """Run built-in plateau and failure-rate detection.

    Returns pause reason string if research should be paused, None otherwise.
    """
    retro_cfg = cfg.get("retrospection", {})
    plateau_window = retro_cfg.get("plateau_window", 200)
    fail_window = retro_cfg.get("fail_window", 100)
    fail_threshold = retro_cfg.get("fail_threshold", 0.5)

    # Derive project's benchmark metric set from report.columns.
    # Each project declares its own full-benchmark coverage (per-dataset
    # metric keys) + primary aggregate. We exclude the primary_metric
    # itself so only per-dataset coverage counts toward min_metric_keys.
    report_cfg = cfg.get("report", {}) or {}
    primary_metric = report_cfg.get("primary_metric")
    benchmark_keys = None
    try:
        cols = report_cfg.get("columns") or []
        bench_set = {c.get("key") for c in cols if isinstance(c, dict)}
        bench_set.discard(primary_metric)
        bench_set.discard("training_time")
        bench_set.discard(None)
        if bench_set:
            benchmark_keys = bench_set
    except Exception:
        benchmark_keys = None

    reasons = []

    is_plateau, plateau_msg = _detect_plateau(
        results_dir, plateau_window,
        primary_metric=primary_metric,
        benchmark_keys=benchmark_keys)
    if is_plateau:
        reasons.append(f"PLATEAU: No improvement in last {plateau_window} experiments ({plateau_msg})")
        logger.warning("Retrospection: %s", reasons[-1])

    is_high_fail, fail_msg = _detect_high_failure_rate(
        results_dir, fail_window, fail_threshold)
    if is_high_fail:
        reasons.append(f"HIGH FAILURE RATE: {fail_msg}")
        logger.warning("Retrospection: %s", reasons[-1])

    # Family concentration check
    max_consec = retro_cfg.get("max_consecutive_family", 5)
    if max_consec > 0:
        try:
            from orze.engine.family_guard import (
                get_recent_winning_families, check_family_concentration,
            )
            report_cfg = cfg.get("report", {})
            primary = report_cfg.get("primary_metric", "")
            sort_dir = report_cfg.get("sort", "descending")
            if primary:
                recent = get_recent_winning_families(
                    results_dir, primary, n=max_consec, sort=sort_dir)
                conc_msg = check_family_concentration(recent, max_consec)
                if conc_msg:
                    reasons.append(f"FAMILY CONCENTRATION: {conc_msg}")
                    logger.warning("Retrospection: %s", reasons[-1])
        except Exception as e:
            logger.debug("Family concentration check skipped: %s", e)

    if reasons:
        return "; ".join(reasons)
    return None


_DEFAULT_DISPATCH = {
    "plateau": "code_evolution",
    "family_imbalance": "meta_research",
    "high_failure_rate": "meta_research",
    "persistent_failure": "pause",
}


def _classify_signals(pause_reason: str) -> list:
    """Extract individual signal types from a combined pause reason string."""
    signals = []
    if "PLATEAU" in pause_reason:
        signals.append("plateau")
    if "HIGH FAILURE RATE" in pause_reason:
        signals.append("high_failure_rate")
    if "FAMILY CONCENTRATION" in pause_reason:
        signals.append("family_imbalance")
    return signals or ["persistent_failure"]


def _dispatch_signal(signal: str, results_dir: Path, cfg: dict,
                     retro_state: dict) -> str:
    """Dispatch a signal to the appropriate role or pause.

    Writes a trigger file for evolution/meta-research roles, or
    falls back to pause after evolution attempts are exhausted.

    Returns the action taken: role name or "pause".
    """
    retro_cfg = cfg.get("retrospection", {})
    dispatch = retro_cfg.get("dispatch", _DEFAULT_DISPATCH)
    evolution_enabled = cfg.get("evolution", {}).get("enabled", False)
    max_attempts = retro_cfg.get("evolution_attempts_before_pause",
                                 cfg.get("evolution", {}).get(
                                     "max_attempts_per_plateau", 2))

    action = dispatch.get(signal, "pause")

    # If evolution is disabled or action is pause, go straight to pause
    if not evolution_enabled or action == "pause":
        _write_pause(results_dir, signal)
        return "pause"

    # Track evolution attempts per signal type
    attempts_key = f"{signal}_evolution_attempts"
    attempts = retro_state.get(attempts_key, 0)

    if attempts >= max_attempts:
        logger.warning("Signal '%s': exhausted %d evolution attempts — escalating to pause",
                       signal, attempts)
        _write_pause(results_dir, f"{signal} (after {attempts} evolution attempts)")
        return "pause"

    # Write trigger file for the role
    trigger_file = results_dir / f"_trigger_{action}"
    try:
        trigger_file.write_text(
            json.dumps({
                "signal": signal,
                "attempt": attempts + 1,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, indent=2),
            encoding="utf-8",
        )
        retro_state[attempts_key] = attempts + 1
        logger.info("Dispatched signal '%s' → role '%s' (attempt %d/%d)",
                    signal, action, attempts + 1, max_attempts)
    except OSError as e:
        logger.error("Could not write trigger for %s: %s", action, e)
        _write_pause(results_dir, signal)
        return "pause"

    return action


def _write_pause(results_dir: Path, reason: str) -> None:
    """Write the pause sentinel."""
    sentinel = results_dir / PAUSE_SENTINEL
    sentinel.write_text(
        json.dumps({
            "paused_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reason": reason,
        }, indent=2),
        encoding="utf-8",
    )
    logger.warning("Research PAUSED by retrospection: %s", reason)


def run_retrospection(results_dir: Path, cfg: dict,
                      completed_count: int, last_count: int,
                      retro_state: Optional[dict] = None) -> int:
    """Run retrospection if interval threshold crossed.

    Runs the user's custom script (if configured) AND built-in checks.
    When signals are detected, dispatches to evolution roles before
    falling back to pause as a last resort.

    Args:
        retro_state: mutable dict for tracking evolution attempts across
            retrospection cycles. If None, falls back to pause-only behavior.

    Returns updated last_count (unchanged if not triggered).
    """
    retro_cfg = cfg.get("retrospection", {})
    if not retro_cfg.get("enabled"):
        return last_count

    interval = retro_cfg.get("interval", 50)
    if completed_count < last_count + interval:
        return last_count

    logger.info("Retrospection triggered: %d completed (last run at %d, interval %d)",
                completed_count, last_count, interval)

    # 1. Run custom script (if configured)
    script = retro_cfg.get("script")
    if script:
        timeout = retro_cfg.get("timeout", 120)
        try:
            env = os.environ.copy()
            env["ORZE_RESULTS_DIR"] = str(results_dir)
            env["ORZE_COMPLETED_COUNT"] = str(completed_count)
            result = subprocess.run(
                [cfg.get("python", sys.executable), script],
                capture_output=True, text=True, timeout=timeout, env=env,
            )
            if result.returncode == 0:
                output_file = results_dir / "_retrospection.txt"
                if result.stdout.strip():
                    output_file.write_text(result.stdout, encoding="utf-8")
                    logger.info("Retrospection script output written to %s",
                                output_file)
            else:
                logger.warning("Retrospection script failed (rc=%d): %s",
                               result.returncode, result.stderr[:300])
        except subprocess.TimeoutExpired:
            logger.warning("Retrospection script timed out after %ds", timeout)
        except Exception as e:
            logger.warning("Retrospection script error: %s", e)

    # 1b. Cross-experiment analysis (writes insights for research agent)
    try:
        from orze.engine.experiment_analysis import (
            analyze_experiments, format_insights,
        )
        analysis = analyze_experiments(results_dir, cfg)
        if analysis:
            insights = format_insights(analysis)
            insights_file = results_dir / "_experiment_insights.txt"
            insights_file.write_text(insights, encoding="utf-8")
            logger.info("Experiment analysis: %d regressions, %d improvements, "
                        "%d patterns, %d actions",
                        len(analysis.get("regressions", [])),
                        len(analysis.get("improvements", [])),
                        len(analysis.get("patterns", [])),
                        len(analysis.get("suggested_actions", [])))
            for action in analysis.get("suggested_actions", []):
                logger.info("  → %s", action[:120])

            # Smart Suggestions: convert insights into ideas (no LLM needed)
            from orze.extensions import has_pro
            if not has_pro():
                try:
                    from orze.engine.smart_suggestions import (
                        suggest_ideas, write_suggestions,
                    )
                    best_cfg = analysis.get("best", {}).get("config_summary", {})
                    # Load full best config from idea_config.yaml
                    best_id = analysis["best"].get("id", "")
                    best_cfg_path = results_dir / best_id / "idea_config.yaml"
                    if best_cfg_path.exists():
                        import yaml as _yaml
                        best_cfg = _yaml.safe_load(
                            best_cfg_path.read_text()) or {}
                    if best_cfg:
                        suggestions = suggest_ideas(analysis, best_cfg, cfg)
                        if suggestions:
                            ideas_path = Path(cfg.get("ideas_file", "ideas.md"))
                            write_suggestions(ideas_path, suggestions)
                except Exception as e:
                    logger.debug("Smart Suggestions skipped: %s", e)
    except Exception as e:
        logger.debug("Experiment analysis skipped: %s", e)

    # 2. Built-in detection + dispatch
    auto_pause = retro_cfg.get("auto_pause", True)
    if auto_pause:
        pause_reason = _run_builtin_checks(results_dir, cfg, completed_count)
        sentinel = results_dir / PAUSE_SENTINEL

        if pause_reason:
            if retro_state is not None:
                # Dispatch mode: route signals to evolution roles
                signals = _classify_signals(pause_reason)
                for signal in signals:
                    _dispatch_signal(signal, results_dir, cfg, retro_state)
            else:
                # Legacy mode: direct pause
                _write_pause(results_dir, pause_reason)
        else:
            # Clear sentinel if conditions improved
            if sentinel.exists():
                sentinel.unlink()
                logger.info("Research RESUMED — plateau/failure conditions cleared")
            # Reset evolution attempt counters on improvement
            if retro_state is not None:
                for key in list(retro_state.keys()):
                    if key.endswith("_evolution_attempts"):
                        retro_state[key] = 0

    return completed_count
