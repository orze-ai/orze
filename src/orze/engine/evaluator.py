"""Post-training evaluation subprocess launcher and monitor.

CALLING SPEC:
    launch_eval(idea_id, gpu, results_dir, cfg) -> EvalProcess | None
        idea_id: str — experiment identifier
        gpu: int — physical CUDA device; child sees it as local device 0
        results_dir: Path — parent dir for experiment results
        cfg: dict — requires 'eval_script'; optional 'eval_args', 'eval_timeout', 'eval_output',
                     'python', 'train_extra_env'
        returns: EvalProcess if launched, None if eval_script missing, already evaluated,
                 or training status != COMPLETED
        side effects: spawns subprocess, creates results_dir/idea_id/eval_output.log

    check_active_evals(active_evals, results_dir, cfg) -> list[(idea_id, gpu)]
        active_evals: Dict[int, EvalProcess] — gpu -> running eval; MUTATED in-place (finished entries removed)
        results_dir: Path
        cfg: dict — uses 'eval_output' (default "eval_report.json")
        returns: list of (idea_id, gpu) tuples for evals that finished this cycle
        side effects: kills timed-out evals, writes failure marker JSON if eval dies without output

    run_eval(idea_id, gpu, results_dir, cfg) -> None
        Blocking wrapper around launch_eval; waits for completion.
        Used in --once mode. Writes failure marker on error/timeout.
        side effects: same as launch_eval, but blocks until done

    run_post_scripts(idea_id, gpu, results_dir, cfg) -> None
        idea_id: str
        gpu: int — set as CUDA_VISIBLE_DEVICES for post-scripts
        results_dir: Path
        cfg: dict — uses 'post_scripts' (list of {script, args, timeout, output, name}),
                     'python', 'train_extra_env'
        side effects: runs each post-script sequentially (blocking), skips if output file exists,
                      skips entirely if training status != COMPLETED
"""
import socket
import datetime
import json
import logging
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from orze.engine.process import EvalProcess, _new_process_group, _terminate_and_reap
from orze.engine.launcher import (
    _assert_gpu_authorized, _assert_launch_authorized, _format_args,
    _launch_min_free_vram, _verify_gpu_free,
)
from orze.core.fs import tail_file
from orze.core.benchmark_contract import (
    BenchmarkContractError,
    load_benchmark_values,
    prepare_benchmark_evaluation,
    validate_benchmark_receipt,
)

logger = logging.getLogger("orze")


def is_training_complete_for_downstream(
    idea_dir: Path, cfg: Optional[dict] = None,
) -> tuple[bool, str]:
    """Return whether immutable training output is eligible for eval/postwork."""
    metrics_path = Path(idea_dir) / "metrics.json"
    if not metrics_path.is_file():
        return False, "training_metrics_missing"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False, "training_metrics_invalid"
    if not isinstance(metrics, dict):
        return False, "training_metrics_invalid"
    if metrics.get("status") != "COMPLETED":
        return False, "training_not_completed"
    lineage = (cfg or {}).get("model_lineage", {})
    if isinstance(lineage, dict) and lineage.get("enabled") is True:
        try:
            from orze.core.model_lineage import (
                validate_model_lineage_for_evaluation,
            )
            validate_model_lineage_for_evaluation(Path(idea_dir), cfg or {})
        except Exception:
            return False, "training_model_lineage_invalid"
    return True, "training_completed"


def _record_eval_audit(idea_dir: Path, action: str, reason: str,
                       **extra) -> None:
    """Append one JSONL line to ``<idea_dir>/_eval_audit.jsonl``.

    Surfaces silent-skip cases (eval output already exists, post-script
    output already exists) that previously only logged at debug level
    and left no persistent trace for cross-cycle forensics. Closes the
    Stage-3 silent-skip class where 4 distinct root causes produced the
    same observable.
    """
    try:
        audit_path = Path(idea_dir) / "_eval_audit.jsonl"
        entry = {
            "ts": datetime.datetime.now().isoformat(),
            "action": action,
            "reason": reason,
            **extra,
        }
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # audit must never block the eval pipeline


def launch_eval(idea_id: str, gpu: int, results_dir: Path,
                cfg: dict, lake=None) -> Optional[EvalProcess]:
    """Launch a non-blocking eval subprocess. Returns EvalProcess or None.

    Args:
        lake: IdeaLake instance for FSM transition recording (optional)
    """
    eval_script = cfg.get("eval_script")
    if not eval_script:
        return None

    eval_output = cfg.get("eval_output") or "eval_report.json"
    idea_dir = results_dir / idea_id
    managed_lineage = (
        isinstance(cfg.get("model_lineage"), dict)
        and cfg["model_lineage"].get("enabled") is True
    )

    def reject_ineligible() -> bool:
        eligible, eligibility_reason = is_training_complete_for_downstream(
            idea_dir, cfg)
        if eligible:
            return False
        logger.warning(
            "[EVAL_SKIP] idea=%s reason=%s", idea_id, eligibility_reason)
        _record_eval_audit(idea_dir, "skip", eligibility_reason)
        return True

    if managed_lineage and reject_ineligible():
        return None

    output_path = results_dir / idea_id / eval_output
    if output_path.exists():
        logger.info(
            "[EVAL_SKIP] idea=%s reason=output_exists path=%s",
            idea_id, output_path,
        )
        _record_eval_audit(
            results_dir / idea_id, "skip", "output_exists",
            output_path=str(output_path),
        )
        return None

    if not managed_lineage and reject_ineligible():
        return None

    python = cfg.get("python", sys.executable)
    eval_args = cfg.get("eval_args") or []
    eval_timeout = cfg.get("eval_timeout", 3600)

    _assert_launch_authorized(idea_id, results_dir, cfg)
    _assert_gpu_authorized(gpu, cfg)

    cmd = [python, eval_script]
    cmd.extend(_format_args(eval_args, {
        "idea_id": idea_id, "gpu": 0, "physical_gpu": gpu,
    }))

    log_path = results_dir / idea_id / "eval_output.log"
    logger.info("Launching eval for %s on GPU %s", idea_id, gpu)

    proc = None
    ep = None
    try:
        benchmark_env = prepare_benchmark_evaluation(idea_dir, cfg)
        _verify_gpu_free(gpu, _launch_min_free_vram(cfg))
        env = os.environ.copy()
        for k, v in (cfg.get("train_extra_env") or {}).items():
            env[k] = str(v)
        # Expose only the authorized physical device. Within the child it is
        # local CUDA device 0; {physical_gpu} remains available for non-CUDA
        # tooling that needs the host index.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.update(benchmark_env)
        log_fh = open(log_path, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
                preexec_fn=_new_process_group,
            )
        except Exception:
            log_fh.close()
            raise

        # v4.5: No explicit FSM transition for eval launch.
        # Training + eval both occur within IN_PROGRESS state.
        # Only final COMPLETE/FAILED is recorded (below).

        ep = EvalProcess(
            idea_id=idea_id, gpu=gpu, process=proc,
            start_time=time.time(), log_path=log_path,
            timeout=eval_timeout, attempt_id=secrets.token_hex(16),
            _log_fh=log_fh,
        )
        from orze.engine.accounting import record_compute_start
        record_compute_start(ep, idea_dir, phase="evaluation")
        return ep
    except Exception as e:
        if proc is not None and proc.poll() is None:
            _terminate_and_reap(proc, f"eval {idea_id}", timeout=3)
        if ep is not None:
            try:
                from orze.engine.accounting import record_compute_terminal
                record_compute_terminal(
                    ep, idea_dir, "failed",
                    "evaluation_launch_initialization_failed",
                    phase="evaluation", return_code=proc.poll())
            except Exception:
                pass
            ep.close_log()
        if isinstance(e, BenchmarkContractError):
            _record_eval_audit(
                idea_dir, "reject", "benchmark_contract_preflight_failed",
                detail=str(e),
            )
        logger.warning("Failed to launch eval for %s: %s", idea_id, e)
        return None


def run_eval(idea_id: str, gpu: int, results_dir: Path, cfg: dict, lake=None):
    """Run post-training evaluation (blocking). Used in --once mode.

    Args:
        lake: IdeaLake instance for FSM transition recording (optional)
    """
    ep = launch_eval(idea_id, gpu, results_dir, cfg, lake=lake)
    if ep is None:
        return
    eval_output = cfg.get("eval_output") or "eval_report.json"
    reason = ""
    outcome = "failed"
    reason_code = "evaluation_error"
    try:
        ep.process.wait(timeout=ep.timeout)
        if ep.process.returncode == 0:
            contract_ok, contract_reason = validate_benchmark_receipt(
                results_dir / idea_id, cfg,
                values=load_benchmark_values(results_dir / idea_id, cfg),
            )
            if contract_ok:
                logger.info("Eval completed for %s", idea_id)
                outcome = "completed"
                reason_code = "evaluation_process_completed"
            else:
                reason = f"Benchmark contract failed: {contract_reason}"
                reason_code = "evaluation_benchmark_contract_failed"
                logger.error("Eval contract failed for %s: %s",
                             idea_id, contract_reason)
                _record_eval_audit(
                    results_dir / idea_id, "reject",
                    "benchmark_contract_validation_failed",
                    detail=contract_reason,
                )
        else:
            reason = f"Exit code {ep.process.returncode}"
            reason_code = "evaluation_process_nonzero"
            logger.warning("Eval failed for %s (exit %d)",
                           idea_id, ep.process.returncode)
    except subprocess.TimeoutExpired:
        reason = f"Timed out after {ep.timeout}s"
        outcome = "interrupted"
        reason_code = "evaluation_timeout"
        logger.warning("Eval timed out for %s after %ds",
                       idea_id, ep.timeout)
        _terminate_and_reap(ep.process, f"eval {idea_id}")
    except Exception as e:
        reason = str(e)
        logger.warning("Eval error for %s: %s", idea_id, e)
    finally:
        ep.close_log()
        from orze.engine.accounting import record_compute_terminal
        record_compute_terminal(
            ep, results_dir / idea_id, outcome, reason_code,
            phase="evaluation", return_code=ep.process.poll())
        if reason:
            _write_eval_failure_marker(results_dir, idea_id, eval_output, reason, lake=lake)


def _write_eval_failure_marker(results_dir: Path, idea_id: str,
                               eval_output: str, reason: str, lake=None) -> None:
    """Safety net: write failure marker if eval process died without one.

    The marker file is the eval_output itself so the backlog scanner
    won't re-queue this idea.  The eval script is responsible for
    writing domain-specific reports; this is a generic fallback.

    Args:
        lake: IdeaLake instance for FSM transition recording (optional)
    """
    report_path = results_dir / idea_id / eval_output
    if report_path.exists():
        return  # Script already wrote a report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "status": "FAILED",
        "reason": reason[:500],
    }, indent=2))
    logger.info("Wrote eval failure marker for %s", idea_id)

    # Record FSM transition: IN_PROGRESS → FAILED (v4.5: generic for all SOP types)
    if lake:
        try:
            lake.record_state_transition(
                idea_id,
                from_state="IN_PROGRESS",
                to_state="FAILED",
                reason=reason[:100],
                host=socket.gethostname(),
                pid=os.getpid(),
                sop_type="training",
            )
        except Exception as e:
            logger.warning("FSM transition failed (non-blocking): %s", e)


def check_active_evals(active_evals: Dict[int, EvalProcess],
                       results_dir: Path, cfg: dict, lake=None) -> list:
    """Check running eval processes. Returns list of (idea_id, gpu) for finished evals.

    Args:
        lake: IdeaLake instance for FSM transition recording (optional)
    """
    eval_output = cfg.get("eval_output") or "eval_report.json"
    from orze.engine.accounting import record_compute_terminal
    finished = []
    for gpu in list(active_evals.keys()):
        ep = active_evals[gpu]
        ret = ep.process.poll()
        elapsed = time.time() - ep.start_time

        if ret is None:
            # Still running — check timeout
            if elapsed > ep.timeout:
                logger.warning("[EVAL TIMEOUT] %s after %.0fm — killing",
                               ep.idea_id, elapsed / 60)
                _terminate_and_reap(ep.process, f"eval {ep.idea_id}")
                ep.close_log()
                _write_eval_failure_marker(
                    results_dir, ep.idea_id, eval_output,
                    f"Timed out after {elapsed/60:.0f}m")
                record_compute_terminal(
                    ep, results_dir / ep.idea_id, "interrupted",
                    "evaluation_timeout", phase="evaluation",
                    return_code=ep.process.poll())
                del active_evals[gpu]
                finished.append((ep.idea_id, gpu))
            continue

        # Process exited
        ep.close_log()
        eval_success = False
        if ret == 0:
            logger.info("[EVAL OK] %s on GPU %s in %.1fm",
                        ep.idea_id, gpu, elapsed / 60)
            # Verify sealed files and validate metrics
            sealed_ok = True
            sealed_files = cfg.get("sealed_files", [])
            if sealed_files:
                from orze.engine.sealed import load_sealed_manifest, verify_sealed_files
                manifest = load_sealed_manifest(results_dir)
                changed = verify_sealed_files(sealed_files, manifest)
                if changed:
                    sealed_ok = False
                    logger.error("[SEALED VIOLATION] %s modified sealed files: %s",
                                 ep.idea_id, changed)
                    from orze.engine.failure_analysis import write_failure_analysis
                    write_failure_analysis(
                        results_dir / ep.idea_id, "sealed_violation",
                        f"Sealed files modified: {', '.join(changed)}")
                    _write_eval_failure_marker(
                        results_dir, ep.idea_id, eval_output,
                        f"Sealed file violation: {', '.join(changed)}", lake=lake)
            if sealed_ok:
                # Validate metric values (NaN, inf, range)
                metrics_path = results_dir / ep.idea_id / "metrics.json"
                if metrics_path.exists():
                    try:
                        import json as _json
                        metrics = _json.loads(metrics_path.read_text(encoding="utf-8"))
                        from orze.engine.sealed import validate_metrics
                        valid, reason = validate_metrics(metrics, cfg)
                        if not valid:
                            logger.warning("[METRIC INVALID] %s: %s", ep.idea_id, reason)
                            from orze.engine.failure_analysis import write_failure_analysis
                            write_failure_analysis(
                                results_dir / ep.idea_id, "eval_failure", reason)
                            # Durably fail the idea, SYMMETRICALLY with the
                            # sealed-violation path above: write the eval_output
                            # marker so the backlog scanner (engine/phases.py
                            # ~line 631: metrics.json present + eval_output absent
                            # => re-queue) does NOT re-queue this idea forever.
                            # Invalid metrics are deterministic for a given
                            # checkpoint, so re-evaluating never resolves them.
                            _write_eval_failure_marker(
                                results_dir, ep.idea_id, eval_output,
                                f"Metric validation failed: {reason}", lake=lake)
                        else:
                            contract_ok, contract_reason = (
                                validate_benchmark_receipt(
                                    results_dir / ep.idea_id, cfg,
                                    values=load_benchmark_values(
                                        results_dir / ep.idea_id, cfg),
                                )
                            )
                            if contract_ok:
                                eval_success = True
                            else:
                                logger.error(
                                    "[BENCHMARK CONTRACT INVALID] %s: %s",
                                    ep.idea_id, contract_reason,
                                )
                                from orze.engine.failure_analysis import (
                                    write_failure_analysis,
                                )
                                write_failure_analysis(
                                    results_dir / ep.idea_id, "eval_failure",
                                    f"Benchmark contract failed: {contract_reason}",
                                )
                                _record_eval_audit(
                                    results_dir / ep.idea_id, "reject",
                                    "benchmark_contract_validation_failed",
                                    detail=contract_reason,
                                )
                                _write_eval_failure_marker(
                                    results_dir, ep.idea_id, eval_output,
                                    "Benchmark contract failed: "
                                    f"{contract_reason}", lake=lake,
                                )
                    except Exception as exc:
                        _write_eval_failure_marker(
                            results_dir, ep.idea_id, eval_output,
                            "Metric validation could not read a valid metrics "
                            f"document: {type(exc).__name__}", lake=lake)
                else:
                    _write_eval_failure_marker(
                        results_dir, ep.idea_id, eval_output,
                        "Evaluation process exited successfully without "
                        "metrics.json", lake=lake)
        else:
            # Log tail of eval output for diagnosis
            eval_tail = tail_file(ep.log_path, 2048).strip()
            logger.warning("[EVAL FAILED] %s on GPU %s — exit %d\n%s",
                           ep.idea_id, gpu, ret,
                           eval_tail[-500:] if eval_tail else "(no output)")
            _write_eval_failure_marker(
                results_dir, ep.idea_id, eval_output,
                f"Exit code {ret}: {eval_tail[-300:] if eval_tail else 'no output'}", lake=lake)

        record_compute_terminal(
            ep,
            results_dir / ep.idea_id,
            "completed" if eval_success else "failed",
            ("evaluation_validated" if eval_success
             else "evaluation_failed_validation_or_process"),
            phase="evaluation",
            return_code=ret,
        )

        # Record FSM transition: IN_PROGRESS → COMPLETE (v4.5: generic for all SOP types)
        if eval_success and lake:
            try:
                lake.record_state_transition(
                    ep.idea_id,
                    from_state="IN_PROGRESS",
                    to_state="COMPLETE",
                    reason=f"training_and_eval completed on gpu {gpu}",
                    host=socket.gethostname(),
                    pid=os.getpid(),
                    sop_type=cfg.get("sop", "training"),
                )
            except Exception as e:
                logger.warning("FSM transition failed (non-blocking): %s", e)

        del active_evals[gpu]
        finished.append((ep.idea_id, gpu))

    return finished


def run_post_scripts(idea_id: str, gpu: int, results_dir: Path, cfg: dict):
    """Run additional post-training scripts (beyond eval_script).
    Each entry in post_scripts is a dict with: script, args, timeout, output."""
    post_scripts = (cfg.get("post_scripts") or [])
    if not post_scripts:
        return

    idea_dir = results_dir / idea_id
    eligible, eligibility_reason = is_training_complete_for_downstream(
        idea_dir, cfg)
    if not eligible:
        logger.warning(
            "[POST_SCRIPT_SKIP] idea=%s reason=%s",
            idea_id, eligibility_reason)
        _record_eval_audit(idea_dir, "skip", eligibility_reason)
        return

    python = cfg.get("python", sys.executable)
    env = os.environ.copy()
    for k, v in (cfg.get("train_extra_env") or {}).items():
        env[k] = str(v)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for i, ps in enumerate(post_scripts):
        script = ps.get("script")
        if not script:
            continue

        # Skip if output already exists
        output_file = ps.get("output", "")
        if output_file:
            output_path = results_dir / idea_id / output_file
            if output_path.exists():
                name = ps.get("name", f"post-script-{i}")
                logger.info(
                    "[POST_SCRIPT_SKIP] idea=%s script=%s reason=output_exists "
                    "path=%s",
                    idea_id, name, output_path,
                )
                _record_eval_audit(
                    results_dir / idea_id, "skip", "output_exists",
                    script=name, output_path=str(output_path),
                )
                continue

        _assert_launch_authorized(idea_id, results_dir, cfg)
        _assert_gpu_authorized(gpu, cfg)
        _verify_gpu_free(gpu, _launch_min_free_vram(cfg))
        args = ps.get("args") or []
        timeout = ps.get("timeout", 3600)
        name = ps.get("name", f"post-script-{i}")

        cmd = [python, script]
        cmd.extend(_format_args(args, {
            "idea_id": idea_id, "gpu": 0, "physical_gpu": gpu,
        }))

        log_path = results_dir / idea_id / f"{name}.log"
        logger.info("Running %s for %s", name, idea_id)

        try:
            with open(log_path, "w", encoding="utf-8") as log_fh:
                result = subprocess.run(
                    cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            if result.returncode == 0:
                logger.info("%s completed for %s", name, idea_id)
            else:
                logger.warning("%s failed for %s (exit %d)",
                               name, idea_id, result.returncode)
        except subprocess.TimeoutExpired:
            logger.warning("%s timed out for %s after %ds",
                           name, idea_id, timeout)
        except Exception as e:
            logger.warning("%s error for %s: %s", name, idea_id, e)
