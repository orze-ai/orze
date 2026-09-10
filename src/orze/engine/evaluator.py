"""Post-training evaluation subprocess launcher and monitor.

CALLING SPEC:
    launch_eval(idea_id, gpu, results_dir, cfg) -> EvalProcess | None
        idea_id: str — experiment identifier
        gpu: int — physical CUDA device; child sees it as local device 0
        results_dir: Path — parent dir for experiment results
        cfg: dict — requires 'eval_script'; optional 'eval_args', 'eval_timeout', 'eval_output',
                     'python', 'train_extra_env'
        returns: EvalProcess if launched, None if eval_script missing, already evaluated,
                 or admission rejected. None never itself proves completion;
                 scheduling callers must inspect evaluation/global stage state.
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
    LaunchIntegrityError, _assert_controller_runtime_attested,
    _assert_campaign_evidence_authorized,
    _assert_gpu_authorized, _authorized_gpu_environment,
    _assert_launch_authorized, _format_args, _launch_min_free_vram,
    _verify_gpu_free,
)
from orze.core.fs import tail_file
from orze.core.gpu_lease import gpu_execution_lease
from orze.core.evaluation_bundle import (
    EvaluationBundleError,
    get_evaluation_bundle_config,
    stage_evaluation_bundle,
)
from orze.core.benchmark_contract import (
    BenchmarkContractError,
    prepare_benchmark_evaluation,
)
from orze.engine.evaluation_contract import validate_evaluation_result
from orze.reporting.evaluation_output import evaluation_output_path
from orze.engine.accounting import (
    ComputeAccountingError, record_compute_start, record_compute_terminal,
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

    def reject_ineligible() -> bool:
        eligible, eligibility_reason = is_training_complete_for_downstream(
            idea_dir, cfg)
        if eligible:
            return False
        logger.warning(
            "[EVAL_SKIP] idea=%s reason=%s", idea_id, eligibility_reason)
        _record_eval_audit(idea_dir, "skip", eligibility_reason)
        return True

    if reject_ineligible():
        return None

    output_path = evaluation_output_path(idea_dir, cfg)
    if output_path is None:
        _record_eval_audit(idea_dir, "reject", "evaluation_output_path_invalid")
        return None
    # With legacy in-place evaluators, the training artifact already exists.
    # Presence is not evidence that evaluation has run; only its stage can
    # authorize reconciliation instead of a new launch.
    aliases_training = output_path == idea_dir / "metrics.json"
    evaluated_alias = (lake is not None and
                       lake.get_stage_state(idea_id, "evaluation") == "COMPLETE")
    if output_path.exists() and (not aliases_training or evaluated_alias):
        contract_ok, contract_reason = validate_evaluation_result(idea_dir, cfg)
        if lake is not None and lake.get_fsm_state(idea_id) == "IN_PROGRESS":
            training_stage = lake.get_stage_state(idea_id, "training")
            if training_stage != "COMPLETE" and training_stage in (
                    "NOT_STARTED", "PENDING", "IN_PROGRESS"):
                lake.record_stage_transition(
                    idea_id,
                    stage="training",
                    from_state=training_stage,
                    to_state="COMPLETE",
                    reason="reconcile_validated_training_output",
                    host=socket.gethostname(),
                    pid=os.getpid(),
                )
            evaluation_stage = lake.get_stage_state(idea_id, "evaluation")
            target_stage = "COMPLETE" if contract_ok else "FAILED"
            if evaluation_stage != target_stage:
                if evaluation_stage not in (
                        "NOT_STARTED", "PENDING", "IN_PROGRESS") or not (
                        lake.record_stage_transition(
                            idea_id,
                            stage="evaluation",
                            from_state=evaluation_stage,
                            to_state=target_stage,
                            reason=(
                                "reconcile_existing_valid_evaluation_output"
                                if contract_ok else
                                "reconcile_existing_invalid_evaluation_output"
                            ),
                            host=socket.gethostname(),
                            pid=os.getpid(),
                        )):
                    raise RuntimeError(
                        "existing_evaluation_stage_reconciliation_failed")
            target_global = "COMPLETE" if contract_ok else "FAILED"
            if not lake.record_state_transition(
                    idea_id,
                    from_state="IN_PROGRESS",
                    to_state=target_global,
                    reason=(
                        "reconcile_existing_valid_evaluation_output"
                        if contract_ok else
                        f"reconcile_existing_evaluation_output:{contract_reason}"
                    ),
                    host=socket.gethostname(),
                    pid=os.getpid(),
                    sop_type="training",
            ):
                raise RuntimeError(
                    "existing_evaluation_global_reconciliation_failed")
            _record_eval_audit(
                idea_dir,
                "reconcile",
                (
                    "existing_evaluation_output_valid"
                    if contract_ok else "existing_evaluation_output_invalid"
                ),
                detail=contract_reason,
            )
        logger.info(
            "[EVAL_SKIP] idea=%s reason=output_exists path=%s",
            idea_id, output_path,
        )
        _record_eval_audit(
            results_dir / idea_id, "skip", "output_exists",
            output_path=str(output_path),
        )
        return None

    python = cfg.get("python", sys.executable)
    eval_args = cfg.get("eval_args") or []
    eval_timeout = cfg.get("eval_timeout", 3600)

    _assert_launch_authorized(idea_id, results_dir, cfg)
    _assert_campaign_evidence_authorized(cfg, lake)
    _assert_gpu_authorized(gpu, cfg)

    log_path = results_dir / idea_id / "eval_output.log"
    logger.info("Launching eval for %s on GPU %s", idea_id, gpu)

    proc = None
    ep = None
    stage_started = False
    try:
        bundle = None
        entrypoint = eval_script
        if get_evaluation_bundle_config(cfg) is not None:
            bundle = stage_evaluation_bundle(idea_dir, cfg)
            entrypoint = str(bundle.entrypoint)
        cmd = [python, entrypoint]
        cmd.extend(_format_args(eval_args, {
            "idea_id": idea_id, "gpu": 0, "physical_gpu": gpu,
        }))
        benchmark_env = prepare_benchmark_evaluation(idea_dir, cfg)
        _assert_controller_runtime_attested(cfg)
        with gpu_execution_lease(gpu, require_idle=True) as lease_fds:
            _verify_gpu_free(gpu, _launch_min_free_vram(cfg))
            if lake is not None:
                training_stage = lake.get_stage_state(idea_id, "training")
                if training_stage != "COMPLETE":
                    if training_stage not in (
                            "NOT_STARTED", "PENDING", "IN_PROGRESS") or not (
                            lake.record_stage_transition(
                                idea_id,
                                stage="training",
                                from_state=training_stage,
                                to_state="COMPLETE",
                                reason="reconcile_validated_training_output",
                                host=socket.gethostname(),
                                pid=os.getpid(),
                            )):
                        raise RuntimeError(
                            "training_stage_not_ready_for_evaluation")
                evaluation_stage = lake.get_stage_state(
                    idea_id, "evaluation")
                if evaluation_stage not in ("NOT_STARTED", "PENDING") or not (
                        lake.record_stage_transition(
                            idea_id,
                            stage="evaluation",
                            from_state=evaluation_stage,
                            to_state="IN_PROGRESS",
                            reason=f"evaluation_launched on gpu {gpu}",
                            host=socket.gethostname(),
                            pid=os.getpid(),
                        )):
                    raise RuntimeError(
                        "evaluation_stage_transition_rejected")
                stage_started = True
            env = os.environ.copy()
            for k, v in (cfg.get("train_extra_env") or {}).items():
                env[k] = str(v)
            # Expose only the authorized physical device. Within the child it
            # is local CUDA device 0; {physical_gpu} remains available for
            # non-CUDA tooling that needs the host index.
            env = _authorized_gpu_environment(gpu, cfg, env)
            env.update(benchmark_env)
            if bundle is not None:
                env.update(bundle.environment(Path(
                    cfg.get("_project_root") or ".")))
            log_fh = open(log_path, "w", encoding="utf-8")
            try:
                proc = subprocess.Popen(
                    cmd, env=env, stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    preexec_fn=_new_process_group, pass_fds=lease_fds,
                )
            except Exception:
                log_fh.close()
                raise

        ep = EvalProcess(
            idea_id=idea_id, gpu=gpu, process=proc,
            start_time=time.time(), log_path=log_path,
            timeout=eval_timeout, attempt_id=secrets.token_hex(16),
            _log_fh=log_fh,
        )
        from orze.engine.accounting import record_compute_start
        record_compute_start(ep, idea_dir, phase="evaluation")
        return ep
    except LaunchIntegrityError:
        # A controller-identity failure is an authorization rejection, not a
        # best-effort evaluator failure. Let the scheduler stop rather than
        # silently continuing under a drifted runtime.
        raise
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
        elif isinstance(e, EvaluationBundleError):
            _record_eval_audit(
                idea_dir, "reject", "evaluation_bundle_preflight_failed",
                detail=str(e),
            )
        if stage_started and lake is not None:
            _write_eval_failure_marker(
                results_dir, idea_id, eval_output,
                f"Evaluation launch failed: {type(e).__name__}", lake=lake,
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
    try:
        ep.process.wait(timeout=ep.timeout)
    except subprocess.TimeoutExpired:
        reason = f"Timed out after {ep.timeout}s"
        outcome = "interrupted"
        reason_code = "evaluation_timeout"
    except Exception as e:
        reason = f"Evaluation wait failed: {type(e).__name__}"
        outcome = "failed"
        reason_code = "evaluation_error"
    else:
        # Normal exits (including nonzero) use the exact asynchronous closure:
        # sealed/source validation, one receipt, and the pipeline transition.
        # Do not catch closure/storage errors and reinterpret them as a second
        # process failure or write a contradictory terminal receipt.
        check_active_evals({gpu: ep}, results_dir, cfg, lake=lake)
        return
    logger.warning("Eval error for %s: %s", idea_id, reason)
    _terminate_and_reap(ep.process, f"eval {idea_id}")
    if ep.process.returncode is None:
        raise RuntimeError("evaluation_termination_unconfirmed")
    ep.close_log()
    _write_eval_failure_marker(
        results_dir, idea_id, eval_output, reason, lake=lake)
    record_compute_terminal(
        ep, results_dir / idea_id, outcome, reason_code,
        phase="evaluation", return_code=ep.process.poll())


def _write_eval_failure_marker(results_dir: Path, idea_id: str,
                               eval_output: str, reason: str, lake=None) -> None:
    """Safety net: write failure marker if eval process died without one.

    The fallback never overwrites existing output, training metrics, or an
    unsafe path. Lifecycle failure is independent of whether a marker can be
    written. The evaluator remains responsible for its domain report.

    Args:
        lake: IdeaLake instance for FSM transition recording (optional)
    """
    report_path = evaluation_output_path(
        results_dir / idea_id, {"eval_output": eval_output})
    if (report_path is not None and
            report_path != results_dir / idea_id / "metrics.json" and
            not report_path.exists()):
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also preserves output that appears between
            # the presence check and this best-effort fallback.
            with report_path.open("x", encoding="utf-8") as handle:
                json.dump({"status": "FAILED", "reason": reason[:500]},
                          handle, indent=2)
            logger.info("Wrote eval failure marker for %s", idea_id)
        except OSError as exc:
            logger.warning("Could not write eval failure marker for %s: %s",
                           idea_id, type(exc).__name__)
            _record_eval_audit(
                results_dir / idea_id, "diagnostic",
                "evaluation_failure_marker_unavailable",
                error_type=type(exc).__name__)

    # A script may have written its own failed report before exiting. The
    # lifecycle transition is independent evidence and must still be closed.
    if lake:
        try:
            current_state = lake.get_fsm_state(idea_id)
            if current_state == "IN_PROGRESS":
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
        try:
            ret = ep.process.poll()
        except OSError as exc:
            # A failed process observation is not a successful evaluation.
            # Reap this owned child before closing its accounting/lifecycle.
            _terminate_and_reap(ep.process, f"eval {ep.idea_id}")
            if ep.process.returncode is None:
                raise RuntimeError("evaluation_termination_unconfirmed") from exc
            ep.close_log()
            _write_eval_failure_marker(
                results_dir, ep.idea_id, eval_output,
                "Evaluation process observation failed", lake=lake)
            record_compute_terminal(
                ep, results_dir / ep.idea_id, "failed",
                "evaluation_process_observation_failed", phase="evaluation",
                return_code=ep.process.returncode)
            del active_evals[gpu]
            finished.append((ep.idea_id, gpu))
            continue
        elapsed = time.time() - ep.start_time

        if ret is None:
            # Still running — check timeout
            if elapsed > ep.timeout:
                logger.warning("[EVAL TIMEOUT] %s after %.0fm — killing",
                               ep.idea_id, elapsed / 60)
                _terminate_and_reap(ep.process, f"eval {ep.idea_id}")
                if ep.process.returncode is None:
                    raise RuntimeError("evaluation_termination_unconfirmed")
                ep.close_log()
                _write_eval_failure_marker(
                    results_dir, ep.idea_id, eval_output,
                    f"Timed out after {elapsed/60:.0f}m", lake=lake)
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
            eval_success, reason = validate_evaluation_result(
                results_dir / ep.idea_id, cfg)
            if not eval_success:
                logger.warning("[EVAL EVIDENCE INVALID] %s: %s",
                               ep.idea_id, reason)
                from orze.engine.failure_analysis import write_failure_analysis
                write_failure_analysis(
                    results_dir / ep.idea_id,
                    ("sealed_violation" if reason == "evaluation_sealed_files_changed"
                     else "eval_failure"), reason)
                _record_eval_audit(
                    results_dir / ep.idea_id, "reject",
                    ("benchmark_contract_validation_failed"
                     if reason.startswith("benchmark_") else
                     "evaluation_evidence_validation_failed"), detail=reason)
                _write_eval_failure_marker(
                    results_dir, ep.idea_id, eval_output,
                    f"Evaluation evidence validation failed: {reason}", lake=lake)
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


def run_post_scripts(
        idea_id: str, gpu: int, results_dir: Path, cfg: dict, lake=None):
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

    _assert_campaign_evidence_authorized(cfg, lake)

    python = cfg.get("python", sys.executable)
    env = os.environ.copy()
    for k, v in (cfg.get("train_extra_env") or {}).items():
        env[k] = str(v)
    env = _authorized_gpu_environment(gpu, cfg, env)

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
        _assert_controller_runtime_attested(cfg)
        args = ps.get("args") or []
        timeout = ps.get("timeout", 3600)
        name = ps.get("name", f"post-script-{i}")

        cmd = [python, script]
        cmd.extend(_format_args(args, {
            "idea_id": idea_id, "gpu": 0, "physical_gpu": gpu,
        }))

        log_path = results_dir / idea_id / f"{name}.log"
        logger.info("Running %s for %s", name, idea_id)

        proc = None
        handle = None
        log_fh = None
        try:
            with gpu_execution_lease(gpu, require_idle=True) as lease_fds:
                _verify_gpu_free(gpu, _launch_min_free_vram(cfg))
                log_fh = open(log_path, "w", encoding="utf-8")
                started = time.time()
                proc = subprocess.Popen(
                    cmd, env=env, stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    preexec_fn=_new_process_group,
                    pass_fds=lease_fds,
                )
                ep = EvalProcess(
                    idea_id=idea_id,
                    gpu=gpu,
                    process=proc,
                    start_time=started,
                    log_path=log_path,
                    timeout=float(timeout),
                    attempt_id=secrets.token_hex(16),
                    _log_fh=log_fh,
                )
                handle = ep
                record_compute_start(
                    handle, idea_dir, phase="post_script")
            return_code = proc.wait(timeout=timeout)
            record_compute_terminal(
                handle, idea_dir,
                "completed" if return_code == 0 else "failed",
                ("post_script_completed" if return_code == 0
                 else "post_script_nonzero"),
                phase="post_script", return_code=return_code,
            )
            if return_code == 0:
                logger.info("%s completed for %s", name, idea_id)
            else:
                logger.warning("%s failed for %s (exit %d)",
                               name, idea_id, return_code)
        except subprocess.TimeoutExpired:
            _terminate_and_reap(proc, f"post-script {idea_id}:{name}")
            if handle is not None:
                record_compute_terminal(
                    handle, idea_dir, "interrupted", "post_script_timeout",
                    phase="post_script", return_code=proc.poll(),
                )
            logger.warning("%s timed out for %s after %ds",
                           name, idea_id, timeout)
        except ComputeAccountingError:
            if proc is not None and proc.poll() is None:
                _terminate_and_reap(
                    proc, f"post-script {idea_id}:{name}", timeout=3)
            if handle is not None:
                try:
                    record_compute_terminal(
                        handle, idea_dir, "failed", "post_script_error",
                        phase="post_script", return_code=proc.poll(),
                    )
                except ComputeAccountingError:
                    pass
            raise
        except Exception as e:
            if proc is not None and proc.poll() is None:
                _terminate_and_reap(
                    proc, f"post-script {idea_id}:{name}", timeout=3)
            if handle is not None:
                try:
                    record_compute_terminal(
                        handle, idea_dir, "failed", "post_script_error",
                        phase="post_script", return_code=proc.poll(),
                    )
                except Exception:
                    pass
            logger.warning("%s error for %s: %s", name, idea_id, e)
        finally:
            if log_fh is not None:
                log_fh.close()
