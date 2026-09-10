def graceful_shutdown(results_dir: Path, cfg: dict,
                      active: dict, active_evals: dict, active_roles: dict,
                      iteration: int, state_dict: dict, lake,
                      hostname: str, instance_uuid: str,
                      kill_all: bool = False, *, managed: bool = False,
                      pid_file_path: Path | None = None) -> None:
    """Terminate roles, detach or kill training/eval, save state, clean up.

    Args:
        results_dir: Path to results directory.
        cfg: Config dict.
        active: Dict of gpu -> TrainingProcess.
        active_evals: Dict of gpu -> EvalProcess.
        active_roles: Dict of role_name -> RoleProcess.
        iteration: Current iteration number.
        state_dict: Pre-built state dict for persistence.
        lake: IdeaLake instance (or None).
        hostname: This node's hostname.
        instance_uuid: This node's instance UUID.
        kill_all: If True, kill training and eval processes too (not just
                  detach them). Used by `orze --stop` to fully stop everything.
    """
    logger.info("Shutting down gracefully (kill_all=%s)...", kill_all)
    training_count = len(active)
    eval_count = len(active_evals)
    held_training, held_evals = {}, {}

    def close_interrupted_evaluation(ep) -> None:
        """Close compute/stage evidence after a controlled evaluator stop."""
        try:
            from orze.engine.accounting import record_compute_terminal
            record_compute_terminal(
                ep, results_dir / ep.idea_id, "interrupted",
                "evaluation_controller_shutdown", phase="evaluation",
                return_code=ep.process.poll(),
            )
        except Exception as exc:
            logger.warning(
                "Could not persist evaluation interruption receipt for %s: %s",
                ep.idea_id, type(exc).__name__,
            )
        if lake is not None:
            try:
                stage = lake.get_stage_state(ep.idea_id, "evaluation")
                if stage == "IN_PROGRESS" and not lake.record_stage_transition(
                    ep.idea_id,
                    stage="evaluation",
                    from_state="IN_PROGRESS",
                    to_state="PENDING",
                    reason="evaluation_controller_shutdown_retry",
                    host=hostname,
                    pid=os.getpid(),
                ):
                    raise RuntimeError("evaluation_stage_retry_rejected")
            except Exception as exc:
                logger.warning(
                    "Could not reset evaluation stage for %s: %s",
                    ep.idea_id, type(exc).__name__,
                )

    # 0. Write "shutting_down" heartbeat so other nodes know our state
    if not managed:
        try:
            write_shutdown_heartbeat(
                results_dir, hostname, instance_uuid, active)
        except Exception:
            pass

    if kill_all:
        # Kill ALL child processes: training, eval, and roles
        for gpu, tp in active.items():
            logger.info("Killing training %s on GPU %s (PID %d)",
                        tp.idea_id, gpu, tp.process.pid)
            phase = "posthoc" if getattr(tp, "is_posthoc", False) else "training"
            if not _stop_for_shutdown(tp, results_dir, phase):
                held_training[gpu] = tp
                continue
            tp.close_log()
            try:
                write_interruption_receipt(
                    tp, results_dir, cfg, reason="orze_stop",
                    terminating_signal="SIGTERM", return_code=tp.process.poll())
            except Exception as exc:
                logger.warning("Could not persist interruption receipt for %s: %s",
                               tp.idea_id, type(exc).__name__)
        for gpu, ep in active_evals.items():
            logger.info("Killing eval %s on GPU %s (PID %d)",
                        ep.idea_id, gpu, ep.process.pid)
            if not _stop_for_shutdown(ep, results_dir, "evaluation"):
                held_evals[gpu] = ep
                continue
            ep.close_log()
            close_interrupted_evaluation(ep)
        for role_name, rp in active_roles.items():
            logger.info("Killing role '%s' (PID %d)",
                        role_name, rp.process.pid)
            reaped = terminate_role_process(rp, f"role {role_name}")
            rp.close_log()
            if reaped:
                _fs_unlock(rp.lock_dir)

    else:
        # Training has a durable process/claim recovery path and may safely
        # detach. Evaluators do not: detaching left an unowned allocation that
        # could never write its terminal receipt. Interrupt evals cleanly and
        # return their stage to PENDING so the next controller can retry.
        for gpu, tp in active.items():
            if getattr(tp, "_termination_unconfirmed", False) is True:
                held_training[gpu] = tp
                continue
            logger.info("Detaching training %s on GPU %s (PID %d) "
                        "-- will finish in background",
                        tp.idea_id, gpu, tp.process.pid)
            tp.close_log()
        for gpu, ep in active_evals.items():
            logger.info("Interrupting eval %s on GPU %s (PID %d) "
                        "-- next controller will retry",
                        ep.idea_id, gpu, ep.process.pid)
            if not _stop_for_shutdown(ep, results_dir, "evaluation"):
                held_evals[gpu] = ep
                continue
            ep.close_log()
            close_interrupted_evaluation(ep)
        for role_name, rp in active_roles.items():
            logger.info("Terminating role '%s' (PID %d)...",
                        role_name, rp.process.pid)
            reaped = terminate_role_process(rp, f"role {role_name}")
            rp.close_log()
            if reaped:
                _fs_unlock(rp.lock_dir)

    # 3. Write shutdown sentinel (tells the watchdog not to restart us)
    if not managed:
        sentinel = results_dir / ".orze_shutdown"
        try:
            sentinel.write_text(
                f"pid={os.getpid()} iteration={iteration} "
                f"time={datetime.datetime.now().isoformat()}\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    # 4. Save state for restart recovery
    if not managed:
        save_state(results_dir, state_dict)

    # 5. Notify (best effort)
    if not managed:
        try:
            notify("shutdown", {
                "host": hostname,
                "message": (f"Graceful shutdown after iteration "
                            f"{iteration}"),
            }, cfg)
        except Exception:
            pass

    # 6. Close IdeaLake (flushes WAL on shared filesystems)
    if lake:
        try:
            lake.close()
        except Exception:
            pass

    # 7. Clean up PID file
    if managed:
        try:
            if pid_file_path is not None:
                pid_file_path.unlink(missing_ok=True)
        except OSError:
            pass
    else:
        remove_pid_file(results_dir / f".orze.pid.{hostname}", results_dir)

    logger.info(
        "Shutdown complete%s after iteration %d. %d training and %d eval "
        "process(es) handled.",
        " (managed; campaign state unchanged)" if managed else "; state saved",
        iteration, training_count if not kill_all else 0,
        eval_count if not kill_all else 0)

    # Detached children must no longer be visible to atexit_cleanup, whose
    # last-resort contract is to kill every process still tracked here.
    active_roles.clear()
    active.clear()
    active.update(held_training)
    active_evals.clear()
    active_evals.update(held_evals)
