def atexit_cleanup(active: dict, active_evals: dict,
                   active_roles: dict,
                   results_dir: Path | None = None) -> None:
    """Last-resort cleanup with allocation closure when context is available.

    This path handles unhandled exceptions and a second shutdown signal, where
    ``graceful_shutdown`` may never run. A child that was already given a GPU
    must still receive a framework-owned terminal receipt; otherwise campaign
    accounting cannot distinguish an interrupted allocation from missing work.
    """
    def close_compute(process, phase: str, reason_code: str) -> None:
        if results_dir is None:
            return
        try:
            from orze.engine.accounting import record_compute_terminal
            record_compute_terminal(
                process, Path(results_dir) / process.idea_id, "interrupted",
                reason_code, phase=phase,
                return_code=process.process.poll(),
            )
        except Exception as exc:
            logger.warning(
                "Could not persist atexit %s receipt for %s: %s",
                phase, process.idea_id, type(exc).__name__,
            )

    for gpu, tp in list(active.items()):
        phase = "posthoc" if getattr(tp, "is_posthoc", False) else "training"
        if not _stop_for_shutdown(tp, results_dir, phase):
            continue
        close_compute(tp, phase, "training_atexit_cleanup")
        tp.close_log()
    for gpu, ep in list(active_evals.items()):
        if not _stop_for_shutdown(ep, results_dir, "evaluation"):
            continue
        close_compute(ep, "evaluation", "evaluation_atexit_cleanup")
        ep.close_log()
    for role_name, rp in list(active_roles.items()):
        terminate_role_process(rp, f"atexit role {role_name}", timeout=2)
        rp.close_log()
