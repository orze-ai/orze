def _write_eval_failure_marker(results_dir: Path, idea_id: str,
                               eval_output: str, reason: str, lake=None,
                               *, effect_lease=None) -> None:
    """Safety net: write failure marker if eval process died without one.

    The fallback never overwrites existing output, training metrics, or an
    unsafe path. Lifecycle failure is independent of whether a marker can be
    written. The evaluator remains responsible for its domain report.

    Args:
        lake: IdeaLake instance for FSM transition recording (optional)
    """
    require_no_unconfirmed_stop(results_dir / idea_id)
    from orze.engine.attempt_effect_lock import AttemptEffectBusy, require_effect_lease
    if effect_lease is not None:
        require_effect_lease(effect_lease, results_dir / idea_id)
    else:
        from orze.engine.attempt_effect_receipts import require_closed_effects
        from orze.engine.execution_catalog import declared_catalog
        require_closed_effects(results_dir / idea_id)
        if declared_catalog(results_dir / idea_id) is not None:
            raise AttemptEffectBusy("evaluation_owned_publication_required")
        if lake is not None:
            from orze.core.execution_attempts import current_attempt
            if current_attempt(lake.conn, idea_id, "evaluation") is not None:
                raise AttemptEffectBusy("evaluation_owned_publication_required")
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
