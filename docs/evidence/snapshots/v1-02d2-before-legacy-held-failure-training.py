def _write_failure(idea_dir: Path, reason: str, lake=None, idea_id=None, cfg=None,
                   *, effect_lease=None):
    """Write owned diagnostics; the native caller owns lifecycle acceptance."""
    require_no_unconfirmed_stop(idea_dir)
    from orze.engine.attempt_effect_lock import AttemptEffectBusy, require_effect_lease
    if effect_lease is not None:
        require_effect_lease(effect_lease, idea_dir)
        if lake is not None:
            raise AttemptEffectBusy("training_owned_writer_cannot_commit_lifecycle")
    else:
        from orze.engine.attempt_effect_receipts import require_closed_effects
        from orze.engine.execution_catalog import declared_catalog
        require_closed_effects(idea_dir)
        if declared_catalog(idea_dir) is not None:
            raise AttemptEffectBusy("training_owned_publication_required")
        if lake is not None:
            from orze.core.execution_attempts import current_attempt
            if current_attempt(lake.conn, idea_id or idea_dir.name, "training") is not None:
                raise AttemptEffectBusy("training_owned_publication_required")
    metrics = {
        "status": "FAILED",
        "error": reason,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    atomic_write(idea_dir / "metrics.json", json.dumps(metrics, indent=2))

    # Record FSM transition: current_state → FAILED (v4.5: determine actual state)
    if lake and idea_id:
        try:
            # Query the actual current state from the FSM
            current_state = lake.get_fsm_state(idea_id)
            if current_state:
                lake.record_state_transition(
                    idea_id,
                    from_state=current_state,
                    to_state="FAILED",
                    reason=reason,
                    host=socket.gethostname(),
                    pid=os.getpid(),
                    sop_type=(cfg or {}).get("sop", "training"),
                )
            else:
                # No state found, default to IN_PROGRESS for backward compatibility
                lake.record_state_transition(
                    idea_id,
                    from_state="IN_PROGRESS",
                    to_state="FAILED",
                    reason=reason,
                    host=socket.gethostname(),
                    pid=os.getpid(),
                    sop_type=(cfg or {}).get("sop", "training"),
                )
        except Exception as e:
            logger.warning("FSM transition failed (non-blocking): %s", e)

    return metrics
