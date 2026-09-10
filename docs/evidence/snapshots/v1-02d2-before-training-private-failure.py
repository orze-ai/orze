def _write_failure(idea_dir: Path, reason: str, lake=None, idea_id=None, cfg=None):
    """Write a failure metrics.json atomically and record FSM transition."""
    require_no_unconfirmed_stop(idea_dir)
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
