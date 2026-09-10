def finish_evaluation(lake, ep, idea_dir, cfg, ret, *, forced=None, not_started=False):
    """Native terminal writer; no legacy shared-output fallback or inference."""
    import socket
    from orze.core.execution_attempts import (
        AttemptAuthorityError, StaleAttempt, finish_attempt, mark_running)
    from orze.core.research_artifacts import register_artifacts
    from orze.core.research_observations import register_observations, observations_for_attempt
    from orze.engine.accounting import record_compute_terminal
    from orze.engine.execution_authority import execution_transaction, lifecycle_fence
    from orze.engine.native_evaluation import _owned, _ref, _verify_compute
    from orze.engine.completion_events import CompletionEvent

    row = _owned(lake, ep, states=("LAUNCHING", "RUNNING"))
    io = validate_bound_execution(row, idea_dir, cfg)
    ref = _ref(ep)
    prepared = None
    outcome, reason, detail = "failed", "observation_process_failed", f"Exit code {ret}"
    if forced is not None:
        outcome, reason, detail = forced
    elif ret == 0:
        try:
            prepared = prepare_observations(lake, ref, idea_dir, cfg, row)
        except (OSError, ValueError, UnicodeError, RecursionError) as exc:
            reason, detail = "observation_output_invalid", type(exc).__name__
        else:
            outcome, reason, detail = "completed", "observation_envelope_recorded", ""
    success = prepared is not None
    try:
        with execution_transaction(lake, idea_dir) as tx:
            row = _owned(lake, ep, states=("LAUNCHING", "RUNNING"))
            validate_bound_execution(row, idea_dir, cfg)
            records, observations = [], []
            if prepared is not None:
                records, observations = verify_observations(prepared, ref, idea_dir, cfg, row)
            plan = {"operation": "observation_evaluation_terminal", "outcome": outcome,
                    "return_code": ret, "reason_code": reason,
                    "artifact_ids": [record["artifact_id"] for record in records],
                    "observation_ids": [record["observation_id"] for record in observations]}
            digest = tx.prepare(ref, plan)
            if row["state"] == "LAUNCHING" and not not_started:
                mark_running(tx.conn, ref)
            artifact_ids = observation_ids = ()
            if prepared is not None:
                verify_observations(prepared, ref, idea_dir, cfg, row)
                artifact_ids = register_artifacts(tx.conn, ref, records)
                observation_ids = register_observations(tx.conn, ref, observations)
            else:
                # Diagnostic publication is attempt-local and never a worker
                # observation. No metrics.json/eval_output fallback is called.
                diagnostic = Path(io["attempt_dir"]) / "failure.json"
                if not diagnostic.exists():
                    _write_once(diagnostic, _encode({"schema": 1, "status": "FAILED",
                        "reason_code": reason, "detail": str(detail)[:500]}))
            if not not_started:
                receipt = record_compute_terminal(ep, idea_dir, outcome, reason,
                    phase="evaluation", return_code=ret)
                _verify_compute(idea_dir, receipt, process=ep, phase="evaluation", event="terminal",
                    outcome=outcome, reason_code=reason, return_code=ret,
                    require_start=row["state"] == "RUNNING")
            if not lake._record_state_transition_in_tx(ep.idea_id, "IN_PROGRESS",
                    "COMPLETE" if success else "FAILED", reason=reason,
                    host=socket.gethostname(), pid=os.getpid(), sop_type="training"):
                raise AttemptAuthorityError("observation_terminal_lifecycle_rejected")
            terminal = {"outcome": outcome, "reason_code": reason, "return_code": ret,
                "effect_receipt_sha256": digest, "artifact_ids": list(artifact_ids),
                "observation_ids": list(observation_ids),
                "lifecycle": lifecycle_fence(lake, ep.idea_id, "evaluation")}
            if finish_attempt(tx.conn, ref, terminal, not_started=not_started) != "committed":
                raise AttemptAuthorityError("observation_terminal_not_new")
            if (not _same(artifacts_for_attempt(tx.conn, ref),
                          sorted(records, key=lambda record: record["logical_name"]))
                    or not _same(observations_for_attempt(tx.conn, ref),
                                 sorted(observations, key=lambda record: record["name"]))):
                raise AttemptAuthorityError("observation_terminal_records_changed")
            if prepared is not None:
                verify_observations(prepared, ref, idea_dir, cfg, row)
    except StaleAttempt:
        return None
    return CompletionEvent(ep.idea_id, ep.gpu, ref)
