def reconcile_running_dead_pids(cfg: dict) -> int:
    """F7: For every status='running' idea, verify a python training
    process exists on this host with ``--idea-id <id>`` in its cmdline.
    If the process is gone but metrics.json shows COMPLETED, mark as
    'completed'. Otherwise mark 'failed' with reason 'orphaned_pid'.

    Multi-host safety: only acts on rows whose claim.json says they
    belong to THIS host. Rows owned by another host are left alone.

    Returns the number of rows reconciled.
    """
    import json as _json
    import socket as _socket
    hostname = _socket.gethostname()
    results_dir = Path(cfg.get("results_dir", "orze_results"))
    lake_path = Path(cfg.get("idea_lake_db") or results_dir / "idea_lake.db")
    if not lake_path.exists():
        return 0

    alive_ideas = _running_idea_pids()
    # Grace period: skip reconcile for rows whose claim.json or
    # idea-dir activity is younger than this. Covers the launch
    # race window (claim.json written → subprocess spawn) and the
    # post-completion metrics flush window. Cycle-092 cross-domain row
    # 65: standard distributed-task-queue grace is 120s.
    grace_seconds = 180
    now_ts = time.time()
    evaluation_required = bool(cfg.get("eval_script"))

    n_completed = 0
    n_orphaned = 0
    n_requeued = 0
    n_skipped_grace = 0
    n_warned = 0
    lake = None
    try:
        from orze.idea_lake import IdeaLake
        lake = IdeaLake(str(lake_path))
        rows = lake.conn.execute(
            "SELECT idea_id, eval_metrics FROM ideas "
            "WHERE status = 'running'").fetchall()
        for idea_id, em_raw in rows:
            idea_dir = results_dir / idea_id
            claim_path = idea_dir / "claim.json"

            def completed_on_disk() -> bool:
                metrics_path = idea_dir / "metrics.json"
                if not metrics_path.is_file():
                    return False
                try:
                    metrics = _json.loads(
                        metrics_path.read_text(encoding="utf-8"))
                except (_json.JSONDecodeError, OSError, UnicodeDecodeError):
                    return False
                return (isinstance(metrics, dict)
                        and metrics.get("status") == "COMPLETED")

            def reconcile_completed_training() -> bool:
                if not evaluation_required:
                    return lake.set_status(idea_id, "completed")
                current = lake.get_fsm_state(idea_id)
                if current == "CLAIMED":
                    if not lake.record_state_transition(
                            idea_id, "CLAIMED", "IN_PROGRESS",
                            reason="reconcile_training_process_started",
                            host=hostname, pid=os.getpid(),
                            sop_type="training"):
                        return False
                    current = "IN_PROGRESS"
                if current != "IN_PROGRESS":
                    return False
                stage = lake.get_stage_state(idea_id, "training")
                training_ok = stage == "COMPLETE"
                if stage in ("NOT_STARTED", "PENDING", "IN_PROGRESS"):
                    training_ok = lake.record_stage_transition(
                        idea_id,
                        stage="training",
                        from_state=stage,
                        to_state="COMPLETE",
                        reason=(
                            "reconcile_training_completed_"
                            "evaluation_pending"
                        ),
                        host=hostname,
                        pid=os.getpid(),
                    )
                if not training_ok:
                    return False
                evaluation_stage = lake.get_stage_state(
                    idea_id, "evaluation")
                if evaluation_stage in ("NOT_STARTED", "IN_PROGRESS"):
                    return lake.record_stage_transition(
                        idea_id,
                        stage="evaluation",
                        from_state=evaluation_stage,
                        to_state="PENDING",
                        reason=(
                            "reconcile_evaluation_pending"
                            if evaluation_stage == "NOT_STARTED" else
                            "reconcile_interrupted_evaluation_pending"
                        ),
                        host=hostname,
                        pid=os.getpid(),
                    )
                return evaluation_stage == "PENDING"

            # ---- Ownership check (multi-host safety) ----
            if claim_path.exists():
                try:
                    claim = _json.loads(claim_path.read_text(encoding="utf-8"))
                    if claim.get("claimed_by") != hostname:
                        continue
                except (_json.JSONDecodeError, OSError):
                    # Ownership cannot be proved. Never mutate another host's
                    # possible work from a corrupt claim.
                    continue
            else:
                # Completion may flush immediately before claim cleanup. It is
                # the only terminal filesystem evidence accepted here.
                if completed_on_disk():
                    if reconcile_completed_training():
                        n_completed += 1
                elif lake.set_status(idea_id, "queued"):
                    n_requeued += 1
                continue

            # ---- Live process? ----
            if idea_id in alive_ideas:
                continue

            # ---- Grace window: ----
            # If the idea_dir was modified within `grace_seconds`, the
            # subprocess may be in the launch race window or the
            # completion-flush window. Skip this iteration.
            try:
                latest = claim_path.stat().st_mtime
                # Also consider real trainer progress and terminal metrics.
                # A submission alone is not lifecycle evidence.
                for fname in ("metrics.json", "train_output.log"):
                    fp = idea_dir / fname
                    if fp.exists():
                        latest = max(latest, fp.stat().st_mtime)
                if (now_ts - latest) < grace_seconds:
                    n_skipped_grace += 1
                    continue
            except OSError:
                pass

            # ---- Completed-on-disk check ----
            if completed_on_disk():
                if reconcile_completed_training():
                    n_completed += 1
                continue

            try:
                em = _json.loads(em_raw) if em_raw else {}
                if not isinstance(em, dict):
                    em = {}
            except (ValueError, TypeError):
                em = {}
            # Require 2 consecutive liveness misses before orphan-marking
            # (Phi-Accrual anti-flap). Persist the first miss immediately;
            # the previous implementation forgot to commit warning-only
            # cycles, so an orphan could remain "running" forever.
            miss_count = em.get("liveness_misses", 0) + 1
            em["liveness_misses"] = miss_count
            if miss_count >= 2:
                em["failure_reason"] = "orphaned_pid"
            lake.conn.execute(
                "UPDATE ideas SET eval_metrics = ? WHERE idea_id = ?",
                (_json.dumps(em), idea_id))
            lake.conn.commit()
            if miss_count >= 2:
                if lake.set_status(idea_id, "failed"):
                    n_orphaned += 1
            else:
                n_warned += 1
        if n_completed:
            target = (
                "evaluation pending" if evaluation_required else "completed")
            logger.info(
                "Reconciled %d 'running' rows (training completed on disk) "
                "-> %s", n_completed, target)
        if n_requeued:
            logger.info(
                "Reconciled %d 'running' rows (no claim.json) -> queued",
                n_requeued)
        if n_skipped_grace:
            logger.debug(
                "Reconcile: %d 'running' rows skipped (within %ds grace)",
                n_skipped_grace, grace_seconds)
        if n_warned:
            logger.info(
                "Reconcile: %d 'running' rows — 1st liveness miss (need 2 to orphan)",
                n_warned)
        if n_orphaned:
            logger.info(
                "Reconciled %d orphaned 'running' rows (dead PID, 2+ misses) -> failed",
                n_orphaned)
    except Exception as e:
        logger.warning("Failed to reconcile dead-PID rows: %s", e)
    finally:
        if lake is not None:
            lake.close()
    return n_completed + n_orphaned


