def _launch_evals(self, finished, eval_finished, ideas):
    """Dispatch each idea at most once per tick; None is not a success.

    A configured evaluator's no-process result is delivered only after its
    real evaluation/global terminal states agree. Other outcomes remain
    pending for a later tick. Only training-only projects may skip eval.
    """
    cfg = self.cfg
    managed_idea = cfg.get("_managed_idea_id")
    if managed_idea:
        finished = [item for item in finished if item[0] == managed_idea]
        eval_finished = [
            item for item in eval_finished if item[0] == managed_idea]
    max_evals = cfg.get("max_concurrent_evals",
                         len(self.gpu_ids))
    delivered_ids = {idea_id for idea_id, _ in eval_finished}
    attempted_ids = set(delivered_ids)
    attempted_ids.update(ep.idea_id for ep in self.active_evals.values())

    def defer(idea_id, gpu):
        if idea_id not in {iid for iid, _ in self.pending_evals}:
            self.pending_evals.append((idea_id, gpu))

    # Explicit retry admissions are durable evaluation work, not QUEUED
    # training ideas. Recover them even with an empty inbox after restart.
    if self.lake is not None and cfg.get("eval_script") and self.gpu_ids:
        from orze.engine.evaluation_retry import pending_evaluation_retries
        for idea_id in pending_evaluation_retries(self.lake):
            if not managed_idea or idea_id == managed_idea:
                defer(idea_id, self.gpu_ids[0])

    def deliver(idea_id, gpu):
        require_no_unconfirmed_stop(self.results_dir / idea_id)
        if idea_id not in delivered_ids:
            eval_finished.append((idea_id, gpu))
            delivered_ids.add(idea_id)

    def finish_without_process(idea_id, gpu):
        # Neither missing eval configuration nor existing output can
        # resolve a persisted, unconfirmed execution stop.
        require_no_unconfirmed_stop(self.results_dir / idea_id)
        if not cfg.get("eval_script"):
            if self.lake:
                state = self.lake.get_fsm_state(idea_id)
                if state == "IN_PROGRESS":
                    if not self.lake.record_state_transition(
                            idea_id, from_state="IN_PROGRESS",
                            to_state="COMPLETE",
                            reason="training_completed_no_eval",
                            host=socket.gethostname(), pid=os.getpid(),
                            sop_type=cfg.get("sop", "training")):
                        return False
                elif state != "COMPLETE":
                    return False
        elif self.lake:
            state = self.lake.get_fsm_state(idea_id)
            evaluation = self.lake.get_stage_state(idea_id, "evaluation")
            if (state, evaluation) not in (
                    ("COMPLETE", "COMPLETE"), ("FAILED", "FAILED")):
                return False
        else:
            # A None return cannot establish an evaluated lifecycle when
            # no authoritative state is available to this controller.
            return False
        deliver(idea_id, gpu)
        return True

    for idea_id, gpu in finished:
        if idea_id in attempted_ids:
            continue
        metrics_path = self.results_dir / idea_id / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(
                    metrics_path.read_text(encoding="utf-8"))
                if (isinstance(metrics, dict)
                        and metrics.get("status") == "COMPLETED"):
                    if not cfg.get("eval_script"):
                        attempted_ids.add(idea_id)
                        if not finish_without_process(idea_id, gpu):
                            defer(idea_id, gpu)
                        continue
                    if len(self.active_evals) < max_evals:
                        if hasattr(self, 'slot_mgr'):
                            eval_busy = (self.slot_mgr.gpu_ids_in_use()
                                         | set(self.active_evals.keys()))
                        else:
                            eval_busy = (set(self.active.keys())
                                         | set(self.active_evals.keys()))
                        free_for_eval = [g for g in self.gpu_ids
                                         if g not in eval_busy]
                        if free_for_eval:
                            use_gpu = free_for_eval[0]
                            attempted_ids.add(idea_id)
                            ep = launch_eval(
                                idea_id, use_gpu,
                                self.results_dir, cfg, lake=self.lake)
                            if ep is not None:
                                self.active_evals[use_gpu] = ep
                            elif not finish_without_process(idea_id, use_gpu):
                                defer(idea_id, use_gpu)
                        else:
                            defer(idea_id, gpu)
                            logger.info(
                                "Eval deferred for %s (no free GPU)",
                                idea_id)
                    else:
                        defer(idea_id, gpu)
                        logger.info(
                            "Eval deferred for %s (limit %d)",
                            idea_id, max_evals)
                else:
                    deliver(idea_id, gpu)
            except TerminationUnconfirmed:
                raise
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                if idea_id in attempted_ids:
                    # Dispatch/lifecycle I/O failures are not evidence that
                    # evaluation completed. Keep the attempted work.
                    defer(idea_id, gpu)
                else:
                    deliver(idea_id, gpu)
        else:
            deliver(idea_id, gpu)

    # 6a. Launch pending evals from previous iterations
    still_pending = []
    active_eval_ids = {ep.idea_id for ep in self.active_evals.values()}
    for p_idea, p_gpu in dict(self.pending_evals).items():
        if p_idea in delivered_ids or p_idea in active_eval_ids:
            continue
        if managed_idea and p_idea != managed_idea:
            still_pending.append((p_idea, p_gpu))
            continue
        if p_idea in attempted_ids:
            still_pending.append((p_idea, p_gpu))
            continue
        if not cfg.get("eval_script"):
            attempted_ids.add(p_idea)
            eligible, _ = is_training_complete_for_downstream(
                self.results_dir / p_idea, cfg)
            if not eligible or not finish_without_process(p_idea, p_gpu):
                still_pending.append((p_idea, p_gpu))
            continue
        if len(self.active_evals) >= max_evals:
            still_pending.append((p_idea, p_gpu))
            continue
        if hasattr(self, 'slot_mgr'):
            eval_busy = (self.slot_mgr.gpu_ids_in_use()
                         | set(self.active_evals.keys()))
        else:
            eval_busy = (set(self.active.keys())
                         | set(self.active_evals.keys()))
        free_for_eval = [g for g in self.gpu_ids
                         if g not in eval_busy]
        if free_for_eval:
            use_gpu = free_for_eval[0]
            attempted_ids.add(p_idea)
            ep = launch_eval(
                p_idea, use_gpu, self.results_dir, cfg, lake=self.lake)
            if ep is not None:
                self.active_evals[use_gpu] = ep
            elif not finish_without_process(p_idea, use_gpu):
                still_pending.append((p_idea, use_gpu))
        else:
            still_pending.append((p_idea, p_gpu))
    self.pending_evals = still_pending

    # 6b. Backlog scan: fill remaining eval slots with
    #     completed-but-unevaluated ideas (newest first)
    backlog = []
    if cfg.get("eval_script") and len(self.active_evals) < max_evals:
        eval_output = cfg.get("eval_output") or "eval_report.json"
        pending_ids = {pi for pi, _ in self.pending_evals}
        active_eval_ids = {ep.idea_id
                           for ep in self.active_evals.values()}
        ckpt_dir = _get_checkpoint_dir(cfg)
        known_ideas = set(ideas.keys())
        backlog = []
        for d in self.results_dir.iterdir():
            if not d.is_dir() or not d.name.startswith("idea-"):
                continue
            iid = d.name
            if managed_idea and iid != managed_idea:
                continue
            if (iid in pending_ids or iid in active_eval_ids
                    or iid in attempted_ids or iid in delivered_ids):
                continue
            # Only evaluate ideas for which this controller has a config.
            if iid not in known_ideas:
                continue
            mpath = d / "metrics.json"
            rpath = d / eval_output
            if self.lake:
                # An existing output may still need real reconciliation;
                # metrics.json can be training output awaiting evaluation.
                needs_evaluation = self.lake.get_fsm_state(iid) == "IN_PROGRESS"
            else:
                needs_evaluation = (
                    not rpath.exists()
                    or Path(eval_output) == Path("metrics.json"))
            if mpath.exists() and needs_evaluation:
                # Skip ideas without checkpoints
                ckpt_name = cfg.get("eval_checkpoint", "best_model.pt")
                has_ckpt = (d / ckpt_name).exists()
                if ckpt_dir and not (
                        ckpt_dir / iid / "best.pt").exists() and not has_ckpt:
                    continue
                eligible, _ = is_training_complete_for_downstream(d, cfg)
                if eligible:
                    try:
                        num = int(iid.split("-", 1)[1])
                    except (IndexError, ValueError):
                        num = 0
                    backlog.append((num, iid))
        if backlog:
            backlog.sort(reverse=True)
            if hasattr(self, 'slot_mgr'):
                eval_busy = (self.slot_mgr.gpu_ids_in_use()
                             | set(self.active_evals.keys()))
            else:
                eval_busy = (set(self.active.keys())
                             | set(self.active_evals.keys()))
            mem_thresh = cfg.get("gpu_mem_threshold", 2000)
            free_for_eval = [
                g for g in self.gpu_ids
                if g not in eval_busy
                and (get_gpu_memory_used(g) or 0) <= mem_thresh
            ]
            launched_backlog = 0
            for _, iid in backlog:
                if (len(self.active_evals) >= max_evals
                        or not free_for_eval):
                    break
                # Skip if an orphaned eval is already running
                if _eval_already_running(iid, cfg):
                    continue
                use_gpu = free_for_eval.pop(0)
                attempted_ids.add(iid)
                ep = launch_eval(
                    iid, use_gpu, self.results_dir, cfg, lake=self.lake)
                if ep is not None:
                    self.active_evals[use_gpu] = ep
                    launched_backlog += 1
                elif not finish_without_process(iid, use_gpu):
                    defer(iid, use_gpu)
            if launched_backlog:
                logger.info(
                    "Launched %d backlog evals (%d remaining)",
                    launched_backlog,
                    len(backlog) - launched_backlog)

    return eval_finished, backlog
