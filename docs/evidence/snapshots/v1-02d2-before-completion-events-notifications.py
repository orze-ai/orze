def process(self, finished: list, completed_rows: list, ideas: dict,
            counts: dict, active_count: int,
            save_config_hash_fn, build_machine_status_fn):
    """Consume a finished batch independently of delivery settings.

    Reconcile current evidence even without a finished batch. Selection
    changes alone are not objective improvements. Repeated batches and
    result revisions still require a separate observation identity model;
    this boundary does not promise exactly-once observations or delivery.
    """
    try:
        cfg = self.cfg
        ncfg = cfg.get("notifications") or {}
        if finished:
            logger.info("Processing completion evidence for %d finished items",
                        len(finished))
        from orze.reporting.notification_evidence import (
            qualified_notification_rows,
        )
        candidates = list(completed_rows)
        candidates.extend({"id": idea_id,
                           "title": ideas.get(idea_id, {}).get("title", idea_id)}
                          for idea_id, _ in finished)
        if self._best_idea_id:
            candidates.append({"id": self._best_idea_id})
        completed_rows = qualified_notification_rows(
            self.results_dir, cfg, candidates, self.lake)
        primary = (cfg.get("report") or {}).get("primary_metric")

        # Build rank lookup and top-10 leaderboard
        rank_lookup, leaderboard = {}, []
        for rank, r in enumerate(completed_rows, 1):
            rank_lookup[r["id"]] = rank
            if rank <= 10:
                leaderboard.append({"id": r["id"],
                                    "title": r.get("title", r["id"]),
                                    "value": r.get("primary_val")})

        view_lbs = self._build_view_leaderboards(cfg, completed_rows)
        row_lookup = {r["id"]: r for r in completed_rows}

        for idea_id, gpu in finished:
            try:
                self._notify_finished(
                    idea_id, gpu, cfg, primary, row_lookup, rank_lookup,
                    leaderboard, view_lbs, ideas, save_config_hash_fn)
            except Exception as exc:
                logger.warning("Completion diagnostic skipped for %s: %s",
                               idea_id, type(exc).__name__)

        # New best detection + plateau tracking
        new_best = self._check_new_best(
            completed_rows, primary, leaderboard, view_lbs, cfg)
        n_completed = len({idea_id for idea_id, _ in finished
                           if idea_id in row_lookup})
        if new_best:
            self._completions_since_best = 0
            self._plateau_notified = False
        else:
            self._completions_since_best += n_completed

        if finished and ncfg.get("enabled", False):
            self._check_plateau(completed_rows, cfg)
            self._periodic_report(ncfg, cfg, primary, counts, active_count,
                                  leaderboard, view_lbs, build_machine_status_fn)
    except Exception as e:
        logger.warning("Notification processing error: %s", e)
