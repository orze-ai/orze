def _notify_finished(self, idea_id, gpu, cfg, primary, row_lookup,
                     rank_lookup, leaderboard, view_lbs, ideas,
                     save_config_hash_fn):
    if (not isinstance(idea_id, str) or idea_id in ("", ".", "..")
            or Path(idea_id).parts != (idea_id,)):
        return
    row = row_lookup.get(idea_id, {})
    m = row.get("metrics")
    if m is None:
        m_path = self.results_dir / idea_id / "metrics.json"
        if m_path.is_symlink() or m_path.parent.is_symlink():
            return
        try:
            m = json.loads(m_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return
    if not isinstance(m, dict):
        return

    status = m.get("status", "UNKNOWN")
    title = ideas.get(idea_id, {}).get("title", idea_id)

    if status == "COMPLETED":
        if not row or row.get("primary_val") is None:
            return
        self._notify_completed(idea_id, title, m, cfg, primary,
                               row_lookup, rank_lookup,
                               leaderboard, view_lbs)
    elif status == "FAILED":
        error_msg = m.get("error")
        if not isinstance(error_msg, str):
            error_msg = "unclassified failure"
        # Suppress notifications for config/argparse errors (exit code 2)
        # and fast crashes (<10s, typically import errors). These are
        # research-agent-generated junk, not worth spamming Telegram.
        is_config_error = "code 2" in error_msg or "code 1" in error_msg
        training_time = m.get("training_time")
        fast_crash = (
            isinstance(training_time, (int, float))
            and not isinstance(training_time, bool)
            and 0 <= training_time < 10
        )
        if is_config_error and fast_crash:
            logger.info("Suppressed notification for %s: config error (%s)",
                        idea_id, error_msg)
        else:
            notify("failed", {"idea_id": idea_id, "title": title,
                              "error": error_msg,
                              "evidence_scope": "artifact_observed_unverified",
                              "leaderboard": leaderboard,
                              "view_leaderboards": view_lbs}, cfg)

    if status == "COMPLETED":
        from orze.reporting.notification_evidence import refresh_metric_snapshot
        refresh_metric_snapshot(self.lake, row)
        try:
            # Config dedup hash MUST be over the same canonical key-set
            # that ingest checks (engine/phases.py: _config_override_hash
            # over idea["config"], i.e. user OVERRIDES only). Previously
            # this hashed the FULL resolved_config.yaml — a different,
            # much larger key-set — so the stored hash never matched the
            # ingest-time override hash and dedup NEVER fired. We now hash
            # the idea's overrides so the two sides agree.
            overrides = ideas.get(idea_id, {}).get("config")
            if overrides is None:
                # Fallback for archived/stub idea records that don't carry
                # the config in memory: recover overrides from the idea's
                # resolved_config.yaml minus the base config, so we still
                # register the SAME override-keyed hash (no silent skip).
                overrides = self._recover_overrides(idea_id, cfg)
            if overrides is None:
                logger.warning(
                    "Config dedup hash NOT stored for %s: could not "
                    "resolve config overrides (no in-memory config and "
                    "no resolved_config.yaml).", idea_id)
            else:
                save_config_hash_fn(idea_id, overrides)
        except Exception as exc:
            logger.debug("Config hash save failed for %s: %s",
                         idea_id, exc)
