def _run_leased(self):
    cfg = self.cfg
    managed_idea = cfg.get("_managed_idea_id")

    # Log pro status
    from orze.extensions import has_pro, pro_version
    if has_pro() and _run_all_roles_impl is not None:
        logger.info("orze-pro %s detected — autopilot features enabled", pro_version())
    elif has_pro() and _run_all_roles_impl is None:
        logger.error(
            "orze-pro licensed but role_runner failed to import — "
            "version mismatch? Try: pip install --upgrade orze orze-pro"
        )
    elif _role_mod is not None:
        logger.info("Using built-in agent modules (install orze-pro to upgrade)")
    else:
        roles = cfg.get("roles", {})
        if roles:
            logger.warning(
                "Roles configured (%s) but no agent support available. "
                "Install orze-pro for autonomous research agents.",
                ", ".join(roles.keys()))

    if managed_idea:
        # A one-idea run must not perform daemon-wide recovery, stale-lock
        # cleanup, symlink normalization, or upgrade cleanup.
        from orze.engine.health import HealthMonitor
        self._health_monitor = HealthMonitor(self.results_dir)
    else:
        self._startup_checks()
        self._kill_orphans()
    # Clear any stale shutdown sentinels from a previous run
    if not managed_idea:
        for sentinel_name in [".orze_shutdown", ".orze_stop_all"]:
            sentinel = self.results_dir / sentinel_name
            if sentinel.exists():
                sentinel.unlink(missing_ok=True)
    # Clear upgrade sentinel if we're already at the target version
    upgrade_sentinel = self.results_dir / ".orze_upgrade"
    if upgrade_sentinel.exists() and not managed_idea:
        try:
            target = upgrade_sentinel.read_text(encoding="utf-8").strip()
            def _ver(s):
                try:
                    return tuple(int(x) for x in s.split(".")[:3])
                except (ValueError, AttributeError):
                    return (0,)
            if _ver(__version__) >= _ver(target):
                upgrade_sentinel.unlink(missing_ok=True)
        except Exception:
            pass
    logger.info("Starting orze v%s on GPUs %s (PID %d)",
                 __version__, self.gpu_ids, os.getpid())
    logger.info("Ideas: %s | Results: %s | Timeout: %ds | Poll: %ds",
                 cfg["ideas_file"], cfg["results_dir"],
                 cfg["timeout"], cfg["poll"])
    for rname, rcfg in (cfg.get("roles") or {}).items():
        if not isinstance(rcfg, dict):
            continue
        rmode = rcfg.get("mode", "script")
        if rmode == "claude":
            skills = rcfg.get("skills") or []
            rtarget = f"{len(skills)} skill(s)" if skills else None
        elif rmode == "research":
            skills = rcfg.get("skills") or []
            rtarget = f"{rcfg.get('backend', '?')} + {len(skills)} skill(s)"
        else:
            rtarget = rcfg.get("script")
        if rtarget:
            logger.info("Role '%s' [%s]: %s (cooldown: %ds, timeout: %ds)",
                        rname, rmode, rtarget,
                        rcfg.get("cooldown", 300),
                        rcfg.get("timeout", 600))

    # Lifecycle notification: started
    n_roles = len([r for r in (cfg.get("roles") or {}).values()
                   if isinstance(r, dict)])
    if not managed_idea:
        notify("started", {
            "host": socket.gethostname(),
            "message": (f"v{__version__} | {len(self.gpu_ids)} GPUs | "
                        f"{n_roles} roles | pid {os.getpid()}"),
        }, cfg)

    # Boot-time delivery canary. Closes the meta-audit blind spot
    # that notification delivery was trust-based — a misconfigured
    # webhook URL or revoked Telegram token would silently swallow
    # every alert. Runs once on leader boot; per-channel result is
    # stashed on self for write_status to surface under
    # ``notification_health``. When ``notifications.startup_canary``
    # is true (default true) any delivery failure exits the daemon
    # nonzero so systemd / loop-restart picks it up.
    ncfg = cfg.get("notifications") or {}
    self.notification_health = (
        {} if managed_idea else startup_canary(cfg))
    if (ncfg.get("enabled") and ncfg.get("startup_canary", True)
            and self.notification_health):
        failed = [lbl for lbl, st in self.notification_health.items()
                  if not st.get("delivered")]
        if failed:
            logger.error(
                "Startup canary FAILED on %d/%d channel(s): %s — "
                "exiting nonzero so the supervisor restarts. Set "
                "notifications.startup_canary: false in orze.yaml "
                "to disable this check.",
                len(failed), len(self.notification_health),
                ", ".join(failed))
            raise SystemExit(
                f"startup_canary failed for: {', '.join(failed)}")

    # Initialize milestone from current state (avoid spurious on restart)
    if not managed_idea:
        try:
            init_ideas = parse_ideas(cfg["ideas_file"])
            init_counts = _count_statuses(
                init_ideas, self.results_dir, lake=self.lake)
            milestone_every = (cfg.get("notifications") or {}).get(
                "milestone_every", 100)
            if milestone_every > 0:
                self._last_milestone = (
                    init_counts.get("COMPLETED", 0) // milestone_every
                ) * milestone_every
                self._hb_completed_count = init_counts.get("COMPLETED", 0)
        except Exception:
            pass

    # Full reconcile at startup: clear ALL stale queued ideas at once
    if self.lake and not managed_idea:
        try:
            n = self.lake.reconcile_statuses(
                str(self.results_dir),
                evaluation_required=bool(cfg.get("eval_script")),
            )
            if n:
                logger.info("Startup reconcile: updated %d stale ideas", n)
        except Exception as e:
            logger.warning("Startup reconcile failed: %s", e)

    # Rebuild config dedup hash cache from completed ideas
    if not managed_idea:
        try:
            self._rebuild_config_hashes()
        except Exception as e:
            logger.error("Config hash cache rebuild failed: %s", e)
            notify("config_hash_failure", {"error": str(e)}, self.cfg)

    # Initialize code change detector (removed in v4.0)

    # Compute sealed file manifest for metric integrity
    sealed_files = cfg.get("sealed_files", [])
    if sealed_files and not managed_idea:
        from orze.engine.sealed import compute_sealed_hashes, write_sealed_manifest
        hashes = compute_sealed_hashes(sealed_files)
        # Explicit pins replace startup observations.  This detects source
        # drift that predates the current Orze process, rather than blessing
        # the already-drifted content as the new baseline.
        hashes.update({
            str(path): str(digest).lower()
            for path, digest in (cfg.get("sealed_hashes") or {}).items()
        })
        write_sealed_manifest(self.results_dir, hashes)

    while self.running:
        self.iteration += 1
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        logger.info("--- Iteration %d [%s] ---", self.iteration, ts)

        # Hot-reload config every 10 iterations (~5 min)
        if not managed_idea and self.iteration % 10 == 0:
            self._hot_reload_config()

        # 0a. Early heartbeat — keeps nodes UI alive even when
        #     iterations are slow (large results_dir scans).
        if not managed_idea:
            try:
                busy = (self.slot_mgr.gpu_ids_in_use()
                        | set(self.active_evals.keys()))
                free_early = [g for g in self.gpu_ids if g not in busy]
                write_host_heartbeat(
                    self.results_dir, socket.gethostname(),
                    self.active, free_early)
            except Exception:
                pass

        # 0b. Auto-upgrade check (rate-limited PyPI + sentinel from other nodes)
        if not managed_idea:
            self._check_auto_upgrade()
            self._check_upgrade_sentinel()

        # 0c. Version compatibility check (updates _incompatible_hosts)
        if not managed_idea:
            try:
                self._check_cluster_versions()
            except Exception:
                pass

        # 0d. Filesystem health check — pause if FS is not writable
        if not self._health_monitor.check_before_write():
            self._stop_event.wait(self._health_monitor.retry_delay)
            continue

        # 0. Check for filesystem stop/disable signals (multi-machine)
        if self._check_stop_all() or self._check_disabled():
            break

        # Round-2 E1: honor .orze_reset_role_state markers dropped by
        # `orze admin reset-role-state --all-hosts`. Each host clears
        # its own per-host state file once per marker; the marker
        # self-deletes after every host has claimed it.
        if not managed_idea:
            try:
                from orze.admin.reset_role_state import consume_marker_on_this_host
                consume_marker_on_this_host(self.results_dir)
            except Exception as _e:  # pragma: no cover — non-fatal hook
                logger.debug("reset-role-state marker hook failed: %s", _e)

        # 1. Check disk space (only gates launches, never skips reaping)
        disk_ok = check_disk_space(self.results_dir,
                                   cfg.get("min_disk_gb", 0))
        if not disk_ok:
            logger.warning(
                "Low disk space (< %dGB free). Pausing launches.",
                cfg["min_disk_gb"])

        # 2. Periodic maintenance (orphans + GC, locked for multi-machine)
        cleanup_cfg = cfg.get("cleanup") or {}
        cleanup_interval = cleanup_cfg.get("interval", 100)
        if (not managed_idea and cleanup_interval > 0
                and self.iteration % cleanup_interval == 0):
            cleanup_lock = self.results_dir / "_cleanup_lock"
            if _fs_lock(cleanup_lock, stale_seconds=300):
                try:
                    orphan_hours = cfg.get("orphan_timeout_hours", 0)
                    if orphan_hours > 0:
                        cleaned = cleanup_orphans(
                            self.results_dir, orphan_hours,
                            lake=self.lake)
                        if cleaned:
                            logger.info("Cleaned %d orphaned claims",
                                        cleaned)
                    run_cleanup(self.results_dir, cfg)
                finally:
                    _fs_unlock(cleanup_lock)
            else:
                logger.debug("Cleanup lock held by another host, skipping")

        # 2b. Periodic orphan cleanup (every 10 iterations ≈ 5 min)
        if not managed_idea and self.iteration % 10 == 0:
            try:
                self._kill_orphans()
            except Exception:
                pass

        # 2b''. FSM dead-PID reaper (every 10 iterations ≈ 5 min, v4.5)
        if (not managed_idea and self.lake
                and self.iteration % 10 == 0):
            try:
                self.lake.reap_dead_claims(max_age_minutes=15)
            except Exception as e:
                logger.debug("FSM dead-PID reaper failed: %s", e)

        # 2b'''. FSM catch-up for missing terminal transitions (every 20 iterations ≈ 10 min)
        if (not managed_idea and self.lake
                and self.iteration % 20 == 0):
            try:
                self.lake.catch_up_missing_terminals(
                    self.results_dir,
                    evaluation_required=bool(self.cfg.get("eval_script")),
                )
            except Exception as e:
                logger.debug("FSM catch-up failed: %s", e)

        # 2b'. F7: every 30 min, mark 'running' rows whose training
        # process has died as 'failed' with reason orphaned_pid.
        if not managed_idea and self.iteration % 60 == 0:
            try:
                reconcile_running_dead_pids(cfg)
            except Exception as e:
                logger.debug("reconcile_running_dead_pids: %s", e)

        # 2c. Periodic metric harvest (every 20 iterations ≈ 5 min).
        # Training scripts that log per-epoch metrics to stdout but
        # never emit metrics.json otherwise leave the leaderboard
        # blind to mid-run progress. Harvester fills in metrics.json
        # from train_output.log so professor/leaderboard see reality.
        # When the bundled regex defaults miss (exotic log formats),
        # orze-pro's pattern_inference can learn patterns via LLM
        # and cache them keyed by train_script mtime — one call per
        # new script, free thereafter.
        if not managed_idea and self.iteration % 20 == 0:
            try:
                from orze.engine.metric_harvester import harvest_running_ideas
                mh_cfg = cfg.get("metric_harvest") or {}
                if mh_cfg.get("enabled", True):
                    primary = (cfg.get("report") or {}).get(
                        "primary_metric", "map")
                    extra = mh_cfg.get("patterns") or []
                    maximize = mh_cfg.get("maximize", True)
                    inferrer = None
                    if mh_cfg.get("llm_fallback", True):
                        try:
                            from orze_pro.agents.pattern_inference import (
                                infer_metric_patterns,
                            )
                            inf_model = mh_cfg.get(
                                "inference_model", "haiku")
                            inf_timeout = int(mh_cfg.get(
                                "inference_timeout", 60))
                            inf_bin = mh_cfg.get("claude_bin", "") or ""

                            def inferrer(script, text, metric,
                                         _fn=infer_metric_patterns,
                                         _m=inf_model,
                                         _t=inf_timeout,
                                         _b=inf_bin):
                                return _fn(script, text, metric,
                                           model=_m, timeout=_t,
                                           claude_bin=_b)
                        except ImportError:
                            inferrer = None
                    ts_str = cfg.get("train_script")
                    ts_path = Path(ts_str) if ts_str else None
                    n = harvest_running_ideas(
                        self.results_dir, primary, extra,
                        maximize=maximize,
                        pattern_inferrer=inferrer,
                        train_script=ts_path)
                    if n > 0:
                        logger.info(
                            "Metric harvest: updated %d running idea(s)", n)
            except Exception as e:
                logger.debug("Metric harvest failed: %s", e)

        # 3. Check active training processes (with health monitoring)
        finished = []
        if self.active:
            finished = check_active(self.active, self.results_dir,
                                    cfg, self.failure_counts,
                                    self.fix_counts, lake=self.lake)

        # 3-auto. After a SUCCESSFUL job finishes, check if GPU mode
        # should upgrade from exclusive to VRAM packing.
        # Critical: only check on success — failed jobs (exit code 2,
        # argparse errors) use 0 VRAM and would falsely trigger packing
        # mode, cascading into 90 launches per GPU.
        if finished and self._auto_gpu_mode:
            from orze.engine.gpu_slots import _query_all_gpu_usage
            # Find the first successful completion (has metrics.json
            # with status=COMPLETED and ran for >30 seconds)
            success_gpu = None
            for idea_id, gpu_key in finished:
                metrics_path = self.results_dir / idea_id / "metrics.json"
                if metrics_path.exists():
                    try:
                        m = json.loads(metrics_path.read_text(encoding="utf-8"))
                        if m.get("status") == "COMPLETED" and m.get("training_time", 0) > 30:
                            success_gpu = gpu_key
                            break
                    except (json.JSONDecodeError, OSError):
                        pass

            if success_gpu is not None:
                try:
                    usage = _query_all_gpu_usage(self.gpu_ids)
                    if usage:
                        gpu_id = int(str(success_gpu).split(":")[0]) if ":" in str(success_gpu) else success_gpu
                        if gpu_id in usage:
                            used, total = usage[gpu_id]
                            pct = used / total * 100 if total > 0 else 100
                            if pct < 30:
                                logger.info(
                                    "Auto GPU mode: successful job used %d/%d MiB (%.0f%%) — "
                                    "upgrading to VRAM packing for higher throughput",
                                    used, total, pct)
                                self.slot_mgr.mode = "vram"
                                self.slot_mgr.max_jobs_per_gpu = max(
                                    int(90 / max(pct, 1)), 2)
                                logger.info("  max_jobs_per_gpu set to %d",
                                            self.slot_mgr.max_jobs_per_gpu)
                            else:
                                logger.info(
                                    "Auto GPU mode: successful job used %.0f%% VRAM — "
                                    "keeping exclusive mode", pct)
                    self._auto_gpu_mode = False  # only check once
                except Exception:
                    self._auto_gpu_mode = False
            # Don't disable auto_gpu_mode on failures — wait for a
            # real success to make the decision.

        # 3a. Check active eval processes
        eval_finished = []
        if self.active_evals:
            eval_finished = check_active_evals(
                self.active_evals, self.results_dir, cfg, lake=self.lake)

        # Reap existing children first, then reject every *new* GPU launch
        # (post-script, evaluation, or training) unless the exact campaign
        # evidence window is active. Clearing pause controls alone cannot
        # create unmeasured work.
        campaign_efficiency_cfg = cfg.get("campaign_efficiency") or {}
        if (self.lake is not None
                and campaign_efficiency_cfg.get("required_for_launch", False)
                and not _is_launcher_paused(cfg, self.results_dir)):
            require_active_campaign_registration(
                self.lake,
                campaign_id=campaign_efficiency_cfg.get("campaign_id"),
                physical_scope=list(self.gpu_ids),
                poll_seconds=cfg["poll"],
            )

        # 3b. Run post-scripts for evals that just completed
        for idea_id, gpu in eval_finished:
            run_post_scripts(
                idea_id, gpu, self.results_dir, cfg, lake=self.lake)

        if not self.running:
            break

        # 3c. Upgrade notification (user-triggered only, no auto-restart)
        # Sentinel-triggered upgrades (another node already pip-installed)
        # are handled in _check_upgrade_sentinel() above.

        # 4. Run agent roles (research, documenter, etc.)
        if not managed_idea:
            try:
                self._run_all_roles()
            except Exception as e:
                logger.error("Error in _run_all_roles: %s — continuing", e)
                notify("role_management_error", {"error": str(e)}, cfg)

        if not self.running:
            break

        # 4b. Mid-iteration heartbeat — keeps nodes alive during long iterations
        if not managed_idea:
            try:
                busy = (self.slot_mgr.gpu_ids_in_use()
                        | set(self.active_evals.keys()))
                free_mid = [g for g in self.gpu_ids if g not in busy]
                write_host_heartbeat(
                    self.results_dir, socket.gethostname(),
                    self.active, free_mid)
            except Exception:
                pass

        # 5. Sync ideas + expand sweeps + build unclaimed queue
        if managed_idea:
            ideas, unclaimed, skipped, raw_ideas = (
                self._sync_managed_idea(cfg, managed_idea))
        else:
            ideas, unclaimed, skipped, raw_ideas = self._sync_ideas(cfg)

        # 6. Launch evals (finished training, pending, backlog)
        eval_finished, backlog = self._launch_evals(
            finished, eval_finished, ideas)

        # 7. Launch training on free GPUs + circuit breaker
        free = self._launch_training(unclaimed, disk_ok, ideas)

        # 7b. Persist one local, privacy-safe scheduler/GPU observation.
        # The sampler queries exactly this controller's physical scope;
        # failures are retained as incomplete evidence, never retried with
        # an unscoped query, and halt required-evidence campaigns.
        self._capture_campaign_efficiency_evidence(
            unclaimed, backlog, disk_ok
        )

        # 8. Update report
        if managed_idea:
            completed_rows = []
            self._capture_campaign_progress_evidence(
                completed_rows, unclaimed, disk_ok, backlog
            )
        else:
            completed_rows = update_report(
                self.results_dir, ideas, cfg, lake=self.lake,
                role_states=self.role_states)

            # 9-10. Daemon-wide reporting and notification are deliberately
            # absent from an exact one-idea invocation.
            write_host_heartbeat(
                self.results_dir, socket.gethostname(), self.active, free)
            counts = _count_statuses(
                ideas, self.results_dir, lake=self.lake)
            self._report_and_notify(
                completed_rows, ideas, counts, eval_finished,
                free, unclaimed, skipped, disk_ok, backlog)

        if self.once:
            all_once_finished = []
            # Wait for active training
            if self.active:
                logger.info("--once mode: waiting for active training...")
                while self.active:
                    time.sleep(5)
                    if (self._check_stop_all()
                            or self._check_disabled()
                            or not self.running):
                        break
                    once_finished = check_active(
                        self.active, self.results_dir,
                        cfg, self.failure_counts,
                        self.fix_counts, lake=self.lake)
                    for idea_id, gpu in once_finished:
                        m_path = self.results_dir / idea_id / "metrics.json"
                        if m_path.exists():
                            try:
                                m = json.loads(m_path.read_text(encoding="utf-8"))
                                if m.get("status") == "COMPLETED":
                                    if managed_idea:
                                        immediate, _ = self._launch_evals(
                                            [(idea_id, gpu)], [],
                                            {idea_id: {}})
                                        if immediate:
                                            run_post_scripts(
                                                idea_id, gpu,
                                                self.results_dir, cfg,
                                                lake=self.lake)
                                    else:
                                        run_eval(
                                            idea_id, gpu,
                                            self.results_dir, cfg,
                                            lake=self.lake)
                                        run_post_scripts(
                                            idea_id, gpu,
                                            self.results_dir, cfg,
                                            lake=self.lake)
                            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                                pass
                        all_once_finished.append((idea_id, gpu))
                    if managed_idea and not (
                            self._capture_managed_wait_campaign_evidence(
                                disk_ok
                            )):
                        break
            # Wait for active evals (launched this iteration or earlier)
            if self.active_evals and self.running:
                logger.info("--once mode: waiting for %d active evals...",
                            len(self.active_evals))
                while self.active_evals:
                    time.sleep(5)
                    if (self._check_stop_all()
                            or self._check_disabled()
                            or not self.running):
                        break
                    ef = check_active_evals(
                        self.active_evals, self.results_dir, cfg, lake=self.lake)
                    for idea_id, gpu in ef:
                        run_post_scripts(
                            idea_id, gpu, self.results_dir, cfg,
                            lake=self.lake)
                        all_once_finished.append((idea_id, gpu))
                    if managed_idea and not (
                            self._capture_managed_wait_campaign_evidence(
                                disk_ok
                            )):
                        break
            if all_once_finished and not managed_idea:
                ideas = parse_ideas(cfg["ideas_file"])
                once_rows = update_report(
                    self.results_dir, ideas, cfg, lake=self.lake)
                once_counts = _count_statuses(ideas, self.results_dir, lake=self.lake)
                self._process_notifications(
                    all_once_finished, once_rows or [], ideas,
                    once_counts)
                # The ordinary status snapshot was written before --once
                # waited for its launched job.  Publish a terminal
                # heartbeat/status after reaping so machine readers do
                # not observe a completed run as still active.
                final_free = list(self.gpu_ids)
                write_host_heartbeat(
                    self.results_dir, socket.gethostname(),
                    self.active, final_free)
                primary = cfg["report"].get(
                    "primary_metric", "test_accuracy")
                top_results = [
                    {
                        "idea_id": row["id"],
                        "title": row["title"][:60],
                        primary: row.get("primary_val"),
                    }
                    for row in (once_rows or [])[:10]
                ]
                write_status_json(
                    self.results_dir, self.iteration, self.active,
                    final_free, once_counts.get("QUEUED", 0),
                    once_counts.get("COMPLETED", 0),
                    once_counts.get("FAILED", 0),
                    once_counts.get("SKIPPED", 0), top_results, cfg,
                    role_states=self.role_states,
                    active_roles=self.active_roles,
                    notification_health=self.notification_health,
                )
                save_state(self.results_dir, self._build_state_dict())
            logger.info("Done.")
            break

        # Interruptible sleep — write heartbeat every 60s while waiting
        poll_remaining = cfg["poll"]
        while poll_remaining > 0 and not self._stop_event.is_set():
            tick = min(poll_remaining, 60)
            self._stop_event.wait(tick)
            poll_remaining -= tick
            if not self._stop_event.is_set():
                try:
                    busy = self.slot_mgr.gpu_ids_in_use() | set(self.active_evals.keys())
                    free = [g for g in self.gpu_ids if g not in busy]
                    write_host_heartbeat(self.results_dir,
                                         socket.gethostname(),
                                         self.active, free)
                except Exception:
                    pass
        if self._stop_event.is_set():
            break

    # Main loop exited (signal received or --once finished)
    if self.active or self.active_evals or self.active_roles:
        self._graceful_shutdown(
            kill_all=(True if managed_idea else getattr(
                self, '_stop_kill_all', False)))
    else:
        # Nothing running, just save state and clean up
        if not managed_idea:
            save_state(self.results_dir, self._build_state_dict())
        if self.lake:
            try:
                self.lake.close()
            except Exception:
                pass
        self._remove_pid_file()
    try:
        if self._leader_handle is not None:
            self._leader_handle.release()
    except Exception:
        pass
    logger.info("Exited after %d iterations.", self.iteration)
