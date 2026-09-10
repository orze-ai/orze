def run_post_scripts(
        idea_id: str, gpu: int, results_dir: Path, cfg: dict, lake=None, *, source_event=None):
    """Run additional post-training scripts (beyond eval_script).
    Each entry in post_scripts is a dict with: script, args, timeout, output."""
    post_scripts = (cfg.get("post_scripts") or [])
    if not post_scripts:
        return

    idea_dir = results_dir / idea_id
    from orze.engine.completion_events import completion_is_current
    if not completion_is_current(source_event or (idea_id, gpu), lake, results_dir):
        return
    eligible, eligibility_reason = is_training_complete_for_downstream(
        idea_dir, cfg)
    if not eligible:
        logger.warning(
            "[POST_SCRIPT_SKIP] idea=%s reason=%s",
            idea_id, eligibility_reason)
        _record_eval_audit(idea_dir, "skip", eligibility_reason)
        return

    _assert_campaign_evidence_authorized(cfg, lake)

    python = cfg.get("python", sys.executable)
    env = os.environ.copy()
    for k, v in (cfg.get("train_extra_env") or {}).items():
        env[k] = str(v)
    env = _authorized_gpu_environment(gpu, cfg, env)

    for i, ps in enumerate(post_scripts):
        script = ps.get("script")
        if not script:
            continue

        # Skip if output already exists
        output_file = ps.get("output", "")
        if output_file:
            output_path = results_dir / idea_id / output_file
            if output_path.exists():
                name = ps.get("name", f"post-script-{i}")
                logger.info(
                    "[POST_SCRIPT_SKIP] idea=%s script=%s reason=output_exists "
                    "path=%s",
                    idea_id, name, output_path,
                )
                _record_eval_audit(
                    results_dir / idea_id, "skip", "output_exists",
                    script=name, output_path=str(output_path),
                )
                continue

        _assert_launch_authorized(idea_id, results_dir, cfg)
        _assert_gpu_authorized(gpu, cfg)
        _assert_controller_runtime_attested(cfg)
        args = ps.get("args") or []
        timeout = ps.get("timeout", 3600)
        name = ps.get("name", f"post-script-{i}")

        cmd = [python, script]
        cmd.extend(_format_args(args, {
            "idea_id": idea_id, "gpu": 0, "physical_gpu": gpu,
        }))

        log_path = results_dir / idea_id / f"{name}.log"
        if not completion_is_current(source_event or (idea_id, gpu), lake, results_dir):
            return
        if getattr(source_event, "attempt_ref", None) is not None:
            from orze.engine.native_post_script import run_native_post_script
            run_native_post_script(source_event, idea_id, gpu, results_dir, cfg, lake,
                                   cmd, timeout, log_path, env)
            continue
        logger.info("Running %s for %s", name, idea_id)

        proc = None
        handle = None
        log_fh = None
        try:
            with gpu_execution_lease(gpu, require_idle=True) as lease_fds:
                _verify_gpu_free(gpu, _launch_min_free_vram(cfg))
                log_fh = open(log_path, "w", encoding="utf-8")
                started = time.time()
                proc = subprocess.Popen(
                    cmd, env=env, stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    preexec_fn=_new_process_group,
                    pass_fds=lease_fds,
                )
                ep = EvalProcess(
                    idea_id=idea_id,
                    gpu=gpu,
                    process=proc,
                    start_time=started,
                    log_path=log_path,
                    timeout=float(timeout),
                    attempt_id=secrets.token_hex(16),
                    _log_fh=log_fh,
                )
                handle = ep
                record_compute_start(
                    handle, idea_dir, phase="post_script")
            return_code = proc.wait(timeout=timeout)
            record_compute_terminal(
                handle, idea_dir,
                "completed" if return_code == 0 else "failed",
                ("post_script_completed" if return_code == 0
                 else "post_script_nonzero"),
                phase="post_script", return_code=return_code,
            )
            if return_code == 0:
                logger.info("%s completed for %s", name, idea_id)
            else:
                logger.warning("%s failed for %s (exit %d)",
                               name, idea_id, return_code)
        except subprocess.TimeoutExpired:
            _terminate_and_reap(proc, f"post-script {idea_id}:{name}")
            if handle is not None:
                record_compute_terminal(
                    handle, idea_dir, "interrupted", "post_script_timeout",
                    phase="post_script", return_code=proc.poll(),
                )
            logger.warning("%s timed out for %s after %ds",
                           name, idea_id, timeout)
        except ComputeAccountingError:
            if proc is not None and proc.poll() is None:
                _terminate_and_reap(
                    proc, f"post-script {idea_id}:{name}", timeout=3)
            if handle is not None:
                try:
                    record_compute_terminal(
                        handle, idea_dir, "failed", "post_script_error",
                        phase="post_script", return_code=proc.poll(),
                    )
                except ComputeAccountingError:
                    pass
            raise
        except Exception as e:
            if proc is not None and proc.poll() is None:
                _terminate_and_reap(
                    proc, f"post-script {idea_id}:{name}", timeout=3)
            if handle is not None:
                try:
                    record_compute_terminal(
                        handle, idea_dir, "failed", "post_script_error",
                        phase="post_script", return_code=proc.poll(),
                    )
                except Exception:
                    pass
            logger.warning("%s error for %s: %s", name, idea_id, e)
        finally:
            if log_fh is not None:
                log_fh.close()
