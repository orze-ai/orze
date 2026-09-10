def admit_resume(idea_id: str, results_dir: Path, cfg: dict,
                 checkpoint_override: str) -> dict:
    """Explicitly re-admit one attested checkpoint without deleting evidence."""
    results_dir = Path(results_dir)
    idea_dir = _idea_dir(idea_id, results_dir)
    receipt, checkpoint, receipt_sha = validate_resume_evidence(
        idea_id, results_dir, cfg, checkpoint_override)
    claim_path = idea_dir / "claim.json"
    if claim_path.exists():
        try:
            claim, _ = _read_json(claim_path)
            pid = int(claim.get("trainer_pid") or 0)
            if pid <= 0:
                raise ResumeValidationError("claim_identity_missing")
            start_ticks = claim.get("trainer_start_ticks")
            if pid and process_is_running(pid, start_ticks):
                raise ResumeValidationError("trainer_still_running")
        except ResumeValidationError:
            raise
        except (TypeError, ValueError):
            raise ResumeValidationError("claim_identity_invalid")

    request = {
        "schema_version": 1,
        "idea_id": idea_id,
        "created_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "created_by_host": socket.gethostname(),
        "checkpoint": _display_path(checkpoint, _project_root(cfg, results_dir)),
        "checkpoint_sha256": receipt["checkpoint"]["sha256"],
        "interruption_receipt_sha256": receipt_sha,
    }
    # Requeue in the audited DB before releasing filesystem admission. A crash
    # before the request is written remains fail-closed because claim/metrics
    # still block the scheduler.
    lake = None
    try:
        from orze.idea_lake import IdeaLake
        db_path = cfg.get("idea_lake_db")
        if db_path and Path(db_path).exists():
            lake = IdeaLake(str(db_path))
            if not lake.set_status(idea_id, "queued"):
                raise ResumeValidationError("idea_lake_requeue_failed")
    finally:
        if lake is not None:
            lake.close()

    atomic_write(
        idea_dir / "resume_request.json",
        json.dumps(request, indent=2, sort_keys=True) + "\n",
    )

    stamp = int(time.time())
    for source, label in ((idea_dir / "metrics.json", "metrics"),
                          (claim_path, "claim")):
        if source.exists():
            target = idea_dir / f"{label}.interrupted.{stamp}.json"
            suffix = 1
            while target.exists():
                target = idea_dir / f"{label}.interrupted.{stamp}.{suffix}.json"
                suffix += 1
            os.replace(source, target)
    return request
