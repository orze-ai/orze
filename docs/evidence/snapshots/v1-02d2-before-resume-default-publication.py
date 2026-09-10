def _open_catalog(folder, cfg, claim):
    if declared_catalog(folder) is not None:
        _reject("native_resume_admission_required")
    configured = cfg.get("idea_lake_db")
    bound = (claim or {}).get("lifecycle_db")
    paths = []
    for value in (configured, bound):
        if value is None:
            continue
        if type(value) is not str or not value:
            _reject("resume_catalog_unavailable")
        path = Path(value)
        if not path.is_absolute():
            from orze.engine.resume import _project_root
            path = _project_root(cfg, folder.parent) / path
        paths.append(path.absolute())
    if paths and any(path != paths[0] for path in paths[1:]):
        _reject("resume_catalog_scope_mismatch")
    if not paths:
        from orze.reporting.evidence import report_lifecycle_db_path
        candidate = report_lifecycle_db_path(folder.parent, cfg)
        if candidate.exists() or candidate.is_symlink():
            paths.append(candidate.absolute())
    if not paths:
        return None
    try:
        return open_existing_lake(paths[0])
    except (ValueError, OSError, sqlite3.Error):
        _reject("resume_catalog_unavailable")
