def load_project_config(path: Optional[str] = None) -> dict:
    """Load orze.yaml and merge with defaults. Returns full config dict."""
    _load_dotenv(path)
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if not path and Path("orze.yaml").exists():
        path = "orze.yaml"

    if path and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for k, v in raw.items():
            if k == "report" and isinstance(v, dict):
                cfg["report"] = {**cfg["report"], **v}
            else:
                cfg[k] = v
        # Fix YAML 'on:' boolean parsing — YAML interprets 'on' as True
        # so notifications.on becomes notifications[True] instead of notifications["on"]
        ncfg = cfg.get("notifications")
        if isinstance(ncfg, dict) and True in ncfg and "on" not in ncfg:
            ncfg["on"] = ncfg.pop(True)
            logger.info("Fixed YAML 'on:' boolean key in notifications config")

        logger.info("Loaded config from %s", path)
    elif path:
        logger.warning("Config file %s not found, using defaults", path)

    # Expand ${VAR} references in config values using os.environ
    cfg = _expand_env_vars(cfg)

    # Loud-warn on unresolved ${VAR} placeholders. Calls relying on these
    # (notifications, webhooks) will silently fail at runtime — make the
    # diagnosis a one-liner instead of a 5-day silent campaign.
    unresolved = _find_unresolved_env_vars(cfg)
    for path, raw in unresolved:
        logger.warning(
            "Unresolved ${VAR} placeholder at %s: %r — env var not set; "
            "any call relying on this value will silently fail "
            "(set the variable in .env or shell, then reload).",
            path, raw,
        )

    # Round-2 F1: top-level ``evolution.enabled`` /
    # ``evolution.max_attempts_per_plateau`` were a separate naming for
    # the same concept the ``code_evolution`` role implements. Migrate
    # them into ``roles.code_evolution`` and emit one DeprecationWarning
    # at config-load so operators have a release to update orze.yaml.
    # Old keys keep working until removed in a future release.
    legacy_evolution = cfg.get("evolution")
    if isinstance(legacy_evolution, dict):
        import warnings as _warnings
        _warnings.warn(
            "Top-level 'evolution:' is deprecated; move 'evolution.enabled' "
            "and 'evolution.max_attempts_per_plateau' under "
            "'roles.code_evolution.{enabled, max_attempts_per_plateau}'. "
            "Old keys will be honored for one release.",
            DeprecationWarning, stacklevel=2,
        )
        logger.warning(
            "Deprecated config key 'evolution:' detected — fold into "
            "roles.code_evolution.{enabled,max_attempts_per_plateau}. "
            "Migrating in-place for this run.")
        if not isinstance(cfg.get("roles"), dict):
            cfg["roles"] = {}
        roles = cfg["roles"]
        ce = roles.setdefault("code_evolution", {})
        if isinstance(ce, dict):
            for src, dst in (("enabled", "enabled"),
                              ("max_attempts_per_plateau",
                               "max_attempts_per_plateau"),
                              ("model", "model"),
                              ("timeout", "timeout"),
                              ("claude_bin", "claude_bin")):
                if src in legacy_evolution and dst not in ce:
                    ce[dst] = legacy_evolution[src]

    # Migrate legacy research: into roles: dict
    if "research" in cfg and isinstance(cfg["research"], dict):
        logger.warning("Migrating legacy 'research:' config to 'roles: {research: ...}'. "
                        "Update orze.yaml to use the 'roles:' format directly.")
        if not cfg.get("roles"):
            cfg["roles"] = {"research": cfg["research"]}
        elif "research" not in cfg["roles"]:
            cfg["roles"]["research"] = cfg["research"]

    # Compute project_root and orze_dir from results_dir
    results_path = Path(cfg["results_dir"])
    if not results_path.is_absolute():
        results_path = Path.cwd() / results_path
    project_root = results_path.parent
    orze_dir = project_root / ".orze"
    
    cfg["_orze_dir"] = str(orze_dir)
    cfg["_project_root"] = str(project_root)
    
    # Resolve ideas_file: None → .orze/ideas.md
    if not cfg.get("ideas_file"):
        cfg["ideas_file"] = str(orze_dir / "ideas.md")
    
    # Resolve idea_lake_db default → .orze/idea_lake.db (NOT results_dir)
    if not cfg.get("idea_lake_db"):
        cfg["idea_lake_db"] = str(orze_dir / "idea_lake.db")
    
    # Environment variable exposures for subprocess injection
    cfg["_env_ORZE_DIR"] = str(orze_dir)
    cfg["_env_ORZE_RESULTS_DIR"] = str(results_path)
    cfg["_env_ORZE_IDEAS_FILE"] = cfg["ideas_file"]
    cfg["_env_ORZE_RULES_DIR"] = str(orze_dir / "rules")
    cfg["_env_ORZE_METHODS_DIR"] = str(results_path / "methods")
    cfg["_env_ORZE_KNOWLEDGE_DIR"] = str(results_path / "knowledge")
    cfg["_env_ORZE_FEEDBACK_DIR"] = str(orze_dir / "feedback")

    # Auto-discover research backends from environment API keys.
    # Only activates if NO roles are explicitly configured at all.
    # If the user defined any roles (even mode: script), respect that
    # and don't inject auto-discovered backends alongside them.
    roles = cfg.get("roles") or {}
    if not roles:
        _AUTO_BACKENDS = [
            ("GEMINI_API_KEY", "gemini", "gemini-2.5-flash"),
            ("OPENAI_API_KEY", "openai", "gpt-4o"),
            ("ANTHROPIC_API_KEY", "anthropic", None),
        ]
        discovered = []
        for env_var, backend, default_model in _AUTO_BACKENDS:
            if os.environ.get(env_var):
                role_name = f"research_{backend}"
                role_cfg = {"mode": "research", "backend": backend}
                if default_model:
                    role_cfg["model"] = default_model
                if "roles" not in cfg:
                    cfg["roles"] = {}
                cfg["roles"][role_name] = role_cfg
                discovered.append(f"{backend} ({env_var})")
        if discovered:
            logger.info("Auto-discovered research backends: %s",
                        ", ".join(discovered))
        else:
            logger.info("No API keys found in environment — research agent will not run. "
                        "Add GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY to .env")

    # Auto-seal eval scripts (data leakage guardrail). Any file in the project
    # root matching eval_*.py or eval_*.sh is added to sealed_files unless
    # auto_seal_eval is explicitly set to false.
    if cfg.get("auto_seal_eval", True):
        sealed = list(cfg.get("sealed_files") or [])
        existing = set(sealed)
        auto_added = []
        try:
            for pattern in ("eval_*.py", "eval_*.sh"):
                for match in sorted(Path(".").glob(pattern)):
                    name = str(match)
                    if name not in existing:
                        sealed.append(name)
                        existing.add(name)
                        auto_added.append(name)
        except Exception as e:
            logger.warning("auto_seal_eval glob failed: %s", e)
        if auto_added:
            cfg["sealed_files"] = sealed
            logger.info("auto_seal_eval: sealed %d eval script(s): %s",
                        len(auto_added), ", ".join(auto_added))

    # A plain sealed_files entry snapshots whatever happens to exist at
    # startup.  sealed_hashes additionally pins a preregistered SHA-256 so
    # drift that happened before startup is caught as well.  Pinned paths are
    # automatically included in every existing sealed-file check.
    pinned = cfg.get("sealed_hashes") or {}
    if isinstance(pinned, dict):
        sealed = list(cfg.get("sealed_files") or [])
        existing = set(sealed)
        for fpath in pinned:
            if fpath not in existing:
                sealed.append(fpath)
                existing.add(fpath)
        if sealed:
            cfg["sealed_files"] = sealed

    return cfg
