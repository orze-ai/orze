def main():
    parser = argparse.ArgumentParser(
        description="orze gc — garbage collect experiment checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Dry run (show what would be deleted)
  python orze/orze_gc.py -c orze.yaml --dry-run

  # Delete non-top-50 checkpoints
  python orze/orze_gc.py -c orze.yaml --keep-top 50

  # In orze.yaml:
  gc:
    enabled: true
    checkpoints_dir: checkpoints
    keep_top: 50
    keep_recent: 20
    min_free_gb: 100
""",
    )

    parser.add_argument("-c", "--config", default="orze.yaml",
                        help="Path to orze.yaml")
    parser.add_argument("--checkpoints-dir", default=None,
                        help="Checkpoint override, relative to the invocation directory")
    parser.add_argument("--keep-top", type=int, default=None,
                        help="Keep top N by primary metric (overrides orze.yaml)")
    parser.add_argument("--keep-recent", type=int, default=None,
                        help="Also keep N most recently completed")
    parser.add_argument("--min-free-gb", type=float, default=None,
                        help="Only run if disk free < this (0 = always run)")
    parser.add_argument("--lake-db", default=None,
                        help="Database override, relative to the invocation directory")
    parser.add_argument("--gc-results", action="store_true",
                        help="Delete large artifacts (.pt) from results/ too")
    parser.add_argument("--archive-dir", default=None,
                        help="Archive override, relative to the invocation directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without deleting")

    args = parser.parse_args()

    # Destructive maintenance never falls back after a missing/bad selected
    # configuration. Read a bounded UTF-8 document without dumping its values
    # into parser errors (a YAML failure can contain credentials).
    import math
    config_path = Path(os.path.abspath(args.config))
    try:
        with config_path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            parser.error("gc_configuration_too_large")
        cfg = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError, RecursionError):
        parser.error("gc_configuration_unavailable_or_invalid")
    if cfg is None:
        cfg = {}
    if type(cfg) is not dict:
        parser.error("gc_configuration_requires_mapping")
    gc_cfg, report_cfg = cfg.get("gc"), cfg.get("report")
    if gc_cfg is None:
        gc_cfg = {}
    if report_cfg is None:
        report_cfg = {}
    if type(gc_cfg) is not dict or type(report_cfg) is not dict:
        parser.error("gc_configuration_sections_require_mapping")
    project_root, invocation_root = config_path.parent, Path.cwd()

    def selected_path(value, root, *, optional=False):
        if optional and value in (None, ""):
            return None
        if type(value) is not str or not value or "\0" in value:
            parser.error("gc_configuration_path_invalid")
        return Path(os.path.abspath(root / value))

    def optional_override(argument, configured):
        return selected_path(argument, invocation_root, optional=True) if argument is not None else selected_path(
            configured, project_root, optional=True)

    results_dir = selected_path(cfg.get("results_dir", "orze_results"), project_root)
    checkpoints_dir = optional_override(args.checkpoints_dir, gc_cfg.get("checkpoints_dir"))
    archive_dir = optional_override(args.archive_dir, gc_cfg.get("archive_dir"))
    control = selected_path(cfg.get("_orze_dir", ".orze"), project_root)
    lake_db_path = (selected_path(args.lake_db, invocation_root) if args.lake_db is not None else
                    selected_path(cfg.get("idea_lake_db", str(control / "idea_lake.db")), project_root))
    keep_top = args.keep_top if args.keep_top is not None else gc_cfg.get("keep_top", 50)
    keep_recent = args.keep_recent if args.keep_recent is not None else gc_cfg.get("keep_recent", 20)
    min_free_gb = args.min_free_gb if args.min_free_gb is not None else gc_cfg.get("min_free_gb", 0)
    if any(type(value) is not int or value < 0 for value in (keep_top, keep_recent)):
        parser.error("gc_keep_counts_require_nonnegative_integers")
    if (type(min_free_gb) not in (int, float) or min_free_gb < 0
            or not math.isfinite(min_free_gb)):
        parser.error("gc_disk_threshold_requires_finite_nonnegative_number")
    results_artifacts = gc_cfg.get("results_artifacts", False)
    if type(results_artifacts) is not bool:
        parser.error("gc_results_artifacts_requires_boolean")
    gc_results_enabled = args.gc_results or results_artifacts
    primary_metric = report_cfg.get("primary_metric", "")
    sort_order = report_cfg.get("sort", "descending")
    if type(primary_metric) is not str or sort_order not in ("ascending", "descending"):
        parser.error("gc_report_selection_invalid")
    cfg.update(_project_root=str(project_root), _config_path=str(config_path),
               _orze_dir=str(control), results_dir=str(results_dir),
               idea_lake_db=str(lake_db_path), gc=gc_cfg, report=report_cfg)

    stats = run_gc(
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
        primary_metric=primary_metric,
        sort_order=sort_order,
        lake_db_path=lake_db_path,
        keep_top=keep_top,
        keep_recent=keep_recent,
        min_free_gb=min_free_gb,
        dry_run=args.dry_run,
        gc_results_enabled=gc_results_enabled,
        archive_dir=archive_dir,
        cfg=cfg,
    )

    print(json.dumps(stats, indent=2))
    return 2 if stats.get("blocked") else 0
