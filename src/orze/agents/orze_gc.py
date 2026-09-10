#!/usr/bin/env python3
"""orze gc: generic garbage collector for experiment checkpoints and artifacts.

Frees disk space by deleting checkpoint directories for non-top experiments.
Determines which experiments to keep from _leaderboard.json and idea_lake.db.

Can be run standalone or wired into the orchestrator's cleanup cycle.

Usage:
    # Standalone
    python orze/orze_gc.py -c orze.yaml --dry-run
    python orze/orze_gc.py -c orze.yaml --keep-top 50

    # In orze.yaml (auto-called by the orchestrator every cleanup.interval iterations):
    gc:
      enabled: true
      checkpoints_dir: checkpoints      # where model checkpoints live
      keep_top: 50                       # keep top N by primary_metric
      keep_recent: 20                    # also keep N most recently completed
      min_free_gb: 100                   # only run GC when disk < this
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orze.gc")


def get_top_idea_ids(results_dir: Path, primary_metric: str,
                     lake_db_path: Optional[Path], keep_top: int,
                     sort_order: str = "descending") -> Set[str]:
    """Collect idea IDs that should be kept (top performers).

    Sources (merged):
    1. _leaderboard.json (top 20 from the orchestrator)
    2. idea_lake.db (top N by primary_metric)
    """
    keep: Set[str] = set()

    # 1. Leaderboard (always trust — these are the current best)
    lb_path = results_dir / "_leaderboard.json"
    if lb_path.exists():
        try:
            data = json.loads(lb_path.read_text(encoding="utf-8"))
            for entry in data.get("top", []):
                iid = entry.get("idea_id", "")
                if iid:
                    keep.add(iid)
        except (json.JSONDecodeError, OSError):
            pass

    # 2. Idea Lake (broader history)
    if lake_db_path and lake_db_path.exists() and primary_metric:
        try:
            uri = lake_db_path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=10)
            # Try primary_metric, then fallback without "adjusted_"
            candidates = [primary_metric]
            if "_adjusted_" in primary_metric:
                candidates.append(primary_metric.replace("_adjusted_", "_"))
            direction = "ASC" if sort_order == "ascending" else "DESC"
            for metric in candidates:
                rows = conn.execute(
                    "SELECT idea_id FROM ideas "
                    "WHERE json_valid(eval_metrics) "
                    "AND json_extract(eval_metrics, ?) IS NOT NULL "
                    f"ORDER BY json_extract(eval_metrics, ?) {direction} LIMIT ?",
                    (f"$.{metric}", f"$.{metric}", keep_top),
                ).fetchall()
                if rows:
                    for r in rows:
                        keep.add(r[0])
                    break
            
            conn.close()
        except Exception as e:
            logger.warning("Lake query failed: %s", e)

    return keep


def get_recent_idea_ids(results_dir: Path, keep_recent: int) -> Set[str]:
    """Get the N most recently completed idea IDs (by metrics.json mtime)."""
    recent: List[tuple] = []  # (mtime, idea_id)

    try:
        with os.scandir(results_dir) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if not entry.name.startswith("idea-"):
                    continue
                metrics_path = Path(entry.path) / "metrics.json"
                if metrics_path.exists():
                    try:
                        data = json.loads(metrics_path.read_text("utf-8"))
                        if data.get("status") == "COMPLETED":
                            recent.append((metrics_path.stat().st_mtime,
                                           entry.name))
                    except (json.JSONDecodeError, OSError):
                        pass
    except OSError:
        pass

    recent.sort(reverse=True)
    return {iid for _, iid in recent[:keep_recent]}


def get_active_idea_ids(results_dir: Path) -> Set[str]:
    """Get idea IDs that are currently being trained (have claim but no metrics)."""
    active: Set[str] = set()
    try:
        with os.scandir(results_dir) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if not entry.name.startswith("idea-"):
                    continue
                claim = Path(entry.path) / "claim.json"
                metrics = Path(entry.path) / "metrics.json"
                if claim.exists() and not metrics.exists():
                    active.add(entry.name)
    except OSError:
        pass
    return active


def gc_checkpoints(checkpoints_dir: Path, keep_ids: Set[str],
                   dry_run: bool = False, *, scope=None) -> Dict[str, Any]:
    """Explicitly scoped checkpoint cleanup; missing scope is a safe refusal."""
    from orze.engine.gc_safety import collect
    return collect(scope, checkpoints_dir, keep_ids, mode="checkpoints", dry_run=dry_run)


def gc_results(results_dir: Path, keep_ids: Set[str],
               dry_run: bool = False, *, scope=None) -> Dict[str, Any]:
    """Prune only scoped, closed, undeclared result artifacts."""
    from orze.engine.gc_safety import collect
    return collect(scope, results_dir, keep_ids, mode="results", dry_run=dry_run)


def archive_to_cold_storage(results_dir: Path, archive_dir: Path, keep_ids: Set[str],
                            dry_run: bool = False, *, scope=None) -> Dict[str, Any]:
    """Same-filesystem no-overwrite archive; no implicit copy/delete fallback."""
    from orze.engine.gc_safety import collect, GCScope, GCRefused
    if not isinstance(scope, GCScope) or Path(archive_dir).absolute() != scope.archive_dir:
        raise GCRefused("gc_archive_scope_required")
    return collect(scope, results_dir, keep_ids, mode="archive", dry_run=dry_run)


def run_gc(
    results_dir: Path,
    checkpoints_dir: Optional[Path],
    primary_metric: str,
    lake_db_path: Optional[Path] = None,
    keep_top: int = 50,
    keep_recent: int = 20,
    min_free_gb: float = 0,
    dry_run: bool = False,
    gc_results_enabled: bool = False,
    archive_dir: Optional[Path] = None,
    extra_keep_ids: Optional[Set[str]] = None,
    sort_order: str = "descending",
    *, cfg: Optional[dict] = None, lake=None,
) -> Dict[str, Any]:
    """Run explicitly scoped GC; unsafe or unknown authority is not deletion."""
    from orze.engine.gc_safety import gc_scope
    try:
        scope = gc_scope(results_dir, cfg, lake=lake, lake_db_path=lake_db_path,
                         checkpoints_dir=checkpoints_dir, archive_dir=archive_dir)
        results_dir = scope.results_dir
        checkpoints_dir, archive_dir = scope.checkpoints_dir, scope.archive_dir
    except Exception as exc:
        logger.warning("GC blocked: %s", exc)
        return {"blocked": True, "reason": str(exc), "checkpoints": {},
                "results": {}, "archive": {}}
    logger.info("=" * 50)
    logger.info("GARBAGE COLLECTION%s", " (DRY RUN)" if dry_run else "")
    logger.info("=" * 50)

    # Check if GC is needed (disk space gate)
    if min_free_gb > 0 and results_dir.exists():
        try:
            usage = shutil.disk_usage(results_dir)
            free_gb = usage.free / (1024 ** 3)
            if free_gb >= min_free_gb:
                logger.info("Disk has %.1fGB free (>= %.0fGB threshold), "
                            "skipping GC", free_gb, min_free_gb)
                return {"skipped": True, "free_gb": round(free_gb, 1)}
        except Exception:
            pass

    # Build the keep set
    keep_ids = get_top_idea_ids(results_dir, primary_metric,
                                lake_db_path, keep_top, sort_order)
    logger.info("Top performers: %d ideas", len(keep_ids))

    recent_ids = get_recent_idea_ids(results_dir, keep_recent)
    keep_ids |= recent_ids
    logger.info("+ Recent completions: %d ideas", len(recent_ids))

    active_ids = get_active_idea_ids(results_dir)
    keep_ids |= active_ids
    if active_ids:
        logger.info("+ Currently active: %d ideas", len(active_ids))

    # Merge externally-known running idea IDs (e.g. from the orchestrator's
    # active training set).  This guards against races where claim.json has
    # not been written yet or metrics.json was partially flushed.
    if extra_keep_ids:
        keep_ids |= extra_keep_ids
        logger.info("+ Orchestrator active (extra_keep_ids): %d ideas",
                     len(extra_keep_ids))

    logger.info("Total protected: %d ideas", len(keep_ids))

    stats: Dict[str, Any] = {"checkpoints": {}, "results": {}, "archive": {}}

    # GC checkpoints
    if checkpoints_dir:
        stats["checkpoints"] = gc_checkpoints(
            checkpoints_dir, keep_ids, dry_run=dry_run, scope=scope)
        cs = stats["checkpoints"]
        logger.info("Checkpoints: deleted=%d, kept=%d, errors=%d",
                     cs["deleted"], cs["kept"], cs["errors"])

    # Cold Storage Archival
    if archive_dir:
        stats["archive"] = archive_to_cold_storage(
            results_dir, archive_dir, keep_ids, dry_run=dry_run, scope=scope)
        as_ = stats["archive"]
        moved_gb = as_.get("moved_bytes", 0) / (1024**3)
        logger.info("Archive: items=%d, logical-moved=%.1fGB, kept=%d",
                     as_["archived_files"], moved_gb, as_["kept"])

    # GC results artifacts (Deletes from hot storage without archive)
    if gc_results_enabled and not archive_dir:
        stats["results"] = gc_results(results_dir, keep_ids, dry_run=dry_run, scope=scope)
        rs = stats["results"]
        freed_mb = rs["freed_bytes"] / (1024 * 1024)
        logger.info("Results artifacts: deleted=%d, logical-removed=%.1fMB, kept=%d",
                     rs["deleted_files"], freed_mb, rs["kept"])

    # Report final disk state
    if results_dir.exists():
        try:
            usage = shutil.disk_usage(results_dir)
            stats["free_gb"] = round(usage.free / (1024 ** 3), 1)
            logger.info("Disk free: %.1fGB", stats["free_gb"])
        except Exception:
            pass

    if any(stats[name].get("errors", 0) for name in ("checkpoints", "results", "archive")):
        stats["blocked"] = True
    return stats


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

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
    if args.lake_db is not None:
        lake_db_path = selected_path(args.lake_db, invocation_root)
    elif "idea_lake_db" in cfg:
        lake_db_path = selected_path(cfg["idea_lake_db"], project_root)
    else:
        default_lake = control / "idea_lake.db"
        lake_db_path = default_lake if default_lake.exists() or default_lake.is_symlink() else None
    keep_top = args.keep_top if args.keep_top is not None else gc_cfg.get("keep_top", 50)
    keep_recent = args.keep_recent if args.keep_recent is not None else gc_cfg.get("keep_recent", 20)
    min_free_gb = args.min_free_gb if args.min_free_gb is not None else gc_cfg.get("min_free_gb", 0)
    if any(type(value) is not int or value < 0 for value in (keep_top, keep_recent)):
        parser.error("gc_keep_counts_require_nonnegative_integers")
    if type(min_free_gb) not in (int, float) or min_free_gb < 0:
        parser.error("gc_disk_threshold_requires_finite_nonnegative_number")
    try:
        finite_threshold = math.isfinite(min_free_gb)
    except OverflowError:
        finite_threshold = False
    if not finite_threshold:
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
               gc=gc_cfg, report=report_cfg)
    if lake_db_path is not None:
        cfg["idea_lake_db"] = str(lake_db_path)

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


if __name__ == "__main__":
    raise SystemExit(main())
