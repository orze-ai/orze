"""Early report-only CLI entry point; observes authority without migrating it.

CALLING SPEC:
    run_report_only(args) -> int
        Load full project config and inbox, read a closed catalog snapshot,
        and regenerate report artifacts. Returns 0 on available authority,
        2 on invalid input or unavailable authority (empty ranks may still be
        written for diagnosis). Never initializes IdeaLake or starts runtime.

Relative config values and CLI path overrides are scoped to the selected
config directory. Temporary cwd changes are confined to this single-threaded
CLI branch and are always restored. Reports are derived writes; their SQLite
authority is inspected read-only, not repaired or completed by observation.
"""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import sys

import yaml

from orze.core.config import load_project_config
from orze.core.ideas import parse_ideas
from orze.reporting.catalog import load_catalog_snapshot
from orze.reporting.evidence import report_lifecycle_db_path
from orze.reporting.leaderboard import update_report


def run_report_only(args) -> int:
    """Regenerate a project's report without compute or database bootstrap."""
    conflicting = (
        "stop", "restart", "disable", "enable", "admin", "upgrade",
        "reinstall", "uninstall", "check", "init", "role_only", "research_only",
    )
    if any(getattr(args, name, None) for name in conflicting):
        print("--report-only cannot be combined with another action flag.", file=sys.stderr)
        return 2

    try:
        config_path = Path(args.config_file or "orze.yaml").absolute()
        if not config_path.is_file():
            raise ValueError("report_config_missing")
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if document is not None and not isinstance(document, dict):
            raise ValueError("report_config_not_mapping")
        caller_cwd = Path.cwd()
        try:
            os.chdir(config_path.parent)
            cfg = load_project_config(str(config_path))
            cfg["_config_path"] = str(config_path)
            # Preserve the loader's paired runtime control root and .orze
            # directory, including defaults under nested/external results.
            for option, key in (("results_dir", "results_dir"),
                                ("ideas_md", "ideas_file"),
                                ("base_config", "base_config"),
                                ("train_script", "train_script"),
                                ("timeout", "timeout"), ("poll", "poll")):
                if getattr(args, option, None) is not None:
                    cfg[key] = getattr(args, option)
            results_dir = Path(cfg["results_dir"]).absolute()
            cfg["results_dir"] = str(results_dir)
            cfg["_env_ORZE_RESULTS_DIR"] = str(results_dir)
            cfg["ideas_file"] = str(Path(cfg["ideas_file"]).absolute())
            cfg["_env_ORZE_IDEAS_FILE"] = cfg["ideas_file"]
            # Only explicit relative DB paths need config-directory anchoring.
            # The loader's default DB is already absolute at results.parent/
            # .orze; moving that authority would break runtime compatibility.
            cfg["idea_lake_db"] = str(Path(cfg["idea_lake_db"]).absolute())
            db_path = report_lifecycle_db_path(results_dir, cfg).absolute()
            cfg["idea_lake_db"] = str(db_path)
            # No live connection is exposed to the report or retained here.
            snapshot = load_catalog_snapshot(db_path)
            ideas = parse_ideas(cfg["ideas_file"])
            update_report(results_dir, ideas, cfg, lake=snapshot)
        finally:
            os.chdir(caller_cwd)
        if not snapshot.available:
            print(f"Report updated without ranking authority: {snapshot.reason}",
                  file=sys.stderr)
            return 2
        print("Report updated.")
        return 0
    except (OSError, ValueError, TypeError, sqlite3.Error, yaml.YAMLError) as exc:
        print(f"Report unavailable: {exc}", file=sys.stderr)
        return 2
