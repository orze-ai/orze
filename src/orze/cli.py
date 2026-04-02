#!/usr/bin/env python3
"""Orze CLI — GPU experiment orchestrator.

Calling spec:
    python -m orze.cli                          # all GPUs, continuous
    python -m orze.cli -c orze.yaml --gpus 0,1  # with project config
    python -m orze.cli --once                   # one cycle then exit
    python -m orze.cli --report-only            # regenerate report
    python -m orze.cli --init                   # initialize new project
    python -m orze.cli --admin                  # launch admin panel

This module contains only:
    setup_logging()  — configure log format
    main()           — argparse + dispatch (imports from cli_* modules)

Extracted modules:
    cli_star.py   — star prompt (maybe_star)
    cli_demo.py   — template strings (BASELINE_TRAIN_PY, RESEARCH_RULES_TEMPLATE)
    cli_setup.py  — install/uninstall/upgrade helpers
    cli_pro.py    — pro license management
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from orze import __version__
from orze.cli_pro import pro_activate, pro_status, pro_deactivate
from orze.cli_setup import (
    do_uninstall, stop_running_instance, do_upgrade,
    do_init, do_check,
)
from orze.cli_star import maybe_star
from orze.core.config import load_project_config
from orze.hardware.gpu import detect_all_gpus

logger = logging.getLogger("orze")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False):
    """Configure logging with timestamps."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    maybe_star()

    parser = argparse.ArgumentParser(
        description="orze: GPU experiment orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m orze.cli                             # all GPUs, continuous
  python -m orze.cli -c orze.yaml --gpus 0,1     # with project config
  python -m orze.cli --once                      # one cycle then exit
  python -m orze.cli --report-only               # regenerate report
        """,
    )
    parser.add_argument("-V", "--version", action="version",
                        version=f"orze {__version__}")
    parser.add_argument("-c", "--config-file", type=str, default=None,
                        help="Path to orze.yaml project config")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (default: auto-detect)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Max training time in seconds")
    parser.add_argument("--poll", type=int, default=None,
                        help="Seconds between iterations")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit")
    parser.add_argument("--stop", action="store_true",
                        help="Gracefully stop a running orze instance")
    parser.add_argument("--restart", action="store_true",
                        help="Stop the running instance and start a new one")
    parser.add_argument("--disable", action="store_true",
                        help="Stop and persistently disable Orze (survives restarts)")
    parser.add_argument("--enable", action="store_true",
                        help="Remove persistent disable flag to allow Orze to run")
    parser.add_argument("--report-only", action="store_true",
                        help="Only regenerate report")
    parser.add_argument("--role-only", type=str, default=None, metavar="NAME",
                        help="Run a single agent role once and exit")
    parser.add_argument("--research-only", action="store_true",
                        help="Alias for --role-only research")
    parser.add_argument("--ideas-md", type=str, default=None,
                        help="Path to ideas markdown file")
    parser.add_argument("--base-config", type=str, default=None,
                        help="Path to base config YAML")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory for results")
    parser.add_argument("--train-script", type=str, default=None,
                        help="Training script to run per idea")
    parser.add_argument("--init", nargs="?", const="__ask__", default=None, metavar="PATH",
                        help="Initialize a new orze project (default: current directory)")
    parser.add_argument("--admin", action="store_true",
                        help="Launch admin panel instead of farm loop")
    parser.add_argument("--upgrade", action="store_true",
                        help="Upgrade orze to the latest version from PyPI")
    parser.add_argument("--check", action="store_true",
                        help="Validate config, check files, API keys, GPUs, .env — then exit")
    parser.add_argument("--uninstall", action="store_true",
                        help="Full uninstall: stop orze, remove runtime files, "
                             "pip uninstall — keeps only research results")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Debug logging")

    # --- subcommands ---
    subparsers = parser.add_subparsers(dest="command")

    # stop
    stop_parser = subparsers.add_parser(
        "stop", help="Stop orze: kill orchestrator + children, "
                     "disable watchdog, clean up GPUs")
    stop_parser.add_argument("-c", "--config-file", type=str, default=None,
                             help="Path to orze.yaml")
    stop_parser.add_argument("--timeout", type=int, default=60,
                             help="Timeout for child processes (default: 60)")
    stop_parser.add_argument("-v", "--verbose", action="store_true")

    # start
    start_parser = subparsers.add_parser(
        "start", help="Start orze as a background daemon")
    start_parser.add_argument("-c", "--config-file", type=str, default=None,
                              help="Path to orze.yaml")
    start_parser.add_argument("--gpus", type=str, default=None,
                              help="Comma-separated GPU IDs (default: auto-detect)")
    start_parser.add_argument("--foreground", action="store_true",
                              help="Run in foreground instead of daemonizing")
    start_parser.add_argument("-v", "--verbose", action="store_true")

    # restart
    restart_parser = subparsers.add_parser(
        "restart", help="Stop then start orze")
    restart_parser.add_argument("-c", "--config-file", type=str, default=None,
                                help="Path to orze.yaml")
    restart_parser.add_argument("--gpus", type=str, default=None,
                                help="Comma-separated GPU IDs (default: auto-detect)")
    restart_parser.add_argument("--timeout", type=int, default=60,
                                help="Timeout for child processes (default: 60)")
    restart_parser.add_argument("--foreground", action="store_true",
                                help="Run in foreground after restart")
    restart_parser.add_argument("-v", "--verbose", action="store_true")

    # service
    svc_parser = subparsers.add_parser("service", help="Manage orze watchdog service")
    svc_sub = svc_parser.add_subparsers(dest="service_action")

    svc_install = svc_sub.add_parser("install", help="Install watchdog service")
    svc_install.add_argument("-c", "--config-file", type=str, default="orze.yaml",
                             help="Path to orze.yaml")
    svc_install.add_argument("--method", choices=["auto", "crontab", "systemd"],
                             default="auto", help="Service method (default: auto)")
    svc_install.add_argument("--stall-threshold", type=int, default=1800,
                             help="Seconds before heartbeat considered stale (default: 1800)")

    svc_sub.add_parser("uninstall", help="Uninstall watchdog service")
    svc_sub.add_parser("status", help="Show watchdog service status")

    svc_logs = svc_sub.add_parser("logs", help="Show watchdog logs")
    svc_logs.add_argument("-n", type=int, default=50,
                          help="Number of log lines (default: 50)")

    # pro
    # reset
    reset_parser = subparsers.add_parser(
        "reset", help="Reset idea lake: purge failed/stale ideas for a fresh start")
    reset_parser.add_argument("-c", "--config-file", type=str, default=None)
    reset_parser.add_argument("--failed", action="store_true",
                              help="Purge all failed ideas")
    reset_parser.add_argument("--all", action="store_true",
                              help="Purge ALL non-completed ideas (queued + failed + partial)")
    reset_parser.add_argument("--full", action="store_true",
                              help="Wipe entire idea lake (backup created)")
    reset_parser.add_argument("-y", "--yes", action="store_true",
                              help="Skip confirmation prompt")

    pro_parser = subparsers.add_parser("pro", help="Manage orze-pro license")
    pro_sub = pro_parser.add_subparsers(dest="pro_action")
    pro_activate_parser = pro_sub.add_parser("activate", help="Activate orze-pro with a license key")
    pro_activate_parser.add_argument("key", nargs="?", default=None, help="License key (or enter interactively)")
    pro_sub.add_parser("status", help="Show orze-pro license status")
    pro_deactivate_parser = pro_sub.add_parser("deactivate", help="Remove saved license key")
    pro_deactivate_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    setup_logging(args.verbose)

    # --- subcommand dispatch ---
    command = getattr(args, "command", None)

    if command == "stop":
        from orze.lifecycle import do_stop
        cfg = load_project_config(args.config_file)
        do_stop(cfg, timeout=args.timeout)
        return

    if command == "start":
        from orze.lifecycle import do_start
        cfg = load_project_config(args.config_file)
        config_path = args.config_file or cfg.get("_config_path", "orze.yaml")
        do_start(cfg, foreground=args.foreground, config_path=config_path,
                 gpus=args.gpus)
        return

    if command == "restart":
        from orze.lifecycle import do_restart
        cfg = load_project_config(args.config_file)
        config_path = args.config_file or cfg.get("_config_path", "orze.yaml")
        do_restart(cfg, timeout=args.timeout, foreground=args.foreground,
                   config_path=config_path, gpus=args.gpus)
        return

    if command == "reset":
        import sqlite3, shutil
        cfg = load_project_config(args.config_file)
        db_path = Path(cfg.get("idea_lake_db") or Path(cfg.get("results_dir", "results")) / "idea_lake.db")
        if not db_path.exists():
            print("No idea_lake.db found.")
            return

        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()

        if args.full:
            if not args.yes:
                c.execute("SELECT COUNT(*) FROM ideas")
                total = c.fetchone()[0]
                resp = input(f"Wipe entire idea lake ({total} ideas)? Backup will be created. [y/N] ")
                if resp.lower() != "y":
                    print("Aborted.")
                    return
            backup = db_path.with_suffix(".db.bak")
            shutil.copy2(db_path, backup)
            c.execute("DELETE FROM ideas")
            print(f"Wiped {c.rowcount} ideas. Backup: {backup}")
        elif args.all:
            c.execute("DELETE FROM ideas WHERE status IN ('queued', 'failed', 'partial', 'running')")
            print(f"Purged {c.rowcount} non-completed ideas.")
        elif args.failed:
            c.execute("DELETE FROM ideas WHERE status = 'failed'")
            print(f"Purged {c.rowcount} failed ideas.")
        else:
            # Default: show status summary
            c.execute("SELECT status, COUNT(*) FROM ideas GROUP BY status")
            for row in c.fetchall():
                print(f"  {row[0]}: {row[1]}")
            print("\nUse --failed, --all, or --full to purge.")

        conn.commit()
        conn.close()

        # Also clear pause sentinel — stale failures shouldn't block research
        results_dir = Path(cfg.get("results_dir", "results"))
        pause_file = results_dir / ".pause_research"
        if pause_file.exists():
            pause_file.unlink()
            print("Cleared .pause_research sentinel.")

        return

    if command == "pro":
        action = getattr(args, "pro_action", None)
        if action == "activate":
            pro_activate(getattr(args, "key", None))
        elif action == "status":
            pro_status()
        elif action == "deactivate":
            pro_deactivate(force=getattr(args, "yes", False))
        else:
            parser.parse_args(["pro", "--help"])
        return

    if command == "service":
        action = getattr(args, "service_action", None)
        if action == "install":
            from orze.service.install import install
            install(args.config_file, method=args.method,
                    stall_threshold=args.stall_threshold)
        elif action == "uninstall":
            from orze.service.install import uninstall
            uninstall()
        elif action == "status":
            from orze.service.status import show_status
            show_status()
        elif action == "logs":
            from orze.service.status import show_logs
            show_logs(n=args.n)
        else:
            parser.parse_args(["service", "--help"])
        return

    # Load project config, then apply CLI overrides
    cfg = load_project_config(args.config_file)
    cfg["_config_path"] = args.config_file or "orze.yaml"  # stored for mode: research

    # --admin: launch web panel
    if args.admin:
        from orze.admin.server import run_admin
        run_admin(cfg)
        return

    # --upgrade: upgrade orze from PyPI (stops + restarts if running)
    if args.upgrade:
        do_upgrade(cfg)
        return

    # --uninstall: full cleanup, keep only research results
    if args.uninstall:
        do_uninstall(cfg)
        return

    # --init: initialize a new project
    if args.init is not None:
        do_init(args.init)
        return

    # --check: validate config and environment, then exit
    if args.check:
        do_check(cfg)
        return

    # Apply CLI overrides
    if args.timeout is not None:
        cfg["timeout"] = args.timeout
    if args.poll is not None:
        cfg["poll"] = args.poll
    if args.ideas_md:
        cfg["ideas_file"] = args.ideas_md
    if args.base_config:
        cfg["base_config"] = args.base_config
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.train_script:
        cfg["train_script"] = args.train_script

    # --stop
    if args.stop:
        import datetime
        from orze.core.fs import atomic_write
        stop_path = Path(cfg["results_dir"]) / ".orze_stop_all"
        atomic_write(stop_path,
                     f"kill {datetime.datetime.now().isoformat()}")
        print(f"Stop signal written to {stop_path}. "
              f"All nodes sharing this results directory will stop within "
              f"~30 seconds. Training, evaluation, and research processes "
              f"will be terminated (SIGTERM, then SIGKILL after 10s). "
              f"The sentinel is cleared automatically on next startup.")
        return

    # --restart: stop running instance, then continue to start a new one
    if args.restart:
        stop_running_instance(Path(cfg["results_dir"]))
        print("Starting new orze instance...")

    # --disable
    if args.disable:
        import datetime
        from orze.core.fs import atomic_write
        disable_path = Path(cfg["results_dir"]) / ".orze_disabled"
        atomic_write(disable_path, f"Disabled at {datetime.datetime.now().isoformat()}")
        print(f"Orze disabled. Remove {disable_path} to re-enable.")
        return

    # --enable
    if args.enable:
        disable_path = Path(cfg["results_dir"]) / ".orze_disabled"
        if disable_path.exists():
            disable_path.unlink()
            print("Orze re-enabled.")
        else:
            print("Orze was not disabled.")
        return

    # --report-only
    if args.report_only:
        from orze.core.ideas import parse_ideas
        from orze.reporting.leaderboard import update_report
        ideas = parse_ideas(cfg["ideas_file"])
        results_dir = Path(cfg["results_dir"])
        lake = None
        lake_path = Path(cfg["idea_lake_db"])
        if lake_path.exists():
            from orze.idea_lake import IdeaLake
            lake = IdeaLake(str(lake_path))
        update_report(results_dir, ideas, cfg, lake=lake)
        print("Report updated.")
        return

    # --research-only is an alias for --role-only research
    if args.research_only:
        args.role_only = "research"

    # Detect GPUs
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    else:
        gpu_ids = detect_all_gpus()

    if not gpu_ids:
        logger.error("No GPUs detected. Use --gpus to specify manually.")
        sys.exit(1)

    # Start admin panel in background thread (unless --role-only or --admin-off)
    if not args.role_only and not getattr(args, 'no_admin', False):
        try:
            import threading
            from orze.admin.server import run_admin as _run_admin_server
            admin_port = int(cfg.get("admin_port") or os.environ.get("ORZE_ADMIN_PORT", "8787"))

            def _admin_thread():
                try:
                    _run_admin_server(cfg, port=admin_port)
                except OSError as e:
                    if "address already in use" in str(e).lower():
                        logger.warning(
                            "Admin port %d already in use by another process. "
                            "Skipping admin panel to avoid killing another instance. "
                            "Set admin_port in orze.yaml to use a different port.",
                            admin_port)
                    else:
                        logger.warning("Admin panel failed to start: %s", e)
                except Exception as e:
                    logger.warning("Admin panel failed to start: %s", e)

            t = threading.Thread(target=_admin_thread, daemon=True)
            t.start()
            logger.info("Admin panel starting on http://0.0.0.0:%d", admin_port)
        except Exception as e:
            logger.warning("Could not start admin panel: %s", e)

    # Launch orchestrator
    from orze.engine.orchestrator import Orze
    orze = Orze(gpu_ids, cfg, once=args.once)

    if args.role_only:
        orze._run_role_once(args.role_only)
    else:
        orze.run()


if __name__ == "__main__":
    main()
