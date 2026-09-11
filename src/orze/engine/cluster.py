"""Multi-machine coordination for Orze.

Calling spec
------------
    from orze.engine.cluster import (
        check_cluster_versions, build_machine_status,
        check_stop_all, check_disabled, kill_orphans,
    )

    heartbeats, bad = check_cluster_versions(results_dir)  # -> (list, set)
    machines = build_machine_status(results_dir)            # -> list[dict]
    should_stop, kill_all = check_stop_all(results_dir)     # -> (bool, bool)
    disabled = check_disabled(results_dir)                  # -> bool
    kill_orphans(results_dir, cfg)                          # side-effect only

All functions are pure (no class state). Side-effects are limited to
filesystem reads and process signals.
"""

import logging
import os
import signal
from pathlib import Path

from orze.reporting.state import _read_all_heartbeats, check_heartbeat_versions

logger = logging.getLogger("orze")


def check_cluster_versions(results_dir: Path) -> tuple:
    """Check version compatibility with other nodes.

    Returns (heartbeats, incompatible_hosts) where incompatible_hosts
    is a set of hostnames with major-version mismatches.
    """
    heartbeats = _read_all_heartbeats(results_dir)
    incompatible_hosts = set(check_heartbeat_versions(heartbeats))
    return heartbeats, incompatible_hosts


def build_machine_status(results_dir: Path) -> list:
    """Build machine status from heartbeats for report notifications.

    Returns list of dicts with keys: host, gpus_busy, gpus_total, utilization.
    """
    heartbeats = _read_all_heartbeats(results_dir)
    machines = []
    for hb in heartbeats:
        host = hb.get("host", "unknown")
        active_list = hb.get("active", [])
        free_list = hb.get("free_gpus", [])
        gpus_busy = len(active_list)
        gpus_total = gpus_busy + len(free_list)
        util = round(gpus_busy / gpus_total * 100) if gpus_total else 0
        machines.append({
            "host": host,
            "gpus_busy": gpus_busy,
            "gpus_total": gpus_total,
            "utilization": util,
        })
    return machines


def check_stop_all(results_dir: Path) -> tuple:
    """Check for filesystem-based stop signal (.orze_stop_all).

    Returns (should_stop, kill_all). should_stop is True if the file
    exists; kill_all is True if the file content contains 'kill'.
    """
    stop_file = results_dir / ".orze_stop_all"
    if stop_file.exists():
        try:
            content = stop_file.read_text(encoding="utf-8").strip()
        except OSError:
            content = ""
        kill_all = "kill" in content.lower()
        logger.info("Found .orze_stop_all — shutting down (kill_all=%s)",
                    kill_all)
        return True, kill_all
    return False, False


def check_disabled(results_dir: Path) -> bool:
    """Check for persistent disable flag (.orze_disabled).

    Returns True if the file exists (Orze should not start).
    """
    disabled_file = results_dir / ".orze_disabled"
    if disabled_file.exists():
        msg = disabled_file.read_text(encoding="utf-8").strip()
        logger.error("Orze is DISABLED: %s", msg)
        logger.error("Remove %s to re-enable", disabled_file)
        return True
    return False


def kill_orphans(results_dir: Path, cfg: dict):
    """Compatibility entry point: inferred orphan ownership cannot authorize kill.

    Native owners stop through their retained, bound execution handles. A
    matching command, parent PID, claim's integer PID or GPU is not such a
    handle. Startup and periodic callers therefore never enumerate or signal
    host processes through this legacy fallback.
    """
    from orze.core.control_outcome import StopOutcome
    logger.debug("Skipping inferred orphan cleanup: owned execution handle required")
    return StopOutcome("hold", "owned_process_handle_required")
