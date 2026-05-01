"""Orchestrator state persistence and host heartbeat management.

CALLING SPEC:
    load_state(results_dir: Path) -> dict
        Load per-host orchestrator state from checkpoint. Returns dict with
        keys: iteration, failure_counts, roles. Handles legacy migration
        and corrupt files gracefully.

    save_state(results_dir: Path, state: dict) -> None
        Atomically write orchestrator state for restart recovery. State is
        per-host (filename includes hostname).

    write_host_heartbeat(results_dir: Path, hostname: str, active, free_gpus: list) -> None
        Write per-host heartbeat JSON with active processes, free GPUs, and
        GPU details. 'active' is a dict of TrainingProcess objects keyed by
        GPU id.

    write_status_json(results_dir: Path, iteration: int,
                      active: Dict[int, TrainingProcess], free_gpus: List[int],
                      queue_depth: int, completed_count: int, failed_count: int,
                      skipped_count: int, top_results: list, cfg: dict,
                      role_states: Optional[dict] = None) -> None
        Write machine-readable status.json merging heartbeats from all hosts.
        Includes per-role status, disk info, and combined multi-machine view.

    _read_all_heartbeats(results_dir: Path, stale_seconds: float = 900) -> list
        Read _host_*.json heartbeat files, keep freshest per host, clean up
        stale/superseded files. Returns list of heartbeat dicts.

    check_heartbeat_versions(heartbeats: list) -> List[str]
        Check version compatibility across cluster heartbeats. Returns list
        of hostnames with incompatible (major version mismatch) orze versions.
"""
import os
import json
import logging
import socket
import time
import datetime
import platform
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from orze import __version__
from orze.core.fs import atomic_write
from orze.hardware.gpu import _query_gpu_details

TrainingProcess = Any  # type alias to avoid circular import

logger = logging.getLogger("orze")


def _parse_version(v: str) -> Tuple[int, ...]:
    """Parse a semver string into a tuple of ints. Returns (0,) on failure."""
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except (ValueError, AttributeError):
        return (0,)


def _heartbeats_dir(results_dir: Path) -> Path:
    """Resolve .orze/heartbeats/ from results_dir (.orze/ sits next to orze_results/)."""
    d = Path(results_dir).parent / ".orze" / "heartbeats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_host_heartbeat(results_dir: Path, hostname: str,
                         active, free_gpus: list):
    """Write per-host heartbeat file with active processes and free GPUs."""
    pid = os.getpid()
    now = time.time()
    heartbeat = {
        "host": hostname,
        "pid": pid,
        "timestamp": datetime.datetime.now().isoformat(),
        "epoch": now,
        "orze_version": __version__,
        "active": [
            {
                "idea_id": tp.idea_id,
                "gpu": tp.gpu,
                "elapsed_min": round((now - tp.start_time) / 60, 1),
            }
            for tp in active.values()
        ],
        "free_gpus": free_gpus,
        "gpu_info": _query_gpu_details(),
        "os": f"{platform.system()} {platform.release()}",
    }
    atomic_write(_heartbeats_dir(results_dir) / f"{hostname}_{pid}.json",
                 json.dumps(heartbeat, indent=2))


def _read_all_heartbeats(results_dir: Path,
                         stale_seconds: float = 900) -> list:
    """Read heartbeat files, keeping only the freshest per host.
    Removes superseded heartbeat files (older duplicates for the same host).
    Only removes stale files after 2x stale_seconds to avoid premature cleanup
    during long iterations on Lustre."""
    now = time.time()
    # Collect all valid heartbeats, keyed by host
    by_host: dict = {}  # host -> (epoch, hb_dict, hb_path)
    superseded_paths = []
    stale_paths = []
    hb_dir = _heartbeats_dir(results_dir)
    # New layout: .orze/heartbeats/<host>_<pid>.json
    # Legacy layout: results_dir/_host_<host>_<pid>.json (read-only fallback)
    hb_paths = list(hb_dir.glob("*.json")) + list(results_dir.glob("_host_*.json"))
    for hb_path in hb_paths:
        try:
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
            age = now - hb.get("epoch", 0)
            host = hb.get("host", "unknown")
            epoch = hb.get("epoch", 0)
            if age > stale_seconds * 2:
                # Truly dead — safe to remove
                stale_paths.append(hb_path)
                continue
            prev = by_host.get(host)
            if prev is None or epoch > prev[0]:
                if prev:
                    superseded_paths.append(prev[2])
                by_host[host] = (epoch, hb, hb_path)
            else:
                superseded_paths.append(hb_path)
        except Exception:
            try:
                if now - hb_path.stat().st_mtime > stale_seconds * 2:
                    stale_paths.append(hb_path)
            except OSError:
                pass
            continue
    # Clean up superseded (same-host duplicates) and truly stale files
    for p in superseded_paths + stale_paths:
        try:
            p.unlink()
        except OSError:
            pass
    return [v[1] for v in by_host.values()]


def check_heartbeat_versions(heartbeats: list) -> List[str]:
    """Check version compatibility across cluster heartbeats.
    Returns list of incompatible hostnames (major version mismatch).
    Logs warnings for any version mismatches."""
    my_ver = _parse_version(__version__)
    incompatible = []
    for hb in heartbeats:
        peer_ver_str = hb.get("orze_version")
        if not peer_ver_str:
            continue  # old node without version — skip silently
        peer_host = hb.get("host", "unknown")
        peer_ver = _parse_version(peer_ver_str)
        if peer_ver == my_ver:
            continue
        if peer_ver[0] != my_ver[0]:
            logger.warning("INCOMPATIBLE: %s runs orze v%s (major %d) vs our v%s (major %d). "
                           "Refusing coordination with this node.",
                           peer_host, peer_ver_str, peer_ver[0],
                           __version__, my_ver[0])
            incompatible.append(peer_host)
        else:
            logger.warning("Version mismatch: %s runs orze v%s, we run v%s",
                           peer_host, peer_ver_str, __version__)
    return incompatible


# ---------------------------------------------------------------------------
# Role health derivation.
#
# Post-mortem (2026-04): the higher-order steering stack (engineer,
# professor, thinker, code_evolution) silently produced 35-byte stubs
# for 5+ days because the orze-claude shim couldn't fall back to the
# API key on auth-failure signals. The dashboard and report.md kept
# rendering as healthy because role-state surfacing only counted
# *runs*, not *outcomes*. This block derives a coarse health verdict
# from existing state + on-disk cycle logs, so any future silent role
# death is visible at a glance.
# ---------------------------------------------------------------------------

# Threshold constants — see derive_role_health docstring.
_HEALTH_LOCKOUT_FAILURES = 5
_HEALTH_LOCKOUT_COOLDOWN_S = 3600
_HEALTH_DEGRADED_LOG_WINDOW = 5
_HEALTH_DEGRADED_TINY_BYTES = 100


def _recent_role_cycle_logs(orze_dir: Path, role_name: str,
                            limit: int = _HEALTH_DEGRADED_LOG_WINDOW) -> list:
    """Return up to ``limit`` most recent cycle log file paths for a role.

    Cycle logs live at ``.orze/logs/<role>/`` (one file per cycle, named
    by timestamp). Returns newest-first. Missing dir → empty list.
    """
    log_dir = orze_dir / "logs" / role_name
    if not log_dir.is_dir():
        return []
    try:
        files = [p for p in log_dir.iterdir() if p.is_file()]
    except OSError:
        return []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def derive_role_health(role_name: str, role_state: dict,
                       orze_dir: Optional[Path]) -> dict:
    """Classify a role as HEALTHY / DEGRADED / LOCKED_OUT.

    Verdict (first-match wins):
      - LOCKED_OUT: ``consecutive_failures >= 5`` OR
                    ``cooldown_override > 3600``  (1h)
      - DEGRADED:   last 5 cycle logs are byte-identical
                    OR all <= 100 bytes (covers the silent-stub case
                    that motivated this fix)
      - HEALTHY:    everything else

    Returns dict with keys: status, last_run_age_min,
    consecutive_failures, cooldown_override_s.
    """
    now = time.time()
    last_run = role_state.get("last_run_time", 0.0) or 0.0
    last_run_age_min = (
        round((now - last_run) / 60, 1) if last_run > 0 else None
    )
    cf = int(role_state.get("consecutive_failures",
                            role_state.get("consecutive_errors", 0)) or 0)
    co = float(role_state.get("cooldown_override", 0) or 0)

    status = "HEALTHY"
    if cf >= _HEALTH_LOCKOUT_FAILURES or co > _HEALTH_LOCKOUT_COOLDOWN_S:
        status = "LOCKED_OUT"
    elif orze_dir is not None:
        recent = _recent_role_cycle_logs(orze_dir, role_name)
        if len(recent) >= _HEALTH_DEGRADED_LOG_WINDOW:
            try:
                blobs = [p.read_bytes() for p in recent]
                if all(len(b) <= _HEALTH_DEGRADED_TINY_BYTES for b in blobs):
                    status = "DEGRADED"
                elif len(set(blobs)) == 1:
                    status = "DEGRADED"
            except OSError:
                pass

    return {
        "status": status,
        "last_run_age_min": last_run_age_min,
        "consecutive_failures": cf,
        "cooldown_override_s": co,
    }


def build_role_health_block(cfg: dict,
                            role_states: dict,
                            orze_dir: Optional[Path]) -> dict:
    """Build {role_name -> health-dict} for every configured role."""
    out: dict = {}
    for rname in (cfg.get("roles") or {}):
        out[rname] = derive_role_health(
            rname, role_states.get(rname, {}) or {}, orze_dir,
        )
    return out


def write_status_json(results_dir: Path, iteration: int,
                      active: Dict[int, TrainingProcess],
                      free_gpus: List[int], queue_depth: int,
                      completed_count: int, failed_count: int,
                      skipped_count: int, top_results: list,
                      cfg: dict,
                      role_states: Optional[dict] = None):
    """Write machine-readable status.json for LLM agents.
    Merges heartbeats from all hosts for a combined multi-machine view."""
    disk_free_gb = 0.0
    try:
        usage = shutil.disk_usage(results_dir)
        disk_free_gb = usage.free / (1024 ** 3)
    except Exception:
        pass

    now = time.time()

    # Merge active processes from all hosts
    heartbeats = _read_all_heartbeats(results_dir)
    all_active = []
    free_gpus_by_host = {}
    for hb in heartbeats:
        host = hb.get("host", "unknown")
        for a in hb.get("active", []):
            a["host"] = host
            all_active.append(a)
        free_gpus_by_host[host] = hb.get("free_gpus", [])

    # Build per-role status
    role_states = role_states or {}
    roles_cfg = cfg.get("roles") or {}
    roles_status = {}
    for rname in roles_cfg:
        rs = role_states.get(rname, {})
        last_run = rs.get("last_run_time", 0.0)
        roles_status[rname] = {
            "enabled": True,
            "cycles": rs.get("cycles", 0),
            "last_run_min_ago": (
                round((now - last_run) / 60, 1) if last_run > 0 else None
            ),
        }

    # Backward compat: flat research_* keys
    research_rs = role_states.get("research", {})
    research_last = research_rs.get("last_run_time", 0.0)

    # Role-health surface (post-mortem fix for 2026-04 silent campaign):
    # classify each role as HEALTHY/DEGRADED/LOCKED_OUT so a human glancing
    # at status.json sees brain-death even when the leaderboard ticks.
    orze_dir_str = cfg.get("_orze_dir")
    orze_dir = Path(orze_dir_str) if orze_dir_str else None
    role_health = build_role_health_block(cfg, role_states, orze_dir)

    hostname = socket.gethostname()
    status = {
        "timestamp": datetime.datetime.now().isoformat(),
        "host": hostname,
        "iteration": iteration,
        "active": all_active,
        "free_gpus": free_gpus,
        "free_gpus_by_host": free_gpus_by_host,
        "queue_depth": queue_depth,
        "completed": completed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "disk_free_gb": round(disk_free_gb, 1),
        "top_results": top_results[:10],
        "roles": roles_status,
        "role_health": role_health,
        "research_enabled": "research" in roles_cfg,
        "research_cycles": research_rs.get("cycles", 0),
        "last_research_min_ago": (
            round((now - research_last) / 60, 1) if research_last > 0
            else None
        ),
    }

    atomic_write(results_dir / "status.json", json.dumps(status, indent=2))


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------


def load_state(results_dir: Path) -> dict:
    """Load orchestrator state from checkpoint (per-host).
    Falls back to legacy .orze_state.json for backward compat.
    Migrates legacy flat research_* keys into roles dict."""
    hostname = socket.gethostname()
    path = results_dir / f".orze_state_{hostname}.json"

    if not path.exists():
        legacy = results_dir / ".orze_state.json"
        if legacy.exists():
            path = legacy

    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            logger.warning("Corrupt state file, starting fresh")
            return {"iteration": 0, "failure_counts": {}, "roles": {}}

        # Migrate legacy flat research state into roles dict
        if "roles" not in state and "research_cycles" in state:
            state["roles"] = {
                "research": {
                    "cycles": state.pop("research_cycles", 0),
                    "last_run_time": state.pop("last_research_time", 0.0),
                }
            }
        state.setdefault("roles", {})
        return state
    return {"iteration": 0, "failure_counts": {}, "roles": {}}


def save_state(results_dir: Path, state: dict):
    """Save orchestrator state for restart recovery (per-host)."""
    hostname = socket.gethostname()
    atomic_write(results_dir / f".orze_state_{hostname}.json",
                 json.dumps(state, indent=2))

