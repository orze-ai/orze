"""Round-2 E1: ``orze admin reset-role-state`` implementation.

Replaces the project-local ``scripts/reset_local_role_state.sh``
workaround. Two modes:

1. Local (default): clear circuit-breaker state in this host's
   ``.orze_state_<host>.json`` directly. Works with the daemon
   running — load_state() re-reads the file on each iteration so the
   change is picked up live.

2. ``--all-hosts``: drop a marker file (``.orze_reset_role_state``) on
   the shared results dir. Every host's daemon checks for the marker
   on each iteration (see ``check_reset_role_state_marker`` in
   ``orze.engine.cluster``); when seen, the host clears its own state
   file and removes its per-host claim from the marker. The marker
   self-deletes after every claim is gone.

Reuses the same filesystem-coordination idiom as ``.orze_stop_all`` so
the multi-host semantics stay familiar to operators.
"""
from __future__ import annotations

import json
import logging
import socket
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orze")


_MARKER_FILENAME = ".orze_reset_role_state"


def _state_files(results_dir: Path) -> list:
    """Return all per-host state files in results_dir."""
    return sorted(results_dir.glob(".orze_state_*.json"))


def _clear_state_file(state_path: Path, role: Optional[str]) -> list:
    """Clear cooldown_override + consecutive_failures for one role
    (or all roles when ``role is None``). Returns list of cleared role
    names."""
    if not state_path.exists():
        return []
    try:
        s = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    cleared = []
    roles = s.get("roles") or {}
    targets = [role] if role else list(roles.keys())
    for r in targets:
        rs = roles.get(r)
        if not isinstance(rs, dict):
            continue
        co = rs.get("cooldown_override", 0) or 0
        cf = rs.get("consecutive_failures", 0) or 0
        ce = rs.get("consecutive_errors", 0) or 0
        if co > 0 or cf > 0 or ce > 0:
            rs.pop("cooldown_override", None)
            rs["consecutive_failures"] = 0
            rs["consecutive_errors"] = 0
            cleared.append(r)
    if cleared:
        try:
            state_path.write_text(json.dumps(s, indent=2),
                                  encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to write %s: %s", state_path, e)
    return cleared


def write_all_hosts_marker(results_dir: Path,
                           role: Optional[str]) -> Path:
    """Drop the .orze_reset_role_state marker on results_dir.

    Marker content lists every host that's expected to claim it (one
    line per peer in ``.orze_state_<host>.json``). Each daemon, on
    seeing the marker, clears its own state then removes its hostname
    line. When the line set becomes empty the marker self-deletes.
    """
    hosts = []
    for sf in _state_files(results_dir):
        # filename: .orze_state_<host>.json
        stem = sf.stem  # .orze_state_<host>
        if stem.startswith(".orze_state_"):
            hosts.append(stem[len(".orze_state_"):])
    if not hosts:
        # No prior state — nothing to clear; still drop the marker so a
        # late-joining daemon picks it up.
        hosts = [socket.gethostname()]
    payload = {
        "created_at": time.time(),
        "role": role,  # None = all roles
        "hosts_pending": sorted(set(hosts)),
    }
    marker = results_dir / _MARKER_FILENAME
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker


def consume_marker_on_this_host(results_dir: Path) -> Optional[list]:
    """Daemon-side hook (one call per iteration). Returns list of
    cleared roles when this host had work to do, or None if no marker
    exists. Self-deletes the marker once every host has claimed it.
    """
    marker = results_dir / _MARKER_FILENAME
    if not marker.exists():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    host = socket.gethostname()
    pending = list(data.get("hosts_pending") or [])
    if host not in pending:
        return None
    role = data.get("role")
    state_path = results_dir / f".orze_state_{host}.json"
    cleared = _clear_state_file(state_path, role)
    pending.remove(host)
    data["hosts_pending"] = pending
    try:
        if pending:
            marker.write_text(json.dumps(data, indent=2),
                              encoding="utf-8")
        else:
            marker.unlink(missing_ok=True)
    except OSError:
        pass
    if cleared:
        logger.warning(
            "reset-role-state marker consumed on %s — cleared roles: %s",
            host, ", ".join(cleared))
    return cleared


def reset_role_state(cfg: dict, *, role: Optional[str] = None,
                     all_hosts: bool = False,
                     force: bool = False) -> int:
    results_dir = Path(cfg.get("results_dir", "orze_results"))
    if not results_dir.is_dir():
        print(f"orze admin reset-role-state: results_dir {results_dir} "
              f"does not exist")
        return 1

    if all_hosts:
        marker = write_all_hosts_marker(results_dir, role)
        print(f"Wrote reset marker: {marker}")
        print("Every host's daemon will clear its own role state on its "
              "next iteration; the marker self-deletes after the last "
              "host claims it.")
        return 0

    # Local mode — clear this host's state file in place.
    host = socket.gethostname()
    state_path = results_dir / f".orze_state_{host}.json"
    cleared = _clear_state_file(state_path, role)
    if not cleared:
        print(f"No stale state to clear in {state_path}")
        return 0
    print(f"Cleared role state on {host}: {', '.join(cleared)}")
    return 0
