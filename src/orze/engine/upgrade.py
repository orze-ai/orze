"""Auto-upgrade logic for Orze.

Calling spec
------------
    from orze.engine.upgrade import UpgradeManager

    mgr = UpgradeManager(results_dir, cfg)

    mgr.pending                                     # -> Optional[str]
    mgr.check_pypi()                                # rate-limited PyPI poll
    mgr.check_sentinel()                            # react to .orze_upgrade file
    mgr.do_upgrade(kill_and_save_fn, remove_pid_fn) # pip install + restart

    kill_and_save_fn: Callable[[], None]
        Orchestrator callback that SIGTERMs children, waits, saves state,
        and closes the idea lake.  Called right before os.execv.

    remove_pid_fn: Callable[[], None]
        Removes the PID file before exec-restart.

Helper:
    parse_version("1.2.3") -> (1, 2, 3)
"""

import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from orze import __version__
from orze.core.fs import _fs_lock, _fs_unlock
from orze.reporting.notifications import notify

logger = logging.getLogger("orze")


def parse_version(s: str) -> tuple:
    """Parse a dotted version string into a comparable tuple of ints."""
    try:
        return tuple(int(x) for x in s.split(".")[:3])
    except (ValueError, AttributeError):
        return (0,)


class UpgradeManager:
    """Owns auto-upgrade state and logic, decoupled from orchestrator internals."""

    def __init__(self, results_dir: Path, cfg: dict):
        self._results_dir = results_dir
        self._cfg = cfg
        self._pending_upgrade: Optional[str] = None
        self._last_upgrade_check: float = 0.0

    @property
    def pending(self) -> Optional[str]:
        return self._pending_upgrade

    @pending.setter
    def pending(self, value: Optional[str]):
        self._pending_upgrade = value

    def check_pypi(self):
        """Check PyPI for a newer orze version. Rate-limited. Never raises."""
        au_cfg = self._cfg.get("auto_upgrade")
        if not au_cfg:
            return
        if isinstance(au_cfg, bool):
            interval = 3600
        else:
            interval = int(au_cfg.get("interval", 3600))

        if time.time() - self._last_upgrade_check < interval:
            return
        self._last_upgrade_check = time.time()

        try:
            import urllib.request
            req = urllib.request.Request(
                "https://pypi.org/pypi/orze/json",
                headers={"User-Agent": f"orze/{__version__}"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            latest = data["info"]["version"]
        except Exception as exc:
            logger.warning("Auto-upgrade: PyPI check failed: %s", exc)
            return

        if parse_version(latest) > parse_version(__version__):
            if self._pending_upgrade != latest:
                logger.info("Auto-upgrade: v%s available (current v%s)",
                            latest, __version__)
            self._pending_upgrade = latest
        else:
            self._pending_upgrade = None

    def check_sentinel(self):
        """Check if another node wrote .orze_upgrade sentinel.

        If the target version is newer, trigger upgrade immediately.
        """
        sentinel = self._results_dir / ".orze_upgrade"
        if not sentinel.exists():
            return
        try:
            target = sentinel.read_text(encoding="utf-8").strip()
        except Exception:
            return

        if parse_version(target) <= parse_version(__version__):
            try:
                sentinel.unlink(missing_ok=True)
            except OSError:
                pass
            return

        logger.info("Auto-upgrade: sentinel found — another node upgraded to v%s, restarting...", target)
        self._pending_upgrade = target

    def do_upgrade(self, kill_and_save_fn, remove_pid_fn):
        """Install pending upgrade, kill everything, and restart via os.execv.

        Args:
            kill_and_save_fn: callback that kills child processes, saves state,
                              and closes the idea lake.
            remove_pid_fn: callback that removes the PID file.
        """
        target = self._pending_upgrade
        if not target:
            return
        logger.info("Auto-upgrade: installing orze==%s (current v%s)...",
                     target, __version__)

        upgrade_lock = self._results_dir / "_upgrade_lock"
        if not _fs_lock(upgrade_lock, stale_seconds=300):
            logger.info("Auto-upgrade: another process is already upgrading, skipping")
            self._pending_upgrade = None
            return

        # After acquiring lock, check if another process already completed
        # this upgrade (our in-memory __version__ is stale but pip is current)
        try:
            from importlib.metadata import version as _pkg_version
            installed = _pkg_version("orze")
        except Exception:
            installed = __version__

        if parse_version(installed) >= parse_version(target):
            logger.info("Auto-upgrade: v%s already installed, restarting without notification", installed)
            _fs_unlock(upgrade_lock)
            self._pending_upgrade = None
            try:
                (self._results_dir / ".orze_shutdown").write_text(
                    f"auto-upgrade restart to v{installed}", encoding="utf-8")
            except OSError:
                pass
            self._restart_in_place(kill_and_save_fn, remove_pid_fn)
            return

        # Write shutdown sentinel so watchdog won't respawn during restart
        try:
            (self._results_dir / ".orze_shutdown").write_text(
                f"auto-upgrade to v{target}", encoding="utf-8")
        except OSError:
            pass

        try:
            notify("upgrading", {
                "host": socket.gethostname(),
                "from_version": __version__,
                "to_version": target,
                "message": f"Upgrading v{__version__} -> v{target}, restarting",
            }, self._cfg)
        except Exception:
            pass

        # Upgrade orze from PyPI
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", f"orze=={target}",
             "--quiet"],
            capture_output=True, text=True,
        )

        # Upgrade orze-pro from private PyPI (if installed + key available)
        try:
            import orze_pro
            from orze.extensions import _find_pro_key
            pro_key = _find_pro_key()
            if pro_key:
                pro_result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "orze-pro",
                     "--upgrade", "--quiet",
                     "--extra-index-url",
                     f"https://__token__:{pro_key}@pypi.orze.ai/simple/"],
                    capture_output=True, text=True,
                )
                if pro_result.returncode == 0:
                    logger.info("Auto-upgrade: orze-pro upgraded from private PyPI")
                else:
                    logger.warning("Auto-upgrade: orze-pro upgrade failed: %s",
                                   pro_result.stderr[:200])
        except ImportError:
            pass
        if result.returncode != 0:
            logger.error("Auto-upgrade pip install failed (rc=%d): %s",
                         result.returncode, result.stderr[:500])
            self._pending_upgrade = None
            _fs_unlock(upgrade_lock)
            try:
                (self._results_dir / ".orze_shutdown").unlink(missing_ok=True)
            except OSError:
                pass
            return

        # Signal other nodes sharing this results_dir to restart
        try:
            (self._results_dir / ".orze_upgrade").write_text(
                target, encoding="utf-8")
        except Exception:
            pass

        _fs_unlock(upgrade_lock)
        self._restart_in_place(kill_and_save_fn, remove_pid_fn)

    def _restart_in_place(self, kill_and_save_fn, remove_pid_fn):
        """Kill children, save state, and replace this process via os.execv."""
        logger.info("Auto-upgrade: killing active processes and restarting...")

        kill_and_save_fn()
        remove_pid_fn()

        config_path = str(self._cfg.get("_config_path", "orze.yaml"))
        logger.info("Auto-upgrade: restarting (config: %s)", config_path)
        try:
            os.execv(sys.executable,
                     [sys.executable, "-m", "orze.cli", "-c", config_path])
        except OSError as exc:
            logger.error("Auto-upgrade: os.execv failed: %s — restart manually", exc)
            sys.exit(1)



# ---------------------------------------------------------------------------
# Upgrade cleanup — scrubs stale state files on version change
# (merged from upgrade_cleanup.py in v4.0)
# ---------------------------------------------------------------------------

import datetime as _datetime
from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
    version as _pkg_version,
)


STAMP_FILENAME = ".orze_version"

# Files that must be scrubbed on version change. Scope: things that
# encode transient / one-shot state the new version cannot safely
# inherit. Experiment outputs (``idea-*/``, ``report.md``,
# ``idea_lake.db``, ``_retrospection.txt``) are NOT listed and must
# never be touched.
_GARBAGE_FILES = (
    ".pause_research",
    "_fsm_state.json",
    "_fsm_activity.jsonl",
    "_fsm_scheduling.json",
    "_fsm_circuit_breakers.json",
    "_fsm_injected_roles.json",
)

# Glob patterns scrubbed alongside the exact-name list. Each matched
# path is additionally filtered via ``_is_live_trigger`` so we don't
# sweep up manually-archived history (``_trigger_bug_fixer.archived.*``)
# or backups.
_GARBAGE_GLOBS = (
    "_trigger_*",
)


def _is_live_trigger(path: Path) -> bool:
    """Only live one-shot trigger files qualify for cleanup.

    A live trigger is named ``_trigger_<role>`` — no extra suffix, no
    dot-separated archival tag. Any file that carries a ``.archived``,
    ``.bak``, timestamped suffix, etc. is a record somebody decided to
    keep, and must not be deleted.
    """
    if "." in path.name:
        return False
    # ``_trigger_<role>`` → two underscore-separated components.
    parts = path.name.split("_")
    # parts looks like ['', 'trigger', 'role'...] after leading underscore
    if len(parts) < 3 or parts[0] != "" or parts[1] != "trigger":
        return False
    return True


def _current_versions() -> dict:
    """Return {"orze": "x.y.z", "orze_pro": "a.b.c" | None}."""
    out = {"orze": None, "orze_pro": None}
    try:
        out["orze"] = _pkg_version("orze")
    except _PackageNotFoundError:
        pass
    try:
        out["orze_pro"] = _pkg_version("orze-pro")
    except _PackageNotFoundError:
        pass
    return out


def _read_stamp(stamp_path: Path) -> Optional[dict]:
    """Read the version stamp. Returns None if absent or unreadable.

    A corrupt stamp is treated identically to a missing stamp — the
    next write overwrites it and cleanup does not fire (we have no
    reliable "from" version to compare against).
    """
    if not stamp_path.exists():
        return None
    try:
        data = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("upgrade_cleanup: stamp at %s unreadable (%s); "
                       "treating as first run", stamp_path, e)
        return None
    if not isinstance(data, dict):
        return None
    return data


def _write_stamp(stamp_path: Path, versions: dict) -> None:
    """Atomic write of the version stamp. Swallows + logs OSError."""
    payload = {
        "orze": versions.get("orze"),
        "orze_pro": versions.get("orze_pro"),
        "stamped_at": _datetime.datetime.utcnow().isoformat(timespec="seconds"),
    }
    tmp = stamp_path.with_suffix(stamp_path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(stamp_path)
    except OSError as e:
        logger.warning("upgrade_cleanup: failed to write stamp %s: %s",
                       stamp_path, e)


def _same_versions(a: Optional[dict], b: dict) -> bool:
    """Compare stamp dicts on the two version fields only. ``stamped_at``
    and any future additive metadata are ignored."""
    if a is None:
        return False
    return (a.get("orze") == b.get("orze")
            and a.get("orze_pro") == b.get("orze_pro"))


def _delete_garbage(results_dir: Path) -> list:
    """Delete every listed garbage file/glob. Returns the names deleted.

    Each unlink is wrapped individually — a single unlinkable file
    must not block cleanup of the rest.
    """
    cleaned: list = []
    for name in _GARBAGE_FILES:
        path = results_dir / name
        if not path.exists():
            continue
        try:
            path.unlink()
            cleaned.append(name)
        except OSError as e:
            logger.warning("upgrade_cleanup: could not remove %s: %s",
                           path, e)
    for pattern in _GARBAGE_GLOBS:
        for path in sorted(results_dir.glob(pattern)):
            if not _is_live_trigger(path):
                continue
            try:
                path.unlink()
                cleaned.append(path.name)
            except OSError as e:
                logger.warning("upgrade_cleanup: could not remove %s: %s",
                               path, e)
    return cleaned


def check_and_clean(results_dir: Path) -> dict:
    """See module docstring."""
    current = _current_versions()
    result = {
        "upgraded": False,
        "from": None,
        "to": {"orze": current["orze"], "orze_pro": current["orze_pro"]},
        "cleaned": [],
    }

    # No results dir yet — nothing to stamp or clean. Startup will
    # create the dir and this check will run cleanly on the next boot.
    if not results_dir.exists():
        return result

    stamp_path = results_dir / STAMP_FILENAME
    previous = _read_stamp(stamp_path)
    result["from"] = previous

    # First run (no prior stamp) or corrupt stamp: just record the
    # current versions. We intentionally do NOT scrub — we have no
    # evidence an upgrade happened, and blowing away state on every
    # first boot would make orze feel unreliable.
    if previous is None:
        _write_stamp(stamp_path, current)
        return result

    if _same_versions(previous, current):
        return result

    # Version change detected (upgrade, downgrade, or orze-pro
    # install/removal). Scrub, then stamp.
    cleaned = _delete_garbage(results_dir)
    _write_stamp(stamp_path, current)
    result["upgraded"] = True
    result["cleaned"] = cleaned

    prev_desc = f"orze={previous.get('orze')} orze_pro={previous.get('orze_pro')}"
    curr_desc = f"orze={current.get('orze')} orze_pro={current.get('orze_pro')}"
    if cleaned:
        logger.info("orze upgrade detected (%s → %s); cleaned %d stale "
                    "file(s): %s", prev_desc, curr_desc, len(cleaned),
                    ", ".join(cleaned))
    else:
        logger.info("orze upgrade detected (%s → %s); no stale files to "
                    "clean", prev_desc, curr_desc)
    return result
