"""Clean up stale state files when orze (or orze-pro) is upgraded.

Background: some state files under ``results/`` are intentionally
persistent across restarts (e.g. ``.pause_research`` — retrospection
writes it, and an active FSM is expected to clear it via state
transitions). When orze or orze-pro is upgraded while those files exist,
the new version can end up stuck on state the old version produced —
a pause sentinel that no FSM transition will ever clear, an FSM state
JSON whose schema has changed, a one-shot ``_trigger_*`` signal that
was already consumed by the previous binary.

On upgrade we scrub a small, explicit set of known-stale files so the
new version starts from a clean slate. Per-experiment results, idea
lake, reports and anything substantive are left untouched.

CALLING SPEC:
    check_and_clean(results_dir: Path) -> dict
        Run once at startup. Compares the installed ``orze`` and
        ``orze-pro`` versions against the version stamp at
        ``results/.orze_version`` and, on mismatch, deletes the known
        garbage files/globs and writes a fresh stamp.

        Returns: {
            "upgraded": bool,           # True only when an upgrade (or
                                        # downgrade, or first-time
                                        # orze-pro install) was detected
            "from": dict | None,        # previous stamp (None on first run)
            "to": dict,                 # the stamp just written
            "cleaned": list[str],       # file names that were removed
        }

        Never raises — all filesystem errors are logged and turned into
        empty cleanup. Startup must not be gated on this module.
"""
from __future__ import annotations

import datetime
import json
import logging
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orze")

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
    except PackageNotFoundError:
        pass
    try:
        out["orze_pro"] = _pkg_version("orze-pro")
    except PackageNotFoundError:
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
        "stamped_at": datetime.datetime.utcnow().isoformat(timespec="seconds"),
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
