"""Data boundary guardrails — catch data leakage by policing file I/O.

CALLING SPEC:
    activate() -> bool
        Reads ORZE_FORBIDDEN_PATHS / ORZE_WATCH_PATHS / ORZE_ACCESS_LOG env
        vars. Monkey-patches builtins.open() to check file paths against a
        forbidden-prefix list; raises OrzeLeakageError on match. Every
        matching forbidden/watched path is appended to ORZE_ACCESS_LOG.
        Returns True if patched, False if env vars unset (no-op).

    class OrzeLeakageError(RuntimeError)
        Raised when training code attempts to open a forbidden path.

Env vars (set by orze launcher via data_boundaries config):
    ORZE_FORBIDDEN_PATHS  colon-separated absolute path prefixes that
                          training MUST NOT read. Match = hard abort.
    ORZE_WATCH_PATHS      colon-separated prefixes to log (post-hoc audit
                          without aborting). Subset of FORBIDDEN by default.
    ORZE_ACCESS_LOG       file path to append watched accesses to.

Design notes:
  - Patches only builtins.open. Every disk read in Python ultimately hits
    it (PyArrow memory_map, HuggingFace datasets, soundfile, torchaudio).
  - String-prefix match is O(P) per open call; P is small (<20 in practice).
  - Opt-in: no env vars = no patching. Existing training scripts unchanged.
  - Does NOT patch network I/O. A user fetching test data over HTTP at
    runtime is not caught — document that as a scope limitation.
  - Path normalization uses os.path.realpath (resolves symlinks), so a
    symlinked alias for a forbidden dir still trips the guardrail.
"""
import builtins
import os
from typing import List, Optional

__all__ = ["OrzeLeakageError", "activate", "is_active"]


class OrzeLeakageError(RuntimeError):
    """Training tried to read a path declared as forbidden for training."""


_FORBIDDEN: List[str] = []
_WATCH: List[str] = []
_ACCESS_LOG: Optional[str] = None
_ACTIVE: bool = False
_REAL_OPEN = builtins.open


def _parse_paths(env_val: str) -> List[str]:
    """Split colon-separated paths, resolve symlinks, drop empties."""
    if not env_val:
        return []
    out = []
    for p in env_val.split(":"):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(os.path.realpath(p))
        except Exception:
            out.append(p)
    return out


def _match_any_prefix(path: str, prefixes: List[str]) -> Optional[str]:
    """Return the matching prefix or None."""
    for pref in prefixes:
        if path == pref or path.startswith(pref.rstrip("/") + "/"):
            return pref
    return None


def _log_access(path: str, matched_prefix: str, forbidden: bool) -> None:
    """Append a single line to the access log. Best-effort, never raises."""
    if not _ACCESS_LOG:
        return
    try:
        tag = "FORBIDDEN" if forbidden else "WATCH"
        line = f"{tag}\t{matched_prefix}\t{path}\n"
        # Use the real open to avoid recursion into our own patch.
        with _REAL_OPEN(_ACCESS_LOG, "a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception:
        pass


def _check(file_arg):
    """Core check: normalize path, match forbidden/watch, raise or log."""
    # Fast path: non-string (e.g. file descriptors) skip check
    if not isinstance(file_arg, (str, bytes, os.PathLike)):
        return
    try:
        path = os.fspath(file_arg)
        if isinstance(path, bytes):
            path = path.decode("utf-8", errors="replace")
        path = os.path.realpath(path)
    except Exception:
        return

    if _FORBIDDEN:
        hit = _match_any_prefix(path, _FORBIDDEN)
        if hit is not None:
            _log_access(path, hit, forbidden=True)
            raise OrzeLeakageError(
                f"[orze data_boundaries] Training process attempted to read "
                f"forbidden path: {path}\n"
                f"  matched prefix: {hit}\n"
                f"  This is the data-leakage guardrail. If this path is "
                f"actually a training split (not a test split), update "
                f"orze.yaml data_boundaries.forbidden_in_training to exclude it."
            )

    if _WATCH:
        hit = _match_any_prefix(path, _WATCH)
        if hit is not None:
            _log_access(path, hit, forbidden=False)


def _patched_open(file, *args, **kwargs):
    _check(file)
    return _REAL_OPEN(file, *args, **kwargs)


def activate() -> bool:
    """Activate the guardrails from env vars. Idempotent. Returns True if on."""
    global _FORBIDDEN, _WATCH, _ACCESS_LOG, _ACTIVE
    if _ACTIVE:
        return True

    _FORBIDDEN = _parse_paths(os.environ.get("ORZE_FORBIDDEN_PATHS", ""))
    _WATCH = _parse_paths(os.environ.get("ORZE_WATCH_PATHS", ""))
    _ACCESS_LOG = os.environ.get("ORZE_ACCESS_LOG") or None

    if not (_FORBIDDEN or _WATCH):
        return False

    builtins.open = _patched_open
    _ACTIVE = True
    return True


def is_active() -> bool:
    return _ACTIVE
