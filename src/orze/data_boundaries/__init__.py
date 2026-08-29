"""Data boundary guardrails with fail-closed kernel isolation.

ORZE data_boundaries supports two modes, set independently per path in
orze.yaml. Pick the right one for your training script architecture:

    data_boundaries:
      forbidden_in_training:   # HARD BLOCK — kernel namespace isolation
        - /path/to/test/data
      watch_paths:             # AUDIT ONLY — log via builtins.open patch
        - /path/to/eval/data
      training_network: deny   # HARD BLOCK — private network namespace

Mode 1: forbidden_in_training (kernel-enforced)
  The orze launcher first proves namespace/mount capability without GPU
  visibility, then wraps training with:
    unshare -U --map-root-user -m bash -c "
      mount -t tmpfs <forbidden-directory>
      exec python -m orze.data_boundaries.wrap train.py ...
    "
  Every prior file rooted at <forbidden> is hidden at the kernel layer.
  Works against any library — pyarrow, h5py, tfrecord, lmdb, C
  extensions — because the block happens below the Python layer.
  Missing tools, unavailable or redirected paths, namespace failures, and
  mount failures reject before GPU telemetry. There is no Python-only fallback.

  USE WHEN: your training subprocess is pure training — it does not
  also evaluate on the forbidden path. If train and eval share a
  subprocess and both read the forbidden path, the namespace hides
  the path from eval too, breaking legitimate evaluation.

Mode 2: watch_paths (audit only)
  The orze launcher launches training via
  `python -m orze.data_boundaries.wrap ...` which monkey-patches
  builtins.open(). Every open() of a watched path is appended to
  ORZE_ACCESS_LOG as a `WATCH\\t<prefix>\\t<path>` line. No block.

  USE WHEN: your training subprocess also performs in-loop eval on
  the same path you want audited. The in-loop eval can still read
  eval files, but any read is logged so you can post-hoc check that
  the training loop (not just the eval loop) didn't touch the
  forbidden data.

  LIMITATION: builtins.open-based audit only catches Python-level
  reads. PyArrow memory_map, h5py, tfrecord, and direct os.open calls
  bypass this log. Use watch_paths for audit completeness on plain
  Python dataloaders; use forbidden_in_training (kernel mode) when
  you need coverage for binary/native dataloaders.

Both modes also populate ORZE_ACCESS_LOG for any builtins.open hits.

CALLING SPEC:
    activate() -> bool
        Reads ORZE_FORBIDDEN_PATHS / ORZE_WATCH_PATHS / ORZE_ACCESS_LOG
        env vars. Monkey-patches builtins.open() to check each read
        against the forbidden / watched lists. Returns True if patched,
        False if env vars unset (no-op).

    class OrzeLeakageError(RuntimeError)
        Raised when training code attempts to open a forbidden path at
        the Python layer. (Kernel-layer blocks raise FileNotFoundError.)

Env vars (set by launcher.py when data_boundaries is configured):
    ORZE_FORBIDDEN_PATHS  colon-separated realpath'd absolute prefixes
                          that training MUST NOT read. Match at the
                          Python layer = hard abort via OrzeLeakageError.
                          Also enforced at the kernel layer when the
                          launcher wraps training with unshare (Linux).
    ORZE_WATCH_PATHS      colon-separated realpath'd prefixes to log
                          (post-hoc audit without aborting).
    ORZE_ACCESS_LOG       file path to append watched/forbidden accesses
                          to. Tab-delimited: TAG\\tprefix\\tfull-path.

Scope limitations:
  - Path isolation cannot find undeclared aliases, hard links, copied samples,
    or contamination already present in a pretrained model.
  - ``training_network: inherit`` permits remote reads. Use ``deny`` to isolate
    training and all descendants in a private network namespace.
  - watch_paths is Python-level only; kernel mode covers all libraries
    but requires a separate subprocess for eval to avoid blocking
    legitimate test-set reads.
"""
import builtins
import os
from pathlib import Path
import stat
from typing import List, Optional

__all__ = [
    "OrzeLeakageError", "activate", "audit_training_access_log", "is_active",
]


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


def audit_training_access_log(
    idea_dir: str | os.PathLike,
    *,
    max_bytes: int = 16 * 1024 * 1024,
    max_entries: int = 100_000,
) -> dict:
    """Fail closed on watched/forbidden training reads without exposing paths."""
    path = Path(idea_dir) / "_access_log.tsv"
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {
            "status": "CLEAN",
            "log_present": False,
            "entries": 0,
            "watch_entries": 0,
            "forbidden_entries": 0,
        }
    except OSError:
        return {"status": "UNVERIFIED", "reason": "access_log_unavailable"}
    if (path.is_symlink() or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1 or not 0 <= info.st_size <= max_bytes):
        return {"status": "UNVERIFIED", "reason": "access_log_unsafe"}

    counts = {"WATCH": 0, "FORBIDDEN": 0}
    try:
        with path.open("rb") as handle:
            for index, raw in enumerate(handle, 1):
                if index > max_entries or len(raw) > 8192:
                    raise ValueError("access log limit exceeded")
                fields = raw.decode("utf-8").rstrip("\n").split("\t")
                if len(fields) != 3 or fields[0] not in counts:
                    raise ValueError("access log row invalid")
                tag, prefix, accessed = fields
                if (not os.path.isabs(prefix) or not os.path.isabs(accessed)
                        or not (accessed == prefix or accessed.startswith(
                            prefix.rstrip(os.sep) + os.sep))):
                    raise ValueError("access log path invalid")
                counts[tag] += 1
    except (OSError, UnicodeDecodeError, ValueError):
        return {"status": "UNVERIFIED", "reason": "access_log_invalid"}

    entries = counts["WATCH"] + counts["FORBIDDEN"]
    return {
        "status": "TAINTED" if entries else "CLEAN",
        "log_present": True,
        "entries": entries,
        "watch_entries": counts["WATCH"],
        "forbidden_entries": counts["FORBIDDEN"],
    }
