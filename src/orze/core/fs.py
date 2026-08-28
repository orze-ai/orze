"""Filesystem utilities: locking, atomic writes, and helpers for shared/Lustre filesystems.

CALLING SPEC:
    _fs_lock(lock_dir: Path, stale_seconds: float = 600) -> bool
        Acquire a filesystem lock via atomic mkdir. Returns True if acquired.
        Auto-breaks stale locks by age or dead local PIDs.

    _fs_unlock(lock_dir: Path) -> None
        Release a filesystem lock (rmtree the lock dir). Silently ignores errors.

    atomic_write(path: Path, content: str) -> None
        Write content atomically via tmp+rename with fsync. Safe for Lustre
        shared filesystems where multiple nodes may read concurrently.

    atomic_create(path: Path, content: str) -> bool
        Publish complete content only when path does not already exist. Uses
        an atomic hard-link commit so concurrent claimers have one winner.

    locked_append(path: Path, content: str, lock_dir: Path,
                  stale_seconds: float = 60, after_append=None) -> bool
        Append and fsync only while holding the named filesystem lock. Returns
        false without writing when another owner holds the lock. An optional
        finalizer runs before unlock; failure truncates the append back.

    deep_get(obj: dict, dotpath: str, default=None) -> Any
        Get nested dict value by dot-separated path, e.g. 'a.b.c'.

    tail_file(path: Path, n_bytes: int = 4096) -> str
        Read the last n_bytes of a file. Returns '' on any error.
"""
import json
import logging
import os
import shutil
import socket
import stat as statlib
import time
import uuid
from pathlib import Path

logger = logging.getLogger("orze")

def _is_pid_alive(host: str, pid: int) -> bool:
    """Check if a process is still alive. For local host, use os.kill(0).
    For remote hosts, assume alive (let stale_seconds handle it)."""
    if host == socket.gethostname():
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            # EPERM — process exists but we can't signal it
            return True
        except OSError:
            # ESRCH — no such process
            return False
    # Remote host — can't check, fall through to stale_seconds timeout
    return False


def _fs_lock(lock_dir: Path, stale_seconds: float = 600) -> bool:
    """Acquire a filesystem lock via atomic mkdir.
    Returns True if acquired, False if held by another.
    Auto-breaks stale locks older than stale_seconds using atomic rename
    to avoid TOCTOU races between nodes.  On the local host, also breaks
    locks whose owning PID has died (regardless of age)."""
    try:
        lock_dir.mkdir(parents=True, exist_ok=False)
        meta = {"host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}
        (lock_dir / "lock.json").write_text(json.dumps(meta), encoding="utf-8")
        return True
    except FileExistsError:
        # A managed-role crash receipt is process authority, not a disposable
        # lock artifact. Only startup reconciliation on its owning host may
        # remove it after nonce-bound stable identities are proven stopped.
        role_receipt = lock_dir / "role-process.json"
        if role_receipt.exists() or role_receipt.is_symlink():
            logger.error(
                "Refusing lock takeover with unresolved managed role "
                "receipt: %s", lock_dir)
            return False
        # Check for stale lock
        try:
            meta_path = lock_dir / "lock.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    dir_age = time.time() - lock_dir.stat().st_mtime
                    if dir_age < 30:
                        return False
                    logger.warning(
                        "Breaking lock with unreadable metadata: %s (age %.0fs)",
                        lock_dir, dir_age)
                else:
                    lock_age = time.time() - meta.get("time", 0)
                    lock_host = meta.get("host", "")
                    lock_pid = meta.get("pid", 0)

                    pid_dead = (lock_host == socket.gethostname()
                                and lock_pid
                                and not _is_pid_alive(lock_host, lock_pid))
                    age_stale = lock_age > stale_seconds

                    if not (age_stale or pid_dead):
                        return False

                    if pid_dead:
                        logger.warning("Breaking dead-pid lock: %s (host=%s pid=%d)",
                                       lock_dir, lock_host, lock_pid)
                    else:
                        logger.warning("Breaking stale lock: %s (age %.0fs)",
                                       lock_dir, lock_age)
            else:
                # Orphaned lock dir with no lock.json — treat as stale
                # unless very recently created (< 30s grace period)
                dir_age = time.time() - lock_dir.stat().st_mtime
                if dir_age < 30:
                    return False
                logger.warning("Breaking orphaned lock (no lock.json): %s (age %.0fs)",
                               lock_dir, dir_age)

            # Atomic takeover: rename the stale lock dir to a unique name.
            # Only one node can succeed at this rename — the loser gets OSError.
            stale_name = lock_dir.with_name(
                f"{lock_dir.name}._stale_{uuid.uuid4().hex[:12]}"
            )
            try:
                os.rename(str(lock_dir), str(stale_name))
            except OSError:
                return False
            try:
                shutil.rmtree(stale_name)
            except OSError:
                pass
            try:
                lock_dir.mkdir(parents=True, exist_ok=False)
                new_meta = {"host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}
                (lock_dir / "lock.json").write_text(json.dumps(new_meta), encoding="utf-8")
                return True
            except FileExistsError:
                return False
        except Exception:
            pass
        return False

def atomic_write(path: Path, content: str):
    """Write content atomically via tmp+rename with fsync for Lustre safety."""
    import errno
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_host = "".join(c if c.isalnum() else "_" for c in socket.gethostname())
    tmp = path.with_name(
        f"{path.name}.{safe_host}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    # Write with explicit fsync before close so other Lustre clients see full content
    fd = os.open(
        str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    try:
        encoded = content.encode("utf-8")
        written = 0
        while written < len(encoded):
            written += os.write(fd, encoded[written:])
        os.fsync(fd)
    except OSError as e:
        os.close(fd)
        # Clean up the partial tmp file
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        if e.errno == errno.ENOSPC:
            logger.warning("atomic_write skipped (ENOSPC): %s", path)
            return
        raise
    finally:
        try:
            os.close(fd)
        except OSError:
            pass  # already closed in the except branch
    try:
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    # fsync parent directory so the rename is durable and visible on other nodes
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_create(path: Path, content: str) -> bool:
    """Atomically create one complete file without replacing a winner.

    The content is first fsynced to a unique file in the destination
    directory. ``link(2)`` then publishes it under ``path`` only if no entry
    exists there; that namespace operation is atomic across processes and
    hosts on Orze's supported shared filesystems. Readers therefore see either
    no file or the complete file, never the partial bytes of an O_EXCL writer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_host = "".join(
        c if c.isalnum() else "_" for c in socket.gethostname()
    )
    tmp = path.with_name(
        f".{path.name}.{safe_host}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    fd = os.open(
        str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    try:
        encoded = content.encode("utf-8")
        written = 0
        while written < len(encoded):
            written += os.write(fd, encoded[written:])
        os.fsync(fd)
    except Exception:
        try:
            os.close(fd)
        finally:
            tmp.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)

    published = False
    try:
        os.link(str(tmp), str(path))
        published = True
    except FileExistsError:
        return False
    finally:
        tmp.unlink(missing_ok=True)

    if published:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    return published


def locked_append(path: Path, content: str, lock_dir: Path,
                  stale_seconds: float = 60, after_append=None) -> bool:
    """Append durable text and finalize it under one acquired lock."""
    path = Path(path)
    lock_dir = Path(lock_dir)
    for candidate in (path, lock_dir):
        absolute = candidate.absolute()
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current = current / part
            if current.is_symlink():
                return False
    locked = _fs_lock(lock_dir, stale_seconds=stale_seconds)
    if not locked:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(path), flags, 0o644)
        try:
            metadata = os.fstat(fd)
            if (not statlib.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1):
                return False
            original_size = metadata.st_size
            encoded = content.encode("utf-8")
            written = 0
            while written < len(encoded):
                count = os.write(fd, encoded[written:])
                if count <= 0:
                    raise OSError("locked_append_short_write")
                written += count
            os.fsync(fd)
            if after_append is not None:
                if not callable(after_append):
                    raise TypeError("locked_append_finalizer_not_callable")
                try:
                    after_append()
                except BaseException:
                    os.ftruncate(fd, original_size)
                    os.fsync(fd)
                    raise
        finally:
            os.close(fd)
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return True
    finally:
        _fs_unlock(lock_dir)


def tail_file(path: Path, n_bytes: int = 4096) -> str:
    """Read last n_bytes of a file."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - n_bytes))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
def deep_get(obj: dict, dotpath: str, default=None):
    """Get nested dict value by dot path: 'a.b.c' -> obj[a][b][c]."""
    keys = dotpath.split(".")
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        else:
            return default
    return obj
def _fs_unlock(lock_dir: Path):
    """Release a filesystem lock."""
    try:
        shutil.rmtree(lock_dir)
    except Exception:
        pass
