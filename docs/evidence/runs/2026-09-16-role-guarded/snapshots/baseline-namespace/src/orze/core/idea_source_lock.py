"""Conservative filesystem ownership for proposal-source critical sections.

CALLING SPEC:
    with idea_source_lock(lock_dir) as lease:
        A non-None immutable lease owns this source namespace until exit.
    idea_source_lock_owned(lease) -> bool
        Recheck the exact directory identity and immutable owner metadata.

A persistent namespace marker is published before mkdir. Generic age-based
locks cannot take over this namespace, including the mkdir/metadata gap.
Existing owners (even dead local or unknown remote ones) are never reclaimed
automatically: append/finalizer outcome may be uncertain. Operator recovery
must first resolve that uncertainty. This is not cross-host exactly-once work.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import stat
import time
import uuid


logger = logging.getLogger("orze")
_PROTOCOL = "orze-idea-source-lock-v1"
_MARKER_BYTES = (_PROTOCOL + "\n").encode("ascii")
_MAX_METADATA_BYTES = 4096


class SourceLockInDoubt(OSError):
    """A source operation could not restore safety; retain its owner directory."""


@dataclass(frozen=True)
class SourceLockLease:
    lock_dir: Path
    owner_nonce: str
    device: int
    inode: int
    metadata_sha256: str


def _marker_path(lock_dir: Path) -> Path:
    return lock_dir.with_name(lock_dir.name + ".source-lock")


def _unredirected(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            return False
    return bool(absolute.name) and absolute.name not in (".", "..")


def _read_regular(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_size > limit):
            raise OSError("idea_source_metadata_invalid")
        chunks, size = [], 0
        while size <= limit:
            chunk = os.read(fd, limit + 1 - size)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        if size > limit:
            raise OSError("idea_source_metadata_oversized")
        return b"".join(chunks)
    finally:
        os.close(fd)


def idea_source_lock_protected(lock_dir: Path) -> bool:
    """A marker only denies generic lock ownership; it never grants any."""
    lock_dir = Path(lock_dir)
    try:
        marker = _marker_path(lock_dir)
        # Invalid/dangling markers still deny generic takeover. The source
        # protocol separately validates a real, single-link marker's bytes.
        if marker.exists() or marker.is_symlink():
            return True
        metadata = json.loads(_read_regular(lock_dir / "lock.json", _MAX_METADATA_BYTES))
        return isinstance(metadata, dict) and metadata.get("protocol") == _PROTOCOL
    except (OSError, ValueError, TypeError):
        return False


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _acquire(lock_dir: Path) -> SourceLockLease | None:
    lock_dir = Path(lock_dir).absolute()
    try:
        marker = _marker_path(lock_dir)
        if not _unredirected(lock_dir) or not _unredirected(marker):
            return None
        # The namespace declaration precedes public lock directory creation.
        # Its persistence is intentional: subsequent owners use this protocol.
        from orze.core.fs import atomic_create
        atomic_create(marker, _MARKER_BYTES.decode("ascii"))
        if _read_regular(marker, len(_MARKER_BYTES)) != _MARKER_BYTES:
            return None
        try:
            lock_dir.mkdir()
        except FileExistsError:
            return None
        info = lock_dir.stat(follow_symlinks=False)
        nonce = uuid.uuid4().hex
        metadata = {
            "protocol": _PROTOCOL, "owner_nonce": nonce,
            "host": socket.gethostname(), "pid": os.getpid(), "time": time.time(),
        }
        encoded = (json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                   + "\n").encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(lock_dir / "lock.json", flags, 0o600)
        try:
            remaining = memoryview(encoded)
            while remaining:
                count = os.write(fd, remaining)
                if count <= 0:
                    raise OSError("idea_source_metadata_short_write")
                remaining = remaining[count:]
            os.fsync(fd)
        finally:
            os.close(fd)
        _sync_directory(lock_dir)
        _sync_directory(lock_dir.parent)
        lease = SourceLockLease(lock_dir, nonce, info.st_dev, info.st_ino,
                                hashlib.sha256(encoded).hexdigest())
        return lease if idea_source_lock_owned(lease) else None
    except (OSError, ValueError, TypeError) as exc:
        # Do not erase a partially initialized ownership directory. Its
        # outcome needs explicit recovery, not an age-triggered second writer.
        logger.warning("Idea source lock unavailable (%s)", type(exc).__name__)
        return None


def idea_source_lock_owned(lease: SourceLockLease | None) -> bool:
    if not isinstance(lease, SourceLockLease):
        return False
    try:
        if not _unredirected(lease.lock_dir):
            return False
        info = lease.lock_dir.stat(follow_symlinks=False)
        if (not stat.S_ISDIR(info.st_mode)
                or (info.st_dev, info.st_ino) != (lease.device, lease.inode)):
            return False
        if _read_regular(_marker_path(lease.lock_dir), len(_MARKER_BYTES)) != _MARKER_BYTES:
            return False
        raw = _read_regular(lease.lock_dir / "lock.json", _MAX_METADATA_BYTES)
        metadata = json.loads(raw)
        return (hashlib.sha256(raw).hexdigest() == lease.metadata_sha256
                and metadata.get("protocol") == _PROTOCOL
                and metadata.get("owner_nonce") == lease.owner_nonce)
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _release(lease: SourceLockLease) -> bool:
    if not idea_source_lock_owned(lease):
        return False
    try:
        # Do not recursively remove unexpected files or a replacement owner.
        if {entry.name for entry in lease.lock_dir.iterdir()} != {"lock.json"}:
            return False
        if not idea_source_lock_owned(lease):
            return False
        (lease.lock_dir / "lock.json").unlink()
        info = lease.lock_dir.stat(follow_symlinks=False)
        if (info.st_dev, info.st_ino) != (lease.device, lease.inode):
            return False
        lease.lock_dir.rmdir()
        _sync_directory(lease.lock_dir.parent)
        return True
    except OSError:
        return False


@contextmanager
def idea_source_lock(lock_dir: Path):
    """Acquire once; an existing source owner is operationally uncertain."""
    lease = _acquire(lock_dir)
    failed = False
    in_doubt = False
    try:
        yield lease
    except BaseException as exc:
        failed = True
        in_doubt = isinstance(exc, SourceLockInDoubt)
        raise
    finally:
        if lease is not None and in_doubt:
            logger.error("Idea source operation outcome uncertain; retaining owner for recovery")
        elif lease is not None and not _release(lease):
            logger.warning("Idea source lock ownership/release unconfirmed")
            if not failed:
                raise OSError("idea_source_lock_ownership_lost")
