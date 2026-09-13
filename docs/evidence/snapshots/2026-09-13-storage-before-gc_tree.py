"""No-follow GC staging and reclamation; never copy-and-delete across devices.

Only the coordinator may authorize a candidate. This module captures bounded
metadata, atomically detaches to same-root quarantine, and reclaims outside the
task guard. Quarantine intents are GC diagnostics, not execution observations.
"""
from dataclasses import dataclass
import ctypes
import errno
import json
import os
from pathlib import Path
import stat
import uuid

from orze.engine.artifact_publication import _open_directory
from orze.engine.attempt_effect_receipts import _publish, _read, _sync


class GCRefused(ValueError):
    pass


def identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def plain_directory(path, *, missing=False):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise GCRefused("gc_path_invalid")
    for current in reversed((path, *path.parents)):
        try:
            info = current.lstat()
        except FileNotFoundError:
            if missing:
                continue
            raise GCRefused("gc_directory_missing")
        if not stat.S_ISDIR(info.st_mode):
            raise GCRefused("gc_directory_redirected")


@dataclass(frozen=True)
class TreeSnapshot:
    path: Path
    root_identity: tuple
    entries: tuple
    size: int


def snapshot(path, expected):
    """Bounded stat walk, outside task guard; no source contents are read."""
    plain_directory(path.parent)
    entries, size = [], 0

    def visit(current, relative, depth):
        nonlocal size
        if depth > 64 or len(entries) >= 16384:
            raise GCRefused("gc_tree_limit")
        info = current.lstat()
        ident = identity(info)
        if relative == "." and ident != expected:
            raise GCRefused("gc_candidate_replaced")
        if not stat.S_ISDIR(info.st_mode) and not (stat.S_ISREG(info.st_mode) and info.st_nlink == 1):
            raise GCRefused("gc_tree_redirected")
        entries.append((relative, ident))
        if stat.S_ISDIR(info.st_mode):
            fd = _open_directory(current)
            try:
                if identity(os.fstat(fd)) != ident:
                    raise GCRefused("gc_candidate_replaced")
                with os.scandir(fd) as iterator:
                    names = []
                    for item in iterator:
                        if len(entries) + len(names) >= 16384:
                            raise GCRefused("gc_tree_limit")
                        names.append(item.name)
                    names.sort()
                for name in names:
                    visit(current / name, name if relative == "." else relative + "/" + name, depth + 1)
                if identity(os.fstat(fd)) != ident:
                    raise GCRefused("gc_tree_changed")
            finally:
                os.close(fd)
        else:
            size += info.st_size
    visit(path, ".", 0)
    return TreeSnapshot(path, expected, tuple(entries), size)


def rename_no_replace(source, destination):
    """Linux atomic no-replace; unavailable platforms refuse, never fallback."""
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise GCRefused("gc_atomic_rename_unavailable")
    function.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    source_fd = _open_directory(source.parent)
    try:
        target_fd = _open_directory(destination.parent)
        try:
            if os.fstat(source_fd).st_dev != os.fstat(target_fd).st_dev:
                raise GCRefused("gc_cross_device_refused")
            if function(source_fd, os.fsencode(source.name), target_fd,
                        os.fsencode(destination.name), 1) != 0:
                error = ctypes.get_errno()
                if error == errno.EXDEV:
                    raise GCRefused("gc_cross_device_refused")
                if error in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
                    raise GCRefused("gc_atomic_rename_unsupported")
                if error == errno.EEXIST:
                    raise GCRefused("gc_archive_destination_exists")
                raise OSError(error, os.strerror(error))
        finally:
            os.close(target_fd)
    finally:
        os.close(source_fd)


def quarantine_root(root, task_id):
    return root / "_orze_gc_quarantine" / task_id


def require_no_pending(root, task_id):
    directory = quarantine_root(root, task_id)
    plain_directory(directory, missing=True)
    if not directory.exists():
        return
    fd = _open_directory(directory)
    try:
        children = []
        with os.scandir(fd) as iterator:
            for item in iterator:
                if len(children) >= 1024:
                    raise GCRefused("gc_quarantine_limit")
                children.append(directory / item.name)
    finally:
        os.close(fd)
    for child in children:
        plain_directory(child)
        try:
            done = json.loads(_read(child / "completed.json"))
        except Exception as exc:
            raise GCRefused("gc_quarantine_hold") from exc
        if done not in ({"status": "deleted"}, {"status": "archived"}):
            raise GCRefused("gc_quarantine_hold")


def prepare_quarantine(root, task_id, tree, destination):
    base = quarantine_root(root, task_id)
    require_no_pending(root, task_id)
    fd = _open_directory(base, create=True)
    try:
        token = uuid.uuid4().hex
        os.mkdir(token, 0o700, dir_fd=fd)
        os.fsync(fd)
    finally:
        os.close(fd)
    directory = base / token
    payload = {"schema": 1, "task_id": task_id, "source": str(tree.path),
               "identity": list(tree.root_identity),
               "destination": str(destination) if destination else None}
    _publish(directory / "intent.json", json.dumps(payload, sort_keys=True,
             separators=(",", ":")).encode())
    return directory


def verify_detached(tree, path):
    root = path.lstat()
    # Rename changes the root inode ctime, but not its identity/content shape.
    if identity(root)[:6] != tree.root_identity[:6]:
        raise GCRefused("gc_quarantine_identity_changed")
    current = snapshot(path, identity(root))
    if len(current.entries) != len(tree.entries):
        raise GCRefused("gc_quarantine_changed")
    for (name, expected), (actual_name, actual) in zip(tree.entries, current.entries):
        if name != actual_name or (expected if name != "." else expected[:6]) != (
                actual if name != "." else actual[:6]):
            raise GCRefused("gc_quarantine_changed")


def reclaim(tree, directory, destination=None):
    """After confirmed detach, operate only on captured quarantine identities."""
    content = directory / "content"
    verify_detached(tree, content)
    if destination is not None:
        fd = _open_directory(destination.parent, create=True)
        os.close(fd)
        rename_no_replace(content, destination)
        _sync(destination.parent)
        _sync(directory)
        verify_detached(tree, destination)
        status = "archived"
    else:
        for relative, expected in reversed(tree.entries):
            path = content if relative == "." else content / relative
            parent = _open_directory(path.parent)
            try:
                info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                if identity(info)[:3] != expected[:3]:
                    raise GCRefused("gc_quarantine_identity_changed")
                if stat.S_ISDIR(info.st_mode):
                    os.rmdir(path.name, dir_fd=parent)
                else:
                    if relative != "." and identity(info) != expected:
                        raise GCRefused("gc_quarantine_file_changed")
                    os.unlink(path.name, dir_fd=parent)
                os.fsync(parent)
            finally:
                os.close(parent)
        status = "deleted"
    _publish(directory / "completed.json", json.dumps({"status": status},
             sort_keys=True, separators=(",", ":")).encode())
