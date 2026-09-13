"""Bounded, actual-route deployment checks, never execution authority.

There is no capability cache or filesystem-name allowlist. Success only says
these operations worked now; the final protected rename and HOLD remain
mandatory. Probe failures cannot authorize cleanup of an existing owner.
"""
from __future__ import annotations

import os
from pathlib import Path
import secrets
import stat

from orze.engine.artifact_publication import _open_directory
from orze.engine.gc_tree import GCRefused, identity, rename_no_replace


class StoragePreflightError(ValueError):
    def __init__(self, reason, *, path=None, probe=None):
        super().__init__(reason)
        self.path = str(path) if path is not None else None
        self.probe = str(probe) if probe is not None else None


def _directories(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise StoragePreflightError('storage_route_invalid', path=path)
    result = []
    for item in reversed((path, *path.parents)):
        info = item.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise StoragePreflightError('storage_route_redirected', path=item)
        result.append((item, (info.st_dev, info.st_ino, info.st_mode)))
    return tuple(result)


def _verify_route(captured, fd):
    for path, expected in captured:
        info = path.lstat()
        if (info.st_dev, info.st_ino, info.st_mode) != expected:
            raise StoragePreflightError('storage_route_changed', path=path)
    info = os.fstat(fd)
    if (info.st_dev, info.st_ino, info.st_mode) != captured[-1][1]:
        raise StoragePreflightError('storage_route_changed', path=captured[-1][0])


class _Probe:
    def __init__(self, root, root_fd):
        self.root = root
        self.root_fd = root_fd
        self.name = '.orze-storage-probe-' + secrets.token_hex(16)
        self.path = root / self.name
        self.entries = {}
        self.fds = []
        self.fd = None
        self.root_identity = None

    def create(self):
        root_fd = self.root_fd
        os.mkdir(self.name, 0o700, dir_fd=root_fd)
        self.root_identity = identity(os.stat(self.name, dir_fd=root_fd, follow_symlinks=False))
        self.fd = os.open(self.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        self.fds.append(self.fd)
        if identity(os.fstat(self.fd)) != self.root_identity:
            raise StoragePreflightError('storage_probe_changed', probe=self.path)
        os.fsync(root_fd)

    def entry(self, parent, name, *, directory=False, raw=None):
        if directory:
            os.mkdir(name, 0o700, dir_fd=parent)
        else:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=parent)
            try:
                offset = 0
                while offset < len(raw):
                    count = os.write(fd, raw[offset:])
                    if count <= 0:
                        raise OSError('storage probe short write')
                    offset += count
                os.fsync(fd)
            finally:
                os.close(fd)
        value = identity(os.stat(name, dir_fd=parent, follow_symlinks=False))
        self.entries[parent, name] = value
        return value

    def directory(self, name):
        expected = self.entry(self.fd, name, directory=True)
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.fd)
        self.fds.append(fd)
        if identity(os.fstat(fd)) != expected:
            raise StoragePreflightError('storage_probe_changed', probe=self.path)
        return fd

    def check(self):
        actual = identity(os.stat(self.name, dir_fd=self.root_fd, follow_symlinks=False))
        if actual[:3] != self.root_identity[:3] or identity(os.fstat(self.fd))[:3] != actual[:3]:
            raise StoragePreflightError('storage_probe_changed', probe=self.path)
        for (parent, name), expected in self.entries.items():
            actual = identity(os.stat(name, dir_fd=parent, follow_symlinks=False))
            if (actual[:3] != expected[:3] if stat.S_ISDIR(expected[2]) else actual != expected):
                raise StoragePreflightError('storage_probe_changed', probe=self.path)

    def cleanup(self):
        # Only exact captured entries through owned directory FDs. No rmtree,
        # glob, stale-probe recovery, symlink following, or user path cleanup.
        self.check()
        for (parent, name), expected in reversed(tuple(self.entries.items())):
            actual = identity(os.stat(name, dir_fd=parent, follow_symlinks=False))
            if (actual[:3] != expected[:3] if stat.S_ISDIR(expected[2]) else actual != expected):
                raise StoragePreflightError('storage_probe_cleanup_unconfirmed', probe=self.path)
            if stat.S_ISDIR(expected[2]):
                os.rmdir(name, dir_fd=parent)
            else:
                os.unlink(name, dir_fd=parent)
            del self.entries[parent, name]
            os.fsync(parent)
        actual = identity(os.stat(self.name, dir_fd=self.root_fd, follow_symlinks=False))
        if actual[:3] != self.root_identity[:3]:
            raise StoragePreflightError('storage_probe_cleanup_unconfirmed', probe=self.path)
        os.rmdir(self.name, dir_fd=self.root_fd)
        os.fsync(self.root_fd)

    def close(self):
        for fd in reversed(self.fds):
            os.close(fd)
        self.fds.clear()


def require_atomic_rename_support(root):
    """Check an existing directory using four tiny real no-replace operations.

Unsupported or uncertain probes refuse, including cleanup/fsync failures.
Only private probe objects are removed; uncertain leftovers are named by the
exception for manual diagnosis and are not subsequent execution authority.
"""
    root = Path(root)
    probe = None
    fd = None
    failure = None
    try:
        captured = _directories(root)
        fd = _open_directory(root)
        _verify_route(captured, fd)
        probe = _Probe(root, fd)
        probe.create()
        for directory in (False, True):
            for collision in (False, True):
                name = 'case-' + str(int(directory)) + str(int(collision))
                parent = probe.directory(name)
                source = probe.entry(parent, 'source', directory=directory, raw=b'probe-source')
                target = (probe.entry(parent, 'target', directory=directory, raw=b'probe-target')
                          if collision else None)
                os.fsync(parent)
                _verify_route(captured, fd)
                probe.check()
                try:
                    rename_no_replace(probe.path / name / 'source', probe.path / name / 'target')
                except GCRefused as exc:
                    if not collision or str(exc) != 'gc_archive_destination_exists':
                        raise
                else:
                    # A successful collision would overwrite something. Even
                    # private probe damage must never report compatibility.
                    if collision:
                        raise StoragePreflightError('storage_collision_not_preserved', probe=probe.path)
                    del probe.entries[parent, 'source']
                    moved = identity(os.stat('target', dir_fd=parent, follow_symlinks=False))
                    if moved[:6] != source[:6]:
                        raise StoragePreflightError('storage_probe_changed', probe=probe.path)
                    probe.entries[parent, 'target'] = moved
                    try:
                        os.stat('source', dir_fd=parent, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        raise StoragePreflightError('storage_probe_not_detached', probe=probe.path)
                if collision:
                    if (identity(os.stat('source', dir_fd=parent, follow_symlinks=False)) != source
                            or identity(os.stat('target', dir_fd=parent, follow_symlinks=False)) != target):
                        raise StoragePreflightError('storage_collision_not_preserved', probe=probe.path)
                if not directory:
                    read_fd = os.open('target', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
                    try:
                        if os.read(read_fd, 64) != (b'probe-target' if collision else b'probe-source'):
                            raise StoragePreflightError('storage_probe_readback_unconfirmed', probe=probe.path)
                    finally:
                        os.close(read_fd)
                os.fsync(parent)
                _verify_route(captured, fd)
                probe.check()
    except BaseException as exc:
        failure = exc
    finally:
        if probe is not None:
            try:
                probe.cleanup()
                _verify_route(captured, fd)
            except BaseException as exc:
                failure = StoragePreflightError('storage_probe_cleanup_unconfirmed', path=root, probe=probe.path)
                failure.__cause__ = exc
            finally:
                probe.close()
        if fd is not None:
            os.close(fd)
    if failure is not None:
        if isinstance(failure, StoragePreflightError):
            raise failure
        if isinstance(failure, GCRefused) and str(failure) in {
                'gc_atomic_rename_unsupported', 'gc_atomic_rename_unavailable'}:
            raise StoragePreflightError('storage_atomic_rename_unsupported', path=root) from failure
        if not isinstance(failure, Exception):
            raise failure
        raise StoragePreflightError('storage_probe_unconfirmed', path=root,
                                    probe=probe.path if probe else None) from failure


def role_uses_atomic_rename(role_cfg):
    if not isinstance(role_cfg, dict) or role_cfg.get('enabled', True) is not True:
        return False
    mode = role_cfg.get('mode', 'script')
    field = {'script': 'script', 'claude': 'skills', 'research': 'backend'}.get(mode)
    return field is not None and bool(role_cfg.get(field))


def _ensure_route(root):
    # Only the explicitly configured administrative/storage directories, no
    # state relocation; traversal refuses redirects before creating children.
    try:
        fd = _open_directory(root, create=True)
        os.close(fd)
    except Exception as exc:
        raise StoragePreflightError('storage_route_unconfirmed', path=root) from exc
    require_atomic_rename_support(root)


def require_role_storage(cfg, role_cfg):
    if not role_uses_atomic_rename(role_cfg):
        return
    control = cfg.get('_orze_dir')
    if not isinstance(control, (str, Path)) or not str(control):
        raise StoragePreflightError('storage_role_route_missing')
    _ensure_route(Path(control).absolute() / 'locks')


def require_deployment_storage(cfg, results_dir):
    """Actual legacy role/emergency-GC routes; CPU-only caller skips this gate."""
    roles = cfg.get('roles') or {}
    if not isinstance(roles, dict):
        raise StoragePreflightError('storage_roles_invalid')
    role = next((value for value in roles.values() if role_uses_atomic_rename(value)), None)
    if role is not None:
        require_role_storage(cfg, role)
    gc = cfg.get('gc') or {}
    if not isinstance(gc, dict):
        raise StoragePreflightError('storage_gc_invalid')
    # This is the actual emergency-GC operation invoked by phases.py. Direct
    # results/archive collection independently validates its own actual roots.
    if gc.get('enabled') and gc.get('checkpoints_dir'):
        from orze.engine.gc_safety import gc_scope
        scope = gc_scope(results_dir, cfg, checkpoints_dir=gc['checkpoints_dir'])
        _ensure_route(scope.checkpoints_dir)
