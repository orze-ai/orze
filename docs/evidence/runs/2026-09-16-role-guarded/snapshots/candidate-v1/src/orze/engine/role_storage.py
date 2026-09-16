"""Owned role locks on storage without no-replace directory rename.

Explicit guarded namespaces have no age/PID takeover. Acquisition and release
use a non-stealable sibling transition directory; interrupted transitions need
operator recovery. A process-local lease never grants process-stop authority.
"""
from dataclasses import dataclass
import hashlib
import itertools
import os
from pathlib import Path
import secrets
import socket
import threading
import time

from orze.engine.artifact_publication import _open_directory
from orze.engine.attempt_effect_receipts import _encoded, _publish, _read, _sync
from orze.engine.gc_tree import identity, plain_directory


class RoleStorageHOLD(RuntimeError):
    pass


_GUARD = threading.RLock()
_LEASES = {}


def marker_path(lock):
    return lock.with_name(lock.name + '.role-lock')


def transition_path(lock):
    return lock.with_name('.' + lock.name + '.role-transition')


def _route(path):
    plain_directory(path)
    return tuple((part, identity(part.lstat())[:3]) for part in (path, *path.parents))


def _check_route(route):
    for path, captured in route:
        if identity(path.lstat())[:3] != captured:
            raise RoleStorageHOLD('role_storage_route_changed')


def _read_pinned(path):
    before = identity(path.lstat())
    raw = _read(path)
    if identity(path.lstat()) != before:
        raise RoleStorageHOLD('role_storage_file_changed')
    return before, raw


def _check_file(path, captured):
    if _read_pinned(path) != captured:
        raise RoleStorageHOLD('role_storage_file_changed')


@dataclass(frozen=True)
class GuardedRoleLease:
    lock_dir: Path
    lock_identity: tuple
    lock_file: tuple
    marker: tuple
    route: tuple
    pid: int
    nonce: str


def _namespace(lease):
    _check_route(lease.route)
    _check_file(marker_path(lease.lock_dir), lease.marker)


def _registered(lease):
    with _GUARD:
        record = _LEASES.get(id(lease))
        if (type(lease) is not GuardedRoleLease or lease.pid != os.getpid()
                or record is None or record[0] is not lease or record[2]):
            raise RoleStorageHOLD('role_storage_lease_unavailable')
        return record


def require_guarded_role(lease):
    _registered(lease)
    _namespace(lease)
    if identity(lease.lock_dir.lstat())[:3] != lease.lock_identity:
        raise RoleStorageHOLD('role_storage_lock_changed')
    _check_file(lease.lock_dir / 'lock.json', lease.lock_file)


class _Transition:
    def __init__(self, lock, route, marker, phase, evidence=None):
        self.lock, self.route, self.marker = lock, route, marker
        self.path = transition_path(lock)
        _check_route(route)
        _check_file(marker_path(lock), marker)
        self.path.mkdir(mode=0o700)
        self.identity = identity(self.path.lstat())[:3]
        _sync(lock.parent)
        _publish(self.path / 'transition.json', _encoded({
            'version': 1, 'phase': phase, 'lock_dir': str(lock),
            'nonce': secrets.token_hex(32), 'host': socket.gethostname(),
            'pid': os.getpid(),
            'evidence': evidence,
        }))
        self.file = _read_pinned(self.path / 'transition.json')
        self.check()

    def check(self):
        _check_route(self.route)
        _check_file(marker_path(self.lock), self.marker)
        if identity(self.path.lstat())[:3] != self.identity:
            raise RoleStorageHOLD('role_storage_transition_changed')
        _check_file(self.path / 'transition.json', self.file)

    def finish(self):
        self.check()
        if set(itertools.islice(self.path.iterdir(), 2)) != {self.path / 'transition.json'}:
            raise RoleStorageHOLD('role_storage_transition_contents_changed')
        fd = _open_directory(self.path)
        try:
            if identity(os.fstat(fd))[:3] != self.identity:
                raise RoleStorageHOLD('role_storage_transition_changed')
            self.check()
            os.unlink('transition.json', dir_fd=fd)
            os.fsync(fd)
            if identity(self.path.lstat())[:3] != self.identity:
                raise RoleStorageHOLD('role_storage_transition_changed')
            self.path.rmdir()
            _sync(self.lock.parent)
            _check_route(self.route)
        finally:
            os.close(fd)


def acquire_guarded_role(lock_dir):
    """Return this process's exclusive lease, or None for an occupied slot."""
    lock = Path(lock_dir).absolute()
    if lock.name in ('', '.', '..') or '..' in lock.parts:
        raise RoleStorageHOLD('role_storage_path_invalid')
    route = _route(lock.parent)
    marker = marker_path(lock)
    raw = _encoded({'version': 1, 'protocol': 'orze-guarded-role-v1', 'lock_dir': str(lock)})
    # Never convert an existing legacy lock into a new protocol's ownership.
    if not marker.exists() and not marker.is_symlink():
        if lock.exists() or lock.is_symlink():
            return None
        try:
            _publish(marker, raw)
        except FileExistsError:
            pass
    pinned = _read_pinned(marker)
    if pinned[1] != raw:
        raise RoleStorageHOLD('role_storage_namespace_invalid')
    try:
        transition = _Transition(lock, route, pinned, 'acquire')
    except FileExistsError:
        return None
    try:
        transition.check()
        try:
            lock.mkdir(mode=0o700)
        except FileExistsError:
            transition.finish()
            return None
        captured = identity(lock.lstat())[:3]
        _sync(lock.parent)
        nonce = secrets.token_hex(32)
        _publish(lock / 'lock.json', _encoded({
            'host': socket.gethostname(), 'pid': os.getpid(), 'time': time.time(),
            'protocol': 'orze-guarded-role-v1', 'owner_nonce': nonce,
        }))
        lease = GuardedRoleLease(lock, captured, _read_pinned(lock / 'lock.json'),
                                 pinned, route, os.getpid(), nonce)
        with _GUARD:
            _LEASES[id(lease)] = [lease, None, False]
        require_guarded_role(lease)
        transition.finish()
        return lease
    except BaseException:
        # Retain every created object. No recursive cleanup or stale retry.
        raise


def bind_guarded_role(lease, owner):
    require_guarded_role(lease)
    if owner.lock_dir != lease.lock_dir:
        raise RoleStorageHOLD('role_storage_owner_route_changed')
    with _GUARD:
        record = _registered(lease)
        if record[1] is not None:
            raise RoleStorageHOLD('role_storage_owner_already_bound')
        record[1] = owner


def _remove_owned(lease, files):
    require_guarded_role(lease)
    transition = _Transition(lease.lock_dir, lease.route, lease.marker, 'release', {
        'lock_identity': list(lease.lock_identity),
        'files': {name: hashlib.sha256(captured[1]).hexdigest()
                  for name, captured in files.items()},
    })
    fd = None
    try:
        require_guarded_role(lease)
        expected = {lease.lock_dir / name for name in files}
        if set(itertools.islice(lease.lock_dir.iterdir(), len(files) + 1)) != expected:
            raise RoleStorageHOLD('role_storage_lock_contents_changed')
        for name, captured in files.items():
            _check_file(lease.lock_dir / name, captured)
        fd = _open_directory(lease.lock_dir)
        if identity(os.fstat(fd))[:3] != lease.lock_identity:
            raise RoleStorageHOLD('role_storage_lock_changed')
        for name, captured in files.items():
            transition.check()
            if identity(lease.lock_dir.lstat())[:3] != lease.lock_identity:
                raise RoleStorageHOLD('role_storage_lock_changed')
            _check_file(lease.lock_dir / name, captured)
            if identity(os.stat(name, dir_fd=fd, follow_symlinks=False)) != captured[0]:
                raise RoleStorageHOLD('role_storage_file_changed')
            os.unlink(name, dir_fd=fd)
            os.fsync(fd)
        transition.check()
        if identity(lease.lock_dir.lstat())[:3] != lease.lock_identity:
            raise RoleStorageHOLD('role_storage_lock_changed')
        lease.lock_dir.rmdir()
        _sync(lease.lock_dir.parent)
        transition.finish()
        with _GUARD:
            _LEASES.pop(id(lease))
    except BaseException:
        with _GUARD:
            record = _LEASES.get(id(lease))
            if record is not None:
                record[2] = True
        raise
    finally:
        if fd is not None:
            os.close(fd)


def cancel_guarded_role(lease):
    """Cancel only the unbound, never-prepared acquisition from this process."""
    if _registered(lease)[1] is not None:
        raise RoleStorageHOLD('role_storage_cancel_after_bind')
    _remove_owned(lease, {'lock.json': lease.lock_file})


def release_guarded_role(lease, owner):
    """Internal final step after RoleLaunch closure and delivery settlement."""
    if _registered(lease)[1] is not owner:
        raise RoleStorageHOLD('role_storage_owner_changed')
    owner._current()
    receipt = _read_pinned(lease.lock_dir / 'role-process.json')
    if receipt[1] != owner._raw:
        raise RoleStorageHOLD('role_storage_receipt_changed')
    _remove_owned(lease, {'lock.json': lease.lock_file, 'role-process.json': receipt})
