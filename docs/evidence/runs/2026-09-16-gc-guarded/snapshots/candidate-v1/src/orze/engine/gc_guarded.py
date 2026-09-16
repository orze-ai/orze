"""Captured-tree GC with durable retirement, exclusive copies and no rename.

The caller retires the closed task under its effect guard, then releases that
short guard before calling this module. Every uncertain operation retains the
pending retirement and GC journal; these routines never retry or roll back.
"""
import hashlib
import json
import os
from pathlib import Path
import stat

from orze.engine.artifact_publication import _open_directory
from orze.engine.attempt_effect_receipts import _publish, _sync
from orze.engine.gc_tree import GCRefused, identity, snapshot


CHUNK_BYTES = 1024 * 1024


def plan_bytes(tree, destination):
    # No payload reads; compute before entering the short task guard.
    return json.dumps({'version': 1, 'storage_mode': 'guarded', 'source': str(tree.path),
                       'destination': str(destination) if destination else None,
                       'entries': tree.entries, 'bytes': tree.size},
                      sort_keys=True, separators=(',', ':')).encode()


def _same_tree(tree):
    if snapshot(tree.path, tree.root_identity) != tree:
        raise GCRefused('gc_guarded_source_changed')


def _copy_file(source, destination, expected, check):
    source_parent = _open_directory(source.parent)
    target_parent = _open_directory(destination.parent)
    source_fd = target_fd = None
    try:
        if os.fstat(source_parent).st_dev != os.fstat(target_parent).st_dev:
            raise GCRefused('gc_cross_device_refused')
        source_fd = os.open(source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source_parent)
        if (identity(os.fstat(source_fd)) != expected
                or identity(os.stat(source.name, dir_fd=source_parent, follow_symlinks=False)) != expected):
            raise GCRefused('gc_guarded_source_changed')
        target_fd = os.open(destination.name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            0o600, dir_fd=target_parent)
        captured = identity(os.fstat(target_fd))[:3]
        digest, size = hashlib.sha256(), 0
        while size < expected[4]:
            check()
            data = os.read(source_fd, min(CHUNK_BYTES, expected[4] - size))
            if not data:
                raise GCRefused('gc_guarded_source_short_read')
            digest.update(data)
            size += len(data)
            remaining = memoryview(data)
            while remaining:
                count = os.write(target_fd, remaining)
                if not 0 < count <= len(remaining):
                    raise OSError('gc_guarded_copy_short_write')
                remaining = remaining[count:]
        if (identity(os.fstat(source_fd)) != expected
                or identity(os.stat(source.name, dir_fd=source_parent, follow_symlinks=False)) != expected):
            raise GCRefused('gc_guarded_source_changed')
        os.fsync(target_fd)
        os.lseek(target_fd, 0, os.SEEK_SET)
        copied = hashlib.sha256()
        while True:
            check()
            data = os.read(target_fd, CHUNK_BYTES)
            if not data:
                break
            copied.update(data)
        target = identity(os.fstat(target_fd))
        if (target[:3] != captured or target[3] != 1 or target[4] != size
                or copied.digest() != digest.digest()
                or identity(os.stat(destination.name, dir_fd=target_parent, follow_symlinks=False)) != target):
            raise GCRefused('gc_guarded_copy_unconfirmed')
        os.fsync(target_parent)
        return digest.hexdigest(), target
    finally:
        for fd in (target_fd, source_fd, target_parent, source_parent):
            if fd is not None:
                os.close(fd)


def copy_new(tree, destination, check):
    parent = _open_directory(destination.parent, create=True)
    try:
        if os.fstat(parent).st_dev != tree.root_identity[0]:
            raise GCRefused('gc_cross_device_refused')
    finally:
        os.close(parent)
    digests, created = {}, {}
    for relative, expected in tree.entries:
        check()
        source = tree.path if relative == '.' else tree.path / relative
        target = destination if relative == '.' else destination / relative
        if identity(source.lstat()) != expected:
            raise GCRefused('gc_guarded_source_changed')
        if stat.S_ISDIR(expected[2]):
            parent = _open_directory(target.parent)
            try:
                os.mkdir(target.name, 0o700, dir_fd=parent)
                created[relative] = identity(os.stat(target.name, dir_fd=parent, follow_symlinks=False))
                os.fsync(parent)
            finally:
                os.close(parent)
        else:
            digests[relative], created[relative] = _copy_file(source, target, expected, check)
    _same_tree(tree)
    copied = snapshot(destination, identity(destination.lstat()))
    if [name for name, _ in copied.entries] != [name for name, _ in tree.entries]:
        raise GCRefused('gc_guarded_copy_tree_changed')
    for name, actual in copied.entries:
        expected = created[name]
        if (actual[:3] != expected[:3] if stat.S_ISDIR(expected[2]) else actual != expected):
            raise GCRefused('gc_guarded_copy_tree_changed')
    return copied, digests


def remove(tree, check, *, copied=None):
    _same_tree(tree)
    if copied is not None:
        _same_tree(copied)
    targets = dict(copied.entries) if copied else {}
    for relative, expected in reversed(tree.entries):
        check()
        if copied is not None:
            target = copied.path if relative == '.' else copied.path / relative
            if identity(target.lstat()) != targets[relative]:
                raise GCRefused('gc_guarded_archive_changed')
        path = tree.path if relative == '.' else tree.path / relative
        parent = _open_directory(path.parent)
        try:
            actual = identity(os.stat(path.name, dir_fd=parent, follow_symlinks=False))
            if (actual[:3] != expected[:3] if stat.S_ISDIR(expected[2]) else actual != expected):
                raise GCRefused('gc_guarded_source_changed')
            if stat.S_ISDIR(expected[2]):
                os.rmdir(path.name, dir_fd=parent)
            else:
                os.unlink(path.name, dir_fd=parent)
            os.fsync(parent)
        finally:
            os.close(parent)


def _publish_plan(path, raw, check):
    # Inventory can exceed the small-control-record limit; bound I/O chunks,
    # check the created identity, and read back the exact bytes through its FD.
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        captured = identity(os.fstat(fd))[:3]
        offset = 0
        while offset < len(raw):
            check()
            count = os.write(fd, memoryview(raw)[offset:offset + CHUNK_BYTES])
            if not 0 < count <= min(CHUNK_BYTES, len(raw) - offset):
                raise OSError('gc_guarded_plan_short_write')
            offset += count
        os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            check()
            block = os.read(fd, CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
        current = identity(os.fstat(fd))
        if (current[:3] != captured or current[3] != 1 or current[4] != len(raw)
                or identity(path.lstat()) != current or digest.digest() != hashlib.sha256(raw).digest()):
            raise GCRefused('gc_guarded_plan_unconfirmed')
    finally:
        os.close(fd)
    _sync(path.parent)


def execute(tree, directory, destination, check, plan):
    check()
    _publish_plan(directory / 'guarded-plan.json', plan, check)
    copied = None
    if destination is not None:
        copied, digests = copy_new(tree, destination, check)
        # One small record per file avoids turning a large tree into an
        # oversized control document. Files contain no original payload.
        manifests = directory / 'archive-files'
        manifests.mkdir(mode=0o700)
        _sync(directory)
        for number, (name, digest) in enumerate(digests.items()):
            _publish(manifests / (str(number) + '.json'), json.dumps({
                'path': name, 'sha256': digest}, sort_keys=True).encode())
        _sync(manifests)
    remove(tree, check, copied=copied)
    check()
    _publish(directory / 'completed.json', json.dumps({
        'status': 'archived' if destination else 'deleted'}, sort_keys=True, separators=(',', ':')).encode())
