"""Actual source-fd failures must not publish partial proposal bytes."""
import errno
import os

import pytest

from orze.core import fs


@pytest.mark.parametrize("failure", ["partial_write", "first_fsync"])
def test_source_io_failure_restores_exact_prior_bytes(tmp_path, monkeypatch, failure):
    source = tmp_path / "ideas.md"
    before = b"# Existing proposals\nkeep every byte\n"
    source.write_bytes(before)
    inode = (source.stat().st_dev, source.stat().st_ino)
    lock_dir = tmp_path / ".ideas.lock"
    write, fsync = os.write, os.fsync
    writes, syncs, finalizers = [], [], []

    def is_source(fd):
        info = os.fstat(fd)
        return (info.st_dev, info.st_ino) == inode

    def failed_write(fd, data):
        if is_source(fd) and failure == "partial_write":
            writes.append(bytes(data))
            if len(writes) == 1:
                return write(fd, data[:5])
            raise OSError(errno.ENOSPC, "injected source write failure")
        return write(fd, data)

    def failed_fsync(fd):
        if is_source(fd) and failure == "first_fsync":
            syncs.append(fd)
            if len(syncs) == 1:
                raise OSError(errno.ENOSPC, "injected source fsync failure")
        return fsync(fd)

    monkeypatch.setattr(os, "write", failed_write)
    monkeypatch.setattr(os, "fsync", failed_fsync)
    with pytest.raises(OSError):
        fs.locked_append(source, "new uncommitted proposal\n", lock_dir,
                         after_append=lambda: finalizers.append(True))
    assert source.read_bytes() == before
    assert finalizers == []
    assert not lock_dir.exists(), "a fully restored source can release its lease"


def test_rollback_failure_holds_source_owner_against_subsequent_consumers(tmp_path, monkeypatch):
    source = tmp_path / "ideas.md"
    source.write_bytes(b"prior\n")
    inode = (source.stat().st_dev, source.stat().st_ino)
    lock_dir = tmp_path / ".ideas.lock"
    write, truncate = os.write, os.ftruncate
    writes, finalizers = [], []

    def is_source(fd):
        info = os.fstat(fd)
        return (info.st_dev, info.st_ino) == inode

    def failed_write(fd, data):
        if is_source(fd):
            writes.append(bytes(data))
            if len(writes) == 1:
                return write(fd, data[:4])
            raise OSError(errno.ENOSPC, "source write failed")
        return write(fd, data)

    def failed_rollback(fd, size):
        if is_source(fd):
            raise OSError(errno.EIO, "rollback failed")
        return truncate(fd, size)

    monkeypatch.setattr(os, "write", failed_write)
    monkeypatch.setattr(os, "ftruncate", failed_rollback)
    with pytest.raises(OSError):
        fs.locked_append(source, "uncommitted\n", lock_dir,
                         after_append=lambda: finalizers.append(True))
    assert finalizers == []
    assert lock_dir.is_dir(), "uncertain rollback cannot release the source to a second owner"
    assert not fs._fs_lock(lock_dir, stale_seconds=-1)
    from orze.core.idea_source_lock import idea_source_lock
    with idea_source_lock(lock_dir) as lease:
        assert lease is None
