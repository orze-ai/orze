"""A secondary close error cannot turn uncertain rollback into released work."""
import errno
import os

import pytest

from orze.core.fs import locked_append


def test_close_failure_cannot_override_uncertain_rollback_hold(tmp_path, monkeypatch):
    source = tmp_path / "ideas.md"
    source.write_bytes(b"before\n")
    info = source.stat()
    inode = (info.st_dev, info.st_ino)
    lock_dir = tmp_path / ".ideas.lock"
    truncate, close = os.ftruncate, os.close

    def is_source(fd):
        current = os.fstat(fd)
        return (current.st_dev, current.st_ino) == inode

    def failed_rollback(fd, size):
        if is_source(fd):
            raise OSError(errno.EIO, "rollback failed")
        return truncate(fd, size)

    def failed_close(fd):
        target = is_source(fd)
        close(fd)
        if target:
            raise OSError(errno.EIO, "source close also failed")

    def failed_finalizer():
        raise ValueError("append was not accepted")

    monkeypatch.setattr(os, "ftruncate", failed_rollback)
    monkeypatch.setattr(os, "close", failed_close)
    with pytest.raises(OSError):
        locked_append(source, "uncertain\n", lock_dir, after_append=failed_finalizer)
    assert source.read_bytes() == b"before\nuncertain\n"
    assert lock_dir.is_dir(), "secondary cleanup failure must preserve the source HOLD"
