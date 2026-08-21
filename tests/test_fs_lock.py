import json
import os
import time

from orze.core.fs import _fs_lock, _fs_unlock


def test_old_corrupt_lock_metadata_is_recovered(tmp_path):
    lock_dir = tmp_path / "role"
    lock_dir.mkdir()
    (lock_dir / "lock.json").write_bytes(b"\0" * 32)
    old = time.time() - 60
    os.utime(lock_dir, (old, old))

    assert _fs_lock(lock_dir, stale_seconds=600)
    metadata = json.loads((lock_dir / "lock.json").read_text())
    assert metadata["pid"] == os.getpid()
    _fs_unlock(lock_dir)


def test_recent_corrupt_lock_metadata_keeps_race_grace(tmp_path):
    lock_dir = tmp_path / "role"
    lock_dir.mkdir()
    (lock_dir / "lock.json").write_bytes(b"\0" * 32)

    assert not _fs_lock(lock_dir, stale_seconds=0)
