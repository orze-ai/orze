import json
import os
import time

import pytest

from orze.core.fs import _fs_lock, _fs_unlock, locked_append


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


def test_managed_role_receipt_prevents_stale_lock_takeover(tmp_path):
    lock_dir = tmp_path / "role"
    lock_dir.mkdir()
    (lock_dir / "lock.json").write_bytes(b"\0" * 32)
    receipt = lock_dir / "role-process.json"
    receipt.write_text("{}\n", encoding="utf-8")
    old = time.time() - 60
    os.utime(lock_dir, (old, old))

    assert not _fs_lock(lock_dir, stale_seconds=0)
    assert receipt.exists()


def test_locked_append_writes_and_fsyncs_under_owned_lock(tmp_path):
    path = tmp_path / "ideas.md"
    lock_dir = tmp_path / ".ideas.lock"
    assert locked_append(path, "first\n", lock_dir) is True
    assert locked_append(path, "second\n", lock_dir) is True
    assert path.read_text(encoding="utf-8") == "first\nsecond\n"
    assert not lock_dir.exists()


def test_locked_append_refuses_to_write_without_lock(tmp_path):
    path = tmp_path / "ideas.md"
    path.write_text("before\n", encoding="utf-8")
    lock_dir = tmp_path / ".ideas.lock"
    assert _fs_lock(lock_dir, stale_seconds=600)
    try:
        assert locked_append(path, "forbidden\n", lock_dir) is False
        assert path.read_text(encoding="utf-8") == "before\n"
    finally:
        _fs_unlock(lock_dir)


def test_locked_append_rejects_redirected_or_hardlinked_target(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    redirected = tmp_path / "redirected.md"
    redirected.symlink_to(outside)
    assert locked_append(
        redirected, "forbidden\n", tmp_path / ".redirect.lock") is False
    assert outside.read_text(encoding="utf-8") == "outside\n"

    hardlink = tmp_path / "hardlink.md"
    os.link(outside, hardlink)
    assert locked_append(
        hardlink, "forbidden\n", tmp_path / ".hardlink.lock") is False
    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_locked_append_finalizes_before_unlock(tmp_path):
    path = tmp_path / "ideas.md"
    lock_dir = tmp_path / ".ideas.lock"
    observed = []

    def finalize():
        observed.append(path.read_text(encoding="utf-8"))
        assert _fs_lock(lock_dir, stale_seconds=600) is False

    assert locked_append(
        path, "admitted\n", lock_dir, after_append=finalize) is True
    assert observed == ["admitted\n"]
    assert not lock_dir.exists()


def test_locked_append_rolls_back_when_finalizer_fails(tmp_path):
    path = tmp_path / "ideas.md"
    path.write_text("before\n", encoding="utf-8")
    lock_dir = tmp_path / ".ideas.lock"

    def fail():
        raise ValueError("receipt admission failed")

    with pytest.raises(ValueError, match="receipt admission failed"):
        locked_append(path, "unadmitted\n", lock_dir, after_append=fail)
    assert path.read_text(encoding="utf-8") == "before\n"
    assert not lock_dir.exists()
