"""Source-only lease mechanisms; these are new-API, not old-API red tests."""
import json
import os
from pathlib import Path
import socket

import pytest

from orze.core import fs, idea_source_lock as source_locks


def test_namespace_is_protected_before_owner_metadata_exists(tmp_path, monkeypatch):
    lock_dir = tmp_path / ".ideas.lock"
    mkdir = Path.mkdir
    attempts = []

    def observed_mkdir(path, *args, **kwargs):
        result = mkdir(path, *args, **kwargs)
        if path == lock_dir:
            assert not (lock_dir / "lock.json").exists()
            assert source_locks.idea_source_lock_protected(lock_dir)
            attempts.append(fs._fs_lock(lock_dir, stale_seconds=-1))
        return result

    monkeypatch.setattr(Path, "mkdir", observed_mkdir)
    with source_locks.idea_source_lock(lock_dir) as lease:
        assert source_locks.idea_source_lock_owned(lease)
    assert attempts == [False]
    assert not lock_dir.exists()
    assert not fs._fs_lock(lock_dir, stale_seconds=-1)


def test_generic_contender_rechecks_marker_after_initial_check_race(tmp_path, monkeypatch):
    lock_dir = tmp_path / ".ideas.lock"
    protected = source_locks.idea_source_lock_protected
    held = []
    first = [True]
    clock = [fs.time.time()]
    monkeypatch.setattr(fs.time, "time", lambda: clock[0])

    def raced_check(path):
        before = protected(path)
        if first[0]:
            first[0] = False
            manager = source_locks.idea_source_lock(lock_dir)
            lease = manager.__enter__()
            assert lease is not None
            held.append(manager)
            clock[0] += 120
        return before

    monkeypatch.setattr(source_locks, "idea_source_lock_protected", raced_check)
    release_errors = []
    try:
        acquired = fs._fs_lock(lock_dir, stale_seconds=60)
    finally:
        for manager in held:
            try:
                manager.__exit__(None, None, None)
            except OSError as exc:
                release_errors.append(str(exc))
    assert acquired is False, "marker may appear after generic contender's first check"
    assert release_errors == []


@pytest.mark.parametrize("replacement", ["nonce", "directory"])
def test_release_never_deletes_replacement_owner(tmp_path, replacement):
    lock_dir = tmp_path / ".ideas.lock"
    with pytest.raises(OSError, match="ownership_lost"):
        with source_locks.idea_source_lock(lock_dir) as lease:
            assert lease is not None
            metadata = (lock_dir / "lock.json").read_bytes()
            if replacement == "nonce":
                changed = json.loads(metadata)
                changed["owner_nonce"] = "different-owner"
                metadata = json.dumps(changed).encode()
                (lock_dir / "lock.json").write_bytes(metadata)
            else:
                lock_dir.rename(tmp_path / "displaced-owner")
                lock_dir.mkdir()
                (lock_dir / "lock.json").write_bytes(metadata)
            assert not source_locks.idea_source_lock_owned(lease)
    assert lock_dir.is_dir()
    assert (lock_dir / "lock.json").read_bytes() == metadata


@pytest.mark.parametrize("owner", ["dead_local", "unknown_remote"])
def test_existing_uncertain_source_owner_is_not_automatically_recovered(tmp_path, owner):
    lock_dir = tmp_path / ".ideas.lock"
    lock_dir.mkdir()
    metadata = json.dumps({
        "host": socket.gethostname() if owner == "dead_local" else "remote-unknown",
        "pid": 2147483647, "time": 1,
    }).encode()
    (lock_dir / "lock.json").write_bytes(metadata)
    with source_locks.idea_source_lock(lock_dir) as lease:
        assert lease is None
    assert (lock_dir / "lock.json").read_bytes() == metadata
    assert not fs._fs_lock(lock_dir, stale_seconds=0)


@pytest.mark.parametrize("link_type", ["symlink", "hardlink"])
def test_redirected_namespace_marker_cannot_grant_ownership(tmp_path, link_type):
    lock_dir = tmp_path / ".ideas.lock"
    marker = tmp_path / ".ideas.lock.source-lock"
    outside = tmp_path / "outside"
    contents = b"orze-idea-source-lock-v1\n"
    outside.write_bytes(contents)
    if link_type == "symlink":
        marker.symlink_to(outside)
    else:
        os.link(outside, marker)
    with source_locks.idea_source_lock(lock_dir) as lease:
        assert lease is None
    assert not lock_dir.exists()
    assert outside.read_bytes() == contents
    assert not fs._fs_lock(lock_dir, stale_seconds=0)
