"""New reservation storage mechanisms, not historical baseline regressions.

Only temporary ledgers and exact descriptor/lock-directory I/O are faulted.
The public prepare function runs; the evaluator script is never executed.
"""
import errno
import json
import os
from pathlib import Path

import pytest

from orze.core import benchmark_contract as benchmark
from orze.core import idea_source_lock as source_locks
from test_benchmark_reservation_fencing import reservation


def _is_ledger(fd, ledger):
    try:
        descriptor = os.fstat(fd)
        named = ledger.stat(follow_symlinks=False)
    except OSError:
        return False
    return (descriptor.st_dev, descriptor.st_ino) == (named.st_dev, named.st_ino)


def _refused(idea, cfg):
    with pytest.raises(benchmark.BenchmarkContractError) as caught:
        benchmark.prepare_benchmark_evaluation(idea, cfg)
    assert not (idea / benchmark.PROVENANCE_FILE).exists()
    return str(caught.value)


@pytest.mark.parametrize("fault", ["half_write", "fsync", "close", "readback"])
def test_uncertain_ledger_io_retains_bytes_and_owner(reservation, monkeypatch, fault):
    cfg, ideas, ledger = reservation
    lock_dir = ledger.parent / benchmark.EXPOSURE_LOCK_DIR
    real_write, real_fsync = os.write, os.fsync
    real_close, real_read = os.close, os.read
    observed = {"wrote": False, "fault": False}

    def write(fd, data):
        if not _is_ledger(fd, ledger):
            return real_write(fd, data)
        if fault == "half_write":
            if not observed["wrote"]:
                count = real_write(fd, data[:max(1, len(data) // 2)])
                observed["wrote"] = True
                return count
            observed["fault"] = True
            raise OSError(errno.ENOSPC, "owned ledger partial append")
        count = real_write(fd, data)
        observed["wrote"] = True
        return count

    def fsync(fd):
        target = _is_ledger(fd, ledger)
        result = real_fsync(fd)
        if target and fault == "fsync" and observed["wrote"]:
            observed["fault"] = True
            raise OSError(errno.EIO, "owned ledger fsync acknowledgement")
        return result

    def close(fd):
        target = _is_ledger(fd, ledger)
        result = real_close(fd)
        if target and fault == "close" and observed["wrote"] and not observed["fault"]:
            observed["fault"] = True
            raise OSError(errno.EIO, "owned ledger close acknowledgement")
        return result

    def read(fd, size):
        if (_is_ledger(fd, ledger) and fault == "readback"
                and observed["wrote"] and not observed["fault"]):
            observed["fault"] = True
            return b""  # An incorrect early EOF cannot confirm persisted bytes.
        return real_read(fd, size)

    with monkeypatch.context() as patch:
        patch.setattr(os, "write", write)
        patch.setattr(os, "fsync", fsync)
        patch.setattr(os, "close", close)
        patch.setattr(os, "read", read)
        reason = _refused(ideas[0], cfg)
    assert observed == {"wrote": True, "fault": True}
    assert "unconfirmed" in reason
    preserved = ledger.read_bytes()
    assert preserved
    assert lock_dir.is_dir()
    metadata = (lock_dir / "lock.json").read_bytes()
    if fault == "half_write":
        assert not preserved.endswith(b"\n")
    else:
        record, = [json.loads(line) for line in preserved.splitlines()]
        assert record["exposure_ordinal"] == 1
    assert _refused(ideas[1], cfg) == "benchmark_exposure_ledger_locked"
    assert ledger.read_bytes() == preserved
    assert (lock_dir / "lock.json").read_bytes() == metadata


def test_release_ack_failure_denies_env_without_refunding_committed_look(
        reservation, monkeypatch):
    cfg, ideas, ledger = reservation
    lock_dir = ledger.parent / benchmark.EXPOSURE_LOCK_DIR
    real_sync = source_locks._sync_directory
    injected = []

    def sync(path):
        result = real_sync(path)
        if Path(path) == ledger.parent and not lock_dir.exists():
            injected.append("after_owned_directory_removed")
            raise OSError(errno.EIO, "owned release directory fsync acknowledgement")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(source_locks, "_sync_directory", sync)
        reason = _refused(ideas[0], cfg)
    assert injected == ["after_owned_directory_removed"]
    assert "unconfirmed" in reason
    preserved = ledger.read_bytes()
    record, = [json.loads(line) for line in preserved.splitlines()]
    assert record["exposure_ordinal"] == 1
    # Late release failure may happen after rmdir; it does not refund the look.
    assert not lock_dir.exists()
    assert lock_dir.with_name(lock_dir.name + ".source-lock").is_file()
    assert _refused(ideas[1], cfg) == "benchmark_exposure_budget_exhausted:1/1"
    assert ledger.read_bytes() == preserved


def test_replaced_owner_inode_is_never_deleted_after_append(reservation, monkeypatch):
    cfg, ideas, ledger = reservation
    lock_dir = ledger.parent / benchmark.EXPOSURE_LOCK_DIR
    displaced = ledger.parent / "owned-displaced-reservation-lock"
    real_write = os.write
    captured = {}

    def write(fd, data):
        target = _is_ledger(fd, ledger)
        count = real_write(fd, data)
        if target and not captured:
            captured["bytes"] = (lock_dir / "lock.json").read_bytes()
            captured["inode"] = lock_dir.stat().st_ino
            lock_dir.rename(displaced)
            lock_dir.mkdir()
            (lock_dir / "lock.json").write_bytes(captured["bytes"])
            captured["replacement_inode"] = lock_dir.stat().st_ino
        return count

    with monkeypatch.context() as patch:
        patch.setattr(os, "write", write)
        reason = _refused(ideas[0], cfg)
    assert captured["inode"] != captured["replacement_inode"]
    assert "unconfirmed" in reason
    preserved = ledger.read_bytes()
    assert json.loads(preserved)["exposure_ordinal"] == 1
    assert (lock_dir / "lock.json").read_bytes() == captured["bytes"]
    assert (displaced / "lock.json").read_bytes() == captured["bytes"]
    assert _refused(ideas[1], cfg) == "benchmark_exposure_ledger_locked"
    assert ledger.read_bytes() == preserved
    assert lock_dir.stat().st_ino == captured["replacement_inode"]
    assert (lock_dir / "lock.json").read_bytes() == captured["bytes"]


def test_partial_legacy_migration_keeps_source_and_never_overwrites_winner(
        reservation, monkeypatch):
    cfg, ideas, ledger = reservation
    first = benchmark.prepare_benchmark_evaluation(ideas[0], cfg)
    assert first["ORZE_BENCHMARK_EXPOSURE_ORDINAL"] == "1"
    original = ledger.read_bytes()
    legacy = ideas[0].parent / benchmark.EXPOSURE_LEDGER_FILE
    ledger.rename(legacy)
    assert not ledger.exists()
    lock_dir = ledger.parent / benchmark.EXPOSURE_LOCK_DIR
    real_write = os.write
    writes = []

    def write(fd, data):
        if not _is_ledger(fd, ledger):
            return real_write(fd, data)
        writes.append(len(data))
        if len(writes) == 1:
            return real_write(fd, data[:max(1, len(data) // 2)])
        raise OSError(errno.ENOSPC, "owned project-ledger migration partial write")

    with monkeypatch.context() as patch:
        patch.setattr(os, "write", write)
        reason = _refused(ideas[1], cfg)
    assert len(writes) == 2
    assert "unconfirmed" in reason
    partial = ledger.read_bytes()
    assert partial and partial != original
    assert original.startswith(partial)
    assert legacy.read_bytes() == original
    assert lock_dir.is_dir()
    assert _refused(ideas[1], cfg) == "benchmark_exposure_ledger_locked"
    assert ledger.read_bytes() == partial
    assert legacy.read_bytes() == original
