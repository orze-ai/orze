"""Live idea-source owners are not leases that expire after sixty seconds.

Exercise existing locked_append/_sync_ideas/_fs_lock public behavior using real
directories, files and IdeaLake. Only wall time/contender hostname are simulated;
no provider, subprocess or GPU is used and no new API absence is a red assertion.
"""

from pathlib import Path

import pytest

from orze.core import fs

from test_idea_ingress_contract import engine, _block


def _clock(monkeypatch):
    clock = [fs.time.time()]
    monkeypatch.setattr(fs.time, "time", lambda: clock[0])
    return clock


def test_live_source_owner_past_sixty_seconds_cannot_be_replaced_by_append(engine, monkeypatch):
    instance, _, source = engine
    lock_dir = instance.results_dir / ".ideas_md.lock"
    clock = _clock(monkeypatch)
    competing = []

    def long_finalizer():
        clock[0] += 120
        competing.append(fs.locked_append(source, _block("idea-competitor", 2), lock_dir))

    assert fs.locked_append(source, _block("idea-owner", 1), lock_dir,
                            after_append=long_finalizer)
    assert competing == [False], "elapsed time does not prove the live source owner is gone"
    assert "idea-owner" in source.read_text(encoding="utf-8")
    assert "idea-competitor" not in source.read_text(encoding="utf-8")


def test_consumer_cannot_read_or_admit_while_long_source_finalizer_is_active(engine, monkeypatch):
    instance, cfg, source = engine
    lock_dir = instance.results_dir / ".ideas_md.lock"
    clock = _clock(monkeypatch)
    reads = []
    in_consumer = [False]
    read_text, read_bytes = Path.read_text, Path.read_bytes

    def observed_text(path, *args, **kwargs):
        if in_consumer[0] and path == source:
            reads.append("read_text")
        return read_text(path, *args, **kwargs)

    def observed_bytes(path, *args, **kwargs):
        if in_consumer[0] and path == source:
            reads.append("read_bytes")
        return read_bytes(path, *args, **kwargs)

    def long_finalizer():
        clock[0] += 120
        in_consumer[0] = True
        try:
            instance._sync_ideas(cfg)
        finally:
            in_consumer[0] = False

    monkeypatch.setattr(Path, "read_text", observed_text)
    monkeypatch.setattr(Path, "read_bytes", observed_bytes)
    assert fs.locked_append(source, _block("idea-uncommitted", 3), lock_dir,
                            after_append=long_finalizer)
    assert reads == [], "consumer must acquire source ownership before reading proposal bytes"
    assert instance.lake.get("idea-uncommitted") is None
    assert "idea-uncommitted" in source.read_text(encoding="utf-8")


@pytest.mark.parametrize("contender", ["same_host", "other_host"])
def test_generic_age_takeover_cannot_bypass_live_source_protocol(engine, monkeypatch, contender):
    instance, _, source = engine
    lock_dir = instance.results_dir / ".ideas_md.lock"
    clock = _clock(monkeypatch)
    attempts = []

    def long_finalizer():
        clock[0] += 120
        with monkeypatch.context() as competing:
            if contender == "other_host":
                competing.setattr(fs.socket, "gethostname", lambda: "different-contender-host")
            attempts.append(fs._fs_lock(lock_dir, stale_seconds=60))

    assert fs.locked_append(source, _block("idea-owner", 4), lock_dir,
                            after_append=long_finalizer)
    assert attempts == [False]


def test_ordinary_role_lock_retains_its_existing_age_takeover_policy(tmp_path, monkeypatch):
    lock_dir = tmp_path / "ordinary-role-lock"
    clock = _clock(monkeypatch)
    assert fs._fs_lock(lock_dir, stale_seconds=60)
    clock[0] += 120
    try:
        assert fs._fs_lock(lock_dir, stale_seconds=60)
    finally:
        fs._fs_unlock(lock_dir)
