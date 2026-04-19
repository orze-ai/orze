"""Tests for orze.engine.leader — flock-based leader election."""
import json
import os
import time
from pathlib import Path

import pytest

from orze.engine import leader


def test_first_caller_wins(tmp_path):
    h = leader.try_acquire(tmp_path)
    assert h is not None
    assert h.fd > 0
    assert (tmp_path / leader.LOCK_NAME).exists()
    assert (tmp_path / leader.HEARTBEAT_NAME).exists()
    h.release()


def test_second_caller_returns_none(tmp_path):
    h1 = leader.try_acquire(tmp_path)
    assert h1 is not None
    h2 = leader.try_acquire(tmp_path)
    assert h2 is None
    h1.release()


def test_release_allows_reacquire(tmp_path):
    h1 = leader.try_acquire(tmp_path)
    assert h1 is not None
    h1.release()
    assert not (tmp_path / leader.HEARTBEAT_NAME).exists()
    h2 = leader.try_acquire(tmp_path)
    assert h2 is not None
    h2.release()


def test_heartbeat_records_identity(tmp_path):
    h = leader.try_acquire(tmp_path, host="test-host", pid=12345)
    try:
        record = leader.read_current_leader(tmp_path)
        assert record is not None
        assert record["host"] == "test-host"
        assert record["pid"] == 12345
        assert "ts" in record
    finally:
        h.release()


def test_stale_lease_takeover(tmp_path, monkeypatch):
    """If heartbeat is stale > STALE_LEASE_S, a new caller can take over."""
    h1 = leader.try_acquire(tmp_path, host="ghost", pid=999)
    assert h1 is not None
    # Manually age the heartbeat.
    hb = tmp_path / leader.HEARTBEAT_NAME
    stale = json.dumps({"host": "ghost", "pid": 999,
                        "ts": time.time() - leader.STALE_LEASE_S - 10})
    hb.write_text(stale, encoding="utf-8")

    h2 = leader.try_acquire(tmp_path, host="new", pid=1)
    # The in-process re-lock should succeed on the recreated lockfile.
    assert h2 is not None, "should take over stale lease"
    h1.release()
    h2.release()


def test_should_skip_role_as_follower():
    assert leader.should_skip_role_as_follower("professor")
    assert leader.should_skip_role_as_follower("data_analyst")
    assert leader.should_skip_role_as_follower("research_gemini")
    assert not leader.should_skip_role_as_follower("custom_nonllm_role")
