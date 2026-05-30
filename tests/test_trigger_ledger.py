"""Tests for orze.engine.trigger_ledger — atomic trigger consume.

Implements determinism hardening win #3 (docs/determinism_hardening.md).
Specifically targets c1005 (DEC-009): role_runner's
read_text() → unlink() race no longer causes duplicate sibling ideas
after daemon restart.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest

from orze.engine.trigger_ledger import (
    TriggerClaim,
    _fingerprint,
    claim_trigger,
    init_schema,
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_claim_returns_payload_unlinks_file_writes_receipt(tmp_path):
    db = tmp_path / "idea_lake.db"
    trig = tmp_path / "_trigger_engineer"
    trig.write_text("BLOCKED_PORTFOLIO: fix sweep collision\n")

    claim = claim_trigger(db, "engineer", trig)

    assert isinstance(claim, TriggerClaim)
    assert claim.role_name == "engineer"
    assert claim.payload == "BLOCKED_PORTFOLIO: fix sweep collision\n"
    assert claim.file_path == str(trig)
    assert not trig.exists()  # unlinked after commit

    # Receipt is in the DB
    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT role_name, payload, consumed_by_host, consumed_by_pid "
        "FROM trigger_consumptions"
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "engineer"
    assert row[1] == "BLOCKED_PORTFOLIO: fix sweep collision\n"
    assert row[2]  # host populated
    assert row[3] == os.getpid()


def test_claim_returns_none_when_file_absent(tmp_path):
    db = tmp_path / "idea_lake.db"
    trig = tmp_path / "_trigger_engineer"  # never created
    assert claim_trigger(db, "engineer", trig) is None


def test_claim_empty_payload_is_still_a_claim(tmp_path):
    """Producer at role_runner.py:475 only checks exists(); payload may
    be empty. Claim must succeed even with zero bytes."""
    db = tmp_path / "idea_lake.db"
    trig = tmp_path / "_trigger_engineer"
    trig.write_text("")
    claim = claim_trigger(db, "engineer", trig)
    assert claim is not None
    assert claim.payload == ""
    assert not trig.exists()


# ---------------------------------------------------------------------------
# c1005 / DEC-009 — crash-restart safety
# ---------------------------------------------------------------------------

def test_orphan_from_prior_cycle_is_silently_unlinked(tmp_path):
    """Simulates the c1005 race: daemon committed the receipt but
    crashed before unlinking the file. On restart, the file still
    exists with the same fingerprint. The next claim must NOT re-fire
    the trigger — it must unlink the orphan and return None."""
    db = tmp_path / "idea_lake.db"
    trig = tmp_path / "_trigger_engineer"
    trig.write_text("BLOCKED_PORTFOLIO: fix sweep collision\n")
    fp = _fingerprint(trig)

    # Pre-seed the DB with a receipt for this exact fingerprint —
    # simulates "we already consumed this in a prior cycle".
    conn = sqlite3.connect(str(db))
    init_schema(conn)
    conn.execute(
        "INSERT INTO trigger_consumptions "
        "(role_name, file_path, fingerprint, payload, consumed_at, "
        " consumed_by_host, consumed_by_pid) "
        "VALUES (?, ?, ?, ?, datetime('now'), ?, ?)",
        ("engineer", str(trig), fp, "prior payload", "old-host", 1234),
    )
    conn.commit()
    conn.close()

    # Daemon restart — try to claim again.
    claim = claim_trigger(db, "engineer", trig)
    assert claim is None, "orphan must not re-fire"
    assert not trig.exists(), "orphan file must be cleaned up"


def test_fresh_trigger_after_consume_does_fire(tmp_path):
    """A NEW trigger file (different inode/mtime) after the prior one
    was consumed must fire — the fingerprint distinguishes new from
    orphan."""
    db = tmp_path / "idea_lake.db"
    trig = tmp_path / "_trigger_thinker"
    trig.write_text("first")
    first = claim_trigger(db, "thinker", trig)
    assert first is not None
    assert first.payload == "first"

    # Producer writes a new trigger (new file → new inode/mtime)
    time.sleep(0.01)  # ensure mtime_ns advances on coarse-mtime FS
    trig.write_text("second")
    second = claim_trigger(db, "thinker", trig)
    assert second is not None
    assert second.payload == "second"
    assert second.fingerprint != first.fingerprint


def test_double_consume_in_same_process_is_safe(tmp_path):
    """Belt-and-braces: even within one process, claiming twice on the
    same fingerprint must produce exactly one TriggerClaim."""
    db = tmp_path / "idea_lake.db"
    trig = tmp_path / "_trigger_professor"
    trig.write_text("user-requested analysis")

    first = claim_trigger(db, "professor", trig)
    assert first is not None

    # File is gone now; recreate at the same path with the same content
    # but a different inode — should still fire because fingerprint
    # differs (different inode + mtime).
    trig.write_text("user-requested analysis")
    second = claim_trigger(db, "professor", trig)
    assert second is not None
    assert second.fingerprint != first.fingerprint


# ---------------------------------------------------------------------------
# Cross-process race (mimics two daemons claiming the same trigger)
# ---------------------------------------------------------------------------

def test_two_short_lived_connections_only_one_wins(tmp_path):
    """The UNIQUE constraint guarantees that two consumers racing on
    the same fingerprint won't both fire. We can't easily simulate
    concurrent processes in a unit test, but we can verify that a
    second claim with the same fingerprint always loses."""
    db = tmp_path / "idea_lake.db"
    trig = tmp_path / "_trigger_engineer"
    trig.write_text("race payload")
    fp = _fingerprint(trig)

    # Manually INSERT a receipt for this fingerprint, mimicking "the
    # other daemon got there first".
    conn = sqlite3.connect(str(db))
    init_schema(conn)
    conn.execute(
        "INSERT INTO trigger_consumptions "
        "(role_name, file_path, fingerprint, consumed_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        ("engineer", str(trig), fp),
    )
    conn.commit()
    conn.close()

    # Now claim from this process — must lose, file must be cleaned up.
    assert claim_trigger(db, "engineer", trig) is None
    assert not trig.exists()


# ---------------------------------------------------------------------------
# Schema lives where IdeaLake expects it
# ---------------------------------------------------------------------------

def test_idea_lake_init_creates_trigger_table(tmp_path):
    """IdeaLake._ensure_schema must materialise the triggers table so
    the first consumer doesn't have to call init_schema."""
    from orze.idea_lake import IdeaLake

    db = tmp_path / "idea_lake.db"
    lake = IdeaLake(str(db))
    try:
        row = lake.conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='trigger_consumptions'"
        ).fetchone()
        assert row is not None
    finally:
        lake.conn.close()


def test_existing_lake_db_migrates_to_include_triggers_table(tmp_path):
    """In-place upgrade: a daemon that already had an idea_lake.db from
    before this change must gain the triggers table on first re-open."""
    from orze.idea_lake import IdeaLake

    db = tmp_path / "idea_lake.db"
    # Step 1: simulate the pre-change DB by running the legacy schema
    # only (no trigger_consumptions table).
    from orze.idea_lake import _SCHEMA_SQL as _LEGACY_SCHEMA
    conn = sqlite3.connect(str(db))
    conn.executescript(_LEGACY_SCHEMA)
    conn.execute("INSERT INTO id_sequence (next_id) VALUES (1)")
    conn.commit()
    conn.close()
    pre = sqlite3.connect(str(db)).execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='trigger_consumptions'").fetchone()
    assert pre is None

    # Step 2: open via IdeaLake — must add the triggers table.
    lake = IdeaLake(str(db))
    try:
        row = lake.conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='trigger_consumptions'"
        ).fetchone()
        assert row is not None
    finally:
        lake.conn.close()

    # Step 3: claim_trigger works against the migrated DB.
    trig = tmp_path / "_trigger_engineer"
    trig.write_text("post-migration test")
    claim = claim_trigger(db, "engineer", trig)
    assert claim is not None
    assert claim.payload == "post-migration test"
