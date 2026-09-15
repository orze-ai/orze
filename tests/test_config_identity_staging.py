"""Legacy preparation must bound collections without weakening write semantics."""
import hashlib
import sqlite3

import pytest
import yaml

from orze.core import config_identity_repair as staging
from orze.core.integrity import hash_config
from orze.core.proposal_admission import admit_proposal_in_tx
from orze.idea_lake import IdeaLake


@pytest.fixture
def lake(tmp_path):
    instance = IdeaLake(tmp_path / "lake.db")
    try:
        yield instance
    finally:
        if instance.conn.in_transaction:
            instance.conn.rollback()
        instance.close()


def legacy(lake, count=600):
    lake.conn.executemany(
        "INSERT INTO ideas(idea_id,title,config,raw_markdown,status) VALUES (?,?,?,?,?)",
        ((f"legacy-{n:04d}", "Imported metadata", f"seed: {n}\n", "", "completed")
         for n in range(count)),
    )
    lake.conn.commit()


def missing(lake):
    return lake.conn.execute(
        "SELECT COUNT(*) FROM ideas WHERE config_hash IS NULL OR config_source_sha256 IS NULL"
    ).fetchone()[0]


class SourceCursor:
    def __init__(self, cursor, owner):
        self.cursor, self.owner = cursor, owner

    def fetchall(self):
        raise AssertionError("unbounded_legacy_source_read")

    def fetchmany(self, size):
        assert 0 < size <= 128
        if self.owner.fail_after_first and self.owner.sizes and not self.owner.failed:
            self.owner.failed = True
            raise sqlite3.OperationalError("database is locked")
        rows = self.cursor.fetchmany(size)
        self.owner.sizes.append(len(rows))
        return rows

    def close(self):
        self.owner.closes += 1
        self.cursor.close()


class SourceConnection:
    def __init__(self, connection, *, fail_after_first=False):
        self.connection, self.fail_after_first = connection, fail_after_first
        self.sizes = []
        self.closes = 0
        self.failed = False

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def execute(self, sql, *args):
        cursor = self.connection.execute(sql, *args)
        if "SELECT idea_id, config FROM ideas" in " ".join(sql.split()):
            return SourceCursor(cursor, self)
        return cursor


def test_source_rows_are_bounded_but_main_write_is_one_atomic_transaction(lake):
    legacy(lake)
    proxy = SourceConnection(lake.conn)
    lake.conn = proxy
    statements = []
    lake.conn.set_trace_callback(statements.append)
    assert lake._repair_admitted_config_hashes() == 600
    lake.conn.set_trace_callback(None)
    assert proxy.sizes == [128, 128, 128, 128, 88, 0]
    assert proxy.closes == 1
    assert [sql for sql in statements if sql in {"BEGIN IMMEDIATE", "COMMIT", "ROLLBACK"}] == [
        "BEGIN IMMEDIATE", "COMMIT",
    ]
    for row in lake.conn.execute("SELECT config,config_hash,config_source_sha256 FROM ideas"):
        assert row[1] == hash_config(yaml.safe_load(row[0]))
        assert row[2] == hashlib.sha256(row[0].encode()).hexdigest()


def test_late_main_write_failure_rolls_back_every_prior_batch(lake):
    legacy(lake, 300)
    lake.conn.execute(
        "CREATE TEMP TRIGGER reject_late BEFORE UPDATE OF config_hash ON ideas "
        "WHEN NEW.idea_id='legacy-0256' BEGIN SELECT RAISE(ABORT, 'late_repair_failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="late_repair_failure"):
        lake._repair_admitted_config_hashes()
    assert not lake.conn.in_transaction
    assert missing(lake) == 300
    lake.conn.execute("DROP TRIGGER reject_late")
    assert lake._repair_admitted_config_hashes() == 300
    assert missing(lake) == 0


@pytest.mark.parametrize("failure", ["decode", "hash"])
def test_late_derivation_failure_cannot_publish_earlier_batches(lake, monkeypatch, failure):
    legacy(lake, 300)
    calls = []
    real_load, real_hash = yaml.safe_load, hash_config

    def checked(value):
        calls.append(value)
        if len(calls) == 257:
            raise TypeError("late_derivation_failure")
        return (real_load if failure == "decode" else real_hash)(value)

    monkeypatch.setattr("orze.idea_lake.yaml.safe_load" if failure == "decode"
                        else "orze.idea_lake.hash_config", checked)
    with pytest.raises(TypeError, match="late_derivation_failure"):
        lake._repair_admitted_config_hashes()
    assert len(calls) == 257
    assert not lake.conn.in_transaction
    assert missing(lake) == 300


def test_source_snapshot_closes_before_decode_and_compare_write_rejects_changed_text(lake, monkeypatch):
    legacy(lake, 300)
    peer = sqlite3.connect(lake.db_path, timeout=0.1)
    original = yaml.safe_load
    seen = []
    changed = "seed: 987654\n"

    def edit_during_decode(value):
        if not seen:
            peer.execute("UPDATE ideas SET config=? WHERE idea_id='legacy-0299'", (changed,))
            peer.commit()  # A still-open source SELECT in DELETE mode blocks this.
        seen.append(value)
        return original(value)

    try:
        monkeypatch.setattr("orze.idea_lake.yaml.safe_load", edit_during_decode)
        assert lake._repair_admitted_config_hashes() == 300  # Legacy return counts attempts.
        assert seen[-1] == "seed: 299\n"  # One snapshot, not fresh later batches.
        row = lake.conn.execute(
            "SELECT config,config_hash,config_source_sha256 FROM ideas WHERE idea_id='legacy-0299'"
        ).fetchone()
        assert tuple(row) == (changed, None, None)
        proposal = lake.prepare_proposal("new-proposal", title="Duplicate", config_yaml=changed,
                                         raw_markdown="")
        lake.conn.execute("BEGIN IMMEDIATE")
        assert admit_proposal_in_tx(lake, proposal)["status"] == "config_duplicate"
        lake.conn.rollback()
        monkeypatch.setattr("orze.idea_lake.yaml.safe_load", original)
        assert lake._repair_admitted_config_hashes() == 1
        assert missing(lake) == 0
    finally:
        peer.close()


def test_failed_begin_leaves_caller_transaction_and_uncommitted_changes_owned(lake):
    legacy(lake, 2)
    peer = sqlite3.connect(lake.db_path, timeout=0.1)
    old = peer.execute("SELECT next_id FROM id_sequence").fetchone()[0]
    try:
        lake.conn.execute("BEGIN IMMEDIATE")
        lake.conn.execute("UPDATE id_sequence SET next_id=12345")
        with pytest.raises(sqlite3.OperationalError, match="within a transaction"):
            lake._repair_admitted_config_hashes()
        assert lake.conn.in_transaction
        assert lake.conn.execute("SELECT next_id FROM id_sequence").fetchone()[0] == 12345
        assert peer.execute("SELECT next_id FROM id_sequence").fetchone()[0] == old
        assert missing(lake) == 2
        lake.conn.rollback()
    finally:
        peer.close()


def test_invalid_and_large_legacy_sources_keep_old_eligibility_and_hash_rules(lake):
    cases = [
        ("invalid", "a: [", "completed", False),
        ("sequence", "[1, 2]", "pending", False),
        ("empty", "", "RUNNING", True),
        ("scalar_false", "false", "QUEUED", True),
        ("large", "payload: " + "z" * 70000 + "\n", "completed", True),
        ("failed", "seed: 77", "failed", False),
        ("archived", "seed: 78", "archived", False),
    ]
    lake.conn.executemany(
        "INSERT INTO ideas(idea_id,title,config,raw_markdown,status) VALUES (?,?,?,?,?)",
        ((name, name, config, "", status) for name, config, status, _ in cases),
    )
    lake.conn.commit()
    assert lake._repair_admitted_config_hashes() == 3
    for name, config, _, repaired in cases:
        row = lake.conn.execute(
            "SELECT config_hash,config_source_sha256 FROM ideas WHERE idea_id=?", (name,)
        ).fetchone()
        assert tuple(row) == ((hash_config(yaml.safe_load(config) or {}),
                              hashlib.sha256(config.encode()).hexdigest())
                             if repaired else (None, None))


def test_no_missing_identity_does_not_create_temporary_storage(lake, monkeypatch):
    lake.insert("ready", "Ready", "seed: 7", "", status="completed")

    def forbidden(*args, **kwargs):
        raise AssertionError("unnecessary_temporary_storage")

    monkeypatch.setattr(staging.tempfile, "TemporaryDirectory", forbidden)
    assert lake._repair_admitted_config_hashes() == 0


def test_partial_source_busy_retry_restarts_complete_snapshot(lake, monkeypatch):
    legacy(lake, 300)
    proxy = SourceConnection(lake.conn, fail_after_first=True)
    lake.conn = proxy
    monkeypatch.setattr("orze.idea_lake.time.sleep", lambda _: None)
    assert lake._repair_admitted_config_hashes() == 300
    assert proxy.failed
    assert proxy.closes == 2
    assert proxy.sizes == [128, 128, 128, 44, 0]
    assert missing(lake) == 0


@pytest.mark.parametrize("failure", ["copy", "derive", "apply"])
def test_failed_stage_or_write_cleans_private_storage_and_preserves_main(lake, monkeypatch, failure):
    from pathlib import Path

    legacy(lake, 300)
    real_temporary = staging.tempfile.TemporaryDirectory
    directories = []

    def track_directory(*args, **kwargs):
        result = real_temporary(*args, **kwargs)
        directories.append(Path(result.name))
        return result

    def fail(*args):
        raise OSError("fixture_storage_failure")

    monkeypatch.setattr(staging.tempfile, "TemporaryDirectory", track_directory)
    monkeypatch.setattr(staging, {"copy": "_copy_snapshot", "derive": "_derive", "apply": "_apply"}[failure], fail)
    with pytest.raises(OSError, match="fixture_storage_failure"):
        lake._repair_admitted_config_hashes()
    assert len(directories) == 1 and not directories[0].exists()
    assert not lake.conn.in_transaction
    assert missing(lake) == 300
