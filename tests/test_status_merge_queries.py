"""Ordered current rows without a dense-history sort before the first result."""
import hashlib

import pytest
import yaml

from orze.core.integrity import hash_config
from orze.core.proposal_admission import admit_proposal
from test_config_identity_staging import lake
from test_proposal_admission import _snapshot

STATUSES = ("queued", "PENDING", "running", "COMPLETED")


def test_first_source_batch_does_not_require_sorting_whole_dense_history(lake):
    lake.conn.executemany(
        "INSERT INTO ideas(idea_id,title,config,raw_markdown,status) VALUES (?,?,?,'',?)",
        ((f"legacy-{n:05d}", "Metadata", f"seed: {n}\n", STATUSES[n % 4]) for n in range(5000)),
    )
    lake.conn.commit()
    original, steps, first = lake.conn, [0], []

    def progress():
        steps[0] += 100
        return 0

    class Cursor:
        def __init__(self, cursor):
            self.cursor = cursor

        def fetchmany(self, size):
            rows = self.cursor.fetchmany(size)
            if not first:
                first.append(steps[0])
                original.set_progress_handler(None, 0)
                assert [r["idea_id"] for r in rows] == [f"legacy-{n:05d}" for n in range(128)]
                # A wide opcode margin detects the full 5000-row pre-sort.
                # This is a local read-work check, not a product deadline.
                assert steps[0] < 20000, "first_source_batch_sorts_dense_history"
            return rows

        def close(self):
            original.set_progress_handler(None, 0)
            self.cursor.close()

    class Connection:
        def __getattr__(self, name):
            return getattr(original, name)

        def execute(self, sql, *args):
            source = " ".join(sql.split()).startswith("SELECT idea_id, config") and "FROM ideas" in sql
            if source:
                original.set_progress_handler(progress, 100)
            cursor = original.execute(sql, *args)
            return Cursor(cursor) if source else cursor

    lake.conn = Connection()
    try:
        assert lake._repair_admitted_config_hashes() == 5000
        assert len(first) == 1
        assert original.execute("SELECT COUNT(*) FROM ideas WHERE config_hash IS NULL").fetchone()[0] == 0
    finally:
        original.set_progress_handler(None, 0)


@pytest.mark.parametrize("group", ["matching", "missing"])
def test_capacity_probe_can_produce_first_row_without_sorting_all_candidates(lake, monkeypatch, group):
    target = hash_config({"seed": 13})
    lake.conn.executemany(
        "INSERT INTO ideas(idea_id,title,config,raw_markdown,status,config_hash,config_source_sha256) "
        "VALUES (?,'Metadata',?,'',?,?,?)",
        ((f"history-{n:05d}", f"seed: {n}\n", STATUSES[n % 4],
          target if group == "matching" else None, "source") for n in range(5000)),
    )
    lake.conn.commit()
    before = _snapshot(lake)
    prepared = lake.prepare_proposal("idea-new", "New", "seed: 13\n", "")
    original, counts = lake.conn, []

    class Connection:
        def __getattr__(self, name):
            return getattr(original, name)

        def execute(self, sql, *args):
            if not sql.startswith("SELECT rowid AS candidate_order"):
                return original.execute(sql, *args)
            steps = [0]

            def progress():
                steps[0] += 100
                return 0

            original.set_progress_handler(progress, 100)
            try:
                cursor = original.execute(sql, *args)
            finally:
                original.set_progress_handler(None, 0)
            counts.append(steps[0])
            try:
                assert steps[0] < 20000, "candidate_first_row_sorts_dense_history"
            except BaseException:
                cursor.close()
                raise
            return cursor

    def no_qualification(value):
        raise AssertionError("over_capacity_must_not_qualify_rows")

    lake.conn = Connection()
    monkeypatch.setattr(yaml, "safe_load", no_qualification)
    result = admit_proposal(lake, prepared)
    assert result["status"] == "rejected" and result["reason"] == "proposal_dedup_capacity"
    assert len(counts) == (1 if group == "matching" else 2)
    assert not lake.conn.in_transaction
    assert _snapshot(lake) == before


@pytest.mark.parametrize("drop_indexes", [False, True])
def test_repair_keeps_rowid_order_status_and_missing_filters_with_gaps(lake, monkeypatch, drop_indexes):
    expected = []
    for n in range(400):
        status = (*STATUSES, "failed", "archived")[n % 6]
        identity = "filled" if n % 9 == 0 else None
        source = "filled"
        if n % 5 == 0 and n % 9:
            identity, source = "filled", None
        config = f"seed: {n}\n"
        lake.conn.execute(
            "INSERT INTO ideas(idea_id,title,config,raw_markdown,status,config_hash,config_source_sha256) "
            "VALUES (?,'Metadata',?,'',?,?,?)", (f"reverse-{400-n:04d}", config, status, identity, source),
        )
        if n % 11 == 0:
            lake.conn.execute("DELETE FROM ideas WHERE idea_id=?", (f"reverse-{400-n:04d}",))
        elif n % 6 < 4 and n % 9:
            expected.append(config)
    if drop_indexes:
        lake.conn.execute("DROP INDEX idx_status_config_hash_nocase")
        lake.conn.execute("DROP INDEX idx_missing_config_identity")
    lake.conn.commit()
    real_load, seen = yaml.safe_load, []

    def observe(value):
        seen.append(value)
        return real_load(value)

    monkeypatch.setattr(yaml, "safe_load", observe)
    assert lake._repair_admitted_config_hashes() == len(expected)
    assert seen == expected


@pytest.mark.parametrize("first_status", STATUSES)
@pytest.mark.parametrize("first_missing", [False, True])
@pytest.mark.parametrize("first_unavailable", [False, True])
def test_first_owner_and_unavailable_precedence_cross_status_and_hash_groups(
        lake, first_status, first_missing, first_unavailable):
    match = hash_config({"seed": 13})
    source = hashlib.sha256(b"seed: 13\n").hexdigest()
    first_config = "seed: 13\n#" + "x" * 65536 if first_unavailable else "{seed: 13}\n"
    second_status = STATUSES[(STATUSES.index(first_status) + 1) % len(STATUSES)]
    for key, config, status, missing in (("idea-z-first", first_config, first_status, first_missing),
                                        ("idea-a-second", "seed: 13\n", second_status, not first_missing)):
        lake.conn.execute(
            "INSERT INTO ideas(idea_id,title,config,raw_markdown,status,config_hash,config_source_sha256) "
            "VALUES (?,'Metadata',?,'',?,?,?)", (key, config, status, None if missing else match, source),
        )
    lake.conn.commit()
    before = _snapshot(lake)
    prepared = lake.prepare_proposal("idea-new", "New", "seed: 13\n", "")
    result = admit_proposal(lake, prepared)
    if first_unavailable:
        assert result["status"] == "rejected" and result["reason"] == "proposal_dedup_config_unavailable"
    else:
        assert result["status"] == "config_duplicate" and result["existing_id"] == "idea-z-first"
    assert _snapshot(lake) == before
