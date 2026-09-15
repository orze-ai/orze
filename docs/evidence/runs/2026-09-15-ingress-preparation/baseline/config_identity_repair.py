"""Bounded collections for legacy config identity preparation, using disk staging.

CALLING SPEC:
    repair_admitted_identities(lake, *, retry, hasher, logger) -> int
        Preserve the legacy source-text compare and one atomic main-DB update.
        Copy one source SELECT to private temporary SQLite storage in 128-row
        batches, close that source cursor, derive identities in bounded batches,
        then apply all valid repairs in one main writer transaction. No caller
        transaction is committed or rolled back if BEGIN itself cannot succeed.

No source/admission result or execution right is cached. The temporary directory
is owned by this call and removed on exit. Both databases keep rollback journaling
and FULL sync. This bounds row collections, not total I/O, temporary disk usage,
one legacy config's size, YAML expansion, or returned all-history lookup maps.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import tempfile

import yaml

from orze.core.sqlite_policy import apply_shared_database_policy

BATCH_ROWS = 128
STAGE_CACHE_KIB = 512
_STATUSES = ("queued", "pending", "running", "completed")
_MISSING = ("FROM ideas WHERE status COLLATE NOCASE IN (?, ?, ?, ?) "
            "AND (config_hash IS NULL OR config_source_sha256 IS NULL) ")
# Each disjoint status can use the existing partial index in rowid order.
# One compound SELECT preserves the source snapshot and global order while
# allowing SQLite to merge indexed streams instead of sorting config payloads.
# An absent optional index may change the plan, never the source semantics.
_ORDERED_MISSING = " UNION ALL ".join(
    "SELECT idea_id, config, rowid AS source_rowid FROM ideas "
    "WHERE status COLLATE NOCASE = ? "
    "AND (config_hash IS NULL OR config_source_sha256 IS NULL)"
    for _ in _STATUSES
) + " ORDER BY source_rowid"


def _has_missing(connection):
    with closing(connection.execute("SELECT 1 " + _MISSING + "LIMIT 1", _STATUSES)) as cursor:
        return cursor.fetchone() is not None


def _copy_snapshot(connection, stage):
    """Keep the source SELECT consistent; release it before parsing or syncing."""
    try:
        stage.execute("DELETE FROM inputs")
        with closing(connection.execute(_ORDERED_MISSING, _STATUSES)) as cursor:
            while True:
                rows = cursor.fetchmany(BATCH_ROWS)
                if not rows:
                    break
                stage.executemany("INSERT INTO inputs(idea_id, config_yaml) VALUES (?, ?)",
                                  ((row["idea_id"], row["config"] or "") for row in rows))
        stage.commit()
    except BaseException:
        stage.rollback()
        raise


def _derive(stage, hasher, logger):
    last = count = 0
    while True:
        rows = stage.execute("SELECT seq, idea_id, config_yaml FROM inputs WHERE seq > ? ORDER BY seq LIMIT ?",
                             (last, BATCH_ROWS)).fetchall()
        if not rows:
            return count
        repairs = []
        for sequence, idea_id, config_yaml in rows:
            source_sha256 = hashlib.sha256(config_yaml.encode("utf-8")).hexdigest()
            try:
                config = yaml.safe_load(config_yaml) or {}
            except yaml.YAMLError:
                logger.warning("Cannot derive config identity for %s: invalid YAML", idea_id)
                continue
            if not isinstance(config, dict):
                logger.warning("Cannot derive config identity for %s: config is not a mapping", idea_id)
                continue
            repairs.append((hasher(config), source_sha256, sequence))
        stage.executemany("UPDATE inputs SET config_hash=?, source_sha256=? WHERE seq=?", repairs)
        stage.commit()
        count += len(repairs)
        last = rows[-1][0]


def _apply(connection, stage):
    # BEGIN stays outside the handler: never roll back a caller-owned writer.
    connection.execute("BEGIN IMMEDIATE")
    try:
        with closing(stage.execute("SELECT config_hash, source_sha256, idea_id, config_yaml "
                                   "FROM inputs WHERE config_hash IS NOT NULL ORDER BY seq")) as cursor:
            while True:
                repairs = cursor.fetchmany(BATCH_ROWS)
                if not repairs:
                    break
                for repair in repairs:
                    connection.execute("UPDATE ideas SET config_hash=?, config_source_sha256=? "
                                       "WHERE idea_id=? AND config=?", repair)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def repair_admitted_identities(lake, *, retry, hasher, logger):
    """Prepare every missing identity without retaining all configs or repairs."""
    if not retry(lambda: _has_missing(lake.conn)):
        return 0
    with tempfile.TemporaryDirectory(prefix="orze-config-repair-") as directory:
        with closing(sqlite3.connect(str(Path(directory) / "stage.db"))) as stage:
            apply_shared_database_policy(stage)
            stage.execute(f"PRAGMA cache_size=-{STAGE_CACHE_KIB}")
            if stage.execute("PRAGMA cache_size").fetchone()[0] != -STAGE_CACHE_KIB:
                raise RuntimeError("config_identity_stage_cache_unconfirmed")
            stage.execute("CREATE TABLE inputs(seq INTEGER PRIMARY KEY, idea_id, config_yaml, config_hash, source_sha256)")
            retry(lambda: _copy_snapshot(lake.conn, stage))
            count = _derive(stage, hasher, logger)
            if count:
                retry(lambda: _apply(lake.conn, stage))
                logger.info("Backfilled %d admitted config identities", count)
            return count
