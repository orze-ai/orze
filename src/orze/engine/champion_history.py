"""Scoped, mutable operational history; not independent replication receipts.

CALLING SPEC:
    objective_scope(cfg) -> str
        Fingerprint the objective and qualification policy, not display labels.
    read_history(db_path, scope, limit) -> list[dict]
        Read the last first-accepted IDs without creating a project database.
    record_history(db_path, scope, idea_id, value, identity, limit) -> None
        Atomically upsert one idea revision and bound this scope's row count.

Legacy _champion_history.json is never read, migrated, or overwritten.
An accepted revision updates its ID's value but does not renew its window
position; repeated checks are not new independent samples.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from orze.core.sqlite_policy import apply_shared_database_policy
from orze.reporting.evidence import _open_authoritative_lifecycle

_TABLE = "champion_guard_history_v1"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS champion_guard_history_v1 (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,
    idea_id TEXT NOT NULL,
    metric REAL NOT NULL,
    evidence_identity TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(scope, idea_id)
)
"""


def objective_scope(cfg: dict) -> str:
    report = cfg["report"]
    sources = {
        str(column["key"]): str(column.get("source") or "")
        for column in report.get("columns") or []
        if isinstance(column, dict) and column.get("key")
    }
    policy = {
        "schema": 1,
        "primary": report["primary_metric"],
        "secondary": report.get("secondary_metric") or None,
        "sort": report.get("sort", "descending"),
        "sources": sources,
        "min_datasets": report.get("min_datasets", 0),
        "benchmark_contract": report.get("benchmark_contract"),
        "qualification": {
            key: cfg.get(key) for key in (
                "metric_validation", "model_lineage", "data_boundaries",
                "data_separation",
            )
        },
        "require_clean_training_access_log": (cfg.get("managed_run") or {}).get(
            "require_clean_training_access_log", False),
    }
    return hashlib.sha256(json.dumps(
        policy, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def read_history(db_path: Path, scope: str, limit: int) -> list[dict]:
    connection, reason = _open_authoritative_lifecycle(db_path)
    if connection is None:
        raise RuntimeError(reason)
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (_TABLE,)
        ).fetchone()
        if not exists:
            return []
        rows = connection.execute(
            "SELECT idea_id, metric, evidence_identity FROM champion_guard_history_v1 "
            "WHERE scope=? ORDER BY sequence DESC LIMIT ?", (scope, limit),
        ).fetchall()
        return [dict(zip(("idea_id", "metric", "evidence_identity"), row))
                for row in rows]
    finally:
        connection.close()


def record_history(db_path: Path, scope: str, idea_id: str, value: float,
                   identity: str, limit: int) -> None:
    # mode=rw never creates a missing project database. Qualification already
    # checked the scoped lifecycle database and its shared-filesystem policy.
    connection = sqlite3.connect(
        Path(db_path).absolute().as_uri() + "?mode=rw", uri=True, timeout=5)
    try:
        apply_shared_database_policy(connection)
        with connection:
            connection.execute(_SCHEMA)
            connection.execute(
                "INSERT INTO champion_guard_history_v1 "
                "(scope, idea_id, metric, evidence_identity, updated_at) "
                "VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) "
                "ON CONFLICT(scope, idea_id) DO UPDATE SET metric=excluded.metric, "
                "evidence_identity=excluded.evidence_identity, "
                "updated_at=excluded.updated_at",
                (scope, idea_id, value, identity),
            )
            connection.execute(
                "DELETE FROM champion_guard_history_v1 WHERE scope=? "
                "AND sequence NOT IN (SELECT sequence FROM champion_guard_history_v1 "
                "WHERE scope=? ORDER BY sequence DESC LIMIT ?)",
                (scope, scope, limit),
            )
    finally:
        connection.close()
