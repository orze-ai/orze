"""Read-only boundary for legacy, tokenless executor repair.

``require_legacy_executor_scope(idea_dir, cfg)`` grants no native repair
authority. Any durable task routing declaration, claim DB route, or registered
execution-attempt history for this task requires a future explicit repair
action instead. Even a closed ``legacy_import`` row is registered history;
mutable origin labels never authorize tokenless repair.

Only a genuinely absent default database, or a readable pre-native database
without this task's attempt history, retains the old legacy behavior. Explicit
missing/invalid databases and unverifiable paths fail closed. SQLite is opened
mode=ro, without IdeaLake construction, schema creation, migration, or writes.
The check is repeatable at entry and immediately before GO; it is not a durable
repair reservation, a held database lock, or cross-restart process adoption.
"""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import sqlite3

from orze.core.execution_attempts import current_attempt
from orze.engine.claim_authority import read_claim, safe_file
from orze.engine.execution_catalog import declared_catalog
from orze.engine.termination_hold import TerminationUnconfirmed
from orze.reporting.evidence import report_lifecycle_db_path


def _check(idea_dir, cfg):
    if not isinstance(cfg, Mapping):
        raise ValueError("configuration")
    folder = Path(idea_dir).absolute()
    if declared_catalog(folder) is not None:
        raise TerminationUnconfirmed("executor_native_scope_requires_explicit_repair")
    claim = read_claim(folder / "claim.json")
    if claim is not None and "lifecycle_db" in claim:
        raise TerminationUnconfirmed("executor_native_scope_requires_explicit_repair")

    explicit = "idea_lake_db" in cfg
    if explicit:
        route = cfg["idea_lake_db"]
        if not isinstance(route, (str, os.PathLike)) or not os.fspath(route):
            raise ValueError("database route")
    path = report_lifecycle_db_path(folder.parent, cfg).absolute()
    before = safe_file(path, missing=not explicit)
    if before is None:
        return
    if before.st_size < 100:
        raise ValueError("database file")

    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)
    try:
        conn.execute("PRAGMA query_only=ON")
        # This public reader validates the exact main schema without DDL.
        # Missing table is the explicitly supported pre-native database case.
        current_attempt(conn, folder.name, "training")
        table = conn.execute(
            "SELECT 1 FROM main.sqlite_master WHERE name='execution_attempts'"
        ).fetchone()
        if table is not None and conn.execute(
                "SELECT 1 FROM main.execution_attempts WHERE task_id=? COLLATE BINARY LIMIT 1",
                (folder.name,)).fetchone() is not None:
            raise TerminationUnconfirmed("executor_native_scope_requires_explicit_repair")
        after = safe_file(path, missing=False)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("database changed")
    finally:
        conn.close()


def require_legacy_executor_scope(idea_dir, cfg):
    """Return None only for readable legacy scope; otherwise raise HOLD.

    Diagnostics contain fixed reason codes, never arbitrary input or database
    error strings. Native rejection is independent of max-fix/tool-policy flags
    so callers must invoke this gate before their ordinary False early returns.
    """
    try:
        _check(idea_dir, cfg)
    except TerminationUnconfirmed:
        raise
    except Exception as exc:
        raise TerminationUnconfirmed("executor_legacy_scope_unverifiable") from exc
