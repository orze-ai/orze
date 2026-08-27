"""Fail-closed SQLite policy for Orze databases on shared filesystems.

SQLite recommends avoiding database files over network filesystems where
possible and says rollback mode can mitigate some network sync/locking hazards.
WAL additionally requires a shared-memory wal-index and is not Orze's supported
multi-host mode. See https://sqlite.org/useovernet.html and
https://sqlite.org/wal.html.
"""

from __future__ import annotations

import sqlite3


POLICY_VERSION = 1
REQUIRED_JOURNAL_MODE = "delete"
REQUIRED_SYNCHRONOUS = 2  # FULL
REQUIRED_LOCKING_MODE = "normal"


class SQLitePolicyError(RuntimeError):
    """Stable policy failure; message contains a closed reason code only."""

    def __init__(self, code: str):
        self.code = str(code)
        super().__init__(self.code)


def _scalar(connection: sqlite3.Connection, pragma: str):
    row = connection.execute(pragma).fetchone()
    return row[0] if row else None


def apply_shared_database_policy(connection: sqlite3.Connection) -> dict:
    """Apply and verify Orze's conservative shared-filesystem policy.

    This must run before a transaction begins. It verifies SQLite's returned
    values instead of assuming a PRAGMA request succeeded. No database content
    is read, logged, or returned.
    """
    if connection.in_transaction:
        raise SQLitePolicyError("sqlite_policy_applied_inside_transaction")
    try:
        journal_mode = str(
            _scalar(connection, "PRAGMA journal_mode=DELETE") or ""
        ).lower()
    except sqlite3.Error:
        raise SQLitePolicyError(
            "sqlite_journal_mode_transition_failed"
        ) from None
    if journal_mode != REQUIRED_JOURNAL_MODE:
        raise SQLitePolicyError("sqlite_delete_journal_unavailable")

    try:
        connection.execute("PRAGMA synchronous=FULL")
        synchronous = _scalar(connection, "PRAGMA synchronous")
        locking_mode = str(
            _scalar(connection, "PRAGMA locking_mode=NORMAL") or ""
        ).lower()
    except sqlite3.Error:
        raise SQLitePolicyError("sqlite_rollback_policy_unavailable") from None
    if synchronous != REQUIRED_SYNCHRONOUS:
        raise SQLitePolicyError("sqlite_full_synchronous_unavailable")
    if locking_mode != REQUIRED_LOCKING_MODE:
        raise SQLitePolicyError("sqlite_normal_locking_unavailable")
    return {
        "policy_version": POLICY_VERSION,
        "journal_mode": journal_mode,
        "synchronous": "full",
        "locking_mode": locking_mode,
    }


def inspect_shared_database_policy(connection: sqlite3.Connection) -> dict:
    """Inspect current connection state without attempting a mode transition."""
    try:
        journal_mode = str(
            _scalar(connection, "PRAGMA journal_mode") or ""
        ).lower()
        synchronous = _scalar(connection, "PRAGMA synchronous")
        locking_mode = str(
            _scalar(connection, "PRAGMA locking_mode") or ""
        ).lower()
    except sqlite3.Error:
        raise SQLitePolicyError("sqlite_policy_inspection_failed") from None
    return {
        "policy_version": POLICY_VERSION,
        "journal_mode": journal_mode,
        "synchronous": synchronous,
        "locking_mode": locking_mode,
        "compliant": (
            journal_mode == REQUIRED_JOURNAL_MODE
            and synchronous == REQUIRED_SYNCHRONOUS
            and locking_mode == REQUIRED_LOCKING_MODE
        ),
    }
