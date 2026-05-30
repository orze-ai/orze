"""Atomic trigger consumption ledger backed by idea_lake.db.

CALLING SPEC:
    claim_trigger(db_path, role_name, trigger_file, host=None, pid=None)
        -> Optional[TriggerClaim]
        Atomically claim a `_trigger_<role>` file. Returns a TriggerClaim
        on first-time consume; None if the file is absent OR an orphan
        from a prior crash (in which case it is unlinked).

    init_schema(conn) -> None
        Idempotent CREATE TABLE for ``trigger_consumptions``. Called
        defensively from claim_trigger and from IdeaLake._ensure_schema.

PURPOSE:
    Replaces the racy ``read_text() → unlink()`` pattern in orze-pro's
    role_runner.py (lines 472-480, 555-564, 1036-1041). Resolves c1005
    (DEC-009): the role_runner band-aid only protected the engineer
    channel; this ledger protects every consumer uniformly.

MODEL:
    * The trigger FILE is the message — producers continue to write
      ``_trigger_<role>`` files unchanged; no producer-side changes
      required.
    * The DB ROW is the receipt — INSERTed atomically by the consumer
      under a UNIQUE constraint on (role_name, fingerprint). A second
      consume attempt against the same file fingerprint raises
      IntegrityError, which we treat as "orphan from prior cycle" and
      silently unlink instead of re-firing.
    * The FINGERPRINT is (inode, size, mtime_ns). A fresh trigger
      written after the receipt was already recorded will have a new
      inode and mtime, so it is correctly treated as a NEW trigger.

CRASH RECOVERY:
    * Crash between INSERT and unlink → next claim sees the file,
      recomputes the same fingerprint, hits IntegrityError, unlinks
      via the orphan-cleanup path. No duplicate consume.
    * Crash before INSERT → next claim sees the file, INSERTs the
      receipt fresh, acts on it normally.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger("orze")


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trigger_consumptions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    role_name        TEXT    NOT NULL,
    file_path        TEXT    NOT NULL,
    fingerprint      TEXT    NOT NULL,
    payload          TEXT,
    consumed_at      TEXT    NOT NULL,
    consumed_by_host TEXT,
    consumed_by_pid  INTEGER,
    UNIQUE(role_name, fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_trigger_consumptions_role
    ON trigger_consumptions(role_name, consumed_at);
"""


@dataclass(frozen=True)
class TriggerClaim:
    role_name: str
    payload: str
    fingerprint: str
    file_path: str


def init_schema(conn: sqlite3.Connection) -> None:
    """Idempotently create the trigger_consumptions table + index."""
    conn.executescript(_SCHEMA_SQL)
    conn.commit()


def _open_short_lived(db_path: Union[str, Path]) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    # DELETE journal mode is required on Lustre/NFS where WAL's shared
    # mmap is unsupported. Mirrors orze.idea_lake.IdeaLake.__init__.
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def _fingerprint(p: Path) -> Optional[str]:
    try:
        st = p.stat()
    except OSError:
        return None
    return f"ino={st.st_ino}:size={st.st_size}:mtime_ns={st.st_mtime_ns}"


def claim_trigger(
    db_path: Union[str, Path],
    role_name: str,
    trigger_file: Union[str, Path],
    *,
    host: Optional[str] = None,
    pid: Optional[int] = None,
) -> Optional[TriggerClaim]:
    """Atomically claim a trigger file.

    Returns:
        TriggerClaim on first-time consume (receipt INSERTed, file unlinked).
        None on any of:
            * file does not exist
            * file is an orphan from a prior cycle (receipt already
              present for this fingerprint) — file is unlinked before
              returning
            * I/O error reading the file
    """
    trigger_file = Path(trigger_file)
    if not trigger_file.exists():
        return None
    fp = _fingerprint(trigger_file)
    if fp is None:
        return None
    try:
        payload = trigger_file.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(
            "[TRIGGER_CLAIM] read_failed role=%s path=%s err=%s",
            role_name, trigger_file, e,
        )
        return None

    host = host or os.uname().nodename
    pid = pid if pid is not None else os.getpid()

    conn = _open_short_lived(db_path)
    try:
        init_schema(conn)
        try:
            with conn:
                conn.execute(
                    "INSERT INTO trigger_consumptions "
                    "(role_name, file_path, fingerprint, payload, "
                    " consumed_at, consumed_by_host, consumed_by_pid) "
                    "VALUES (?, ?, ?, ?, datetime('now'), ?, ?)",
                    (role_name, str(trigger_file), fp, payload, host, pid),
                )
        except sqlite3.IntegrityError:
            # Receipt already exists — orphan from prior cycle or lost
            # race. The payload was already acted on by whoever
            # committed first; clean up the file so it doesn't re-fire.
            try:
                trigger_file.unlink(missing_ok=True)
            except OSError:
                pass
            logger.info(
                "[TRIGGER_CLAIM] orphan_cleaned role=%s path=%s fp=%s",
                role_name, trigger_file, fp,
            )
            return None
    finally:
        conn.close()

    # Receipt committed; unlink the file. A crash here is recoverable
    # via the orphan-cleanup path above.
    try:
        trigger_file.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(
            "[TRIGGER_CLAIM] unlink_failed_after_commit role=%s path=%s err=%s",
            role_name, trigger_file, e,
        )

    logger.info(
        "[TRIGGER_CLAIM] consumed role=%s path=%s payload_len=%d fp=%s "
        "host=%s pid=%d",
        role_name, trigger_file, len(payload), fp, host, pid,
    )
    return TriggerClaim(
        role_name=role_name,
        payload=payload,
        fingerprint=fp,
        file_path=str(trigger_file),
    )
