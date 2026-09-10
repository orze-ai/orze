"""Explicit, atomic evaluation-only retry against existing lifecycle authority.

CALLING SPEC:
    open_existing_lake(db_path) -> IdeaLake
        Open mode=rw only after path, policy, and schema checks. Never create,
        bootstrap, migrate, or change journal policy. Caller must close it.
    retry_evaluation(lake, idea_id, reason='evaluation_retry_requested',
                     *, prepare_artifacts=None) -> bool
        Reopen FAILED / training COMPLETE / evaluation FAILED as IN_PROGRESS /
        training COMPLETE / evaluation PENDING in one BEGIN IMMEDIATE transaction.
        Repeated pending requests are no-ops only after an audited retry edge.

The optional trusted coordinator callback receives the failed global transition
ID under the write lock, before any lifecycle changes. It must not commit or
mutate the database or launch work. Its filesystem operations must be recoverable
and idempotent by that ID: SQL rollback cannot undo files, and a busy transaction
may be retried. This is not protection against hostile concurrent path replacement.
"""

from __future__ import annotations

import os
import socket
import sqlite3
import stat
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from orze.core.sqlite_policy import SQLitePolicyError, inspect_shared_database_policy

if TYPE_CHECKING:
    from orze.idea_lake import IdeaLake


_REQUIRED_COLUMNS = {
    "ideas": {"idea_id", "status"},
    "idea_state": {
        "idea_id", "current_state", "updated_by_host", "updated_by_pid",
        "sop_type", "updated_at", "started_at", "terminal_at", "completed_at",
    },
    "idea_stage_state": {
        "idea_id", "stage", "current_state", "updated_at", "started_at", "terminal_at",
    },
    "idea_transitions": {
        "id", "idea_id", "from_state", "to_state", "reason", "host", "pid", "sop_type", "ts",
    },
    "idea_stage_transitions": {
        "id", "idea_id", "stage", "from_state", "to_state", "reason", "host", "pid", "ts",
    },
}
_PRIMARY_KEYS = {
    "ideas": ("idea_id",),
    "idea_state": ("idea_id",),
    "idea_stage_state": ("idea_id", "stage"),
    "idea_transitions": ("id",),
    "idea_stage_transitions": ("id",),
}


def open_existing_lake(db_path: str | Path) -> "IdeaLake":
    """Open compatible existing authority; reject unsafe paths without creation."""
    from orze.idea_lake import IdeaLake

    path = Path(db_path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("evaluation_retry_database_redirected")
    if not path.exists():
        raise ValueError("evaluation_retry_database_missing")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("evaluation_retry_database_not_regular")
    if metadata.st_nlink != 1:
        raise ValueError("evaluation_retry_database_redirected")

    connection = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=5)
    try:
        try:
            policy = inspect_shared_database_policy(connection)
        except SQLitePolicyError:
            raise ValueError("evaluation_retry_database_policy_invalid") from None
        if not policy["compliant"]:
            raise ValueError("evaluation_retry_database_policy_invalid")
        for table, required in _REQUIRED_COLUMNS.items():
            entry = connection.execute(
                "SELECT type FROM sqlite_master WHERE name=?", (table,),
            ).fetchone()
            columns = connection.execute(f"PRAGMA table_info({table})").fetchall()
            primary_key = tuple(
                column[1] for column in sorted(columns, key=lambda item: item[5])
                if column[5]
            )
            if (entry is None or entry[0] != "table"
                    or not required.issubset({column[1] for column in columns})
                    or primary_key != _PRIMARY_KEYS[table]):
                raise ValueError("evaluation_retry_database_schema_invalid")
        connection.row_factory = sqlite3.Row
        lake = IdeaLake.__new__(IdeaLake)
        lake.db_path = str(path)
        lake.conn = connection
        lake.schema_bootstrap_cache_hit = False
        return lake
    except Exception:
        connection.close()
        raise


def retry_evaluation(
    lake: "IdeaLake",
    idea_id: str,
    reason: str = "evaluation_retry_requested",
    *,
    prepare_artifacts: Callable[[int], None] | None = None,
) -> bool:
    """Reopen only a failed evaluation; never use the general training retry."""
    from orze.idea_lake import _retry_on_busy

    if not isinstance(idea_id, str) or not idea_id:
        return False
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("evaluation_retry_reason_invalid")
    if prepare_artifacts is not None and not callable(prepare_artifacts):
        raise ValueError("evaluation_retry_prepare_invalid")
    host, pid = socket.gethostname(), os.getpid()
    connection = lake.conn

    def transact():
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT i.status, s.current_state FROM ideas i "
                "JOIN idea_state s ON s.idea_id=i.idea_id WHERE i.idea_id=?",
                (idea_id,),
            ).fetchone()
            latest = connection.execute(
                "SELECT id, from_state, to_state FROM idea_transitions "
                "WHERE idea_id=? ORDER BY id DESC LIMIT 1", (idea_id,),
            ).fetchone()
            training = lake._stage_state_in_tx(idea_id, "training")
            evaluation = lake._stage_state_in_tx(idea_id, "evaluation")
            if row is None or latest is None or training != "COMPLETE":
                connection.rollback()
                return False
            status, state = str(row[0]).lower(), row[1]
            if (status == "running" and state == "IN_PROGRESS"
                    and evaluation == "PENDING"
                    and latest[1] == "FAILED" and latest[2] == "IN_PROGRESS"):
                connection.rollback()
                return True
            if not (status == "failed" and state == "FAILED"
                    and evaluation == "FAILED" and latest[2] == "FAILED"):
                connection.rollback()
                return False

            if prepare_artifacts is not None:
                prepare_artifacts(int(latest[0]))
                if not connection.in_transaction:
                    raise ValueError("evaluation_retry_prepare_ended_transaction")
            at = lake._transition_time(connection)
            if not lake._write_state_row(
                    connection, idea_id, "IN_PROGRESS", host, pid, "training", at,
                    expected_state="FAILED"):
                raise ValueError("evaluation_retry_state_transition_rejected")
            if not lake._record_stage_transition_in_tx(
                    idea_id, "evaluation", "FAILED", "PENDING", reason, host, pid, at):
                raise ValueError("evaluation_retry_stage_transition_rejected")
            connection.execute(
                "INSERT INTO idea_transitions "
                "(idea_id, from_state, to_state, reason, host, pid, sop_type, ts) "
                "VALUES (?, 'FAILED', 'IN_PROGRESS', ?, ?, ?, 'training', ?)",
                (idea_id, reason, host, pid, at),
            )
            updated = connection.execute(
                "UPDATE ideas SET status='running' WHERE idea_id=? AND lower(status)='failed'",
                (idea_id,),
            )
            if updated.rowcount != 1:
                raise ValueError("evaluation_retry_status_transition_rejected")
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise

    return bool(_retry_on_busy(transact))
