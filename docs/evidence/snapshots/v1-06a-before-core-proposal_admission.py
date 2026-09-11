"""Atomic, non-replacing admission of queued task proposals.

CALLING SPEC:
    admit_proposal(lake, prepared) -> dict
        Internal IdeaLake.insert(if_absent=True) branch. Own BEGIN IMMEDIATE;
        return status inserted / already_present_exact / conflict /
        config_duplicate / rejected, plus idea_id and a content-free reason.
        Existing source identity is checked before config-dedup policy. Exact
        replay changes no metadata, result mirrors, lifecycle, or timestamps.

This is task admission, not an observation/attempt ledger. The caller owns source
acknowledgment. Only inserted or already_present_exact can acknowledge source
bytes; neither rejected nor config_duplicate is a durable rejection receipt.
The default IdeaLake.insert remains the legacy import/update API. No filesystem
operation or provider call occurs here. The short busy timeout is restored and
no transaction supplied by the caller is committed or rolled back.
"""
from __future__ import annotations

import os
import re
import socket
import sqlite3

import yaml

from orze.core.integrity import hash_config


_SOURCE_FIELDS = (
    "title", "config", "raw_markdown", "priority", "category", "parent",
    "hypothesis", "approach_family", "kind",
)
_IDEA_FIELDS = (
    "idea_id", "id_num", "title", "priority", "category", "parent", "hypothesis",
    "config", "config_hash", "config_source_sha256", "raw_markdown",
    "config_summary", "eval_metrics", "status", "training_time", "archived_at",
    "created_at", "approach_family", "kind",
)
_ADMITTED_STATUSES = ("queued", "pending", "running", "completed")
_MAX_CONFIG_BYTES = 65536
_MAX_DEDUP_CANDIDATES = 1024
_BUSY_TIMEOUT_MS = 1000


class _Rejected(Exception):
    pass


def _result(prepared, status, reason, **extra):
    return {"status": status, "reason": reason, "idea_id": prepared["idea_id"], **extra}


def _dedup_owner(connection, prepared):
    # Native config-update triggers invalidate derived hashes. Read their null
    # entries in this transaction without calling the repair API (which commits).
    # Check matching cached rows against actual YAML as well, so stale matching
    # hashes cannot authorize a false duplicate. Do not migrate/cache old rows.
    rows = connection.execute(
        "SELECT idea_id, CASE WHEN typeof(config)='text' "
        "AND length(CAST(config AS BLOB)) <= ? THEN config ELSE NULL END AS config "
        "FROM ideas WHERE status COLLATE NOCASE IN (?, ?, ?, ?) "
        "AND (config_hash=? OR config_hash IS NULL OR config_source_sha256 IS NULL) "
        "ORDER BY rowid LIMIT ?",
        (_MAX_CONFIG_BYTES, *_ADMITTED_STATUSES, prepared["config_hash"],
         _MAX_DEDUP_CANDIDATES + 1),
    ).fetchall()
    if len(rows) > _MAX_DEDUP_CANDIDATES:
        raise _Rejected("proposal_dedup_capacity")
    for row in rows:
        if row["config"] is None:
            raise _Rejected("proposal_dedup_config_unavailable")
        try:
            parsed = yaml.safe_load(row["config"])
            if isinstance(parsed, dict) and hash_config(parsed) == prepared["config_hash"]:
                return row["idea_id"]
        except (yaml.YAMLError, TypeError, ValueError, RecursionError):
            # Historical unparseable configs had no dedup identity either.
            continue
    return None


def _new_rows(lake, prepared):
    connection = lake.conn
    idea_id = prepared["idea_id"]
    at = lake._transition_time(connection)
    data = {**prepared, "status": "queued", "archived_at": at,
            "created_at": prepared["created_at"] or at}
    cursor = connection.execute(
        f"INSERT INTO ideas ({', '.join(_IDEA_FIELDS)}) "
        f"VALUES ({', '.join('?' for _ in _IDEA_FIELDS)})",
        tuple(data[field] for field in _IDEA_FIELDS),
    )
    if cursor.rowcount != 1:
        raise _Rejected("proposal_insert_not_applied")

    host, pid = socket.gethostname(), os.getpid()
    state = {
        "idea_id": idea_id, "current_state": "QUEUED", "updated_by_host": host,
        "updated_by_pid": pid, "sop_type": "training", "updated_at": at,
        "first_queued_at": at, "queued_at": at, "claimed_at": None,
        "started_at": None, "terminal_at": None, "completed_at": None,
    }
    cursor = connection.execute(
        f"INSERT INTO idea_state ({', '.join(state)}) "
        f"VALUES ({', '.join('?' for _ in state)})", tuple(state.values()),
    )
    if cursor.rowcount != 1:
        raise _Rejected("proposal_state_not_applied")
    transition = {
        "idea_id": idea_id, "from_state": "UNKNOWN", "to_state": "QUEUED",
        "reason": "proposal_admitted", "host": host, "pid": pid,
        "sop_type": "training", "ts": at,
    }
    cursor = connection.execute(
        f"INSERT INTO idea_transitions ({', '.join(transition)}) "
        f"VALUES ({', '.join('?' for _ in transition)})", tuple(transition.values()),
    )
    if cursor.rowcount != 1:
        raise _Rejected("proposal_transition_not_applied")
    transition_id = cursor.lastrowid
    for table, expected, key, value in (
        ("ideas", data, "idea_id", idea_id),
        ("idea_state", state, "idea_id", idea_id),
        ("idea_transitions", {**transition, "id": transition_id}, "id", transition_id),
    ):
        rows = connection.execute(
            f"SELECT {', '.join(expected)} FROM {table} WHERE {key}=?", (value,),
        ).fetchall()
        if len(rows) != 1 or dict(rows[0]) != expected:
            raise _Rejected("proposal_readback_mismatch")
    return transition_id


def admit_proposal(lake, prepared):
    """Atomically check-and-create one queued task without replacing a winner."""
    connection = lake.conn
    if connection.in_transaction:
        return _result(prepared, "rejected", "proposal_caller_transaction")
    if str(prepared["status"]).lower() != "queued":
        return _result(prepared, "rejected", "proposal_requires_queued")
    if (not isinstance(prepared["idea_id"], str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", prepared["idea_id"]) is None):
        return _result(prepared, "rejected", "proposal_id_invalid")
    if (prepared["config_hash"] is None
            or not isinstance(prepared["config"], str)
            or len(prepared["config"].encode("utf-8")) > _MAX_CONFIG_BYTES):
        return _result(prepared, "rejected", "proposal_config_invalid")
    # The public entry has already prepared the complete metadata; reject
    # non-text identity fields instead of letting SQLite silently coerce them.
    if any(value is not None and not isinstance(value, str)
           for value in (prepared[field] for field in _SOURCE_FIELDS)):
        return _result(prepared, "rejected", "proposal_source_invalid")

    timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    owns_transaction = False
    try:
        connection.execute("BEGIN IMMEDIATE")
        owns_transaction = True
        existing = connection.execute(
            f"SELECT {', '.join(_SOURCE_FIELDS)} FROM ideas WHERE idea_id COLLATE BINARY=?",
            (prepared["idea_id"],),
        ).fetchall()
        if existing:
            exact = (len(existing) == 1 and all(
                existing[0][field] == prepared[field] for field in _SOURCE_FIELDS))
            connection.rollback()
            return _result(prepared, "already_present_exact" if exact else "conflict",
                           "proposal_exact_replay" if exact else "proposal_identity_conflict")
        for table in ("idea_state", "idea_stage_state", "idea_transitions", "idea_stage_transitions"):
            if connection.execute(
                f"SELECT 1 FROM {table} WHERE idea_id=? LIMIT 1", (prepared["idea_id"],),
            ).fetchone() is not None:
                raise _Rejected("proposal_orphan_lifecycle")
        owner = _dedup_owner(connection, prepared)
        if owner is not None:
            connection.rollback()
            return _result(prepared, "config_duplicate", "proposal_config_duplicate", existing_id=owner)
        transition_id = _new_rows(lake, prepared)
        connection.commit()
        return _result(prepared, "inserted", "proposal_admitted", transition_id=transition_id)
    except _Rejected as exc:
        if owns_transaction:
            connection.rollback()
        return _result(prepared, "rejected", str(exc))
    except sqlite3.Error:
        if owns_transaction:
            connection.rollback()
        return _result(prepared, "rejected", "proposal_storage_error")
    except BaseException:
        if owns_transaction:
            connection.rollback()
        raise
    finally:
        connection.execute(f"PRAGMA busy_timeout={int(timeout)}")
