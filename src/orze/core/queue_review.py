"""Version-bound review decisions for existing, unclaimed queue entries.

CALLING SPEC:
    review_batch(db_path, limit=20) -> list[dict]
        Read at most 20 bounded, displayable queued records without creating
        authority or receipts. Scan past previously reviewed records.
    apply_review_decisions(db_path, batch, decisions, *, allow_skip=False,
                           allow_prioritize=False) -> list[dict]
        Recheck all selected revisions under one write transaction. Commit
        decision receipts and permitted mutations together, or reject all.

This is queue admission, not an experiment/observation ledger or a scientific
verdict. It never writes result files and cannot authorize missing lifecycle
rows. The supplied batch is framework-owned, not an untrusted capability.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sqlite3
from typing import Mapping

from orze.core.evaluation_retry_state import open_existing_lake
from orze.reporting.catalog import _display_config
from orze.reporting.evidence import _open_authoritative_lifecycle


MAX_BATCH = 20
_PAGE_SIZE = 128
_REVISION = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_TABLE = "idea_review_decisions"
_RECEIPT_COLUMNS = {
    "idea_id", "revision", "resulting_revision", "decision", "reason", "committed_at",
}
_REQUIRED = {
    "ideas": {"idea_id", "title", "hypothesis", "config", "kind", "priority", "status"},
    "idea_state": {"idea_id", "current_state", "updated_at", "first_queued_at", "queued_at"},
    "idea_stage_state": {"idea_id", "stage", "current_state"},
    "idea_transitions": {"id", "idea_id", "from_state", "to_state"},
}
_PRIMARY = {
    "ideas": ("idea_id",), "idea_state": ("idea_id",),
    "idea_stage_state": ("idea_id", "stage"), "idea_transitions": ("id",),
    _TABLE: ("idea_id", "revision"),
}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS idea_review_decisions (
    idea_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    resulting_revision TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    PRIMARY KEY (idea_id, revision)
)
"""
_SELECT = """
SELECT i.idea_id, i.title, i.hypothesis, i.config, i.kind, i.priority,
       s.updated_at, s.first_queued_at, s.queued_at,
       (SELECT MAX(t.id) FROM idea_transitions t WHERE t.idea_id=i.idea_id) AS last_transition
FROM ideas i JOIN idea_state s ON s.idea_id=i.idea_id
WHERE lower(i.status) IN ('queued', 'pending') AND s.current_state='QUEUED'
  AND NOT EXISTS (
    SELECT 1 FROM idea_stage_state stage WHERE stage.idea_id=i.idea_id
      AND (stage.stage NOT IN ('training', 'evaluation') OR stage.stage IS NULL
           OR stage.current_state IS NULL
           OR stage.current_state NOT IN ('NOT_STARTED', 'PENDING')))
  AND typeof(i.idea_id)='text' AND length(CAST(i.idea_id AS BLOB)) BETWEEN 1 AND 128
  AND typeof(i.config)='text' AND length(CAST(i.config AS BLOB)) <= 65536
  AND typeof(i.title)='text' AND length(CAST(i.title AS BLOB)) <= 4096
  AND (i.hypothesis IS NULL OR (typeof(i.hypothesis)='text'
       AND length(CAST(i.hypothesis AS BLOB)) <= 16384))
  AND typeof(i.kind)='text' AND length(CAST(i.kind AS BLOB)) BETWEEN 1 AND 128
  AND typeof(i.priority)='text' AND length(CAST(i.priority AS BLOB)) BETWEEN 1 AND 128
  AND (s.updated_at IS NULL OR (typeof(s.updated_at)='text'
       AND length(CAST(s.updated_at AS BLOB)) <= 256))
  AND (s.first_queued_at IS NULL OR (typeof(s.first_queued_at)='text'
       AND length(CAST(s.first_queued_at AS BLOB)) <= 256))
  AND (s.queued_at IS NULL OR (typeof(s.queued_at)='text'
       AND length(CAST(s.queued_at AS BLOB)) <= 256))
"""


class QueueReviewError(ValueError):
    """A bounded, content-free queue review refusal."""


def _safe_id(value):
    return isinstance(value, str) and _SAFE_ID.fullmatch(value) is not None


def _table_exists(connection, name):
    row = connection.execute(
        "SELECT type FROM sqlite_master WHERE name=?", (name,)).fetchone()
    if row is None:
        return False
    if row[0] != "table":
        raise QueueReviewError("queue_review_schema_invalid")
    return True


def _check_table(connection, name, required):
    if not _table_exists(connection, name):
        raise QueueReviewError("queue_review_schema_invalid")
    # All table names originate from module constants, not requests.
    columns = connection.execute(f"PRAGMA table_info({name})").fetchall()
    primary = tuple(row[1] for row in sorted(columns, key=lambda item: item[5]) if row[5])
    if not required.issubset({row[1] for row in columns}) or primary != _PRIMARY[name]:
        raise QueueReviewError("queue_review_schema_invalid")


def _check_schema(connection):
    for name, columns in _REQUIRED.items():
        _check_table(connection, name, columns)
    exists = _table_exists(connection, _TABLE)
    if exists:
        _check_table(connection, _TABLE, _RECEIPT_COLUMNS)
    return exists


def _revision(row, db_path):
    return hashlib.sha256(json.dumps(
        {"schema": 1, "database": str(Path(db_path).absolute()), "row": list(row)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def _display(row, db_path):
    if not _safe_id(row[0]):
        return None
    config, available = _display_config(row[3])
    if not available:
        # Omitted from review, never condemned as an invalid scientific idea.
        return None
    return {"id": row[0], "title": row[1], "hypothesis": row[2] or "",
            "config": config, "config_available": True, "kind": row[4],
            "revision": _revision(row, db_path)}


def _reviewed(connection, idea_id, revision):
    return connection.execute(
        "SELECT 1 FROM idea_review_decisions WHERE idea_id=? "
        "AND (revision=? OR resulting_revision=?) LIMIT 1",
        (idea_id, revision, revision),
    ).fetchone() is not None


def review_batch(db_path, limit=20):
    """Observe an existing queue; an unavailable database is not an empty one."""
    if type(limit) is not int or not 1 <= limit <= MAX_BATCH:
        raise QueueReviewError("queue_review_limit_invalid")
    path = Path(db_path).absolute()
    connection, reason = _open_authoritative_lifecycle(path)
    if connection is None:
        raise QueueReviewError("queue_review_" + reason)
    try:
        connection.execute("BEGIN")
        receipts = _check_schema(connection)
        batch, after = [], ""
        while len(batch) < limit:
            rows = connection.execute(
                _SELECT + " AND i.idea_id > ? ORDER BY i.idea_id LIMIT ?",
                (after, _PAGE_SIZE),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                after = row[0]
                item = _display(row, path)
                if item is None or (receipts and _reviewed(connection, row[0], item["revision"])):
                    continue
                batch.append(item)
                if len(batch) == limit:
                    break
            if len(rows) < _PAGE_SIZE:
                break
        return batch
    except (sqlite3.Error, OSError, TypeError) as exc:
        raise QueueReviewError("queue_review_read_failed") from exc
    finally:
        connection.close()


def _validated_decisions(batch, decisions, allow_skip, allow_prioritize):
    if type(allow_skip) is not bool or type(allow_prioritize) is not bool:
        raise QueueReviewError("queue_review_permissions_invalid")
    if not isinstance(batch, list) or not 1 <= len(batch) <= MAX_BATCH:
        raise QueueReviewError("queue_review_batch_invalid")
    submitted = {}
    for row in batch:
        if (not isinstance(row, Mapping) or not _safe_id(row.get("id"))
                or not isinstance(row.get("revision"), str)
                or _REVISION.fullmatch(row["revision"]) is None
                or row["id"] in submitted):
            raise QueueReviewError("queue_review_batch_invalid")
        submitted[row["id"]] = row["revision"]
    if not isinstance(decisions, list) or not len(decisions) <= MAX_BATCH:
        raise QueueReviewError("queue_review_decisions_invalid")
    selected, seen = [], set()
    for entry in decisions:
        if not isinstance(entry, Mapping):
            raise QueueReviewError("queue_review_decisions_invalid")
        idea_id, decision = entry.get("idea_id"), entry.get("decision")
        if (not _safe_id(idea_id) or idea_id not in submitted or idea_id in seen
                or decision not in ("APPROVE", "SKIP", "PRIORITIZE")):
            raise QueueReviewError("queue_review_decisions_invalid")
        if (decision == "SKIP" and not allow_skip
                or decision == "PRIORITIZE" and not allow_prioritize):
            raise QueueReviewError("queue_review_decision_not_permitted")
        revision = submitted[idea_id]
        if "revision" in entry and entry["revision"] != revision:
            raise QueueReviewError("queue_review_revision_mismatch")
        reason = entry.get("reason", "")
        if not isinstance(reason, str) or len(reason.encode("utf-8")) > 4096:
            raise QueueReviewError("queue_review_reason_invalid")
        selected.append({"idea_id": idea_id, "decision": decision,
                         "revision": revision, "reason": reason})
        seen.add(idea_id)
    return selected


def apply_review_decisions(db_path, batch, decisions, *, allow_skip=False,
                           allow_prioritize=False):
    """Commit selected decisions atomically; stale/replayed batches fail closed."""
    selected = _validated_decisions(batch, decisions, allow_skip, allow_prioritize)
    if not selected:
        return []
    try:
        lake = open_existing_lake(db_path)
    except (ValueError, OSError, sqlite3.Error) as exc:
        raise QueueReviewError("queue_review_database_unavailable") from exc
    connection = lake.conn
    try:
        connection.execute("BEGIN IMMEDIATE")
        receipts = _check_schema(connection)
        for decision in selected:
            if receipts and _reviewed(connection, decision["idea_id"], decision["revision"]):
                raise QueueReviewError("queue_review_already_applied")
            row = connection.execute(
                _SELECT + " AND i.idea_id=?", (decision["idea_id"],)).fetchone()
            if (row is None or _display(row, db_path) is None
                    or _revision(row, db_path) != decision["revision"]):
                raise QueueReviewError("queue_review_stale_or_ineligible")
        connection.execute(_SCHEMA)
        host, pid = socket.gethostname(), os.getpid()
        at = lake._transition_time(connection)
        for decision in selected:
            idea_id, action, reason = decision["idea_id"], decision["decision"], decision["reason"]
            resulting_revision = decision["revision"]
            if action == "SKIP":
                if not lake._write_state_row(
                        connection, idea_id, "SKIPPED", host, pid, "queue_review", at,
                        expected_state="QUEUED"):
                    raise QueueReviewError("queue_review_state_conflict")
                updated = connection.execute(
                    "UPDATE ideas SET status='skipped' WHERE idea_id=? "
                    "AND lower(status) IN ('queued', 'pending')", (idea_id,))
                if updated.rowcount != 1:
                    raise QueueReviewError("queue_review_status_conflict")
                transition = connection.execute(
                    "INSERT INTO idea_transitions "
                    "(idea_id,from_state,to_state,reason,host,pid,sop_type,ts) "
                    "VALUES (?,'QUEUED','SKIPPED',?,?,?,'queue_review',?)",
                    (idea_id, "queue_review:" + reason, host, pid, at),
                )
                recorded = connection.execute(
                    "SELECT idea_id,from_state,to_state,reason,host,pid,sop_type,ts "
                    "FROM idea_transitions WHERE id=?", (transition.lastrowid,),
                ).fetchone()
                expected = (idea_id, "QUEUED", "SKIPPED", "queue_review:" + reason,
                            host, pid, "queue_review", at)
                if transition.rowcount != 1 or recorded is None or tuple(recorded) != expected:
                    raise QueueReviewError("queue_review_transition_not_recorded")
            elif action == "PRIORITIZE":
                updated = connection.execute(
                    "UPDATE ideas SET priority='critical' WHERE idea_id=?", (idea_id,))
                row = connection.execute(_SELECT + " AND i.idea_id=?", (idea_id,)).fetchone()
                if updated.rowcount != 1 or row is None or row[5] != "critical":
                    raise QueueReviewError("queue_review_priority_conflict")
                resulting_revision = _revision(row, db_path)
            receipt = connection.execute(
                "INSERT INTO idea_review_decisions "
                "(idea_id,revision,resulting_revision,decision,reason,committed_at) "
                "VALUES (?,?,?,?,?,?)",
                (idea_id, decision["revision"], resulting_revision, action, reason, at),
            )
            recorded = connection.execute(
                "SELECT idea_id,revision,resulting_revision,decision,reason,committed_at "
                "FROM idea_review_decisions WHERE idea_id=? AND revision=?",
                (idea_id, decision["revision"]),
            ).fetchone()
            expected = (idea_id, decision["revision"], resulting_revision, action, reason, at)
            if receipt.rowcount != 1 or recorded is None or tuple(recorded) != expected:
                raise QueueReviewError("queue_review_receipt_not_recorded")
        connection.commit()
        return selected
    except Exception as exc:
        connection.rollback()
        if isinstance(exc, QueueReviewError):
            raise
        raise QueueReviewError("queue_review_commit_failed") from exc
    finally:
        lake.close()
