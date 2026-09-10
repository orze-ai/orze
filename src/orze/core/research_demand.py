"""Closed, read-only persistent lifecycle demand for research cadence.

CALLING SPEC:
    load_persistent_demand(db_path) -> DemandSnapshot
        Read existing policy-compliant authority in one transaction. Never
        create, migrate, reconcile, parse task configurations or admit work.
    snapshot.counts.queued -> int
        A known lower bound when available, even with other unknown tasks.
    snapshot.queue_count / snapshot.waiting_count -> int | None
        Exact classified counts only when available and complete. Waiting is
        queued + evaluation_pending, never active/claimed tasks.

This describes recorded backlog, not executable readiness, process ownership,
GPU/CPU capacity, evaluation artifact validity or permission to launch. Missing
stage history can preserve queued/completed compatibility; an unclassified
IN_PROGRESS row cannot establish that evaluation demand is zero. No result
artifacts or potentially large title/configuration/metric columns are read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sqlite3

from orze.reporting.evidence import (
    _LIFECYCLE_STATUS_STATES, _open_authoritative_lifecycle,
)
from orze.reporting.lifecycle_stages import (
    completed_stages_agree, validate_lifecycle_schema,
)


@dataclass(frozen=True)
class DemandCounts:
    queued: int = 0
    evaluation_pending: int = 0
    claimed: int = 0
    in_progress: int = 0
    inactive: int = 0
    unknown: int = 0

    @property
    def total(self) -> int:
        return (self.queued + self.evaluation_pending + self.claimed
                + self.in_progress + self.inactive + self.unknown)


@dataclass(frozen=True)
class DemandSnapshot:
    available: bool
    complete: bool
    reason: str
    counts: DemandCounts = field(default_factory=DemandCounts)

    @property
    def queue_count(self) -> int | None:
        return self.counts.queued if self.available and self.complete else None

    @property
    def waiting_count(self) -> int | None:
        if not self.available or not self.complete:
            return None
        return self.counts.queued + self.counts.evaluation_pending


_STAGES = ("NOT_STARTED", "PENDING", "IN_PROGRESS", "COMPLETE", "FAILED", "SKIPPED")
_GLOBALS = tuple(sorted({state for states in _LIFECYCLE_STATUS_STATES.values() for state in states}))
_QUEUED_STAGES = (None, "NOT_STARTED", "PENDING")
_CHUNK_SIZE = 256


def _known_value(expression, values):
    # Both identifiers and enum values are internal constants, never input.
    choices = ",".join("'" + value + "'" for value in values)
    return (f"CASE WHEN typeof({expression})='text' AND "
            f"{expression} COLLATE BINARY IN ({choices}) THEN {expression} END")


def _query(schema):
    # SQLite lower() coerces a BLOB into text; inspect the stored type first.
    mirror = ("CASE WHEN typeof(i.status)='text' THEN "
              + _known_value("lower(i.status)", tuple(_LIFECYCLE_STATUS_STATES)) + " END")
    state = _known_value("s.current_state", _GLOBALS)
    if schema["idea_stage_state"] is None:
        stages, joins = "0,NULL,0,NULL,0", ""
    else:
        stages = ("t.idea_id IS NOT NULL," + _known_value("t.current_state", _STAGES)
                  + ",e.idea_id IS NOT NULL," + _known_value("e.current_state", _STAGES))
        # A misspelled case variant of a framework stage is not compatible
        # historical absence. Other exact stage names gain no new semantics.
        stages += (",EXISTS (SELECT 1 FROM idea_stage_state bad "
                   "WHERE bad.idea_id COLLATE BINARY=i.idea_id COLLATE BINARY "
                   "AND lower(bad.stage) COLLATE BINARY IN ('training','evaluation') "
                   "AND bad.stage COLLATE BINARY NOT IN ('training','evaluation'))")
        joins = (
            " LEFT JOIN idea_stage_state t ON t.idea_id COLLATE BINARY=i.idea_id COLLATE BINARY"
            " AND t.stage COLLATE BINARY='training'"
            " LEFT JOIN idea_stage_state e ON e.idea_id COLLATE BINARY=i.idea_id COLLATE BINARY"
            " AND e.stage COLLATE BINARY='evaluation'"
        )
    # SQL normalizes each selected state to a bounded known enum or NULL.
    # Even corrupt huge text values never enter the Python result buffers.
    return (
        "SELECT typeof(i.idea_id)='text' AND length(CAST(i.idea_id AS BLOB)) BETWEEN 1 AND 128,"
        + mirror + "," + state + "," + stages
        + " FROM ideas i LEFT JOIN idea_state s"
        " ON s.idea_id COLLATE BINARY=i.idea_id COLLATE BINARY" + joins
    )


def _category(row):
    valid_id, mirror, state, has_training, training, has_evaluation, evaluation, bad_stage = row
    if (not valid_id or bad_stage
            or state not in _LIFECYCLE_STATUS_STATES.get(mirror, ())
            or (has_training and training is None)
            or (has_evaluation and evaluation is None)):
        return "unknown"
    if state in ("QUEUED", "CLAIMED"):
        if training not in _QUEUED_STAGES or evaluation not in _QUEUED_STAGES:
            return "unknown"
        return "queued" if state == "QUEUED" else "claimed"
    if state == "COMPLETE":
        if not completed_stages_agree(
                1 if has_training else None, training,
                1 if has_evaluation else None, evaluation):
            return "unknown"
        return "inactive"
    if state in ("FAILED", "SKIPPED", "ARCHIVED"):
        if "IN_PROGRESS" in (training, evaluation):
            return "unknown"
        return "inactive"
    if state == "IN_PROGRESS":
        if training == "COMPLETE" and evaluation == "PENDING":
            return "evaluation_pending"
        if training == "IN_PROGRESS" and evaluation in _QUEUED_STAGES:
            return "in_progress"
        if evaluation == "IN_PROGRESS" and training in (None, "COMPLETE"):
            return "in_progress"
        # An old global running row without enough phase evidence could be
        # waiting for evaluation; never manufacture an empty waiting queue.
    return "unknown"


def load_persistent_demand(db_path: str | Path) -> DemandSnapshot:
    """Return a detached bounded-memory observation, with unknown != zero."""
    try:
        connection, reason = _open_authoritative_lifecycle(Path(db_path))
    except (TypeError, ValueError, OSError, sqlite3.Error):
        return DemandSnapshot(False, False, "persistent_demand_database_invalid")
    if connection is None:
        return DemandSnapshot(False, False, reason)
    result = DemandSnapshot(False, False, "persistent_demand_invalid")
    try:
        connection.execute("BEGIN")
        schema = validate_lifecycle_schema(connection)
        counts = {name: 0 for name in DemandCounts.__dataclass_fields__}
        cursor = connection.execute(_query(schema))
        while rows := cursor.fetchmany(_CHUNK_SIZE):
            for row in rows:
                counts[_category(row)] += 1
        complete = counts["unknown"] == 0
        result = DemandSnapshot(
            True, complete,
            "persistent_demand_loaded" if complete else "persistent_demand_incomplete",
            DemandCounts(**counts),
        )
    except (sqlite3.Error, OSError, ValueError, TypeError):
        # No partial lower bound is trusted after a structural/read failure.
        pass
    finally:
        try:
            connection.close()
        except (sqlite3.Error, OSError):
            result = DemandSnapshot(False, False, "persistent_demand_close_unconfirmed")
    return result
