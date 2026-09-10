"""Closed, read-only lifecycle snapshots for observer entry points.

CALLING SPEC:
    load_catalog_snapshot(db_path, *, include_configs=False) -> CatalogSnapshot
        Inspect an existing policy-compliant DB in one read transaction. Never
        create, initialize or migrate a database; returned values hold no conn.
    CatalogSnapshot.get_metadata_index() / get_lifecycle_counts() -> dict
        Report-compatible copies. Pipeline counts retain the existing audited
        FSM basis, distinct from the agreed lifecycle_state used by admin.

Records contain lightweight metadata, never cached metrics. Admin can opt into
bounded JSON-compatible configuration display; unavailable configuration does
not change task state. This is a point-in-time lifecycle view, not current
execution authority or a globally atomic snapshot with result files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import math
from pathlib import Path
import sqlite3

import yaml

from orze.reporting.evidence import (
    _LIFECYCLE_STATUS_STATES, _open_authoritative_lifecycle,
)
from orze.reporting.lifecycle_stages import (
    completed_stages_agree, stage_projection, validate_lifecycle_schema,
)


@dataclass(frozen=True)
class CatalogSnapshot:
    db_path: Path
    reason: str
    records: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.reason == "authoritative_lifecycle_loaded"

    def get_metadata_index(self) -> dict:
        return deepcopy(self.records)

    def get_lifecycle_counts(self) -> dict:
        return dict(self.counts)

    @property
    def unknown_count(self) -> int:
        return sum(row["lifecycle_state"] == "UNKNOWN"
                   for row in self.records.values())


def _agreed_state(status, state, training_id, training, evaluation_id, evaluation):
    normalized_status = str(status or "").strip().lower()
    if state is None:
        return "UNKNOWN", "lifecycle_state_missing"
    if state not in _LIFECYCLE_STATUS_STATES.get(normalized_status, frozenset()):
        return "UNKNOWN", "lifecycle_state_conflict"
    # Missing historical stage rows remain unrecorded, not manufactured as
    # success. Training-only tasks can legitimately skip evaluation.
    if state == "COMPLETE" and not completed_stages_agree(
            training_id, training, evaluation_id, evaluation):
        return "UNKNOWN", "lifecycle_stage_conflict"
    return state, "lifecycle_agreed"


_MAX_CONFIG_BYTES = 64 * 1024


def _display_config(raw):
    """Bound parsing as well as the returned JSON tree; never run YAML tags.

    Alias-bearing YAML remains valid execution configuration, but this optional
    observer display declines it before constructors/merge expansion. Callers
    must not interpret unavailable display data as invalid task configuration.
    """
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > _MAX_CONFIG_BYTES:
        return {}, False
    try:
        depth = events = 0
        for event in yaml.parse(raw, Loader=yaml.SafeLoader):
            events += 1
            if isinstance(event, yaml.events.AliasEvent) or events > 4096:
                return {}, False
            if isinstance(event, (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent)):
                depth += 1
                if depth > 32:
                    return {}, False
            elif isinstance(event, (yaml.events.MappingEndEvent, yaml.events.SequenceEndEvent)):
                depth -= 1
        config = yaml.safe_load(raw)
        if not isinstance(config, dict):
            return {}, False
        pending, visited, nodes, string_bytes = [(config, 1)], set(), 0, 0
        while pending:
            value, level = pending.pop()
            nodes += 1
            if nodes > 2048 or level > 32:
                return {}, False
            if isinstance(value, (dict, list)):
                if id(value) in visited:
                    return {}, False
                visited.add(id(value))
                if isinstance(value, dict):
                    if any(not isinstance(key, str) for key in value):
                        return {}, False
                    pending.extend((key, level + 1) for key in value)
                    pending.extend((item, level + 1) for item in value.values())
                else:
                    pending.extend((item, level + 1) for item in value)
            elif isinstance(value, str):
                string_bytes += len(value.encode("utf-8"))
                if string_bytes > _MAX_CONFIG_BYTES:
                    return {}, False
            elif isinstance(value, float):
                if not math.isfinite(value):
                    return {}, False
            elif value is not None and not isinstance(value, (bool, int)):
                return {}, False
        return config, True
    except (yaml.YAMLError, RecursionError, UnicodeError, ValueError, TypeError):
        return {}, False


def load_catalog_snapshot(db_path: Path | str, *, include_configs=False) -> CatalogSnapshot:
    """Read lightweight catalog data without running an IdeaLake constructor."""
    path = Path(db_path)
    connection, reason = _open_authoritative_lifecycle(path)
    if connection is None:
        return CatalogSnapshot(path, reason)
    try:
        connection.execute("BEGIN")
        schema = validate_lifecycle_schema(connection)
        ideas_columns = schema["ideas"]

        metadata = ", ".join(
            f"i.{name}" if name in ideas_columns else f"NULL AS {name}"
            for name in ("title", "priority", "category", "parent", "hypothesis")
        )
        config_column = "NULL"
        if include_configs and "config" in ideas_columns:
            # SQL bounds the bytes returned to Python. Ordinary reports never
            # select or measure this potentially large column at all.
            config_column = (
                "CASE WHEN length(CAST(i.config AS BLOB)) <= "
                f"{_MAX_CONFIG_BYTES} THEN i.config ELSE NULL END"
            )
        stages, stage_join = stage_projection(schema)
        rows = connection.execute(
            f"SELECT i.idea_id, i.status, s.current_state, {metadata}, {stages}, {config_column} "
            "FROM ideas i LEFT JOIN idea_state s ON s.idea_id=i.idea_id"
            + stage_join,
        ).fetchall()
        records, counts = {}, {}
        from orze.idea_lake import STATUS_TO_STATE
        display = {
            "QUEUED": "QUEUED", "CLAIMED": "IN_PROGRESS",
            "IN_PROGRESS": "IN_PROGRESS", "COMPLETE": "COMPLETED",
            "FAILED": "FAILED", "SKIPPED": "SKIPPED", "ARCHIVED": "ARCHIVED",
        }
        for row in rows:
            (idea_id, status, state, title, priority, category, parent,
             hypothesis, training_id, training, evaluation_id, evaluation, raw_config) = row
            if (not isinstance(idea_id, str) or idea_id in ("", ".", "..")
                    or Path(idea_id).parts != (idea_id,) or idea_id in records):
                raise ValueError("catalog_identity_invalid")
            agreed, row_reason = _agreed_state(
                status, state, training_id, training, evaluation_id, evaluation)
            records[idea_id] = {
                "idea_id": idea_id, "title": str(title or idea_id),
                "status": status, "fsm_state": state,
                "lifecycle_state": agreed, "lifecycle_reason": row_reason,
                "training_state": training, "evaluation_state": evaluation,
                "priority": str(priority or "medium"),
                "category": str(category or "other"), "parent": str(parent or "none"),
                "hypothesis": str(hypothesis or ""),
            }
            if include_configs:
                config, config_available = _display_config(raw_config)
                records[idea_id].update(config=config, config_available=config_available)
            # Preserve the pre-existing report pipeline accounting definition;
            # never reuse these counts as admin's agreed task-state counts.
            counted = state or STATUS_TO_STATE.get(str(status).lower())
            key = display.get(counted, "UNKNOWN")
            counts[key] = counts.get(key, 0) + 1
        return CatalogSnapshot(path, "authoritative_lifecycle_loaded", records, counts)
    except (sqlite3.Error, OSError, ValueError, TypeError):
        return CatalogSnapshot(path, "authoritative_catalog_invalid")
    finally:
        connection.close()
