"""Idea Lake — SQLite archive for completed/failed ideas.

Provides a queryable store so ideas.md can stay small (~500 hot ideas)
while all historical ideas remain accessible for config lookups, dedup,
and leaderboard queries.

Usage:
    lake = IdeaLake("idea_lake.db")
    lake.insert("idea-001", "Zipformer", config_yaml, raw_md, eval_metrics={...})
    idea = lake.get("idea-001")
    top = lake.get_top_models(metric="test_accuracy", n=10)
"""

import datetime
import json
import logging
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger("idea_lake")


def flatten_config(config: dict, prefix: str = "", max_depth: int = 2) -> Dict[str, Any]:
    """Flatten a nested config dict into dot-separated keys.
    Only keeps leaf scalar values (str, int, float, bool).
    """
    result = {}
    if not isinstance(config, dict) or max_depth <= 0:
        return result

    for key, val in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            result.update(flatten_config(val, full_key, max_depth - 1))
        elif not isinstance(val, (dict, list)) and val is not None:
            result[full_key] = val
    return result

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ideas (
    idea_id TEXT PRIMARY KEY,
    id_num INTEGER,
    title TEXT NOT NULL,
    priority TEXT DEFAULT 'medium',
    category TEXT DEFAULT 'architecture',
    parent TEXT,
    hypothesis TEXT,
    config TEXT NOT NULL,
    raw_markdown TEXT NOT NULL,
    config_summary TEXT,
    eval_metrics TEXT,
    status TEXT DEFAULT 'archived',
    training_time REAL,
    archived_at TEXT,
    created_at TEXT,
    approach_family TEXT DEFAULT 'other',
    kind TEXT NOT NULL DEFAULT 'train'
);

CREATE INDEX IF NOT EXISTS idx_status ON ideas(status);

CREATE TABLE IF NOT EXISTS id_sequence (
    next_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS idea_state (
    idea_id TEXT PRIMARY KEY,
    current_state TEXT NOT NULL DEFAULT 'QUEUED',
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_idea_state_current ON idea_state(current_state);

CREATE TABLE IF NOT EXISTS idea_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idea_id TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    sop_type TEXT DEFAULT 'training',
    reason TEXT,
    host TEXT,
    pid INTEGER,
    ts TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_idea_transitions_idea_id ON idea_transitions(idea_id);
CREATE INDEX IF NOT EXISTS idx_idea_transitions_to_state ON idea_transitions(to_state);
CREATE INDEX IF NOT EXISTS idx_idea_transitions_ts ON idea_transitions(ts);

CREATE TABLE IF NOT EXISTS schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# Legacy fixed-metric columns from pre-1.5 schema.
# Auto-detected from PRAGMA table_info during migration.
_KNOWN_META_COLS = {
    "idea_id", "id_num", "title", "priority", "category", "parent",
    "hypothesis", "config", "raw_markdown", "config_summary",
    "eval_metrics", "status", "training_time", "archived_at", "created_at",
    "approach_family", "kind",
}

# F8: closed vocabulary of idea kinds. Anything else is a hard error so the
# launcher / scheduler can safely dispatch on kind.
ALLOWED_KINDS = {
    "train",           # legacy / default: run train_script
    "posthoc_eval",    # inference-only job on an existing ckpt/npz
    "tta_sweep",       # generate a TTA view (subclass of posthoc_eval)
    "agg_search",      # sweep aggregations/calibrators on a bundle
    "bundle_combine",  # combine N views of ONE ckpt
    "audit",           # F14: champion-promotion audit
}


# ``ideas.status`` predates the audited FSM and is still consumed by reports
# and older integrations. Keep the two representations in one transaction;
# otherwise a crash or a legacy ``set_status`` caller can advertise work as
# queued after it has already been claimed or rejected.
STATE_TO_STATUS = {
    "QUEUED": "queued",
    "CLAIMED": "running",
    "IN_PROGRESS": "running",
    "COMPLETE": "completed",
    "FAILED": "failed",
    "SKIPPED": "skipped",
    "ARCHIVED": "archived",
}

STATUS_TO_STATE = {
    "queued": "QUEUED",
    "pending": "QUEUED",
    "claimed": "CLAIMED",
    "running": "IN_PROGRESS",
    "training": "IN_PROGRESS",
    "evaluating": "IN_PROGRESS",
    "completed": "COMPLETE",
    "partial": "COMPLETE",
    "failed": "FAILED",
    "dead": "FAILED",
    "skipped": "SKIPPED",
    "archived": "ARCHIVED",
}

VALID_STATE_TRANSITIONS = {
    "QUEUED": {"CLAIMED", "SKIPPED"},
    "CLAIMED": {"IN_PROGRESS", "FAILED", "QUEUED", "SKIPPED"},
    "IN_PROGRESS": {"COMPLETE", "FAILED", "QUEUED"},
    "COMPLETE": {"ARCHIVED"},
    "FAILED": {"QUEUED", "SKIPPED"},
    "SKIPPED": {"QUEUED", "ARCHIVED"},
    "ARCHIVED": set(),
}


def _retry_on_busy(func, max_retries=10, base_delay=1.0):
    """Retry a callable on SQLITE_BUSY / OperationalError with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), 30)
                logger.warning("SQLite busy (attempt %d/%d), retrying in %.1fs",
                               attempt + 1, max_retries, delay)
                time.sleep(delay)
            else:
                raise


_MEDAL_ORDER = ("none", "below_median", "above_median", "bronze", "silver", "gold")


def _medal_rank(medal) -> int:
    """Rank a medal label so higher == better. Unknown/None -> 0."""
    try:
        return _MEDAL_ORDER.index(medal)
    except (ValueError, TypeError):
        return 0


class IdeaLake:
    """SQLite-backed archive for ideas."""

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        # DELETE journal mode is safe on network filesystems (Lustre/NFS).
        # WAL mode requires shared-memory (mmap) which Lustre does not support.
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self._ensure_schema()

    def _ensure_schema(self):
        self.conn.executescript(_SCHEMA_SQL)
        # Ensure id_sequence has a row
        row = self.conn.execute("SELECT next_id FROM id_sequence LIMIT 1").fetchone()
        if row is None:
            self.conn.execute("INSERT INTO id_sequence (next_id) VALUES (1)")
            self.conn.commit()
        self._migrate_if_needed()
        # Trigger consumption ledger (resolves c1005 / DEC-009). Owned by
        # orze.engine.trigger_ledger but the table is materialised here
        # so it exists from first connect, before any consumer runs.
        try:
            from orze.engine.trigger_ledger import init_schema as _init_trig
            _init_trig(self.conn)
        except Exception as e:
            logger.warning("trigger_ledger schema init failed: %s", e)

    def _migrate_if_needed(self):
        """Migrate from old fixed-column schema to generic JSON blobs."""
        cols = {
            r[1] for r in self.conn.execute("PRAGMA table_info(ideas)").fetchall()
        }
        # Detect legacy schema: any column not in _KNOWN_META_COLS is an old metric/config column
        extra_cols = cols - _KNOWN_META_COLS
        has_old = bool(extra_cols)
        has_new = "eval_metrics" in cols

        if "id_num" not in cols:
            logger.info("Migrating idea_lake schema: adding id_num column")
            self.conn.execute("ALTER TABLE ideas ADD COLUMN id_num INTEGER")
            # Backfill
            rows = self.conn.execute("SELECT idea_id FROM ideas").fetchall()
            for r in rows:
                match = re.search(r"idea-([a-z0-9]+)", r["idea_id"])
                if match:
                    try:
                        num = int(match.group(1))
                    except ValueError:
                        hex_part = re.sub(r"^[^0-9a-f]+", "", match.group(1))
                        try:
                            num = int(hex_part, 16) % (2**31) if hex_part else hash(r["idea_id"]) % (2**31)
                        except ValueError:
                            num = hash(r["idea_id"]) % (2**31)
                    self.conn.execute(
                        "UPDATE ideas SET id_num = ? WHERE idea_id = ?",
                        (num, r["idea_id"])
                    )
            self.conn.commit()
            logger.info("Backfilled id_num for %d ideas", len(rows))
            # Create indexes that need id_num
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_status_priority_id ON ideas(status, priority, id_num)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_id_num ON ideas(id_num)")
            self.conn.commit()

        if has_old and not has_new:
            logger.info("Migrating idea_lake schema: fixed columns → JSON blobs (extra: %s)", extra_cols)
            self.conn.execute("ALTER TABLE ideas ADD COLUMN eval_metrics TEXT")
            self.conn.execute("ALTER TABLE ideas ADD COLUMN config_summary TEXT")

            # Build JSON blobs from old columns
            rows = self.conn.execute("SELECT idea_id, * FROM ideas").fetchall()
            for row in rows:
                rd = dict(row)
                merged = {}
                for c in extra_cols:
                    if c in rd and rd[c] is not None:
                        merged[c] = rd[c]

                self.conn.execute(
                    "UPDATE ideas SET eval_metrics = ? WHERE idea_id = ?",
                    (json.dumps(merged) if merged else None, rd["idea_id"]),
                )
            self.conn.commit()
            logger.info("Migration complete for %d ideas", len(rows))

        elif not has_new:
            # Fresh DB, columns already correct from _SCHEMA_SQL
            pass

        # Add approach_family column if missing
        if "approach_family" not in cols:
            logger.info("Migrating idea_lake schema: adding approach_family column")
            self.conn.execute(
                "ALTER TABLE ideas ADD COLUMN approach_family TEXT DEFAULT 'other'"
            )
            self.conn.commit()

        # F8: add kind column if missing (defaults to 'train' for back-compat).
        if "kind" not in cols:
            logger.info("Migrating idea_lake schema: adding kind column (F8)")
            self.conn.execute(
                "ALTER TABLE ideas ADD COLUMN kind TEXT NOT NULL DEFAULT 'train'"
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_kind ON ideas(kind)")
            self.conn.commit()

        # v4.5: Genericize FSM (orthogonal to SOP type)
        try:
            from orze.migrations.v45_genericize_fsm import migrate_v45
            migrate_v45(self.conn)
        except Exception as e:
            logger.warning("v4.5 FSM migration failed: %s (will retry next init)", e)

        self._reconcile_lifecycle_columns()

    def _reconcile_lifecycle_columns(self) -> None:
        """Idempotently repair legacy/FSM lifecycle divergence.

        Older Orze releases wrote ``ideas.status`` and ``idea_state`` through
        independent paths. Preserve an active FSM claim, but import terminal
        legacy decisions and backfill rows that predate the FSM. Finally make
        the legacy column reflect audited non-queued FSM states. This is a
        bounded startup migration; repeated opens produce no new transitions.
        """
        migration_name = "lifecycle_columns_v1"
        if self.conn.execute(
            "SELECT 1 FROM schema_migrations WHERE name = ?",
            (migration_name,),
        ).fetchone():
            return

        def _do_reconcile():
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                # A second process may have completed the migration while this
                # connection waited for the write lock.
                if self.conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE name = ?",
                    (migration_name,),
                ).fetchone():
                    self.conn.rollback()
                    return

                # Rows created before the FSM get a state derived from their
                # legacy status. Unknown statuses remain legacy-only rather
                # than being invented into the audited lifecycle.
                for status, state in STATUS_TO_STATE.items():
                    self.conn.execute(
                        "INSERT OR IGNORE INTO idea_state (idea_id, current_state) "
                        "SELECT idea_id, ? FROM ideas WHERE lower(status) = ?",
                        (state, status),
                    )

                # Historical admission and terminal writes updated only the
                # legacy column. Import them when the FSM still says QUEUED;
                # SKIPPED also supersedes FAILED because launch validation in
                # older versions wrote a failure marker before classifying the
                # zero-compute rejection as skipped.
                repairs = (
                    ("SKIPPED", ("skipped",), ("QUEUED", "FAILED")),
                    ("COMPLETE", ("completed", "partial"),
                     ("QUEUED", "IN_PROGRESS")),
                    ("FAILED", ("failed", "dead"), ("QUEUED",)),
                )
                for target, statuses, sources in repairs:
                    status_marks = ",".join("?" for _ in statuses)
                    state_marks = ",".join("?" for _ in sources)
                    self.conn.execute(
                        f"UPDATE idea_state SET current_state = ?, "
                        "updated_at = datetime('now') "
                        f"WHERE current_state IN ({state_marks}) AND idea_id IN ("
                        "SELECT idea_id FROM ideas "
                        f"WHERE lower(status) IN ({status_marks}))",
                        (target, *sources, *statuses),
                    )

                # FSM ownership wins for active work and for terminal states
                # when the legacy row still claims the idea is dispatchable.
                for state, status in STATE_TO_STATUS.items():
                    if state == "QUEUED":
                        continue
                    self.conn.execute(
                        "UPDATE ideas SET status = ? WHERE idea_id IN ("
                        "SELECT idea_id FROM idea_state WHERE current_state = ?) "
                        "AND lower(status) IN ('queued', 'pending', 'running')",
                        (status, state),
                    )
                self.conn.execute(
                    "INSERT INTO schema_migrations (name) VALUES (?)",
                    (migration_name,),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

        _retry_on_busy(_do_reconcile)

    def insert(
        self,
        idea_id: str,
        title: str,
        config_yaml: str,
        raw_markdown: str,
        eval_metrics: Optional[Dict[str, Any]] = None,
        config_summary: Optional[Dict[str, Any]] = None,
        status: str = "archived",
        priority: str = "medium",
        category: str = "architecture",
        parent: Optional[str] = None,
        hypothesis: Optional[str] = None,
        training_time: Optional[float] = None,
        created_at: Optional[str] = None,
        approach_family: str = "other",
        kind: str = "train",
    ):
        """Insert or update an idea in the lake."""
        if kind not in ALLOWED_KINDS:
            raise ValueError(
                f"idea kind={kind!r} not in {sorted(ALLOWED_KINDS)}"
            )
        # Evolution contract (soft): a child that declares a parent should also
        # record a rationale (the hypothesis behind the change). We do not reject
        # it — we log the violation so the gap is visible and the search-path
        # visualizer can flag it — but every evolution edge is expected to be
        # justified.
        _real_parent = parent and str(parent).lower() not in ("", "none")
        if _real_parent and not (hypothesis and hypothesis.strip()):
            logger.warning(
                "idea %s declares parent=%s but has no rationale/hypothesis "
                "(unjustified evolution edge)", idea_id, parent
            )
        # Extract numeric ID for indexed sorting (supports both numeric and hex IDs)
        id_num = None
        match = re.search(r"idea-([a-z0-9]+)", idea_id)
        if match:
            try:
                id_num = int(match.group(1))
            except ValueError:
                # Strip non-hex prefix (e.g., "v", "ss") then parse as hex
                hex_part = re.sub(r"^[^0-9a-f]+", "", match.group(1))
                try:
                    id_num = int(hex_part, 16) % (2**31) if hex_part else hash(idea_id) % (2**31)
                except ValueError:
                    id_num = hash(idea_id) % (2**31)

        # Auto-compute summary if missing
        if not config_summary and config_yaml:
            try:
                cfg_obj = yaml.safe_load(config_yaml)
                if isinstance(cfg_obj, dict):
                    config_summary = flatten_config(cfg_obj)
            except yaml.YAMLError:
                pass

        def _do_insert():
            self.conn.execute(
                """INSERT OR REPLACE INTO ideas (
                    idea_id, id_num, title, priority, category, parent, hypothesis,
                    config, raw_markdown,
                    config_summary, eval_metrics,
                    status, training_time, archived_at, created_at,
                    approach_family, kind
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?
                )""",
                (
                    idea_id,
                    id_num,
                    title,
                    priority,
                    category,
                    parent,
                    hypothesis,
                    config_yaml,
                    raw_markdown,
                    json.dumps(config_summary) if config_summary else None,
                    json.dumps(eval_metrics) if eval_metrics else None,
                    status,
                    training_time,
                    datetime.datetime.now().isoformat(),
                    created_at or datetime.datetime.now().isoformat(),
                    approach_family,
                    kind,
                ),
            )
            # New proposals enter through the queue. Imported running or
            # terminal rows may be followed by explicit history replay (the
            # startup recovery contract), so leave those without an implicit
            # transition and let the idempotent startup reconciler backfill
            # them on the next open.
            initial_state = STATUS_TO_STATE.get(str(status).lower())
            if initial_state == "QUEUED":
                self.conn.execute(
                    "INSERT OR IGNORE INTO idea_state (idea_id, current_state) "
                    "VALUES (?, ?)",
                    (idea_id, initial_state),
                )
            self.conn.commit()
        _retry_on_busy(_do_insert)

    def get(self, idea_id: str) -> Optional[dict]:
        """Get a single idea by ID."""
        # SQLITE_BUSY here had been bubbling up through update_report() into
        # cli.main() and crashing the daemon (observed 2026-04-30). Writes in
        # this class are wrapped in _retry_on_busy; reads are not, but they
        # are equally exposed when a competing writer holds the lock past
        # busy_timeout. Wrap the read so a transient lock contention is
        # retried instead of being raised into the orchestrator main loop.
        def _do_get():
            return self.conn.execute(
                "SELECT * FROM ideas WHERE idea_id = ?", (idea_id,)
            ).fetchone()
        row = _retry_on_busy(_do_get)
        if row is None:
            return None
        d = dict(row)
        # Parse JSON blobs for convenience
        for key in ("eval_metrics", "config_summary"):
            if d.get(key) and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def set_status(self, idea_id: str, status: str) -> bool:
        """Update legacy status and audited state atomically.

        Known lifecycle statuses follow the same transition contract as
        :meth:`record_state_transition`. Repeating an already-applied status is
        idempotent and does not append another audit event. Unknown statuses
        retain the historical legacy-only behavior for compatibility.
        """
        def _do():
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                idea = self.conn.execute(
                    "SELECT 1 FROM ideas WHERE idea_id = ?", (idea_id,),
                ).fetchone()
                if idea is None:
                    self.conn.rollback()
                    return False

                target = STATUS_TO_STATE.get(str(status).lower())
                if target is None:
                    self.conn.execute(
                        "UPDATE ideas SET status = ? WHERE idea_id = ?",
                        (status, idea_id),
                    )
                    self.conn.commit()
                    return True

                row = self.conn.execute(
                    "SELECT current_state FROM idea_state WHERE idea_id = ?",
                    (idea_id,),
                ).fetchone()
                current = row[0] if row else "QUEUED"
                if current == target:
                    if row is None:
                        self.conn.execute(
                            "INSERT INTO idea_state "
                            "(idea_id, current_state, updated_by_host, "
                            "updated_by_pid) "
                            "VALUES (?, ?, 'legacy_status', ?)",
                            (idea_id, target, os.getpid()),
                        )
                    self.conn.execute(
                        "UPDATE ideas SET status = ? WHERE idea_id = ?",
                        (status, idea_id),
                    )
                    self.conn.commit()
                    return True

                if target not in VALID_STATE_TRANSITIONS.get(current, set()):
                    logger.warning(
                        "Invalid lifecycle status update: %s %s -> %s (%s)",
                        idea_id, current, target, status,
                    )
                    self.conn.rollback()
                    return False

                if row:
                    cursor = self.conn.execute(
                        "UPDATE idea_state SET current_state = ?, "
                        "updated_by_host = 'legacy_status', "
                        "updated_by_pid = ?, updated_at = datetime('now') "
                        "WHERE idea_id = ? AND current_state = ?",
                        (target, os.getpid(), idea_id, current),
                    )
                    if cursor.rowcount != 1:
                        self.conn.rollback()
                        return False
                else:
                    self.conn.execute(
                        "INSERT INTO idea_state "
                        "(idea_id, current_state, updated_by_host, updated_by_pid) "
                        "VALUES (?, ?, 'legacy_status', ?)",
                        (idea_id, target, os.getpid()),
                    )

                self.conn.execute(
                    "INSERT INTO idea_transitions "
                    "(idea_id, from_state, to_state, reason, host, pid, sop_type) "
                    "VALUES (?, ?, ?, ?, 'legacy_status', ?, 'training')",
                    (idea_id, current, target,
                     f"set_status:{str(status).lower()}", os.getpid()),
                )
                self.conn.execute(
                    "UPDATE ideas SET status = ? WHERE idea_id = ?",
                    (status, idea_id),
                )
                self.conn.commit()
                return True
            except Exception:
                self.conn.rollback()
                raise
        return bool(_retry_on_busy(_do))

    def has(self, idea_id: str) -> bool:
        # SQLITE_BUSY exposure: same as get(). Wrap the read so a transient
        # lock contention is retried instead of raised into the orchestrator.
        def _do_has():
            return self.conn.execute(
                "SELECT 1 FROM ideas WHERE idea_id = ?", (idea_id,)
            ).fetchone()
        row = _retry_on_busy(_do_has)
        return row is not None

    def count(self) -> int:
        def _do_count():
            return self.conn.execute("SELECT COUNT(*) FROM ideas").fetchone()
        row = _retry_on_busy(_do_count)
        return row[0]

    def child_counts(self) -> Dict[str, int]:
        """Return {parent_id: number_of_children} over all parented ideas.

        Used by the free research loop to cap a single parent's fan-out so
        search branches broadly and deepens winning lineages instead of
        spraying every variation off one champion hub (research efficiency)."""
        def _do_counts():
            return self.conn.execute(
                "SELECT parent, COUNT(*) FROM ideas "
                "WHERE parent IS NOT NULL AND parent != '' "
                "AND lower(parent) != 'none' GROUP BY parent"
            ).fetchall()
        rows = _retry_on_busy(_do_counts)
        return {r[0]: r[1] for r in rows}

    def get_all_ids(self, status: Optional[str] = None) -> Set[str]:
        """Return set of all idea IDs in the lake, optionally filtered by status."""
        query = "SELECT idea_id FROM ideas"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        def _do_all_ids():
            return self.conn.execute(query, params).fetchall()
        rows = _retry_on_busy(_do_all_ids)
        return {r[0] for r in rows}

    def get_metadata_index(self) -> Dict[str, Dict[str, Any]]:
        """Return lightweight lifecycle metadata keyed by idea ID."""
        def _do_metadata_index():
            return self.conn.execute(
                "SELECT idea_id, title, status FROM ideas"
            ).fetchall()
        rows = _retry_on_busy(_do_metadata_index)
        return {
            row["idea_id"]: {
                "idea_id": row["idea_id"],
                "title": row["title"],
                "status": row["status"],
            }
            for row in rows
        }

    def get_queue(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Return ideas that both lifecycle representations say are queued."""
        def _do_get_queue():
            return self.conn.execute(
                """SELECT i.idea_id, i.title, i.priority, i.config, i.created_at
                   FROM ideas AS i
                   LEFT JOIN idea_state AS s ON s.idea_id = i.idea_id
                   WHERE (i.status = 'queued' OR i.status = 'pending')
                     AND COALESCE(s.current_state, 'QUEUED') = 'QUEUED'
                   ORDER BY
                     CASE i.priority
                       WHEN 'critical' THEN 0
                       WHEN 'high' THEN 1
                       WHEN 'medium' THEN 2
                       WHEN 'low' THEN 3
                       ELSE 2
                     END,
                     i.id_num ASC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
        rows = _retry_on_busy(_do_get_queue)
        return [dict(r) for r in rows]

    def reconcile_statuses(self, results_dir: str, limit: int = 0) -> int:
        """Reconcile DB queue with filesystem: mark queued ideas that already
        have results dirs as completed/failed. Returns count of updates.

        This prevents stale 'queued' rows from blocking the queue after a
        restart when a previous Orze instance already ran those ideas.

        Args:
            results_dir: path to the results directory
            limit: max rows to scan (0 = all queued ideas)
        """
        import glob
        import shutil
        from pathlib import Path
        rd = Path(results_dir)
        query = """SELECT idea_id FROM ideas
                   WHERE status = 'queued' OR status = 'pending'
                   ORDER BY id_num ASC"""
        if limit > 0:
            query += f" LIMIT {limit}"
        def _do_reconcile_select():
            return self.conn.execute(query).fetchall()
        rows = _retry_on_busy(_do_reconcile_select)
        updated = 0
        for (idea_id,) in rows:
            idea_dir = rd / idea_id
            has_dir = idea_dir.exists()
            has_subs = bool(glob.glob(str(rd / f"{idea_id}-ht-*")))
            if not has_dir and not has_subs:
                continue
            # Determine final status from filesystem
            metrics_path = idea_dir / "metrics.json" if has_dir else None
            if metrics_path and metrics_path.exists():
                try:
                    m = json.loads(metrics_path.read_text(encoding="utf-8"))
                    new_status = "completed" if m.get("status") == "COMPLETED" else "failed"
                except (json.JSONDecodeError, OSError):
                    new_status = "failed"
            elif has_subs:
                # Sweep parent: sub-runs (-ht-*) are the real experiments.
                # Two gaps the original logic missed:
                #   1. We marked the parent 'completed' even when every sub-run
                #      was PARTIAL/failed — the leaderboard then showed a green
                #      row that had no real metrics behind it.
                #   2. We never copied the best sub-run's metrics into the
                #      parent's eval_metrics, so downstream consumers
                #      (report.md, status.json top_results, the campaign-side
                #      "did anything land?" queries) saw NULL even when one
                #      sub-run finished cleanly.
                # Fix: treat parent as completed iff ≥1 sub-run reached
                # status=COMPLETED on disk, and lift that sub-run's metrics
                # onto the parent. Otherwise mark failed.
                sub_dirs = sorted(glob.glob(str(rd / f"{idea_id}-ht-*")))
                best_sub_metrics: Optional[Dict[str, Any]] = None
                any_completed = False
                for sd in sub_dirs:
                    sm_path = Path(sd) / "metrics.json"
                    if not sm_path.exists():
                        continue
                    try:
                        sm = json.loads(sm_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    if not isinstance(sm, dict):
                        continue
                    if sm.get("status") != "COMPLETED":
                        continue
                    any_completed = True
                    # Pick the sub-run with the best primary score we can
                    # see. We don't know the project's primary_metric here,
                    # so prefer common ones in priority order: avg_wer (ASR),
                    # score (generic), test_accuracy (classification). Lower-
                    # is-better for *_wer / *_loss, higher otherwise.
                    if best_sub_metrics is None:
                        best_sub_metrics = sm
                    else:
                        for k, lower_better in (
                            ("avg_wer", True), ("wer", True),
                            ("test_loss", True), ("loss", True),
                            ("score", False), ("test_accuracy", False),
                            ("accuracy", False),
                        ):
                            if k in sm and k in best_sub_metrics:
                                try:
                                    a = float(sm[k])
                                    b = float(best_sub_metrics[k])
                                except (TypeError, ValueError):
                                    continue
                                if (lower_better and a < b) or (
                                        not lower_better and a > b):
                                    best_sub_metrics = sm
                                break
                new_status = "completed" if any_completed else "failed"
                if best_sub_metrics is not None:
                    em = dict(best_sub_metrics)
                    em["_aggregated_from"] = "sweep_sub_runs"
                    self.conn.execute(
                        "UPDATE ideas SET eval_metrics = ? "
                        "WHERE idea_id = ?",
                        (json.dumps(em), idea_id),
                    )
            elif has_dir:
                # Orphan dir (no metrics, no sweep subs).
                # Only remove if claimed by this host — another node may
                # still be training this idea.
                import socket as _socket
                claim_path = idea_dir / "claim.json"
                # Crash recovery intentionally releases claim.json while
                # retaining recovery.json, the prior claim, logs, and attempt
                # ledger. Preserve that evidence; scheduler.claim() can reuse
                # a directory with no live claim and no terminal metrics.
                if ((idea_dir / "recovery.json").exists()
                        or any(idea_dir.glob("claim.recovered.*.json"))):
                    logger.info(
                        "Preserving recovered retry directory for %s", idea_id)
                    continue
                if claim_path.exists():
                    try:
                        claim = json.loads(claim_path.read_text(encoding="utf-8"))
                        if claim.get("claimed_by") != _socket.gethostname():
                            continue  # owned by another host, don't touch
                    except (json.JSONDecodeError, OSError):
                        pass
                try:
                    shutil.rmtree(idea_dir)
                except OSError:
                    pass
                continue  # leave as queued for retry
            else:
                continue
            self.conn.execute(
                "UPDATE ideas SET status = ? WHERE idea_id = ?",
                (new_status, idea_id),
            )
            updated += 1
        if updated:
            self.conn.commit()
            logger.info("Reconciled %d stale queued ideas with filesystem", updated)
        return updated

    def get_max_id_num(self) -> int:
        """Return the highest numeric idea ID in the lake."""
        def _do_max_id():
            return self.conn.execute(
                "SELECT MAX(id_num) FROM ideas WHERE id_num IS NOT NULL"
            ).fetchone()
        row = _retry_on_busy(_do_max_id)
        if row is None or row[0] is None:
            return 0
        return int(row[0])

    def query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        min_metric: Optional[tuple] = None,
        sort_metric: Optional[str] = None,
        limit: int = 20,
    ) -> List[dict]:
        """Filtered query with optional constraints.

        Args:
            filters: dict of {json_path: value} to match in config_summary.
                     e.g. {"backbone_name": "dinov2_vitl14"}
            min_metric: (metric_key, min_value) to filter eval_metrics.
                        e.g. ("test_accuracy", 0.8)
            sort_metric: key in eval_metrics to sort by (descending).
            limit: max results.
        """
        _safe_key = re.compile(r"^[a-zA-Z0-9_.]+$")
        clauses = []
        params = []

        if filters:
            for key, val in filters.items():
                if not _safe_key.match(key):
                    continue
                clauses.append(
                    f"json_extract(config_summary, '$.{key}') = ?"
                )
                params.append(val)

        if min_metric:
            metric_key, min_val = min_metric
            if _safe_key.match(metric_key):
                clauses.append(
                    f"json_extract(eval_metrics, '$.{metric_key}') >= ?"
                )
                params.append(min_val)

        where = " AND ".join(clauses) if clauses else "1=1"
        if sort_metric and _safe_key.match(sort_metric):
            sort_col = f"json_extract(eval_metrics, '$.{sort_metric}')"
        else:
            sort_col = "archived_at"
        params.append(limit)
        def _do_query():
            return self.conn.execute(
                f"SELECT * FROM ideas WHERE {where} "
                f"ORDER BY {sort_col} DESC NULLS LAST LIMIT ?",
                params,
            ).fetchall()
        rows = _retry_on_busy(_do_query)
        return [dict(r) for r in rows]

    def get_top_models(
        self, metric: str = "test_accuracy", n: int = 20
    ) -> List[dict]:
        """Return top N models, ordered by medal tier then raw score (both desc).

        Medal tier sidesteps per-competition metric-direction issues: gold means
        "good" regardless of whether the metric is maximized or minimized.
        Within a tier, rows are sorted by raw `metric` value descending.
        """
        def _do_top():
            return self.conn.execute(
                "SELECT idea_id, title, config_summary, eval_metrics, status "
                "FROM ideas "
                "WHERE json_extract(eval_metrics, ?) IS NOT NULL",
                (f"$.{metric}",),
            ).fetchall()
        rows = _retry_on_busy(_do_top)
        results = []
        for r in rows:
            d = dict(r)
            for key in ("eval_metrics", "config_summary"):
                if d.get(key) and isinstance(d[key], str):
                    try:
                        d[key] = json.loads(d[key])
                    except (json.JSONDecodeError, TypeError):
                        pass
            results.append(d)
        def _key(d):
            em = d.get("eval_metrics") or {}
            if not isinstance(em, dict):
                em = {}
            try:
                score = float(em.get(metric))
            except (TypeError, ValueError):
                score = float("-inf")
            return (_medal_rank(em.get("medal")), score)
        results.sort(key=_key, reverse=True)
        return results[:n]

    def get_next_id(self) -> int:
        """Atomically get and increment the next idea ID number.
        Uses BEGIN IMMEDIATE to prevent concurrent readers from getting the same ID.
        Retries on SQLITE_BUSY for network filesystem safety."""
        def _do_get_next():
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self.conn.execute("SELECT next_id FROM id_sequence LIMIT 1")
                row = cur.fetchone()
                next_id = row[0] if row else 1
                self.conn.execute(
                    "UPDATE id_sequence SET next_id = ?", (next_id + 1,)
                )
                self.conn.commit()
                return next_id
            except Exception:
                self.conn.rollback()
                raise
        return _retry_on_busy(_do_get_next)

    def set_next_id(self, n: int):
        """Set the next ID sequence value."""
        self.conn.execute("UPDATE id_sequence SET next_id = ?", (n,))
        self.conn.commit()

    def bulk_insert(self, ideas: List[Dict[str, Any]]):
        """Insert many ideas in a single transaction."""
        for idea in ideas:
            eval_metrics = idea.get("eval_metrics") or idea.get("metrics")
            config_summary = idea.get("config_summary")
            config_yaml = idea.get("config_yaml", "")

            if not config_summary and config_yaml:
                try:
                    cfg_obj = yaml.safe_load(config_yaml)
                    if isinstance(cfg_obj, dict):
                        config_summary = flatten_config(cfg_obj)
                except yaml.YAMLError:
                    pass

            # Extract numeric ID for sorting (supports both numeric and hex IDs)
            id_num = None
            try:
                match = re.search(r"idea-([a-z0-9]+)", idea["idea_id"])
                if match:
                    try:
                        id_num = int(match.group(1))
                    except ValueError:
                        hex_part = re.sub(r"^[^0-9a-f]+", "", match.group(1))
                        try:
                            id_num = int(hex_part, 16) % (2**31) if hex_part else hash(idea["idea_id"]) % (2**31)
                        except ValueError:
                            id_num = hash(idea["idea_id"]) % (2**31)
            except (AttributeError, ValueError):
                pass

            self.conn.execute(
                """INSERT OR IGNORE INTO ideas (
                    idea_id, id_num, title, priority, category, parent, hypothesis,
                    config, raw_markdown,
                    config_summary, eval_metrics,
                    status, training_time, archived_at, created_at,
                    approach_family
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?
                )""",
                (
                    idea["idea_id"],
                    id_num,
                    idea["title"],
                    idea.get("priority", "medium"),
                    idea.get("category", "architecture"),
                    idea.get("parent"),
                    idea.get("hypothesis"),
                    config_yaml,
                    idea.get("raw_markdown", ""),
                    json.dumps(config_summary) if config_summary else None,
                    json.dumps(eval_metrics) if eval_metrics else None,
                    idea.get("status", "archived"),
                    (eval_metrics or {}).get("training_time"),
                    datetime.datetime.now().isoformat(),
                    (eval_metrics or {}).get("created_at"),
                    idea.get("approach_family", "other"),
                ),
            )
        _retry_on_busy(self.conn.commit)
        logger.info("Bulk inserted %d ideas", len(ideas))

    def ensure_config_summaries(self, force: bool = False):
        """Backfill missing config_summary for all rows by parsing config YAML.
        Highly recommended for performance on large databases.
        """
        query = "SELECT idea_id, config FROM ideas WHERE config IS NOT NULL AND config != ''"
        if not force:
            query += " AND config_summary IS NULL"

        rows = self.conn.execute(query).fetchall()
        if not rows:
            return

        logger.info("Updating config_summary for %d ideas (force=%s)...", len(rows), force)
        count = 0
        for r in rows:
            try:
                cfg_obj = yaml.safe_load(r["config"])
                if isinstance(cfg_obj, dict):
                    summary = flatten_config(cfg_obj)
                    self.conn.execute(
                        "UPDATE ideas SET config_summary = ? WHERE idea_id = ?",
                        (json.dumps(summary), r["idea_id"]),
                    )
                    count += 1
                    if count % 500 == 0:
                        self.conn.commit()
            except Exception:
                continue
        self.conn.commit()
        logger.info("Successfully updated %d config summaries.", count)

    def record_state_transition(self, idea_id: str, from_state: str, to_state: str,
                                reason: Optional[str] = None,
                                host: Optional[str] = None,
                                pid: Optional[int] = None,
                                sop_type: Optional[str] = None) -> bool:
        """Atomically record an FSM state transition with audit trail.

        v4.5+: Generic FSM orthogonal to SOP type.
        Valid transitions regardless of workflow:
          QUEUED → CLAIMED (scheduler claims work)
          QUEUED → SKIPPED (admission rejects work before compute)
          CLAIMED → IN_PROGRESS (launcher starts work)
          CLAIMED → FAILED (pre-launch validation or setup fails)
          IN_PROGRESS → COMPLETE (work succeeds)
          IN_PROGRESS → FAILED (work fails)
          COMPLETE → ARCHIVED (idea retired)
          FAILED → QUEUED (retry)
          FAILED → SKIPPED (classify a zero-compute validation failure)
          CLAIMED → QUEUED (stale recovery)
          SKIPPED → QUEUED (explicit re-admission)
        """
        import socket as _socket
        host = host or _socket.gethostname()
        pid = pid or os.getpid()
        sop_type = sop_type or "training"

        def _do_transition():
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                # Generic FSM validation (SOP-orthogonal)
                if to_state not in VALID_STATE_TRANSITIONS.get(from_state, set()):
                    logger.warning(
                        "Invalid FSM transition: %s %s → %s",
                        idea_id, from_state, to_state)
                    self.conn.rollback()
                    return False

                row = self.conn.execute(
                    "SELECT current_state FROM idea_state WHERE idea_id = ?",
                    (idea_id,),
                ).fetchone()
                actual_state = row[0] if row else "QUEUED"
                if actual_state != from_state:
                    logger.warning(
                        "Stale FSM transition rejected: %s expected=%s actual=%s to=%s",
                        idea_id, from_state, actual_state, to_state,
                    )
                    self.conn.rollback()
                    return False

                if row:
                    cursor = self.conn.execute(
                        "UPDATE idea_state SET current_state = ?, updated_by_host = ?, "
                        "updated_by_pid = ?, sop_type = ?, updated_at = datetime('now') "
                        "WHERE idea_id = ? AND current_state = ?",
                        (to_state, host, pid, sop_type, idea_id, from_state),
                    )
                    if cursor.rowcount != 1:
                        self.conn.rollback()
                        return False
                else:
                    self.conn.execute(
                        "INSERT INTO idea_state "
                        "(idea_id, current_state, updated_by_host, updated_by_pid, sop_type, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, datetime('now'))",
                        (idea_id, to_state, host, pid, sop_type),
                    )

                # Record transition
                self.conn.execute(
                    "INSERT INTO idea_transitions "
                    "(idea_id, from_state, to_state, reason, host, pid, sop_type) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (idea_id, from_state, to_state, reason or "", host, pid, sop_type)
                )

                # Keep the compatibility status and the audited FSM in the
                # same transaction for every lifecycle state, not only the
                # terminals. Dispatch can therefore never see a claimed run
                # as queued after this commit.
                legacy_status = STATE_TO_STATUS.get(to_state)
                if legacy_status:
                    self.conn.execute(
                        "UPDATE ideas SET status = ? WHERE idea_id = ?",
                        (legacy_status, idea_id),
                    )

                self.conn.commit()
                logger.info(
                    "[LIFECYCLE_TRANSITION] idea=%s %s → %s reason=\"%s\"",
                    idea_id, from_state, to_state, reason or "")
                return True
            except Exception as e:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                logger.error("FSM transition error for %s: %s", idea_id, e)
                raise

        return bool(_retry_on_busy(_do_transition))

    def get_fsm_state(self, idea_id: str) -> str:
        """Get current FSM state for an idea."""
        def _do_get():
            return self.conn.execute(
                "SELECT current_state FROM idea_state WHERE idea_id = ?",
                (idea_id,)
            ).fetchone()
        row = _retry_on_busy(_do_get)
        return row[0] if row else "UNKNOWN"

    def get_fsm_history(self, idea_id: str) -> List[Dict[str, any]]:
        """Get complete audit trail for an idea."""
        def _do_history():
            return self.conn.execute(
                "SELECT from_state, to_state, reason, host, pid, ts "
                "FROM idea_transitions WHERE idea_id = ? ORDER BY id ASC",
                (idea_id,)
            ).fetchall()
        rows = _retry_on_busy(_do_history)
        return [dict(row) for row in rows]

    def detect_stale_claims(self, timeout_hours: int = 6) -> List[tuple]:
        """Detect ideas stuck in CLAIMED state beyond timeout."""
        def _do_detect():
            return self.conn.execute(
                "SELECT idea_id, current_state, updated_at FROM idea_state "
                "WHERE current_state = 'CLAIMED' "
                "AND datetime(updated_at, '+' || ? || ' hours') < datetime('now')",
                (timeout_hours,)
            ).fetchall()
        rows = _retry_on_busy(_do_detect)
        return [(r[0], r[1], r[2]) for r in rows]

    def catch_up_missing_terminals(self, results_dir) -> int:
        """Find ideas in IN_PROGRESS with completed metrics.json and record COMPLETE transitions.

        Handles case where training completed but FSM transition was never recorded
        (e.g., when eval_script was not configured).

        Returns count of transitions recorded.
        """
        from pathlib import Path
        import json

        def _do_catch_up():
            rows = self.conn.execute(
                "SELECT idea_id FROM idea_state WHERE current_state = 'IN_PROGRESS'"
            ).fetchall()
            return [r[0] for r in rows]

        in_progress_ids = _retry_on_busy(_do_catch_up)
        if not in_progress_ids:
            return 0

        results_dir = Path(results_dir)
        recorded = 0

        for idea_id in in_progress_ids:
            metrics_path = results_dir / idea_id / "metrics.json"
            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if metrics.get("status") == "COMPLETED":
                        try:
                            self.record_state_transition(
                                idea_id,
                                from_state="IN_PROGRESS",
                                to_state="COMPLETE",
                                reason="catch_up_training_completed",
                                host="catch_up",
                                pid=None,
                                sop_type="training",
                            )
                            recorded += 1
                        except Exception:
                            pass
                except (json.JSONDecodeError, OSError):
                    pass

        if recorded > 0:
            logger.info("Catch-up: recorded %d missing COMPLETE transitions", recorded)
        return recorded

    def reap_dead_claims(self, max_age_minutes: int = 15) -> int:
        """Requeue ideas with dead PIDs in CLAIMED/IN_PROGRESS.

        Returns count of ideas requeued.
        """
        def _is_pid_alive(pid):
            if not pid or pid <= 0:
                return False
            try:
                os.kill(pid, 0)
                return True
            except (OSError, ProcessLookupError):
                return False

        def _do_reap():
            # Find ideas in CLAIMED or IN_PROGRESS older than max_age_minutes
            rows = self.conn.execute(
                "SELECT idea_id, current_state, updated_by_pid FROM idea_state "
                "WHERE current_state IN ('CLAIMED', 'IN_PROGRESS') "
                "AND datetime(updated_at, '+' || ? || ' minutes') < datetime('now')",
                (max_age_minutes,)
            ).fetchall()
            return [(r[0], r[1], r[2]) for r in rows]

        stale = _retry_on_busy(_do_reap)
        if not stale:
            return 0

        requeued = 0
        for idea_id, current_state, pid in stale:
            if not _is_pid_alive(pid):
                try:
                    self.record_state_transition(
                        idea_id,
                        from_state=current_state,
                        to_state="QUEUED",
                        reason=f"reap_dead_pid_{pid}",
                        host="reaper",
                        pid=None,
                    )
                    requeued += 1
                except Exception as e:
                    logger.warning("Failed to reap idea %s (pid %s): %s", idea_id, pid, e)

        if requeued > 0:
            logger.info("Reaped %d ideas with dead PIDs", requeued)
        return requeued

    def _recover_fsm_from_crash(self):
        """Detect and recover ideas stuck in CLAIMED >6h."""
        stale = self.detect_stale_claims(timeout_hours=6)
        if not stale:
            return

        logger.info("FSM crash recovery: found %d stale claims, resetting to QUEUED", len(stale))
        for idea_id, _, _ in stale:
            try:
                self.record_state_transition(
                    idea_id,
                    from_state="CLAIMED",
                    to_state="QUEUED",
                    reason="crash_recovery_timeout_6h",
                    host="recovery",
                    pid=None,
                )
            except Exception as e:
                logger.warning("Failed to recover idea %s: %s", idea_id, e)

    def close(self):
        self.conn.close()
