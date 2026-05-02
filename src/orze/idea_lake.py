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

    def set_status(self, idea_id: str, status: str):
        """Update just the status of an idea. No-op if idea doesn't exist."""
        def _do():
            self.conn.execute(
                "UPDATE ideas SET status = ? WHERE idea_id = ?",
                (status, idea_id),
            )
            self.conn.commit()
        _retry_on_busy(_do)

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

    def get_queue(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Return unclaimed ideas, sorted by priority then ID."""
        def _do_get_queue():
            return self.conn.execute(
                """SELECT idea_id, title, priority, config, created_at
                   FROM ideas
                   WHERE status = 'queued' OR status = 'pending'
                   ORDER BY
                     CASE priority
                       WHEN 'critical' THEN 0
                       WHEN 'high' THEN 1
                       WHEN 'medium' THEN 2
                       WHEN 'low' THEN 3
                       ELSE 2
                     END,
                     id_num ASC
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
        """Return top N models by the given metric (from eval_metrics JSON)."""
        def _do_top():
            return self.conn.execute(
                "SELECT idea_id, title, config_summary, eval_metrics, status "
                "FROM ideas "
                "WHERE json_extract(eval_metrics, ?) IS NOT NULL "
                "ORDER BY json_extract(eval_metrics, ?) DESC LIMIT ?",
                (f"$.{metric}", f"$.{metric}", n),
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
        return results

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

    def close(self):
        self.conn.close()
