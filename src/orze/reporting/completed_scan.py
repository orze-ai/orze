"""Bounded lifecycle enumeration and requested family-label reads.

CALLING SPEC:
    with CompletedIdeaScan(db_path, page_size=128) as scan:
        for page in scan.pages(): ... qualify each ID; retain an aggregate ...
    Alternatively, iterate scan.family_pages(frozenset(ids)) for ordered
    mappings of requested, lifecycle-complete family labels.
    Consume either aggregate only AFTER normal context exit. A peer commit,
    schema/path change or reader error invalidates the whole scan. Pages are
    provisional lifecycle candidates, not qualified evidence or execution rights.

Each query releases its read transaction before artifact qualification. Schema
validation still covers all identities. ID enumeration reads every completed
candidate; family lookup reads every requested ID and refuses missing labels.
There is no cached champion or persistent cursor. Temporary page retention is
O(page_size); SQLite work and individual ID/artifact sizes are not bounded.
"""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import stat

from orze.reporting.evidence import _open_authoritative_lifecycle
from orze.reporting.lifecycle_stages import (
    completed_stage_sql, validate_lifecycle_schema,
)


class CompletedScanUnavailable(ValueError):
    """A content-safe reason why no aggregate may be consumed."""

    def __init__(self, reason, *, retryable=False):
        super().__init__(reason)
        # Only a peer data commit permits starting a fresh scan. No pages
        # from the invalidated scan may be retained by the caller.
        self.retryable = retryable


class CompletedIdeaScan:
    def __init__(self, db_path: Path, *, page_size: int = 128):
        self._conn = None
        if type(page_size) is not int or not 1 <= page_size <= 256:
            raise CompletedScanUnavailable("authoritative_lifecycle_page_size_invalid")
        self._path = Path(db_path).absolute()
        self._size = page_size
        self._started = False
        self._finished = False
        try:
            try:
                self._identity = self._path_identity()
            except FileNotFoundError as exc:
                raise CompletedScanUnavailable("authoritative_lifecycle_database_unavailable") from exc
            self._conn, reason = _open_authoritative_lifecycle(self._path)
            if self._conn is None:
                raise CompletedScanUnavailable(reason)
            self._pid = os.getpid()
            self._revision = self._stamp()
            try:
                self._conn.execute("BEGIN")
                schema = validate_lifecycle_schema(self._conn)
                self._predicate = completed_stage_sql(schema, idea_alias="i")
            finally:
                self._conn.rollback()
            self.verify()
        except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
            self.close()
            if isinstance(exc, CompletedScanUnavailable):
                raise
            raise CompletedScanUnavailable("authoritative_lifecycle_database_invalid") from exc

    def _path_identity(self):
        if any(path.is_symlink() for path in (self._path, *self._path.parents)):
            raise CompletedScanUnavailable("authoritative_lifecycle_database_redirected")
        info = self._path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CompletedScanUnavailable("authoritative_lifecycle_database_redirected")
        return info.st_dev, info.st_ino, info.st_mode

    def _stamp(self):
        if self._conn is None or self._conn.in_transaction:
            raise CompletedScanUnavailable("authoritative_lifecycle_scan_changed")
        return (self._conn.total_changes,
                self._conn.execute("PRAGMA data_version").fetchone()[0],
                self._conn.execute("PRAGMA schema_version").fetchone()[0])

    def verify(self):
        try:
            if self._pid != os.getpid() or self._path_identity() != self._identity:
                raise CompletedScanUnavailable("authoritative_lifecycle_scan_changed")
            revision = self._stamp()
            if revision[0] != self._revision[0] or revision[2] != self._revision[2]:
                raise CompletedScanUnavailable("authoritative_lifecycle_scan_changed")
            if revision[1] != self._revision[1]:
                raise CompletedScanUnavailable(
                    "authoritative_lifecycle_scan_changed", retryable=True)
        except (OSError, sqlite3.Error) as exc:
            raise CompletedScanUnavailable("authoritative_lifecycle_scan_changed") from exc

    def pages(self):
        """Yield each valid legacy-compatible completed ID once, in binary order."""
        if self._started:
            raise CompletedScanUnavailable("authoritative_lifecycle_scan_reused")
        self._started = True
        after = None
        while True:
            self.verify()
            try:
                self._conn.execute("BEGIN")
                where, values = ("", []) if after is None else (
                    " AND i.idea_id COLLATE BINARY > ?", [after])
                rows = self._conn.execute(
                    "SELECT i.idea_id FROM ideas AS i "
                    # Keep the ID scan outermost: a status-index-first plan
                    # sorts the remaining completed history again every page.
                    "CROSS JOIN idea_state AS s ON s.idea_id=i.idea_id "
                    "WHERE typeof(i.idea_id)='text' AND lower(i.status)='completed' "
                    "AND s.current_state COLLATE BINARY='COMPLETE' AND "
                    + self._predicate + where
                    + " ORDER BY i.idea_id COLLATE BINARY LIMIT ?",
                    [*values, self._size],
                ).fetchall()
            except sqlite3.Error as exc:
                raise CompletedScanUnavailable("authoritative_lifecycle_database_invalid") from exc
            finally:
                if self._conn is not None and self._conn.in_transaction:
                    self._conn.rollback()
            self.verify()
            if not rows:
                break
            after = rows[-1][0]
            page = frozenset(row[0] for row in rows
                             if row[0] not in ("", ".", "..")
                             and Path(row[0]).parts == (row[0],))
            # The set is the qualifier's current lifecycle membership proof;
            # consumers sort this bounded page when order affects accumulation.
            if page:
                yield page
            if len(rows) < self._size:
                break
        self.verify()
        self._finished = True

    def family_pages(self, idea_ids: frozenset[str]):
        """Yield requested complete labels in input order, then verify all pages.

        This is an alternative to ``pages()``, not a second pass on one scan.
        Each mapping is provisional until normal exhausted context exit.
        Only an immutable ID set is accepted; IDs retain legacy path rules.
        """
        from itertools import islice
        from orze.reporting.evidence import _SAFE_FAMILY_RE

        if self._started:
            raise CompletedScanUnavailable("authoritative_lifecycle_scan_reused")
        self._started = True
        if type(idea_ids) is not frozenset:
            raise CompletedScanUnavailable("authoritative_lifecycle_idea_ids_invalid")
        pending = iter(idea_ids)
        while True:
            selected = tuple(islice(pending, self._size))
            if not selected:
                break
            if any(not isinstance(key, str) or key in ("", ".", "..")
                   or Path(key).parts != (key,) for key in selected):
                raise CompletedScanUnavailable("authoritative_lifecycle_idea_ids_invalid")
            self.verify()
            try:
                self._conn.execute("BEGIN")
                marks = ",".join("?" for _ in selected)
                rows = self._conn.execute(
                    "SELECT i.idea_id, i.approach_family FROM ideas AS i "
                    "JOIN idea_state AS s ON s.idea_id=i.idea_id "
                    "WHERE lower(i.status)='completed' "
                    "AND s.current_state COLLATE BINARY='COMPLETE' AND "
                    + self._predicate + f" AND i.idea_id IN ({marks})",
                    selected,
                ).fetchall()
            except sqlite3.Error as exc:
                raise CompletedScanUnavailable("authoritative_lifecycle_database_invalid") from exc
            finally:
                if self._conn is not None and self._conn.in_transaction:
                    self._conn.rollback()
            self.verify()
            families = {}
            for key, raw_family in rows:
                family = str(raw_family or "other").strip().lower()
                families[key] = family if _SAFE_FAMILY_RE.fullmatch(family) else "other"
            if families.keys() != set(selected):
                raise CompletedScanUnavailable("authoritative_family_evidence_incomplete")
            yield {key: families[key] for key in selected}
        self.verify()
        self._finished = True


    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        try:
            self.verify()
            return self
        except Exception:
            self.close()
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                self.verify()
                if not self._finished:
                    raise CompletedScanUnavailable("authoritative_lifecycle_scan_incomplete")
        finally:
            self.close()
