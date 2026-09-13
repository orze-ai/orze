"""Bounded metadata-only recorded action results for replaceable CPU policies.

This fresh same-Lake view is neither a content hash check nor source/launch
authority. Analyze dispatch must independently capture_sources again. Negative
outcomes and invalid/unknown observations remain unchanged; no score is made
up for absent evidence. No schema creation, writer transaction, or local
harvest cache is used. The legacy view considers at most 32 current terminals.
An invocation-owned pager can traverse them in bounded, revision-fenced pages.
"""
from dataclasses import asdict
import json
import os
import secrets
import sqlite3

from orze.core.execution_attempts import AttemptRef, _schema, _json, require_current
from orze.core.artifact_contract import validate_artifact_publication_binding
from orze.core.observation_contract import validate_observation_publication_binding
from orze.core.research_artifacts import artifacts_for_attempt
from orze.core.research_observations import observations_for_attempt
from orze.engine import artifact_publication as files
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.cpu_action_sources import _route, _effect
from orze.engine.execution_catalog import declared_catalog
from orze.engine.termination_hold import require_no_unconfirmed_stop
from pathlib import Path


class PolicyEvidenceHOLD(AttemptEffectInDoubt):
    """The result window itself cannot be read safely from this namespace."""


class _Unavailable(ValueError):
    pass


def _raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _id_set(ids, records, key):
    if (type(ids) is not list or any(type(value) is not str for value in ids)
            or len(ids) != len(set(ids)) or set(ids) != {record[key] for record in records}):
        raise _Unavailable("terminal_record_ids_changed")


def _read_sets(lake, scope, database, ref):
    folder = Path(scope) / ref.task_id
    require_no_unconfirmed_stop(folder)
    if declared_catalog(folder) != database:
        raise _Unavailable("evidence_catalog_mismatch")
    row = require_current(lake.conn, ref, states=("TERMINAL",))
    terminal = row["terminal"]
    if (type(terminal) is not dict or type(terminal.get("outcome")) is not str
            or terminal["outcome"] not in ("completed", "failed", "interrupted")):
        raise _Unavailable("evidence_outcome_invalid")
    artifacts = artifacts_for_attempt(lake.conn, ref)
    observations = observations_for_attempt(lake.conn, ref)
    _id_set(terminal.get("artifact_ids"), artifacts, "artifact_id")
    _id_set(terminal.get("observation_ids"), observations, "observation_id")
    if artifacts:
        binding = validate_artifact_publication_binding(row["binding"].get("artifact_publication"))
        if binding["scope"] != scope or any(
                artifact["scope"] != scope or artifact["spec_fingerprint"] != binding["spec_fingerprint"]
                or artifact["path"] != str(Path(binding["root"]) / artifact["artifact_id"] / "content")
                for artifact in artifacts):
            raise _Unavailable("evidence_artifact_scope_invalid")
    if observations:
        binding = validate_observation_publication_binding(row["binding"].get("observation_publication"))
        if binding["scope"] != scope:
            raise _Unavailable("evidence_observation_scope_invalid")
        fields = {key for key in binding if key != "version"}
        for observation in observations:
            if (observation["schema"] != binding.get("version", 1)
                    or _raw({key: observation[key] for key in fields}) != _raw({key: binding[key] for key in fields})
                    or not set(observation["result_artifact_ids"]).issubset({item["artifact_id"] for item in artifacts})):
                raise _Unavailable("evidence_observation_binding_changed")
    return row, artifacts, observations


def _collect(lake, scope, database, refs, more_available, *, validate=_json):
    # Reserve diagnostic space before admitting any potentially large result.
    view = {"results": [], "unavailable": [
        {"ref": asdict(ref), "reason": "evidence_verification_unavailable"} for ref in refs],
        "more_available": more_available}
    validate(view)
    accepted = []
    for ref in refs:
        diagnostic = next(item for item in view["unavailable"] if item["ref"] == asdict(ref))
        try:
            before = _read_sets(lake, scope, database, ref)
            identities = _effect(ref, Path(scope) / ref.task_id, before[0])
            after = _read_sets(lake, scope, database, ref)
            if _raw(before) != _raw(after):
                raise _Unavailable("evidence_changed_during_read")
            files._verify_identities(identities)
            result = {"ref": asdict(ref), "outcome": before[0]["terminal"]["outcome"],
                      "artifact_records": before[1], "observation_records": before[2]}
            trial = {**view, "results": [*view["results"], result],
                     "unavailable": [item for item in view["unavailable"] if item is not diagnostic]}
            try:
                validate(trial)
            except (ValueError, RuntimeError):
                diagnostic["reason"] = "evidence_result_limit"
                view["more_available"] = True
            else:
                view = trial
                accepted.append((ref, _raw(before), identities, result))
        except (OSError, ValueError, TypeError, RuntimeError, sqlite3.Error) as exc:
            diagnostic["reason"] = str(exc) if type(exc) is _Unavailable else "evidence_unconfirmed"
    # A later item must not hide rotation of an earlier current result.
    for ref, expected, identities, result in accepted:
        try:
            if _raw(_read_sets(lake, scope, database, ref)) != expected:
                raise _Unavailable("evidence_changed_during_read")
            files._verify_identities(identities)
        except (OSError, ValueError, TypeError, RuntimeError, sqlite3.Error):
            view["results"] = [item for item in view["results"] if item is not result]
            view["unavailable"].append({"ref": asdict(ref), "reason": "evidence_changed_during_read"})
    return view


def recorded_evidence(lake, results_dir, *, limit=32):
    """Return results/unavailable/more_available; whole JSON obeys _json limits.

    Oversized records are never truncated into misleading observations. Their
    exact ref is reported unavailable and more_available is true. That flag
    indicates omitted evidence, not an authorization or pagination cursor.
    """
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError("cpu_policy_evidence_limit_invalid")
    try:
        if lake.conn.in_transaction:
            raise PolicyEvidenceHOLD("cpu_policy_evidence_caller_transaction_active")
        scope, database, route_ids = _route(lake, results_dir)
        empty = {"results": [], "unavailable": [], "more_available": False}
        if not _schema(lake.conn):
            return empty
        candidates = lake.conn.execute("""
            SELECT a.task_id,a.phase,a.attempt_id,a.generation
            FROM main.execution_attempts a
            WHERE a.phase='action' COLLATE BINARY AND a.state='TERMINAL'
              AND NOT EXISTS (SELECT 1 FROM main.execution_attempts b
                WHERE b.task_id=a.task_id COLLATE BINARY AND b.phase=a.phase COLLATE BINARY
                  AND b.generation>a.generation)
            ORDER BY a.task_id COLLATE BINARY LIMIT ?
        """, (limit + 1,)).fetchall()
        refs = [AttemptRef(*row) for row in candidates[:limit]]
        view = _collect(lake, scope, database, refs, len(candidates) > limit)
        if _route(lake, results_dir) != (scope, database, route_ids):
            raise PolicyEvidenceHOLD("cpu_policy_evidence_route_changed")
        return json.loads(_json(view))
    except PolicyEvidenceHOLD:
        raise
    except (OSError, ValueError, TypeError, RuntimeError, sqlite3.Error) as exc:
        raise PolicyEvidenceHOLD("cpu_policy_evidence_read_unconfirmed") from exc


class EvidencePager:
    """Read-only, process/connection-bound scan; never an execution capability.

    The coordinator owns this object; Policy receives only detached JSON and a
    one-use next token. No persistent cursor, SQL table, all-history cache or
    long-lived read transaction is introduced. Any SQL write invalidates the
    revision, conservatively including unrelated writes and rolled-back local
    writes. A fresh invocation must open a fresh scan.

    traversal_end means enumeration of *current terminal* rows ended in this
    revision, not scientific convergence or current validity of all old files.
    Each page verifies its own effect metadata; selected sources are freshly
    re-read, and normal admission still rechecks content/ownership separately.
    """

    def __init__(self, lake, results_dir, *, limit=32):
        if type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("cpu_policy_evidence_limit_invalid")
        self._lake, self._conn, self._results = lake, lake.conn, results_dir
        self._pid, self._limit = os.getpid(), limit
        self._route = _route(lake, results_dir)
        self._revision = self._stamp()
        self._scan_id = secrets.token_hex(24)
        self._after, self._cursor = None, None
        self._page = self._seen = self._unavailable = 0
        self._end = False
        self.verify()

    def _stamp(self):
        if self._conn.in_transaction:
            raise PolicyEvidenceHOLD("cpu_policy_evidence_caller_transaction_active")
        return (self._conn.total_changes,
                self._conn.execute("PRAGMA main.data_version").fetchone()[0],
                self._conn.execute("PRAGMA main.schema_version").fetchone()[0])

    def verify(self):
        try:
            if (os.getpid() != self._pid or self._lake.conn is not self._conn
                    or _route(self._lake, self._results) != self._route
                    or self._stamp() != self._revision):
                raise PolicyEvidenceHOLD("cpu_policy_evidence_scan_stale")
        except PolicyEvidenceHOLD:
            raise
        except (OSError, ValueError, TypeError, RuntimeError, sqlite3.Error) as exc:
            raise PolicyEvidenceHOLD("cpu_policy_evidence_scan_unconfirmed") from exc

    def read(self, *, cursor=None, refs=None):
        self.verify()
        selection = refs is not None
        if selection:
            if (cursor is not None or self._page == 0 or type(refs) is not list
                    or not 1 <= len(refs) <= 32):
                raise PolicyEvidenceHOLD("cpu_policy_evidence_selection_invalid")
            try:
                requested = []
                for value in refs:
                    if type(value) is not dict or set(value) != {"task_id", "phase", "attempt_id", "generation"}:
                        raise ValueError("ref shape")
                    ref = AttemptRef(**value)
                    if ref.phase != "action" or ref in requested:
                        raise ValueError("ref phase or duplicate")
                    requested.append(ref)
            except (ValueError, TypeError, RuntimeError) as exc:
                raise PolicyEvidenceHOLD("cpu_policy_evidence_selection_invalid") from exc
            after, next_cursor, end = self._after, self._cursor, self._end
            page, seen = self._page, self._seen
        else:
            if ((self._page == 0 and cursor is not None)
                    or (self._page > 0 and (type(cursor) is not str
                        or self._cursor is None or cursor != self._cursor))):
                raise PolicyEvidenceHOLD("cpu_policy_evidence_cursor_invalid")
            rows = []
            if _schema(self._conn):
                extra, args = "", []
                if self._after is not None:
                    extra = " AND (a.task_id COLLATE BINARY,a.phase COLLATE BINARY,a.generation,a.attempt_id COLLATE BINARY) > (?,?,?,?)"
                    args.extend(self._after)
                rows = self._conn.execute("""
                    SELECT a.task_id,a.phase,a.attempt_id,a.generation
                    FROM main.execution_attempts a
                    WHERE a.phase='action' COLLATE BINARY AND a.state='TERMINAL'
                      AND NOT EXISTS (SELECT 1 FROM main.execution_attempts b
                        WHERE b.task_id=a.task_id COLLATE BINARY AND b.phase=a.phase COLLATE BINARY
                          AND b.generation>a.generation)
                    """ + extra + " ORDER BY a.task_id COLLATE BINARY,a.phase COLLATE BINARY,a.generation,a.attempt_id COLLATE BINARY LIMIT ?",
                    (*args, self._limit + 1)).fetchall()
            requested = [AttemptRef(*row) for row in rows[:self._limit]]
            after = self._after
            if requested:
                ref = requested[-1]
                after = (ref.task_id, ref.phase, ref.generation, ref.attempt_id)
            end = len(rows) <= self._limit
            next_cursor = None if end else secrets.token_hex(24)
            page, seen = self._page + 1, self._seen + len(requested)
        meta = {"schema": 1, "scan_id": self._scan_id, "mode": "selection" if selection else "scan",
                "page": page, "seen": seen, "traversal_end": end,
                "unavailable_seen": self._unavailable + len(requested), "next_cursor": next_cursor}
        def validate(view):
            return _json({"recorded_evidence": view, "evidence_page": meta})
        view = _collect(self._lake, self._route[0], self._route[1], requested,
                        selection or page > 1 or not end, validate=validate)
        self.verify()
        unavailable = self._unavailable + (0 if selection else len(view["unavailable"]))
        meta["unavailable_seen"] = unavailable
        output = json.loads(_json({"recorded_evidence": view, "evidence_page": meta}))
        self._after, self._cursor, self._end = after, next_cursor, end
        self._page, self._seen, self._unavailable = page, seen, unavailable
        return output
