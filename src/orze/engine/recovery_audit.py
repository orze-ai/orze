"""Read-only recovery and lifecycle acceptance audit."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import socket
import sqlite3
from pathlib import Path
from typing import Mapping

from orze.core.ideas import IDEA_ID_PATTERN
from orze.core.sqlite_policy import (
    SQLitePolicyError,
    inspect_shared_database_policy,
)
from orze.engine.process import process_is_running


_IDEA_RE = re.compile(IDEA_ID_PATTERN)
_MAX_ROWS = 100_000
_MAX_TRANSITIONS = 1_000_000
_MAX_CLAIM_BYTES = 64 * 1024
_REQUIRED_TABLES = frozenset({
    "ideas",
    "idea_state",
    "idea_transitions",
    "idea_stage_state",
    "idea_stage_transitions",
})
_STATUS_STATES = {
    "queued": frozenset({"QUEUED"}),
    "pending": frozenset({"QUEUED"}),
    "claimed": frozenset({"CLAIMED"}),
    "running": frozenset({"CLAIMED", "IN_PROGRESS"}),
    "training": frozenset({"IN_PROGRESS"}),
    "evaluating": frozenset({"IN_PROGRESS"}),
    "completed": frozenset({"COMPLETE"}),
    "partial": frozenset({"FAILED"}),
    "failed": frozenset({"FAILED"}),
    "dead": frozenset({"FAILED"}),
    "skipped": frozenset({"SKIPPED"}),
    "archived": frozenset({"ARCHIVED"}),
}
_ACTIVE_STATES = frozenset({"CLAIMED", "IN_PROGRESS"})
_TERMINAL_STATES = frozenset({"COMPLETE", "FAILED", "SKIPPED", "ARCHIVED"})
_STAGE_PAIRS = {
    "QUEUED": frozenset({
        ("NOT_STARTED", "NOT_STARTED"),
        ("PENDING", "PENDING"),
    }),
    "CLAIMED": frozenset({
        ("NOT_STARTED", "NOT_STARTED"),
        ("PENDING", "PENDING"),
    }),
    "IN_PROGRESS": frozenset({
        ("IN_PROGRESS", "PENDING"),
        ("COMPLETE", "PENDING"),
        ("COMPLETE", "IN_PROGRESS"),
    }),
    "COMPLETE": frozenset({
        ("COMPLETE", "COMPLETE"),
        ("COMPLETE", "SKIPPED"),
    }),
    "FAILED": frozenset({
        ("FAILED", "SKIPPED"),
        ("COMPLETE", "FAILED"),
    }),
    "SKIPPED": frozenset({("NOT_STARTED", "NOT_STARTED")}),
    "ARCHIVED": frozenset({("NOT_STARTED", "NOT_STARTED")}),
}


def _path_redirected(path: Path) -> bool:
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current = current / part
            if current.is_symlink():
                return True
    except OSError:
        return True
    return False


def _open_database(path: Path) -> sqlite3.Connection:
    path = Path(path).absolute()
    if (_path_redirected(path) or not path.is_file()
            or path.stat().st_nlink != 1):
        raise ValueError("recovery_database_unavailable_or_redirected")
    connection = sqlite3.connect(
        path.as_uri() + "?mode=ro", uri=True, timeout=5,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        policy = inspect_shared_database_policy(connection)
        if not policy["compliant"]:
            raise ValueError("recovery_database_policy_invalid")
        connection.execute("BEGIN")
        return connection
    except (sqlite3.Error, SQLitePolicyError, ValueError):
        connection.close()
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    before = os.fstat(descriptor)
    if before.st_nlink != 1:
        os.close(descriptor)
        raise OSError("recovery_database_identity_invalid")
    with os.fdopen(descriptor, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    current = path.lstat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
            or current.st_nlink != 1):
        raise OSError("recovery_database_changed_during_audit")
    return digest.hexdigest()


def _read_claim(path: Path) -> tuple[dict | None, str | None]:
    try:
        info = path.lstat()
        if (path.is_symlink() or not path.is_file() or info.st_nlink != 1
                or not 1 <= info.st_size <= _MAX_CLAIM_BYTES):
            return None, "claim_file_invalid"
        before = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        payload = json.loads(path.read_text(encoding="utf-8"))
        after_info = path.lstat()
        after = (
            after_info.st_dev, after_info.st_ino, after_info.st_size,
            after_info.st_mtime_ns,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "claim_file_invalid"
    if before != after or not isinstance(payload, dict):
        return None, "claim_file_changed_or_invalid"
    return payload, None


def _process_identity(claim: Mapping, prefix: str) -> tuple[int, int] | None:
    pid_key = "pid" if prefix == "owner" else "trainer_pid"
    ticks_key = (
        "owner_start_ticks" if prefix == "owner" else "trainer_start_ticks"
    )
    pid = claim.get(pid_key)
    ticks = claim.get(ticks_key)
    if (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
            or isinstance(ticks, bool) or not isinstance(ticks, int)
            or ticks <= 0):
        return None
    return pid, ticks


def audit_recovery_state(
    db_path: str | Path,
    results_dir: str | Path,
    *,
    hostname: str | None = None,
) -> dict:
    """Audit production lifecycle/process truth without mutating recovery state."""
    generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt = {
        "schema_version": 1,
        "generated_at": generated_at,
        "status": "UNVERIFIED",
        "reason": "recovery_evidence_incomplete",
        "checks": {},
        "counts": {},
        "contradiction_idea_ids": [],
        "evidence_gap_idea_ids": [],
        "rank_claim_proven": False,
    }
    try:
        source_before = _sha256(Path(__file__))
    except OSError:
        receipt["reason"] = "recovery_auditor_source_invalid"
        return receipt
    receipt["source_sha256"] = source_before
    database = Path(db_path).absolute()
    results = Path(results_dir).absolute()
    if _path_redirected(results):
        receipt["reason"] = "recovery_results_directory_redirected"
        return receipt
    try:
        connection = _open_database(database)
    except (OSError, sqlite3.Error, SQLitePolicyError, ValueError):
        receipt["reason"] = "recovery_database_invalid"
        return receipt

    contradictions = set()
    gaps = set()
    missing_tables = []
    remote_claims = set()
    status_conflicts = set()
    stage_conflicts = set()
    stage_missing = set()
    transition_conflicts = set()
    missing_global_states = set()
    orphan_global_states = set()
    active_count = 0
    live_terminal_count = 0
    dead_active_count = 0
    try:
        tables = {
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing_tables = sorted(_REQUIRED_TABLES - tables)
        if not {"ideas", "idea_state"}.issubset(tables):
            receipt["reason"] = "recovery_lifecycle_schema_missing"
            receipt["checks"]["required_tables"] = {
                "passed": False,
                "missing": missing_tables,
            }
            return receipt
        count = connection.execute(
            "SELECT COUNT(*) FROM ideas"
        ).fetchone()[0]
        if not isinstance(count, int) or count < 0 or count > _MAX_ROWS:
            receipt["reason"] = "recovery_row_limit_exceeded"
            return receipt
        rows = connection.execute(
            "SELECT i.idea_id, i.status, s.current_state "
            "FROM ideas AS i LEFT JOIN idea_state AS s "
            "ON s.idea_id = i.idea_id ORDER BY i.idea_id"
        ).fetchall()
        if len(rows) != count:
            receipt["reason"] = "recovery_lifecycle_rows_incomplete"
            return receipt
        idea_ids = {str(row["idea_id"]) for row in rows}
        state_rows = connection.execute(
            "SELECT idea_id FROM idea_state ORDER BY idea_id"
        ).fetchall()
        if len(state_rows) > _MAX_ROWS:
            receipt["reason"] = "recovery_row_limit_exceeded"
            return receipt
        state_ids = {str(row["idea_id"]) for row in state_rows}
        missing_global_states.update(idea_ids - state_ids)
        orphan_global_states.update(state_ids - idea_ids)
        contradictions.update(missing_global_states)
        contradictions.update(orphan_global_states)

        transition_count = connection.execute(
            "SELECT COUNT(*) FROM idea_transitions"
        ).fetchone()[0]
        if (not isinstance(transition_count, int) or transition_count < 0
                or transition_count > _MAX_TRANSITIONS):
            receipt["reason"] = "recovery_transition_limit_exceeded"
            return receipt
        global_last = {}
        for row in connection.execute(
            "SELECT idea_id, to_state FROM idea_transitions ORDER BY id"
        ):
            transition_idea_id = str(row["idea_id"])
            if transition_idea_id not in idea_ids:
                transition_conflicts.add(transition_idea_id)
                contradictions.add(transition_idea_id)
            global_last[transition_idea_id] = str(row["to_state"])

        stages = {}
        stage_last = {}
        if not missing_tables:
            for row in connection.execute(
                "SELECT idea_id, stage, current_state "
                "FROM idea_stage_state ORDER BY idea_id, stage"
            ):
                stage_idea_id = str(row["idea_id"])
                stage = str(row["stage"])
                if stage_idea_id not in idea_ids or stage not in {
                        "training", "evaluation"}:
                    stage_conflicts.add(stage_idea_id)
                    contradictions.add(stage_idea_id)
                    continue
                stages.setdefault(stage_idea_id, {})[stage] = str(
                    row["current_state"]
                )
            stage_transition_count = connection.execute(
                "SELECT COUNT(*) FROM idea_stage_transitions"
            ).fetchone()[0]
            if (not isinstance(stage_transition_count, int)
                    or stage_transition_count < 0
                    or stage_transition_count > _MAX_TRANSITIONS):
                receipt["reason"] = "recovery_transition_limit_exceeded"
                return receipt
            for row in connection.execute(
                "SELECT idea_id, stage, to_state "
                "FROM idea_stage_transitions ORDER BY id"
            ):
                key = (str(row["idea_id"]), str(row["stage"]))
                if key[0] not in idea_ids or key[1] not in {
                        "training", "evaluation"}:
                    stage_conflicts.add(key[0])
                    contradictions.add(key[0])
                    continue
                stage_last[key] = str(row["to_state"])

        local_host = hostname or socket.gethostname()
        for row in rows:
            idea_id = row["idea_id"]
            status = str(row["status"] or "").strip().lower()
            state = row["current_state"]
            if (not isinstance(idea_id, str)
                    or _IDEA_RE.fullmatch(idea_id) is None
                    or state not in _STATUS_STATES.get(status, frozenset())):
                safe_id = idea_id if isinstance(idea_id, str) else "<invalid>"
                status_conflicts.add(safe_id)
                contradictions.add(safe_id)
                continue
            if state in _ACTIVE_STATES:
                active_count += 1
            last_global = global_last.get(idea_id)
            if last_global is not None and last_global != state:
                transition_conflicts.add(idea_id)
                contradictions.add(idea_id)
            elif last_global is None and state in _ACTIVE_STATES:
                gaps.add(idea_id)

            stage_rows = stages.get(idea_id, {})
            pair = (
                stage_rows.get("training", "NOT_STARTED"),
                stage_rows.get("evaluation", "NOT_STARTED"),
            )
            if missing_tables:
                if state in _ACTIVE_STATES | {"COMPLETE", "FAILED"}:
                    stage_missing.add(idea_id)
                    gaps.add(idea_id)
            elif pair not in _STAGE_PAIRS[state]:
                if pair == ("NOT_STARTED", "NOT_STARTED"):
                    stage_missing.add(idea_id)
                    gaps.add(idea_id)
                else:
                    stage_conflicts.add(idea_id)
                    contradictions.add(idea_id)
            elif not missing_tables:
                for stage, current_state in stage_rows.items():
                    if stage_last.get((idea_id, stage)) != current_state:
                        transition_conflicts.add(idea_id)
                        contradictions.add(idea_id)

            claim_path = results / idea_id / "claim.json"
            if _path_redirected(claim_path):
                gaps.add(idea_id)
                continue
            if not claim_path.exists() and not claim_path.is_symlink():
                if state in _ACTIVE_STATES:
                    contradictions.add(idea_id)
                    dead_active_count += 1
                continue
            claim, claim_error = _read_claim(claim_path)
            if claim_error or claim is None:
                gaps.add(idea_id)
                continue
            claim_host = claim.get("claimed_by")
            if not isinstance(claim_host, str) or not claim_host:
                gaps.add(idea_id)
                continue
            if claim_host != local_host:
                remote_claims.add(idea_id)
                gaps.add(idea_id)
                continue
            identities = [
                identity for identity in (
                    _process_identity(claim, "owner"),
                    _process_identity(claim, "trainer"),
                )
                if identity is not None
            ]
            if not identities:
                gaps.add(idea_id)
                continue
            live = any(process_is_running(pid, ticks) for pid, ticks in identities)
            if state in _TERMINAL_STATES and live:
                contradictions.add(idea_id)
                live_terminal_count += 1
            elif state in _ACTIVE_STATES and not live:
                contradictions.add(idea_id)
                dead_active_count += 1

        database_sha = _sha256(database)
    except (OSError, sqlite3.Error, TypeError, ValueError):
        receipt["reason"] = "recovery_audit_read_failed"
        return receipt
    finally:
        connection.close()

    receipt["checks"] = {
        "required_tables": {
            "passed": not missing_tables,
            "missing": missing_tables,
        },
        "legacy_status_matches_global_state": {
            "passed": not status_conflicts,
            "idea_ids": sorted(status_conflicts),
        },
        "global_state_universe_exact": {
            "passed": (
                not missing_global_states and not orphan_global_states
            ),
            "missing_idea_ids": sorted(missing_global_states),
            "orphan_idea_ids": sorted(orphan_global_states),
        },
        "pipeline_stage_truth_complete": {
            "passed": not stage_missing and not stage_conflicts,
            "missing_idea_ids": sorted(stage_missing),
            "conflicting_idea_ids": sorted(stage_conflicts),
        },
        "transition_ledgers_match_current_state": {
            "passed": not transition_conflicts,
            "idea_ids": sorted(transition_conflicts),
        },
        "process_state_has_no_contradictions": {
            "passed": not contradictions,
        },
        "all_claim_owners_locally_verifiable": {
            "passed": not remote_claims,
            "idea_ids": sorted(remote_claims),
        },
    }
    receipt["counts"] = {
        "ideas": len(rows),
        "active_states": active_count,
        "live_process_terminal_states": live_terminal_count,
        "dead_process_active_states": dead_active_count,
        "status_conflicts": len(status_conflicts),
        "missing_global_states": len(missing_global_states),
        "orphan_global_states": len(orphan_global_states),
        "stage_conflicts": len(stage_conflicts),
        "stage_evidence_missing": len(stage_missing),
        "transition_conflicts": len(transition_conflicts),
        "evidence_gaps": len(gaps),
    }
    receipt["contradiction_idea_ids"] = sorted(contradictions)
    receipt["evidence_gap_idea_ids"] = sorted(gaps)
    receipt["database_sha256"] = database_sha
    try:
        if _sha256(Path(__file__)) != source_before:
            receipt["reason"] = "recovery_auditor_source_changed"
            receipt["evidence_gap_idea_ids"] = ["<auditor_source>"]
            return receipt
    except OSError:
        receipt["reason"] = "recovery_auditor_source_invalid"
        receipt["evidence_gap_idea_ids"] = ["<auditor_source>"]
        return receipt
    if contradictions:
        receipt["status"] = "FAILED"
        receipt["reason"] = "recovery_contradictions_detected"
    elif not gaps and not missing_tables:
        receipt["status"] = "VERIFIED"
        receipt["reason"] = "recovery_state_verified"
    return receipt


def write_recovery_audit(
    db_path: str | Path,
    results_dir: str | Path,
    output_path: str | Path,
) -> dict:
    receipt = audit_recovery_state(db_path, results_dir)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return receipt


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit recovery and lifecycle truth read-only",
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = write_recovery_audit(args.db, args.results_dir, args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
