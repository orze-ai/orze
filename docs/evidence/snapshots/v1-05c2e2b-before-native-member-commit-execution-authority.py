"""Coordinate native attempt/lifecycle transactions with bounded file intents.

Lock order is the task's effect guard, then SQLite BEGIN IMMEDIATE. No process
launch/wait, provider call, or large artifact hash belongs in this transaction.
The low-level attempt store never commits; this coordinator owns that boundary.
File publication is not transactional: once prepare starts, any uncertain exit
retains the guard and intent instead of permitting a different attempt to write.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import sqlite3

from orze.core.execution_attempts import (
    AttemptAuthorityError, AttemptRef, StaleAttempt, require_current,
)
from orze.engine.attempt_effect_lock import (
    AttemptEffectBusy, AttemptEffectInDoubt, AttemptEffectLease,
    attempt_effect_lock, require_effect_lease,
)


@dataclass
class ExecutionTransaction:
    lake: object
    idea_dir: Path
    lease: AttemptEffectLease
    prepared_ref: AttemptRef | None = None
    prepared_sha256: str | None = None
    prepare_started: bool = False
    _watched_attempts: dict = field(default_factory=dict, init=False, repr=False)
    _watched_dependencies: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def conn(self):
        return self.lake.conn

    def prepare(self, ref: AttemptRef, plan: dict) -> str:
        """Persist intent before any callback-controlled file mutation."""
        from orze.engine.attempt_effect_receipts import prepare_effect

        if self.prepare_started or ref.task_id != self.idea_dir.name:
            raise AttemptAuthorityError("execution_effect_prepare_invalid")
        require_effect_lease(self.lease, self.idea_dir)
        require_current(self.conn, ref)
        # Publication can fail after a partial directory/name change. Set this
        # before the call so even its own I/O failure retains the owner.
        self.prepare_started = True
        self.prepared_ref = ref
        self.prepared_sha256 = prepare_effect(self.lease, ref, plan)
        return self.prepared_sha256

    def watch_attempt(self, ref: AttemptRef) -> None:
        """Bind a final row after all intended SQL mutations, before commit.

        Used by intent/started adapters without a terminal file-effect plan.
        It grants no launch permission and assumes no generic phase stages.
        Explicit binding/terminal lifecycle receipts must both be current.
        """
        if (not self.conn.in_transaction or ref.task_id != self.idea_dir.name
                or ref in self._watched_attempts or len(self._watched_attempts) >= 32):
            raise AttemptAuthorityError("execution_attempt_watch_invalid")
        require_effect_lease(self.lease, self.idea_dir)
        self._watched_attempts[ref] = _watched_snapshot(self, ref)

    def _verify_watches(self) -> None:
        for ref, expected in self._watched_attempts.items():
            if _watched_snapshot(self, ref) != expected:
                raise AttemptAuthorityError("execution_watched_attempt_changed")
        for ref, expected in self._watched_dependencies.items():
            if _dependency_snapshot(self, ref) != expected:
                raise AttemptAuthorityError("execution_dependency_changed")

    def watch_dependency(self, ref: AttemptRef) -> None:
        """Pin a closed source attempt without reviving its old lifecycle.

        A later controller action may legitimately change the task lifecycle.
        This watch requires only the exact source row to remain closed/current;
        it is not permission to launch or mutate that historical attempt.
        """
        if (not self.conn.in_transaction or ref.task_id != self.idea_dir.name
                or ref in self._watched_dependencies
                or len(self._watched_dependencies) >= 32):
            raise AttemptAuthorityError("execution_dependency_watch_invalid")
        require_effect_lease(self.lease, self.idea_dir)
        self._watched_dependencies[ref] = _dependency_snapshot(self, ref)


def _canonical(value) -> str:
    # Do not let Python's True == 1 == 1.0 equality authorize a different
    # recorded fence. Attempt storage has already checked JSON size/shape.
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def canonical_identity_equal(left, right) -> bool:
    """Compare bounded JSON-object identities without Python numeric coercion.

    Invalid identities are never equal, including two equally invalid values.
    Reuse the attempt store's depth/node/UTF-8 limits and type validation;
    this helper does not normalize arbitrary objects into authoritative data.
    """
    from orze.core.execution_attempts import _json

    try:
        return _json(left) == _json(right)
    except (AttemptAuthorityError, TypeError, ValueError, RecursionError):
        return False


def _watched_snapshot(tx: ExecutionTransaction, ref: AttemptRef) -> str:
    row = require_current(tx.conn, ref, states=(
        "LAUNCHING", "RUNNING", "TERMINAL", "NOT_STARTED", "IN_DOUBT"))
    for payload in (row["binding"], row["terminal"]):
        if payload is not None and "lifecycle" in payload:
            current = lifecycle_fence(tx.lake, ref.task_id, _lifecycle_phase(payload, ref))
            if not canonical_identity_equal(payload["lifecycle"], current):
                raise AttemptAuthorityError("execution_watched_lifecycle_changed")
    return _canonical(row)


def _dependency_snapshot(tx: ExecutionTransaction, ref: AttemptRef) -> str:
    return _canonical(require_current(tx.conn, ref, states=("TERMINAL", "NOT_STARTED")))


def _lifecycle_phase(payload, ref):
    phase = payload.get("lifecycle_phase", ref.phase)
    if (type(phase) is not str or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", phase)
            or phase in (".", "..")):
        raise AttemptAuthorityError("execution_lifecycle_phase_invalid")
    return phase


def _bound_terminal(tx: ExecutionTransaction) -> str:
    if tx.prepared_ref is None or tx.prepared_sha256 is None:
        raise AttemptEffectInDoubt("execution_effect_prepare_incomplete")
    row = require_current(tx.conn, tx.prepared_ref,
                          states=("TERMINAL", "NOT_STARTED"))
    terminal = row["terminal"]
    if (terminal is None or terminal.get("effect_receipt_sha256")
            != tx.prepared_sha256):
        raise AttemptEffectInDoubt("execution_effect_terminal_not_bound")
    # Generic phases need not claim a framework lifecycle. If a caller binds
    # one, every final table write must still agree with that exact receipt.
    if "lifecycle" in terminal:
        fence = lifecycle_fence(tx.lake, tx.prepared_ref.task_id,
                                _lifecycle_phase(terminal, tx.prepared_ref))
        if not canonical_identity_equal(terminal["lifecycle"], fence):
            raise AttemptEffectInDoubt("execution_effect_lifecycle_not_bound")
    return _canonical(row)


@contextmanager
def execution_transaction(lake, idea_dir: Path, *, lease=None):
    # Membership is outside SQL commit, file confirmation, timeout restoration
    # AND effect-guard exit. finish_attempt alone never marks it SETTLED.
    from orze.engine.controller_members import begin_transaction, end_transaction
    ticket = begin_transaction(lake)
    try:
        with _execution_transaction(lake, idea_dir, lease=lease) as tx:
            yield tx
    except BaseException:
        end_transaction(ticket, success=False)
        raise
    else:
        end_transaction(ticket, success=True)


@contextmanager
def _execution_transaction(lake, idea_dir: Path, *, lease=None):
    """Yield a short writer, committing only its own complete transaction.

    When prepare() was used, the same SQLite transaction must close that
    attempt and include ``effect_receipt_sha256`` in its terminal object.
    The matching filesystem confirmation is published only after SQL commit.
    A returned context does not by itself grant current-attempt ownership:
    callers must verify their ref and lifecycle fence before reads/effects.
    """
    from orze.engine.attempt_effect_receipts import confirm_effect

    conn = lake.conn
    if conn.in_transaction:
        raise AttemptEffectBusy("execution_caller_transaction_active")
    idea_dir = Path(idea_dir).absolute()
    with attempt_effect_lock(idea_dir, lease=lease) as acquired:
        tx = ExecutionTransaction(lake, idea_dir, acquired)
        previous_timeout = None
        began = commit_attempted = False
        failure = None
        cleanup_failures = []
        uncertain = False
        try:
            previous_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            if type(previous_timeout) is not int or previous_timeout < 0:
                raise AttemptAuthorityError("execution_timeout_invalid")
            conn.execute("PRAGMA busy_timeout=1000")
            if conn.execute("PRAGMA busy_timeout").fetchone()[0] != 1000:
                raise AttemptAuthorityError("execution_timeout_not_set")
            conn.execute("BEGIN IMMEDIATE")
            began = True
            yield tx
            if not conn.in_transaction:
                raise AttemptEffectInDoubt("execution_transaction_ended_by_callback")
            require_effect_lease(acquired, idea_dir)
            expected_terminal = _bound_terminal(tx) if tx.prepare_started else None
            tx._verify_watches()
            # An exception from commit cannot prove whether SQL committed.
            # A no-op commit is equally unsuitable as a publication receipt.
            commit_attempted = True
            conn.commit()
            if conn.in_transaction:
                raise AttemptEffectInDoubt("execution_commit_not_confirmed")
            tx._verify_watches()
            if tx.prepare_started:
                require_effect_lease(acquired, idea_dir)
                if _bound_terminal(tx) != expected_terminal:
                    raise AttemptEffectInDoubt("execution_committed_terminal_changed")
                confirm_effect(acquired, tx.prepared_ref, tx.prepared_sha256)
        except BaseException as exc:
            failure = exc
            uncertain = (isinstance(exc, AttemptEffectInDoubt)
                         or tx.prepare_started or commit_attempted
                         or (began and not conn.in_transaction))
            if conn.in_transaction:
                try:
                    conn.rollback()
                    if conn.in_transaction:
                        raise AttemptEffectInDoubt("execution_rollback_not_confirmed")
                except BaseException as cleanup:
                    cleanup_failures.append(cleanup)
                    uncertain = True
        finally:
            if type(previous_timeout) is int and previous_timeout >= 0:
                try:
                    conn.execute(f"PRAGMA busy_timeout={previous_timeout}")
                    restored = conn.execute("PRAGMA busy_timeout").fetchone()[0]
                    if type(restored) is not int or restored != previous_timeout:
                        raise AttemptEffectInDoubt("execution_timeout_restore_not_confirmed")
                except BaseException as cleanup:
                    cleanup_failures.append(cleanup)
                    uncertain = True
        # Never let a secondary cleanup error replace the original cause or
        # downgrade uncertainty to an ordinary retryable pre-effect rejection.
        if failure is not None or cleanup_failures:
            if uncertain:
                error = AttemptEffectInDoubt("execution_transaction_unconfirmed")
                # Keep secondary diagnostics without Exception.add_note,
                # which is unavailable on supported Python 3.9/3.10 runtimes.
                error.cleanup_failures = tuple(cleanup_failures)
                raise error from (failure if failure is not None else cleanup_failures[0])
            if isinstance(failure, (AttemptAuthorityError, sqlite3.Error)) and not isinstance(failure, StaleAttempt):
                raise AttemptEffectBusy("execution_authority_rejected") from failure
            raise failure.with_traceback(failure.__traceback__)


def lifecycle_fence(lake, idea_id: str, phase: str) -> dict:
    """Capture immutable transition identities as well as state labels.

    Reading under the coordinator's write transaction prevents a same-state
    ABA transition from passing the publication fence. Missing/contradictory
    lifecycle records do not become a fabricated historical attempt.
    """
    from orze.idea_lake import STATE_TO_STATUS

    # Explicit main qualification shares the attempt store's namespace. One
    # statement provides a coherent read snapshot even for read-only callers.
    # Fetch two rather than silently accepting the first ambiguous join row.
    try:
        rows = lake.conn.execute(
            "SELECT i.status,s.current_state,g.id,g.to_state,"
            "p.current_state,t.id,t.to_state "
            "FROM main.ideas i JOIN main.idea_state s "
            "ON s.idea_id COLLATE BINARY=i.idea_id COLLATE BINARY "
            "JOIN main.idea_transitions g "
            "ON g.idea_id COLLATE BINARY=i.idea_id COLLATE BINARY "
            "AND g.id=(SELECT MAX(id) FROM main.idea_transitions "
            "WHERE idea_id COLLATE BINARY=i.idea_id COLLATE BINARY) "
            "JOIN main.idea_stage_state p "
            "ON p.idea_id COLLATE BINARY=i.idea_id COLLATE BINARY "
            "AND p.stage COLLATE BINARY=? "
            "JOIN main.idea_stage_transitions t "
            "ON t.idea_id COLLATE BINARY=i.idea_id COLLATE BINARY "
            "AND t.stage COLLATE BINARY=p.stage COLLATE BINARY "
            "AND t.id=(SELECT MAX(id) FROM main.idea_stage_transitions "
            "WHERE idea_id COLLATE BINARY=i.idea_id COLLATE BINARY "
            "AND stage COLLATE BINARY=p.stage COLLATE BINARY) "
            "WHERE i.idea_id COLLATE BINARY=? LIMIT 2", (phase, idea_id),
        ).fetchmany(2)
    except sqlite3.Error as exc:
        raise AttemptAuthorityError("execution_lifecycle_schema_invalid") from exc
    if len(rows) != 1:
        raise AttemptAuthorityError("execution_lifecycle_identity_missing")
    row = rows[0]
    if (type(row[2]) is not int or row[2] <= 0
            or type(row[5]) is not int or row[5] <= 0):
        raise AttemptAuthorityError("execution_lifecycle_identity_missing")
    if (row[1] not in STATE_TO_STATUS or row[0] != STATE_TO_STATUS[row[1]]
            or row[3] != row[1] or row[6] != row[4]
            or row[4] not in {"NOT_STARTED", "PENDING", "IN_PROGRESS",
                              "COMPLETE", "FAILED", "SKIPPED"}):
        raise AttemptAuthorityError("execution_lifecycle_state_conflict")
    return {"legacy_status": row[0], "global_state": row[1],
            "global_transition_id": row[2], "phase_state": row[4],
            "phase_transition_id": row[5]}
