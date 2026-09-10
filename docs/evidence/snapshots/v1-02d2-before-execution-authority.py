"""Coordinate native attempt/lifecycle transactions with bounded file intents.

Lock order is the task's effect guard, then SQLite BEGIN IMMEDIATE. No process
launch/wait, provider call, or large artifact hash belongs in this transaction.
The low-level attempt store never commits; this coordinator owns that boundary.
File publication is not transactional: once prepare starts, any uncertain exit
retains the guard and intent instead of permitting a different attempt to write.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
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


@contextmanager
def execution_transaction(lake, idea_dir: Path, *, lease=None):
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
        previous_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        conn.execute("PRAGMA busy_timeout=1000")
        committed = False
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield tx
            if not conn.in_transaction:
                raise AttemptEffectInDoubt("execution_transaction_ended_by_callback")
            require_effect_lease(acquired, idea_dir)
            if tx.prepare_started:
                if tx.prepared_ref is None or tx.prepared_sha256 is None:
                    raise AttemptEffectInDoubt("execution_effect_prepare_incomplete")
                terminal = require_current(
                    conn, tx.prepared_ref, states=("TERMINAL", "NOT_STARTED"))
                if (terminal["terminal"] is None
                        or terminal["terminal"].get("effect_receipt_sha256")
                        != tx.prepared_sha256):
                    raise AttemptEffectInDoubt("execution_effect_terminal_not_bound")
            conn.commit()
            committed = True
            if tx.prepare_started:
                confirm_effect(acquired, tx.prepared_ref, tx.prepared_sha256)
        except BaseException as exc:
            if conn.in_transaction:
                conn.rollback()
            if tx.prepare_started or committed:
                raise AttemptEffectInDoubt("execution_effect_commit_unconfirmed") from exc
            if isinstance(exc, (AttemptAuthorityError, sqlite3.Error)) and not isinstance(exc, StaleAttempt):
                raise AttemptEffectBusy("execution_authority_rejected") from exc
            raise
        finally:
            try:
                conn.execute(f"PRAGMA busy_timeout={int(previous_timeout)}")
            except sqlite3.Error as exc:
                if tx.prepare_started or committed:
                    raise AttemptEffectInDoubt("execution_transaction_cleanup_unconfirmed") from exc
                raise AttemptEffectBusy("execution_transaction_cleanup_failed") from exc


def lifecycle_fence(lake, idea_id: str, phase: str) -> dict:
    """Capture immutable transition identities as well as state labels.

    Reading under the coordinator's write transaction prevents a same-state
    ABA transition from passing the publication fence. Missing/contradictory
    lifecycle records do not become a fabricated historical attempt.
    """
    row = lake.conn.execute(
        "SELECT i.status,s.current_state,(SELECT MAX(id) FROM idea_transitions "
        "WHERE idea_id=i.idea_id COLLATE BINARY) FROM ideas i JOIN idea_state s "
        "ON s.idea_id=i.idea_id COLLATE BINARY WHERE i.idea_id=? COLLATE BINARY",
        (idea_id,),
    ).fetchone()
    stage = lake.conn.execute(
        "SELECT current_state,(SELECT MAX(id) FROM idea_stage_transitions "
        "WHERE idea_id=? COLLATE BINARY AND stage=? COLLATE BINARY) "
        "FROM idea_stage_state WHERE idea_id=? COLLATE BINARY AND stage=? COLLATE BINARY",
        (idea_id, phase, idea_id, phase),
    ).fetchone()
    if (row is None or type(row[2]) is not int or row[2] <= 0
            or stage is None or type(stage[1]) is not int or stage[1] <= 0):
        raise AttemptAuthorityError("execution_lifecycle_identity_missing")
    return {"legacy_status": row[0], "global_state": row[1],
            "global_transition_id": row[2], "phase_state": stage[0],
            "phase_transition_id": stage[1]}
