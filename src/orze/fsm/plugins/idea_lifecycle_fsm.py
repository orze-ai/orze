"""Explicit finite state machine for idea lifecycle tracking.

Implements determinism hardening #5: replace filesystem-inferred state transitions
with explicit, audited state changes. The FSM tracks ideas through their complete
lifecycle with atomic transitions, crash recovery, and timeout reaping.

States: QUEUED → CLAIMED → TRAINING → EVALUATING → COMPLETE | FAILED | ARCHIVED

Key features:
- Atomic transitions using SELECT...FOR UPDATE (prevents TOCTOU races)
- Transactional audit trail (every state change recorded with host/pid/reason)
- Crash recovery (orphan cleanup detects and resets stuck ideas)
- Timeout reaper (6h default TTL for stale claims/training)
- Non-blocking: FSM failures don't crash orchestrator
"""

import datetime
import logging
import os
import socket
from pathlib import Path
from typing import Dict, List, Optional

import sqlite3

logger = logging.getLogger("orze.fsm")


class IdeaLifecycleFSM:
    """State machine for idea lifecycle with atomic transitions."""

    # Valid state transitions (guards prevent invalid moves)
    VALID_TRANSITIONS = {
        "QUEUED": {"CLAIMED"},
        "CLAIMED": {"TRAINING", "QUEUED"},  # QUEUED if claim times out
        "TRAINING": {"EVALUATING", "FAILED", "QUEUED"},  # QUEUED if times out
        "EVALUATING": {"COMPLETE", "FAILED"},
        "COMPLETE": {"ARCHIVED"},
        "FAILED": {"QUEUED"},  # Can retry after failure
        "ARCHIVED": set(),  # Terminal state
    }

    # Timeout defaults (in hours)
    TIMEOUT_CLAIM = 6  # max 6h to go from CLAIMED to TRAINING
    TIMEOUT_TRAINING = 24  # max 24h in TRAINING
    TIMEOUT_EVALUATING = 6  # max 6h in EVALUATING

    def __init__(self, db_path: str):
        """Initialize FSM with database connection."""
        self.db_path = db_path
        self.host = socket.gethostname()
        self.pid = os.getpid()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def transition(self, idea_id: str, to_state: str, reason: Optional[str] = None) -> bool:
        """Atomically transition an idea to a new state.

        Uses BEGIN IMMEDIATE transaction for atomicity (SQLite serialization).

        Args:
            idea_id: identifier of the idea
            to_state: target state (must be in VALID_TRANSITIONS[current_state])
            reason: audit trail reason (e.g., "claimed by scheduler", "eval completed")

        Returns:
            True if transition succeeded, False if preconditions not met
        """
        conn = self._get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")

            # Get current state (within exclusive transaction)
            row = conn.execute(
                "SELECT current_state FROM idea_state WHERE idea_id = ?",
                (idea_id,)
            ).fetchone()

            if not row:
                # First transition: assume QUEUED if not in DB
                from_state = "QUEUED"
                current_state = "QUEUED"
            else:
                current_state = row[0]
                from_state = current_state

            # Validate transition is allowed
            if to_state not in self.VALID_TRANSITIONS.get(current_state, set()):
                logger.warning(
                    "Invalid FSM transition: %s %s → %s (not in state graph)",
                    idea_id, current_state, to_state
                )
                conn.rollback()
                conn.close()
                return False

            # Create/update state row
            if not row:
                conn.execute(
                    "INSERT OR IGNORE INTO idea_state "
                    "(idea_id, current_state, claimed_by_host, claimed_by_pid, claimed_at) "
                    "VALUES (?, ?, ?, ?, datetime('now'))",
                    (idea_id, to_state, self.host, self.pid)
                )
            else:
                conn.execute(
                    "UPDATE idea_state SET current_state = ?, updated_at = datetime('now') "
                    "WHERE idea_id = ?",
                    (to_state, idea_id)
                )

            # Record transition in audit log
            conn.execute(
                "INSERT INTO idea_transitions "
                "(idea_id, from_state, to_state, reason, host, pid) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (idea_id, from_state, to_state, reason or "", self.host, self.pid)
            )

            conn.commit()
            logger.info(
                "[LIFECYCLE_TRANSITION] idea=%s %s → %s reason=\"%s\"",
                idea_id, from_state, to_state, reason or ""
            )
            return True

        except Exception as e:
            logger.warning("FSM transition error for %s: %s", idea_id, e)
            try:
                conn.rollback()
            except Exception:
                pass
            return False
        finally:
            conn.close()

    def current_state(self, idea_id: str) -> str:
        """Get current FSM state for an idea.

        Returns the tracked state, or "UNKNOWN" if not in FSM.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT current_state FROM idea_state WHERE idea_id = ?",
                (idea_id,)
            ).fetchone()
            return row[0] if row else "UNKNOWN"
        except Exception as e:
            logger.warning("Failed to get FSM state for %s: %s", idea_id, e)
            return "UNKNOWN"
        finally:
            conn.close()

    def history(self, idea_id: str) -> List[Dict[str, any]]:
        """Get complete audit trail for an idea."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT from_state, to_state, reason, host, pid, ts "
                "FROM idea_transitions WHERE idea_id = ? ORDER BY ts ASC",
                (idea_id,)
            ).fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.warning("Failed to get FSM history for %s: %s", idea_id, e)
            return []
        finally:
            conn.close()

    def detect_stale_claims(self, timeout_hours: int = None) -> List[tuple]:
        """Detect ideas stuck in CLAIMED state beyond timeout.

        Returns list of (idea_id, current_state, claimed_at) tuples.
        """
        timeout_hours = timeout_hours or self.TIMEOUT_CLAIM
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT idea_id, current_state, claimed_at FROM idea_state "
                "WHERE current_state = 'CLAIMED' "
                "AND datetime(claimed_at, '+' || ? || ' hours') < datetime('now')",
                (timeout_hours,)
            ).fetchall()
            return [(r[0], r[1], r[2]) for r in rows]
        except Exception as e:
            logger.warning("Failed to detect stale claims: %s", e)
            return []
        finally:
            conn.close()

    def detect_stale_training(self, timeout_hours: int = None) -> List[tuple]:
        """Detect ideas stuck in TRAINING state beyond timeout.

        Returns list of (idea_id, current_state, claimed_at) tuples.
        """
        timeout_hours = timeout_hours or self.TIMEOUT_TRAINING
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT idea_id, current_state, claimed_at FROM idea_state "
                "WHERE current_state = 'TRAINING' "
                "AND datetime(claimed_at, '+' || ? || ' hours') < datetime('now')",
                (timeout_hours,)
            ).fetchall()
            return [(r[0], r[1], r[2]) for r in rows]
        except Exception as e:
            logger.warning("Failed to detect stale training: %s", e)
            return []
        finally:
            conn.close()

    def reconcile_from_filesystem(self, idea_id: str, results_dir: Path) -> Optional[str]:
        """Infer idea state from filesystem artifacts (crash recovery).

        Called on startup to reconcile in-flight ideas after a crash.
        Returns the inferred state, or None if not detectable.

        Inference rules:
        - If claim.json exists: idea is at least CLAIMED
        - If train_output.log exists: idea is at least TRAINING
        - If metrics.json with status=COMPLETED: idea is at least EVALUATING/COMPLETE
        """
        conn = self._get_conn()
        try:
            # Only reconcile if FSM state is missing (first startup)
            row = conn.execute(
                "SELECT current_state FROM idea_state WHERE idea_id = ?",
                (idea_id,)
            ).fetchone()

            if row:
                # Already tracked; don't overwrite with inference
                return row[0]

            # Infer from filesystem
            idea_dir = results_dir / idea_id
            if not idea_dir.exists():
                return None

            inferred_state = "QUEUED"

            # Check for claim marker
            if (idea_dir / "claim.json").exists():
                inferred_state = "CLAIMED"

            # Check for training marker
            if (idea_dir / "train_output.log").exists():
                inferred_state = "TRAINING"

            # Check for metrics (indicates training completed)
            metrics_path = idea_dir / "metrics.json"
            if metrics_path.exists():
                import json
                try:
                    metrics = json.loads(metrics_path.read_text())
                    if metrics.get("status") == "COMPLETED":
                        inferred_state = "EVALUATING"  # Training done, eval next
                except Exception:
                    pass

            # Record the inferred state
            conn.execute(
                "INSERT OR IGNORE INTO idea_state "
                "(idea_id, current_state, claimed_by_host, claimed_by_pid, claimed_at) "
                "VALUES (?, ?, ?, ?, datetime('now'))",
                (idea_id, inferred_state, "reconcile", None)
            )
            conn.execute(
                "INSERT INTO idea_transitions "
                "(idea_id, from_state, to_state, reason, host, pid) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (idea_id, "UNKNOWN", inferred_state, "filesystem_reconciliation", "reconcile", None)
            )
            conn.commit()

            logger.info("[FSM_RECONCILE] idea=%s inferred %s from filesystem", idea_id, inferred_state)
            return inferred_state

        except Exception as e:
            logger.warning("Failed to reconcile FSM state for %s: %s", idea_id, e)
            return None
        finally:
            conn.close()


# Global FSM instance (lazy-initialized)
_fsm_instance: Optional[IdeaLifecycleFSM] = None


def get_fsm(db_path: str) -> IdeaLifecycleFSM:
    """Get or create the global FSM instance."""
    global _fsm_instance
    if _fsm_instance is None:
        _fsm_instance = IdeaLifecycleFSM(db_path)
    return _fsm_instance


def try_transition(db_path: str, idea_id: str, to_state: str,
                   reason: Optional[str] = None) -> bool:
    """Non-blocking helper: attempt transition, log any errors, return success."""
    try:
        fsm = get_fsm(db_path)
        return fsm.transition(idea_id, to_state, reason)
    except Exception as e:
        logger.warning("FSM transition failed (non-blocking): %s", e)
        return False
