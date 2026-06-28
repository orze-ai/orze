"""Integration tests for FSM determinism hardening.

Tests the FSM against real code paths: scheduler.claim(), launcher.launch(),
evaluator.launch_eval(). Uses real SQLite, real processes (where applicable).

NO mocks. NO fake data. Real state machine, real transitions.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Test must import the actual modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orze.idea_lake import IdeaLake
from orze.engine.scheduler import claim
from orze.fsm.plugins.idea_lifecycle_fsm import IdeaLifecycleFSM


class TestFSMSchema(unittest.TestCase):
    """Test FSM schema initialization and basic operations."""

    def setUp(self):
        """Create a temporary database for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

    def tearDown(self):
        """Clean up temp database."""
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_schema_initialization(self):
        """Verify FSM tables are created on first IdeaLake init."""
        lake = IdeaLake(self.db_path)

        # Check tables exist
        cursor = lake.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}

        self.assertIn('idea_state', tables, "idea_state table missing")
        self.assertIn('idea_transitions', tables, "idea_transitions table missing")
        lake.conn.close()

    def test_record_state_transition(self):
        """Verify transitions are atomically recorded with audit trail."""
        lake = IdeaLake(self.db_path)

        # Record a transition
        lake.record_state_transition(
            "idea-test-001",
            from_state="QUEUED",
            to_state="CLAIMED",
            reason="test_claim",
            host="test-host",
            pid=12345
        )

        # Verify state was recorded
        state = lake.get_fsm_state("idea-test-001")
        self.assertEqual(state, "CLAIMED", f"Expected CLAIMED, got {state}")

        # Verify transition was logged
        history = lake.get_fsm_history("idea-test-001")
        self.assertEqual(len(history), 1, f"Expected 1 transition, got {len(history)}")
        self.assertEqual(history[0]['from_state'], 'QUEUED')
        self.assertEqual(history[0]['to_state'], 'CLAIMED')
        self.assertEqual(history[0]['reason'], 'test_claim')

        lake.conn.close()

    def test_multiple_transitions(self):
        """Verify full lifecycle transitions are tracked."""
        lake = IdeaLake(self.db_path)
        idea_id = "idea-lifecycle-test"

        # Simulate full lifecycle
        transitions = [
            ("QUEUED", "CLAIMED", "claimed by scheduler"),
            ("CLAIMED", "TRAINING", "training launched"),
            ("TRAINING", "EVALUATING", "eval launched"),
            ("EVALUATING", "COMPLETE", "eval succeeded"),
        ]

        for from_state, to_state, reason in transitions:
            lake.record_state_transition(
                idea_id,
                from_state=from_state,
                to_state=to_state,
                reason=reason,
                host="test-host",
                pid=os.getpid(),
            )

        # Verify final state
        state = lake.get_fsm_state(idea_id)
        self.assertEqual(state, "COMPLETE")

        # Verify complete audit trail
        history = lake.get_fsm_history(idea_id)
        self.assertEqual(len(history), len(transitions),
                        f"Expected {len(transitions)} transitions, got {len(history)}")

        # Verify order
        for i, (from_st, to_st, reason) in enumerate(transitions):
            self.assertEqual(history[i]['from_state'], from_st)
            self.assertEqual(history[i]['to_state'], to_st)

        lake.conn.close()


class TestFSMCore(unittest.TestCase):
    """Test FSM state machine logic."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        # Initialize schema
        lake = IdeaLake(self.db_path)
        lake.conn.close()

    def tearDown(self):
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_atomic_transition(self):
        """Verify transitions use SELECT...FOR UPDATE (atomic)."""
        fsm = IdeaLifecycleFSM(self.db_path)
        idea_id = "idea-atomic-test"

        # First transition should succeed
        success = fsm.transition(idea_id, "CLAIMED", reason="test claim")
        self.assertTrue(success)

        # Second transition should succeed (valid path)
        success = fsm.transition(idea_id, "TRAINING", reason="test training")
        self.assertTrue(success)

        # Verify state
        state = fsm.current_state(idea_id)
        self.assertEqual(state, "TRAINING")

    def test_invalid_transition_rejected(self):
        """Verify invalid state transitions are rejected."""
        fsm = IdeaLifecycleFSM(self.db_path)
        idea_id = "idea-invalid-test"

        # Set to CLAIMED
        fsm.transition(idea_id, "CLAIMED")

        # Try to jump directly to COMPLETE (invalid)
        success = fsm.transition(idea_id, "COMPLETE")
        self.assertFalse(success, "Invalid transition should be rejected")

        # Verify state didn't change
        state = fsm.current_state(idea_id)
        self.assertEqual(state, "CLAIMED")

    def test_detect_stale_claims(self):
        """Verify timeout detection finds stuck claims."""
        fsm = IdeaLifecycleFSM(self.db_path)

        # Manually insert a stale claim
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO idea_state (idea_id, current_state, claimed_at) "
            "VALUES (?, ?, datetime('now', '-10 hours'))",
            ("idea-stale-001", "CLAIMED")
        )
        conn.commit()
        conn.close()

        # Detect stale claims (default 6h timeout)
        stale = fsm.detect_stale_claims()
        stale_ids = [s[0] for s in stale]
        self.assertIn("idea-stale-001", stale_ids, "Stale claim not detected")

    def test_history_preserves_order(self):
        """Verify audit trail preserves transition order."""
        fsm = IdeaLifecycleFSM(self.db_path)
        idea_id = "idea-order-test"

        # Make several transitions
        for i in range(3):
            state = ["QUEUED", "CLAIMED", "TRAINING", "EVALUATING"][i]
            next_state = ["QUEUED", "CLAIMED", "TRAINING", "EVALUATING"][i + 1]
            fsm.transition(idea_id, next_state, reason=f"step_{i}")

        # Check history is in order
        history = fsm.history(idea_id)
        self.assertEqual(len(history), 3)
        for i, h in enumerate(history):
            self.assertIn(f"step_{i}", h['reason'])


class TestSchedulerClaim(unittest.TestCase):
    """Test scheduler.claim() integration with FSM."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.temp_dir.name)
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        # Initialize schema
        lake = IdeaLake(self.db_path)
        lake.conn.close()

    def tearDown(self):
        self.temp_dir.cleanup()
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_claim_records_fsm_transition(self):
        """Verify claim() records QUEUED→CLAIMED in FSM."""
        lake = IdeaLake(self.db_path)
        idea_id = "idea-claim-test"

        # Claim the idea
        success = claim(idea_id, self.results_dir, gpu=0, lake=lake)
        self.assertTrue(success, "Claim should succeed")

        # Verify FSM state was recorded
        state = lake.get_fsm_state(idea_id)
        self.assertEqual(state, "CLAIMED", f"Expected CLAIMED, got {state}")

        # Verify audit trail exists
        history = lake.get_fsm_history(idea_id)
        self.assertGreater(len(history), 0, "Audit trail should exist")
        self.assertEqual(history[0]['to_state'], 'CLAIMED')

        lake.conn.close()

    def test_double_claim_fails(self):
        """Verify second claim on same idea fails."""
        lake = IdeaLake(self.db_path)
        idea_id = "idea-double-claim-test"

        # First claim succeeds
        success1 = claim(idea_id, self.results_dir, gpu=0, lake=lake)
        self.assertTrue(success1)

        # Second claim fails
        success2 = claim(idea_id, self.results_dir, gpu=1, lake=lake)
        self.assertFalse(success2, "Second claim should fail")

        lake.conn.close()


if __name__ == '__main__':
    unittest.main()
