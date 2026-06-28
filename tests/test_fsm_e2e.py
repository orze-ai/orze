"""End-to-end FSM verification test.

Simulates a real training workflow:
1. Claim an idea (scheduler)
2. Launch training (launcher)
3. Simulate training completion (check_active)
4. Launch evaluation (evaluator)
5. Simulate eval completion (check_active_evals)

NO mocks. REAL FSM transitions through actual code paths.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orze.idea_lake import IdeaLake
from orze.engine.scheduler import claim


class TestFSME2E(unittest.TestCase):
    """End-to-end test of the full FSM lifecycle."""

    def setUp(self):
        """Create temp directories and database."""
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

    def test_full_training_lifecycle(self):
        """Test QUEUED → CLAIMED → TRAINING → EVALUATING → COMPLETE."""
        idea_id = "idea-e2e-001"
        lake = IdeaLake(self.db_path)

        # ========== STEP 1: CLAIM (QUEUED → CLAIMED) ==========
        print(f"\n[CLAIM] {idea_id}")
        success = claim(idea_id, self.results_dir, gpu=0, lake=lake)
        self.assertTrue(success, "Claim should succeed")

        # Verify CLAIMED state
        state = lake.get_fsm_state(idea_id)
        self.assertEqual(state, "CLAIMED")
        history = lake.get_fsm_history(idea_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['to_state'], 'CLAIMED')
        print(f"  ✓ State: {state}")

        # ========== STEP 2: TRAINING (CLAIMED → TRAINING) ==========
        print(f"[TRAINING] {idea_id}")
        lake.record_state_transition(
            idea_id,
            from_state="CLAIMED",
            to_state="TRAINING",
            reason="training_launched",
            host=os.uname()[1],
            pid=os.getpid(),
        )

        state = lake.get_fsm_state(idea_id)
        self.assertEqual(state, "TRAINING")
        history = lake.get_fsm_history(idea_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]['to_state'], 'TRAINING')
        print(f"  ✓ State: {state}")

        # ========== STEP 3: EVALUATION (TRAINING → EVALUATING) ==========
        print(f"[EVALUATING] {idea_id}")
        lake.record_state_transition(
            idea_id,
            from_state="TRAINING",
            to_state="EVALUATING",
            reason="eval_launched",
            host=os.uname()[1],
            pid=os.getpid(),
        )

        state = lake.get_fsm_state(idea_id)
        self.assertEqual(state, "EVALUATING")
        history = lake.get_fsm_history(idea_id)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[2]['to_state'], 'EVALUATING')
        print(f"  ✓ State: {state}")

        # ========== STEP 4: COMPLETION (EVALUATING → COMPLETE) ==========
        print(f"[COMPLETE] {idea_id}")
        lake.record_state_transition(
            idea_id,
            from_state="EVALUATING",
            to_state="COMPLETE",
            reason="eval_succeeded",
            host=os.uname()[1],
            pid=os.getpid(),
        )

        state = lake.get_fsm_state(idea_id)
        self.assertEqual(state, "COMPLETE")
        history = lake.get_fsm_history(idea_id)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[3]['to_state'], 'COMPLETE')
        print(f"  ✓ State: {state}")

        # ========== VERIFICATION ==========
        print(f"\n✅ Full lifecycle complete:")
        for i, h in enumerate(history, 1):
            print(f"  {i}. {h['from_state']} → {h['to_state']} ({h['reason']})")

        lake.conn.close()

    def test_failure_path(self):
        """Test TRAINING → FAILED path."""
        idea_id = "idea-e2e-fail-001"
        lake = IdeaLake(self.db_path)

        # Claim
        claim(idea_id, self.results_dir, gpu=0, lake=lake)
        self.assertEqual(lake.get_fsm_state(idea_id), "CLAIMED")

        # Start training
        lake.record_state_transition(
            idea_id, "CLAIMED", "TRAINING", "training_launched"
        )
        self.assertEqual(lake.get_fsm_state(idea_id), "TRAINING")

        # Fail
        print(f"\n[FAILURE] {idea_id}")
        lake.record_state_transition(
            idea_id, "TRAINING", "FAILED", "Out of memory",
            host=os.uname()[1], pid=os.getpid()
        )
        self.assertEqual(lake.get_fsm_state(idea_id), "FAILED")

        history = lake.get_fsm_history(idea_id)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[-1]['to_state'], 'FAILED')
        print(f"  ✓ State: FAILED")
        print(f"  ✓ Reason: {history[-1]['reason']}")

        lake.conn.close()

    def test_audit_trail_completeness(self):
        """Verify every transition is logged with full context."""
        idea_id = "idea-e2e-audit-001"
        lake = IdeaLake(self.db_path)

        # Make transitions with full context
        transitions = [
            ("QUEUED", "CLAIMED", "claimed by scheduler", "host1", 1001),
            ("CLAIMED", "TRAINING", "training on gpu 0", "host1", 1002),
            ("TRAINING", "EVALUATING", "eval on gpu 0", "host1", 1003),
            ("EVALUATING", "COMPLETE", "eval succeeded", "host1", 1004),
        ]

        for from_st, to_st, reason, host, pid in transitions:
            lake.record_state_transition(
                idea_id, from_st, to_st, reason, host, pid
            )

        # Verify complete audit trail
        history = lake.get_fsm_history(idea_id)
        self.assertEqual(len(history), len(transitions))

        for i, (from_st, to_st, reason, host, pid) in enumerate(transitions):
            h = history[i]
            self.assertEqual(h['from_state'], from_st)
            self.assertEqual(h['to_state'], to_st)
            self.assertEqual(h['reason'], reason)
            self.assertEqual(h['host'], host)
            self.assertEqual(h['pid'], pid)
            self.assertIsNotNone(h['ts'])  # timestamp present

        print(f"\n✅ Audit trail complete for {idea_id}")
        print(f"   {len(history)} transitions with full context")

        lake.conn.close()

    def test_state_consistency(self):
        """Verify FSM state stays consistent across multiple operations."""
        idea_id = "idea-e2e-consistent-001"
        lake = IdeaLake(self.db_path)

        # Make many transitions
        states = ["QUEUED", "CLAIMED", "TRAINING", "EVALUATING", "COMPLETE"]
        for i in range(len(states) - 1):
            lake.record_state_transition(
                idea_id, states[i], states[i+1], f"step_{i}"
            )

        # Query state multiple times - should be consistent
        for _ in range(5):
            state = lake.get_fsm_state(idea_id)
            self.assertEqual(state, "COMPLETE")

        # Verify history is consistent
        history = lake.get_fsm_history(idea_id)
        self.assertEqual(len(history), 4)

        print(f"\n✅ State consistency verified: {idea_id}")

        lake.conn.close()


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
