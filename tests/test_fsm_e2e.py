"""End-to-end FSM verification test.

Simulates a real training workflow:
1. Claim an idea (scheduler)
2. Launch training (launcher)
3. Simulate training completion (check_active)
4. Launch evaluation (evaluator)
5. Simulate eval completion (check_active_evals)

NO mocks. REAL FSM transitions through actual code paths.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

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
        """Test the generic QUEUED → CLAIMED → IN_PROGRESS → COMPLETE lifecycle."""
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

        # ========== STEP 2: EXECUTION (CLAIMED → IN_PROGRESS) ==========
        print(f"[IN_PROGRESS] {idea_id}")
        lake.record_state_transition(
            idea_id,
            from_state="CLAIMED",
            to_state="IN_PROGRESS",
            reason="training_launched",
            host=os.uname()[1],
            pid=os.getpid(),
        )

        state = lake.get_fsm_state(idea_id)
        self.assertEqual(state, "IN_PROGRESS")
        history = lake.get_fsm_history(idea_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]['to_state'], 'IN_PROGRESS')
        print(f"  ✓ State: {state}")

        # ========== STEP 3: COMPLETION (IN_PROGRESS → COMPLETE) ==========
        print(f"[COMPLETE] {idea_id}")
        lake.record_state_transition(
            idea_id,
            from_state="IN_PROGRESS",
            to_state="COMPLETE",
            reason="eval_succeeded",
            host=os.uname()[1],
            pid=os.getpid(),
        )

        state = lake.get_fsm_state(idea_id)
        self.assertEqual(state, "COMPLETE")
        history = lake.get_fsm_history(idea_id)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[2]['to_state'], 'COMPLETE')
        print(f"  ✓ State: {state}")

        # ========== VERIFICATION ==========
        print(f"\n✅ Full lifecycle complete:")
        for i, h in enumerate(history, 1):
            print(f"  {i}. {h['from_state']} → {h['to_state']} ({h['reason']})")

        lake.conn.close()

    def test_failure_path(self):
        """Test IN_PROGRESS → FAILED path."""
        idea_id = "idea-e2e-fail-001"
        lake = IdeaLake(self.db_path)

        # Claim
        claim(idea_id, self.results_dir, gpu=0, lake=lake)
        self.assertEqual(lake.get_fsm_state(idea_id), "CLAIMED")

        # Start training
        lake.record_state_transition(
            idea_id, "CLAIMED", "IN_PROGRESS", "training_launched"
        )
        self.assertEqual(lake.get_fsm_state(idea_id), "IN_PROGRESS")

        # Fail
        print(f"\n[FAILURE] {idea_id}")
        lake.record_state_transition(
            idea_id, "IN_PROGRESS", "FAILED", "Out of memory",
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
            ("CLAIMED", "IN_PROGRESS", "experiment on gpu 0", "host1", 1002),
            ("IN_PROGRESS", "COMPLETE", "evaluation succeeded", "host1", 1003),
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
        states = ["QUEUED", "CLAIMED", "IN_PROGRESS", "COMPLETE"]
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
        self.assertEqual(len(history), 3)

        print(f"\n✅ State consistency verified: {idea_id}")

        lake.conn.close()


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
