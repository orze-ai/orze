"""Integration tests for FSM determinism hardening.

Tests the FSM against real code paths: scheduler.claim(), launcher.launch(),
evaluator.launch_eval(). Uses real SQLite, real processes (where applicable).

NO mocks. NO fake data. Real state machine, real transitions.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

# Test must import the actual modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orze.idea_lake import IdeaLake
from orze.engine.scheduler import claim


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
        self.assertIn('schema_migrations', tables,
                      "schema_migrations table missing")
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
            ("CLAIMED", "IN_PROGRESS", "experiment launched"),
            ("IN_PROGRESS", "COMPLETE", "evaluation succeeded"),
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

    def test_stale_from_state_is_rejected_without_audit_entry(self):
        lake = IdeaLake(self.db_path)
        idea_id = "idea-stale-writer"

        self.assertTrue(lake.record_state_transition(idea_id, "QUEUED", "CLAIMED"))
        self.assertFalse(
            lake.record_state_transition(idea_id, "QUEUED", "CLAIMED"),
            "a second writer must not overwrite an already-claimed experiment",
        )

        self.assertEqual(lake.get_fsm_state(idea_id), "CLAIMED")
        self.assertEqual(len(lake.get_fsm_history(idea_id)), 1)
        lake.conn.close()

    def test_inserted_queue_row_has_matching_fsm_state(self):
        lake = IdeaLake(self.db_path)
        lake.insert("idea-queued", "queued", "{}", "", status="queued")

        self.assertEqual(lake.get_fsm_state("idea-queued"), "QUEUED")
        self.assertEqual(
            [row["idea_id"] for row in lake.get_queue()], ["idea-queued"])
        lake.conn.close()

    def test_skip_is_atomic_and_idempotent(self):
        lake = IdeaLake(self.db_path)
        lake.insert("idea-skip", "skip", "{}", "", status="queued")

        self.assertTrue(lake.set_status("idea-skip", "skipped"))
        self.assertEqual(lake.get_fsm_state("idea-skip"), "SKIPPED")
        self.assertEqual(lake.get("idea-skip")["status"], "skipped")
        self.assertEqual(len(lake.get_fsm_history("idea-skip")), 1)
        self.assertEqual(lake.get_queue(), [])

        self.assertTrue(lake.set_status("idea-skip", "skipped"))
        self.assertEqual(
            len(lake.get_fsm_history("idea-skip")), 1,
            "repeating a status must not append transition spam",
        )
        lake.conn.close()

    def test_claim_transition_hides_legacy_queue_row(self):
        lake = IdeaLake(self.db_path)
        lake.insert("idea-claim-sync", "claim", "{}", "", status="queued")

        self.assertTrue(lake.record_state_transition(
            "idea-claim-sync", "QUEUED", "CLAIMED"))
        self.assertEqual(lake.get("idea-claim-sync")["status"], "running")
        self.assertEqual(lake.get_queue(), [])
        lake.conn.close()

    def test_requeue_backfills_missing_fsm_row_without_fake_transition(self):
        lake = IdeaLake(self.db_path)
        lake.insert("idea-legacy-queue", "queue", "{}", "", status="queued")
        lake.conn.execute(
            "DELETE FROM idea_state WHERE idea_id='idea-legacy-queue'")
        lake.conn.commit()

        self.assertTrue(lake.set_status("idea-legacy-queue", "queued"))
        self.assertEqual(lake.get_fsm_state("idea-legacy-queue"), "QUEUED")
        self.assertEqual(lake.get_fsm_history("idea-legacy-queue"), [])
        lake.conn.close()

    def test_startup_reconciles_legacy_terminal_writes(self):
        lake = IdeaLake(self.db_path)
        lake.insert("idea-old-skip", "skip", "{}", "", status="queued")
        lake.insert("idea-old-complete", "complete", "{}", "", status="queued")
        lake.conn.execute(
            "UPDATE ideas SET status='skipped' WHERE idea_id='idea-old-skip'")
        lake.conn.execute(
            "UPDATE ideas SET status='completed' "
            "WHERE idea_id='idea-old-complete'")
        # Simulate a database produced before lifecycle_columns_v1 existed.
        lake.conn.execute(
            "DELETE FROM schema_migrations "
            "WHERE name='lifecycle_columns_v1'")
        lake.conn.commit()
        lake.conn.close()

        lake = IdeaLake(self.db_path)
        self.assertEqual(lake.get_fsm_state("idea-old-skip"), "SKIPPED")
        self.assertEqual(lake.get_fsm_state("idea-old-complete"), "COMPLETE")
        self.assertEqual(lake.get_queue(), [])
        self.assertEqual(lake.get_fsm_history("idea-old-skip"), [])
        self.assertIsNotNone(lake.conn.execute(
            "SELECT 1 FROM schema_migrations "
            "WHERE name='lifecycle_columns_v1'").fetchone())
        lake.conn.close()


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

    def test_rejected_fsm_claim_removes_filesystem_claim(self):
        lake = IdeaLake(self.db_path)
        idea_id = "idea-stale-queue-claim"
        lake.insert(idea_id, "test", "{}", "", status="queued")
        self.assertTrue(lake.record_state_transition(
            idea_id, "QUEUED", "CLAIMED"))

        self.assertFalse(claim(idea_id, self.results_dir, gpu=0, lake=lake))
        self.assertFalse((self.results_dir / idea_id / "claim.json").exists())
        self.assertEqual(lake.get_fsm_state(idea_id), "CLAIMED")
        lake.conn.close()


if __name__ == '__main__':
    unittest.main()
