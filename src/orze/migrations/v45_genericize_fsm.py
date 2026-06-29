"""Migration v4.5: Genericize FSM (orthogonal to SOP type).

Changes:
  - TRAINING state → IN_PROGRESS
  - EVALUATING state → IN_PROGRESS
  - Add sop_type column to idea_state and idea_transitions
  - Backward compatible: all existing (training) ideas default to sop_type='training'

This migration runs automatically on IdeaLake init.
"""

import logging

logger = logging.getLogger("orze.migrations")


def migrate_v45(conn):
    """Apply v4.5 FSM genericization atomically."""
    cursor = conn.cursor()

    try:
        # Check if sop_type column exists in BOTH tables (comprehensive idempotency check)
        cursor.execute("PRAGMA table_info(idea_state)")
        idea_state_cols = {row[1] for row in cursor.fetchall()}

        cursor.execute("PRAGMA table_info(idea_transitions)")
        idea_transitions_cols = {row[1] for row in cursor.fetchall()}

        if "sop_type" in idea_state_cols and "sop_type" in idea_transitions_cols:
            logger.info("v4.5 migration already applied (both tables have sop_type)")
            return

        logger.info("Applying v4.5 FSM genericization migration...")

        # Ensure explicit transaction (DDL should not autocommit)
        cursor.execute("BEGIN IMMEDIATE")

        # Add missing columns to idea_state (v4.5 uses updated_by_* instead of claimed_*)
        if "updated_by_host" not in idea_state_cols:
            cursor.execute("ALTER TABLE idea_state ADD COLUMN updated_by_host TEXT")
            logger.info("  Added updated_by_host column to idea_state")

        if "updated_by_pid" not in idea_state_cols:
            cursor.execute("ALTER TABLE idea_state ADD COLUMN updated_by_pid INTEGER")
            logger.info("  Added updated_by_pid column to idea_state")

        if "sop_type" not in idea_state_cols:
            cursor.execute("ALTER TABLE idea_state ADD COLUMN sop_type TEXT DEFAULT 'training'")
            logger.info("  Added sop_type column to idea_state")

        # Add sop_type column to idea_transitions (if needed)
        if "sop_type" not in idea_transitions_cols:
            cursor.execute(
                "ALTER TABLE idea_transitions ADD COLUMN sop_type TEXT DEFAULT 'training'"
            )
            logger.info("  Added sop_type column to idea_transitions")

        # Migrate TRAINING → IN_PROGRESS in idea_state
        cursor.execute(
            "UPDATE idea_state SET current_state = 'IN_PROGRESS' WHERE current_state = 'TRAINING'"
        )
        train_count = cursor.rowcount
        if train_count > 0:
            logger.info(f"  Migrated {train_count} ideas from TRAINING → IN_PROGRESS")

        # Migrate EVALUATING → IN_PROGRESS in idea_state
        cursor.execute(
            "UPDATE idea_state SET current_state = 'IN_PROGRESS' WHERE current_state = 'EVALUATING'"
        )
        eval_count = cursor.rowcount
        if eval_count > 0:
            logger.info(f"  Migrated {eval_count} ideas from EVALUATING → IN_PROGRESS")

        # Migrate TRAINING → IN_PROGRESS in idea_transitions
        cursor.execute(
            "UPDATE idea_transitions SET to_state = 'IN_PROGRESS' WHERE to_state = 'TRAINING'"
        )
        trans_train_count = cursor.rowcount
        if trans_train_count > 0:
            logger.info(f"  Migrated {trans_train_count} transitions from TRAINING → IN_PROGRESS")

        # Migrate TRAINING → (anything) to IN_PROGRESS → (anything)
        cursor.execute(
            "UPDATE idea_transitions SET from_state = 'IN_PROGRESS' WHERE from_state = 'TRAINING'"
        )
        trans_from_train_count = cursor.rowcount
        if trans_from_train_count > 0:
            logger.info(
                f"  Migrated {trans_from_train_count} transitions from IN_PROGRESS → *"
            )

        # Migrate EVALUATING → (anything) to IN_PROGRESS → (anything)
        cursor.execute(
            "UPDATE idea_transitions SET from_state = 'IN_PROGRESS' WHERE from_state = 'EVALUATING'"
        )
        trans_from_eval_count = cursor.rowcount
        if trans_from_eval_count > 0:
            logger.info(
                f"  Migrated {trans_from_eval_count} transitions from IN_PROGRESS → *"
            )

        # Migrate EVALUATING → IN_PROGRESS in idea_transitions
        cursor.execute(
            "UPDATE idea_transitions SET to_state = 'IN_PROGRESS' WHERE to_state = 'EVALUATING'"
        )
        trans_eval_count = cursor.rowcount
        if trans_eval_count > 0:
            logger.info(f"  Migrated {trans_eval_count} transitions to IN_PROGRESS")

        # Set all existing ideas to sop_type='training' (backward compat)
        cursor.execute("UPDATE idea_state SET sop_type = 'training' WHERE sop_type IS NULL")
        cursor.execute("UPDATE idea_transitions SET sop_type = 'training' WHERE sop_type IS NULL")

        # Create indexes for efficient SOP-type queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_idea_state_sop_type ON idea_state(sop_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_idea_transitions_sop_type ON idea_transitions(sop_type)"
        )

        conn.commit()
        logger.info("v4.5 FSM genericization migration completed successfully")

    except Exception as e:
        logger.error("v4.5 migration failed: %s", e)
        conn.rollback()
        raise
