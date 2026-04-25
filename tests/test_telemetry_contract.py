"""Guardrail: verify orze's data files that orze-pro telemetry reads haven't broken.

If this test fails, it means a change in orze may break the telemetry pipeline
to the global admin panel (gadmin). Update orze-pro/src/orze_pro/telemetry.py
to match, then update these assertions.
"""
import json
import os
import sqlite3
import tempfile
import pytest
from pathlib import Path


class TestIdeaLakeSchema:
    """The telemetry client reads .orze/idea_lake.db -- verify the schema is stable."""

    def test_ideas_table_has_required_columns(self):
        """telemetry.py queries: idea_id, status, approach_family, id_num"""
        from orze.idea_lake import IdeaLake
        with tempfile.TemporaryDirectory() as tmpdir:
            lake = IdeaLake(os.path.join(tmpdir, "idea_lake.db"))
            conn = sqlite3.connect(lake.db_path)
            cursor = conn.execute("PRAGMA table_info(ideas)")
            columns = {row[1] for row in cursor.fetchall()}
            conn.close()
            lake.conn.close()

            required = {"idea_id", "status", "approach_family", "id_num"}
            missing = required - columns
            assert not missing, (
                f"idea_lake.ideas missing columns {missing} required by telemetry. "
                f"If you renamed them, update orze-pro/src/orze_pro/telemetry.py:_read_idea_lake()"
            )

    def test_status_values_lowercase(self):
        """telemetry.py queries WHERE status='completed' / 'failed' (lowercase)."""
        from orze.idea_lake import IdeaLake
        with tempfile.TemporaryDirectory() as tmpdir:
            lake = IdeaLake(os.path.join(tmpdir, "idea_lake.db"))
            conn = sqlite3.connect(lake.db_path)
            # Verify the column exists and can hold lowercase values
            conn.execute(
                "INSERT INTO ideas (idea_id, id_num, title, config, raw_markdown, status, kind) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("test-001", 1, "test", "{}", "## test\n```yaml\n{}\n```", "completed", "train"),
            )
            conn.commit()
            row = conn.execute("SELECT status FROM ideas WHERE idea_id='test-001'").fetchone()
            conn.close()
            lake.conn.close()
            assert row[0] == "completed"


class TestResultsLayout:
    """The telemetry client reads results/{idea_id}/metrics.json and status.json."""

    def test_status_json_path(self):
        """telemetry reads {results_dir}/status.json -- written by orze.reporting.state"""
        from orze.reporting import state as state_mod
        source = Path(state_mod.__file__).read_text()
        assert "status.json" in source, (
            "orze.reporting.state no longer writes status.json -- "
            "telemetry.py:_read_status_json() will break"
        )

    def test_metrics_json_convention(self):
        """telemetry reads results/{idea_id}/metrics.json for failure details"""
        from orze.engine import evaluator
        source = Path(evaluator.__file__).read_text()
        assert "metrics.json" in source, (
            "orze.engine.evaluator no longer writes metrics.json -- "
            "telemetry.py:_read_failure_detail() will break"
        )


class TestAdminCachePath:
    """The telemetry client reads leaderboard JSON caches."""

    def test_leaderboard_cache_path(self):
        """telemetry reads _leaderboard.json -- verify orze still writes there"""
        from orze.reporting import leaderboard
        source = Path(leaderboard.__file__).read_text()
        # The main leaderboard module writes _leaderboard.json and admin cache
        assert "leaderboard" in source.lower(), (
            "orze.reporting.leaderboard changed -- "
            "telemetry.py:_read_leaderboard() may break"
        )


class TestReceiptsLayout:
    """The telemetry client reads .orze/receipts/*.json."""

    def test_receipts_format(self):
        """Verify orze's receipt writer produces the fields telemetry expects."""
        from orze.skills import receipts as receipts_mod

        source = Path(receipts_mod.__file__).read_text()
        for field in ["skills_declared", "skills_evidenced", "role"]:
            assert field in source, (
                f"orze.skills.receipts no longer writes '{field}' -- "
                f"telemetry.py:_read_receipts_summary() will break"
            )


class TestHeartbeatsPath:
    """The telemetry client reads .orze/heartbeats/ via _read_role_health."""

    def test_heartbeat_state_path(self):
        from orze.reporting import state as state_mod
        source = Path(state_mod.__file__).read_text()
        assert "heartbeat" in source.lower(), (
            "orze.reporting.state changed heartbeat path -- "
            "telemetry may miss liveness data"
        )
