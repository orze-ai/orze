"""Frozen draft regressions for contradictory or non-text lifecycle facts.

These exercise the real demand reader against deliberately persisted bad
rows; they are not missing-API failures. Ordinary terminal compatibility is
an explicit positive control. The original 22 mechanism cases stay frozen.
"""
from pathlib import Path
import sqlite3

import pytest

from orze.core.research_demand import load_persistent_demand
from test_persistent_research_demand import lake, _queued, _running, _training_done


@pytest.mark.parametrize("terminal", ["FAILED", "SKIPPED", "ARCHIVED"])
def test_terminal_with_a_recorded_active_stage_does_not_manufacture_empty_demand(lake, terminal):
    name = _running(lake)
    lake.conn.execute("UPDATE ideas SET status=? WHERE idea_id=?", (terminal.lower(), name))
    lake.conn.execute("UPDATE idea_state SET current_state=? WHERE idea_id=?", (terminal, name))
    lake.conn.commit()
    assert lake.get_stage_state(name, "training") == "IN_PROGRESS"
    before = Path(lake.db_path).read_bytes()

    snapshot = load_persistent_demand(lake.db_path)

    assert snapshot.available and not snapshot.complete
    assert snapshot.counts.unknown == 1 and snapshot.counts.inactive == 0
    assert snapshot.queue_count is None and snapshot.waiting_count is None
    assert Path(lake.db_path).read_bytes() == before


def test_real_terminal_transitions_and_skipped_pending_stage_remain_inactive(lake):
    failed = _running(lake, "idea-failed")
    assert lake.record_state_transition(failed, "IN_PROGRESS", "FAILED", "failure")
    for terminal in ("SKIPPED", "ARCHIVED"):
        name = _running(lake, "idea-" + terminal.lower())
        assert lake.record_state_transition(name, "IN_PROGRESS", "QUEUED", "readmit")
        assert lake.record_state_transition(name, "QUEUED", "SKIPPED", "operator skip")
        if terminal == "ARCHIVED":
            assert lake.record_state_transition(name, "SKIPPED", "ARCHIVED", "archive")
        assert lake.get_stage_state(name, "training") == "PENDING"

    snapshot = load_persistent_demand(lake.db_path)

    assert snapshot.available and snapshot.complete
    assert snapshot.counts.inactive == snapshot.counts.total == 3
    assert snapshot.queue_count == snapshot.waiting_count == 0


@pytest.mark.parametrize("kind", ["queued", "evaluation_running"])
def test_blob_mirror_cannot_be_coerced_into_a_known_text_status(lake, kind):
    name = _queued(lake) if kind == "queued" else _training_done(lake)
    value = b"queued" if kind == "queued" else b"running"
    lake.conn.execute("UPDATE ideas SET status=? WHERE idea_id=?", (sqlite3.Binary(value), name))
    lake.conn.commit()
    assert lake.conn.execute("SELECT typeof(status) FROM ideas WHERE idea_id=?", (name,)).fetchone()[0] == "blob"
    before = Path(lake.db_path).read_bytes()

    snapshot = load_persistent_demand(lake.db_path)

    assert snapshot.available and not snapshot.complete
    assert snapshot.counts.unknown == 1
    assert snapshot.counts.queued == snapshot.counts.evaluation_pending == 0
    assert snapshot.queue_count is None and snapshot.waiting_count is None
    assert Path(lake.db_path).read_bytes() == before
