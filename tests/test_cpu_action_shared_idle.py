"""Actual CLI policy must respect other owners' real durable reservations.

The peer reservation is created by the public budget API, not a fake ledger.
It intentionally remains RESERVED (possible peer/HOLD admission); no process
or claim is manufactured and the controller lifecycle is not replaced.
"""
import json
from pathlib import Path

from test_cpu_product_loop import project
from orze.core import cpu_action_budget as budget
from orze.idea_lake import IdeaLake


def test_empty_local_controller_waits_for_shared_active_reservation(project):
    root, cfg, run = project
    cfg["action_policy"]["idle"] = "stop"
    results = Path(cfg["results_dir"])
    results.mkdir()
    peer = IdeaLake(cfg["idea_lake_db"])
    try:
        scope = budget.initialize(peer, results, cfg["execution"])
        permit = budget.reserve(peer, scope, "peer-active", 2)
        assert permit is not None
        before = tuple(peer.conn.execute("SELECT * FROM cpu_action_reservations").fetchone())
        assert run() in (None, 0)
        records = [json.loads(row[0]) for row in peer.conn.execute(
            "SELECT record_json FROM cpu_action_decisions")]
        assert len(records) == 1
        assert records[0]["kind"] == "Wait"
        assert records[0]["wakeup"] is not None
        status = budget.snapshot(peer, scope)
        assert not status["stopped"] and status["active_reservations"] == 1
        assert status["reserved_wall_seconds"] == 2
        assert tuple(peer.conn.execute("SELECT * FROM cpu_action_reservations").fetchone()) == before
        assert peer.conn.execute("SELECT name FROM sqlite_master WHERE name='execution_attempts'").fetchall() == []
        assert list(results.glob("*/claim.json")) == []
    finally:
        peer.close()
