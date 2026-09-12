"""Actual existing foreground CLI consumes the new default/short lease."""
import json
import sqlite3

import pytest

from test_cpu_product_loop import project, task


@pytest.mark.parametrize("expires", [False, True])
def test_foreground_cli_persists_default_or_short_lease_and_settles(project, expires):
    root, cfg, run = project
    if expires:
        cfg["cpu_runtime_lease"] = {"version": 1, "ttl_seconds": .75}
    task(root, outputs={}, program=(
        "import signal,time; signal.signal(signal.SIGTERM, lambda *_: exit(0)); time.sleep(10)"
        if expires else "pass"))
    assert run() in (None, 0)
    with sqlite3.connect(root / "lake.db") as conn:
        rows = conn.execute("SELECT state,binding_json,terminal_json FROM execution_attempts").fetchall()
        reservations = conn.execute("SELECT state FROM cpu_action_reservations").fetchall()
    assert len(rows) == 1 and rows[0][0] == "TERMINAL"
    bound, terminal = json.loads(rows[0][1]), json.loads(rows[0][2])
    descriptor = bound["runtime_lease"]
    assert bound["process_supervision_protocol"] == "orze.linux_subreaper.v2"
    assert bound["supervision"]["runtime_lease"] == descriptor
    assert descriptor["deadline_ns"] - descriptor["issued_ns"] == (750_000_000 if expires else 2_000_000_000)
    assert terminal["runtime_lease"]["status"] == ("expired" if expires else "authorized")
    assert terminal["outcome"] == ("interrupted" if expires else "completed")
    assert terminal["reason_code"] == ("cpu_runtime_lease_expired" if expires else "cpu_action_completed")
    assert terminal["return_code"] == 0
    assert terminal["artifact_ids"] == terminal["observation_ids"] == []
    assert terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
    assert reservations == [("SETTLED",)]


def test_lease_exceeding_selected_action_is_rejected_before_reserve_or_claim(project):
    root, cfg, run = project
    cfg["cpu_runtime_lease"] = {"version": 1, "ttl_seconds": 3}
    task(root, outputs={}, program="raise AssertionError('must not execute')")
    # The foreground entry reports execution errors; inspect real durable
    # poststate independently of its human-facing return-code convention.
    try:
        run()
    except (SystemExit, ValueError) as exc:
        assert isinstance(exc, SystemExit) or "exceeds action" in str(exc)
    with sqlite3.connect(root / "lake.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM cpu_action_reservations").fetchone()[0] == 0
        assert conn.execute("SELECT status FROM ideas").fetchone()[0] == "queued"
    assert not (root / "results" / "idea-0001" / "claim.json").exists()
