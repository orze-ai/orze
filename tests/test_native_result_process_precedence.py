"""Native proposal sidecars cannot override process or delivery uncertainty."""
import sqlite3
import time

import pytest

from orze.engine import roles
from test_native_role_result_boundaries import project, trigger_project, _publish
from test_trigger_role_completion import _state


@pytest.mark.parametrize("case,expected", [
    ("exit-error", roles.OUTCOME_ERROR),
    ("quota", roles.OUTCOME_RATE_LIMITED),
    ("timeout", roles.OUTCOME_TIMEOUT),
    ("cleanup-unconfirmed", roles.OUTCOME_ERROR),
    ("terminal-unconfirmed", roles.OUTCOME_ERROR),
])
def test_valid_result_cannot_override_actual_completion_authority(project, monkeypatch, case, expected):
    p = project
    _publish(p)
    result_bytes = p.result_path.read_bytes()
    if case == "exit-error":
        p.rp.process.poll = lambda: 1
    elif case == "quota":
        p.rp.process.poll = lambda: 42
    elif case == "timeout":
        p.rp.process.poll = lambda: None
        p.rp.start_time = time.time() - 120
    elif case == "cleanup-unconfirmed":
        monkeypatch.setattr(roles, "_terminate_and_reap", lambda *args, **kwargs: False)
    else:
        with sqlite3.connect(p.lake.db_path) as connection:
            connection.execute(
                "CREATE TRIGGER refuse_terminal BEFORE UPDATE ON trigger_deliveries "
                "WHEN NEW.state='TERMINAL' BEGIN SELECT RAISE(IGNORE); END"
            )
    assert roles.check_active_roles(p.active, ideas_file=str(p.ideas)) == [("worker", expected)]
    assert p.rp.native_result_details["status"] == "error"
    assert p.rp.native_result_details["accepted_count"] == 0
    assert p.result_path.read_bytes() == result_bytes, "keep original evidence, do not rewrite it to match the decision"
    assert not p.active
    if case in ("cleanup-unconfirmed", "terminal-unconfirmed"):
        assert _state(p) in ("STARTED", "IN_DOUBT")
        assert p.receipt.exists()
    else:
        assert _state(p) == "TERMINAL"
        row = p.lake.conn.execute(
            "SELECT outcome FROM trigger_delivery_transitions "
            "WHERE delivery_id=? AND to_state='TERMINAL'", (p.launch["delivery_id"],)
        ).fetchone()
        assert row is not None and row[0] == expected.name.lower()
