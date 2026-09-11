"""Independent real SQLite checks of proposal commit and normal admission fences."""
import json

import pytest

from orze.core import cpu_action_budget as budget
from orze.core import cpu_proposal_requests as store
from orze.engine import cpu_proposals as coordinator
from orze.idea_lake import IdeaLake
from test_cpu_domain_product import request
from test_cpu_proposal_product import proposal


@pytest.fixture
def context(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "lake.db"))
    cfg = {"results_dir": str(results), "idea_lake_db": lake.db_path,
           "execution": {"version": 1, "resource": "cpu", "slots": 1,
                         "wall_budget_seconds": 10},
           "action_domain": {"version": 1, "kind": "command", "config": {}}}
    try:
        yield lake, results, cfg, proposal("idea-proposed", request("pass"))
    finally:
        if lake.conn.in_transaction:
            lake.conn.rollback()
        lake.close()


def _call(context):
    lake, results, cfg, decision = context
    return coordinator.propose(lake, results, cfg, decision, expected_sources=[])


def _count(lake, table):
    if lake.conn.execute("SELECT 1 FROM main.sqlite_master WHERE name=?", (table,)).fetchone() is None:
        return 0
    return lake.conn.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]


def test_normal_proposal_commits_task_state_transition_and_request_together(context):
    lake, results, cfg, decision = context
    result = _call(context)
    assert result["status"] == "inserted"
    assert not lake.conn.in_transaction
    assert [_count(lake, table) for table in (
        "ideas", "idea_state", "idea_transitions", "cpu_proposal_requests",
    )] == [1, 1, 1, 1]
    assert _count(lake, "execution_attempts") == 0
    assert _count(lake, "cpu_action_reservations") == 0
    assert not (results / decision["task_id"]).exists()


def test_request_trigger_cannot_erase_admission_state_then_ack_inserted(context):
    lake, results, cfg, decision = context
    lake.conn.execute("BEGIN IMMEDIATE")
    store.ensure_schema(lake.conn)
    lake.conn.commit()
    lake.conn.executescript("""
        CREATE TRIGGER erase_proposal_state AFTER INSERT ON cpu_proposal_requests
        BEGIN DELETE FROM idea_state WHERE idea_id=NEW.task_id; END;
    """)
    with pytest.raises(coordinator.ProposalHOLD):
        _call(context)
    assert not lake.conn.in_transaction
    assert [_count(lake, table) for table in (
        "ideas", "idea_state", "idea_transitions", "cpu_proposal_requests",
    )] == [0, 0, 0, 0]
    assert not (results / decision["task_id"]).exists()


@pytest.mark.parametrize("mode,durable_count", [("rollback", 0), ("committed_response_lost", 1)])
def test_commit_fault_does_not_return_provisional_outcome(context, mode, durable_count):
    lake, results, cfg, decision = context
    actual = lake.conn
    calls = []

    class CommitFault:
        def __getattr__(self, name):
            return getattr(actual, name)

        def commit(self):
            calls.append(mode)
            if mode == "rollback":
                actual.rollback()
                return None
            actual.commit()
            raise OSError("private fixture lost commit response")

    lake.conn = CommitFault()
    try:
        with pytest.raises(coordinator.ProposalHOLD):
            _call(context)
    finally:
        lake.conn = actual
    assert calls == [mode]
    assert not actual.in_transaction
    assert [_count(lake, table) for table in (
        "ideas", "idea_state", "idea_transitions", "cpu_proposal_requests",
    )] == [durable_count] * 4
    assert _count(lake, "execution_attempts") == 0
    assert _count(lake, "cpu_action_reservations") == 0
    assert not (results / decision["task_id"]).exists()


def test_durable_budget_stop_refuses_proposal_without_refund_or_new_request(context):
    lake, results, cfg, decision = context
    scope = budget.initialize(lake, results, cfg["execution"])
    budget.record_decision(lake, scope, {"kind": "Stop", "reason": "private fixture", "wakeup": None})
    before = list(lake.conn.iterdump())
    with pytest.raises(coordinator.ProposalHOLD):
        _call(context)
    assert list(lake.conn.iterdump()) == before
    assert _count(lake, "cpu_action_reservations") == 0
    assert _count(lake, "cpu_proposal_requests") == 0
    assert not (results / decision["task_id"]).exists()


def test_late_budget_stop_in_request_writer_rolls_back_proposal(context):
    lake, results, cfg, decision = context
    scope = budget.initialize(lake, results, cfg["execution"])
    lake.conn.execute("BEGIN IMMEDIATE")
    store.ensure_schema(lake.conn)
    lake.conn.commit()
    stop = json.dumps({"kind": "Stop", "reason": "private late stop", "wakeup": None})
    lake.conn.execute("CREATE TRIGGER stop_before_ack AFTER INSERT ON cpu_proposal_requests "
        "BEGIN UPDATE cpu_action_scopes SET stop_json=" + "'" + stop + "'" + "; END;")
    lake.conn.commit()
    before = list(lake.conn.iterdump())
    with pytest.raises(coordinator.ProposalHOLD):
        _call(context)
    assert list(lake.conn.iterdump()) == before
    assert _count(lake, "cpu_action_reservations") == 0
    assert not (results / decision["task_id"]).exists()
