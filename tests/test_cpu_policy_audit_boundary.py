"""Real iteration/pagers: one complete audit per read-only continuation.

The 33 prefix attempts are metadata fixtures, never native executions. Real
worker/settlement coverage lives in the unchanged paging product suites.
"""
import copy
import sqlite3

import pytest

from orze.core import cpu_action_budget as budget
from orze.core import research_interfaces as api
from orze.core.cpu_execution import CPUExecutionError
from orze.engine import cpu_phase
from orze.engine.cpu_policy_evidence import EvidencePager
from test_cpu_evidence_paging_product import _metadata_prefix
from test_cpu_product_loop import project


@pytest.fixture
def paging(project, monkeypatch):
    from orze.engine.orchestrator import Orze
    root, cfg, _ = project
    calls = []

    class Policy:
        def __init__(self, declaration):
            pass

        def decide(self, snapshot, allowance):
            calls.append(copy.deepcopy((snapshot, allowance)))
            return {"kind": "ReadEvidence", "cursor": snapshot["evidence_page"]["next_cursor"]}

    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))
    api.register_policy("audit_boundary", "audit.boundary.v1", Policy)
    cfg["action_policy"] = {"version": 2, "kind": "audit_boundary", "idle": "wait",
                            "wait_seconds": .01, "evidence_page_size": 8}
    engine = Orze([], cfg)
    cpu_phase.start(engine)
    _metadata_prefix(engine.lake, engine.results_dir)
    try:
        yield engine, calls
    finally:
        budget._HELD.discard(budget._key(engine._cpu_scope))
        cpu_phase.close(engine)


def test_each_read_only_page_audits_once_without_reusing_previous_budget(paging, monkeypatch):
    engine, calls = paging
    audits = []
    original = budget._totals

    def observe(conn, scope):
        audits.append(scope)
        return original(conn, scope)

    monkeypatch.setattr(budget, "_totals", observe)
    before = tuple(engine.lake.conn.iterdump())
    assert cpu_phase.iteration(engine)
    # Initial ingress can write, so it retains its separate admission audit.
    assert len(audits) == 2
    for page in (2, 3, 4):
        audits.clear()
        assert cpu_phase.iteration(engine)
        assert len(audits) == 1
        assert calls[-1][0]["evidence_page"]["page"] == page
        assert calls[-1][1] == calls[0][1]
    assert tuple(engine.lake.conn.iterdump()) == before


@pytest.mark.parametrize("change", ["peer", "local", "rollback", "schema", "stop",
                                    "held", "signal", "config", "operator"])
def test_change_during_continuation_read_refuses_before_callback(paging, monkeypatch, change):
    engine, calls = paging
    assert cpu_phase.iteration(engine)
    original = EvidencePager.read

    def changed(pager, **kwargs):
        view = original(pager, **kwargs)
        if change == "peer":
            with sqlite3.connect(engine.lake.db_path) as peer:
                peer.execute("UPDATE cpu_action_scopes SET stop_json='broken'")
        elif change in {"local", "rollback"}:
            engine.lake.conn.execute("UPDATE cpu_action_scopes SET stop_json='broken'")
            if change == "rollback":
                engine.lake.conn.rollback()
            else:
                engine.lake.conn.commit()
        elif change == "schema":
            engine.lake.conn.execute("CREATE TABLE changed_schema (id INTEGER)")
        elif change == "stop":
            budget.record_decision(engine.lake, engine._cpu_scope,
                {"kind": "Stop", "reason": "operator", "wakeup": None})
        elif change == "held":
            budget._HELD.add(budget._key(engine._cpu_scope))
        elif change == "signal":
            engine._stop_event.set()
        elif change == "config":
            engine.cfg["execution"]["slots"] = 2
        else:
            (engine.results_dir / ".orze_stop_all").touch()
        return view

    monkeypatch.setattr(EvidencePager, "read", changed)
    with pytest.raises(CPUExecutionError):
        cpu_phase.iteration(engine)
    assert len(calls) == 1
    assert not engine._cpu_handles
    assert engine.lake.conn.execute("SELECT COUNT(*) FROM cpu_action_reservations").fetchone()[0] == 0


def test_corrupt_budget_still_refuses_before_ingress(paging, monkeypatch):
    engine, calls = paging
    from orze.engine import idea_ingress
    from test_cpu_budget_scan import add_row
    add_row(engine.lake.conn, engine._cpu_scope, 1)
    engine.lake.conn.execute("UPDATE cpu_action_reservations SET permit_json='broken'")
    engine.lake.conn.commit()
    monkeypatch.setattr(idea_ingress, "ingest_ideas_source",
                        lambda *args: pytest.fail("ingress ran before complete audit"))
    with pytest.raises(budget.CpuBudgetHOLD):
        cpu_phase.iteration(engine)
    assert calls == []
