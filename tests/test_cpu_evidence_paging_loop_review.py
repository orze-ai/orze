"""Independent phase checks using real CLI/SQLite and the frozen paging fixture.

The reused 33 prefixes are metadata-only transactions, not native workers.
Only the source and the --once action below execute under actual supervision.
"""
import copy
import json
import sqlite3

from test_cpu_product_loop import project
from test_cpu_evidence_paging_product import (
    SOURCE, PREFIX_COUNT, history, _database, _register, _capture, _next,
)
from test_cpu_domain_product import request, submit


def test_peer_sql_write_after_snapshot_cannot_publish_stop(history):
    class Policy:
        def __init__(self, declaration):
            pass

        def decide(self, snapshot, budget):
            _capture(history, snapshot, budget)
            # Real second writer, after the actual pager supplied this view.
            # Stop needs no source/worker authorization, so rejection must not
            # depend on a later source capture or launch discovering the write.
            with sqlite3.connect(history["root"] / "lake.db") as peer:
                changed = peer.execute(
                    "UPDATE main.ideas SET title=title || ' peer-write' WHERE idea_id=?",
                    (SOURCE,),
                ).rowcount
            assert changed == 1
            decision = {"kind": "Stop", "reason": "stale_callback_stop", "wakeup": None}
            history["trace"][-1]["decision"] = copy.deepcopy(decision)
            return decision

    _register(history, Policy)
    code = history["run"](once=False)
    history["outcomes"].append(code)
    assert code == 75
    after = _database(history["root"])
    assert len(history["trace"]) == 1
    assert after["cpu_action_decisions"] == history["before"]["cpu_action_decisions"]
    assert after["cpu_action_reservations"] == history["before"]["cpu_action_reservations"]
    assert after["execution_attempts"] == history["before"]["execution_attempts"]
    assert len(history["processes"]) == 1
    with sqlite3.connect(history["root"] / "lake.db") as conn:
        assert conn.execute("SELECT title FROM ideas WHERE idea_id=?", (SOURCE,)).fetchone()[0].endswith(" peer-write")
        assert conn.execute("SELECT stop_json FROM cpu_action_scopes").fetchone() == (None,)


def test_once_read_chain_reuses_ingress_then_executes_one_real_action(history, monkeypatch):
    from orze.engine import idea_ingress
    task_id = "idea-zzzz-once"
    submit(history["root"], task_id, request(
        "from pathlib import Path; Path('once-output').write_text('actual CPU')",
        outputs={"once": {"path": "once-output", "max_bytes": 128}},
    ))
    real_ingest = idea_ingress.ingest_ideas_source
    calls = []
    engines = []

    def ingest(engine, cfg):
        calls.append(len(history["trace"]))
        engines.append(engine)
        return real_ingest(engine, cfg)

    monkeypatch.setattr(idea_ingress, "ingest_ideas_source", ingest)

    class Policy:
        def __init__(self, declaration):
            pass

        def decide(self, snapshot, budget):
            _capture(history, snapshot, budget)
            seen = any(r["ref"]["task_id"] == SOURCE
                       for r in snapshot["recorded_evidence"]["results"])
            decision = ({"kind": "Execute", "task_id": task_id}
                        if seen else _next(snapshot))
            history["trace"][-1]["decision"] = copy.deepcopy(decision)
            return decision

    _register(history, Policy)
    code = history["run"](once=True)
    history["outcomes"].append(code)
    assert code == 0
    assert calls == [0]
    assert [t["decision"]["kind"] for t in history["trace"]] == ["ReadEvidence"] * 4 + ["Execute"]
    assert [t["snapshot"]["evidence_page"]["page"] for t in history["trace"]] == list(range(1, 6))
    assert len({t["snapshot"]["evidence_page"]["scan_id"] for t in history["trace"]}) == 1
    assert all(any(q["idea_id"] == task_id for q in t["snapshot"]["queue"]) for t in history["trace"])
    after = _database(history["root"])
    assert len(history["processes"]) == 2
    assert len(after["execution_attempts"]) == PREFIX_COUNT + 2
    assert [r["state"] for r in after["cpu_action_reservations"]] == ["SETTLED", "SETTLED"]
    terminal = json.loads(next(r for r in after["execution_attempts"] if r["task_id"] == task_id)["terminal_json"])
    assert terminal["outcome"] == "completed" and terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
    assert after["cpu_action_decisions"] == history["before"]["cpu_action_decisions"]
    assert all(getattr(engines[0], name, None) is None for name in
               ("_cpu_evidence_pager", "_cpu_evidence_queue", "_cpu_evidence_request"))
