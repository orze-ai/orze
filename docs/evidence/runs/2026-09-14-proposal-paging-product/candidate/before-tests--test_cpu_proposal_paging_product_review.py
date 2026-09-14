"""Real proposal receipts drive CLI decisions beyond the first 32 entries.

The fixture creates 35 actual coordinator admissions/replays/conflicts and one
queued task. These are NOT 35 workers, native attempts, or TREE proofs.
The positive case alone launches that task under actual CPU supervision.
"""
import copy
import json
import sqlite3

import pytest

from test_cpu_product_loop import project
from test_cpu_domain_product import request
from test_cpu_evidence_paging_product import _database, _metadata_prefix

TASK = "idea-proposal-page-tail"
EXACT = "zz-tail-exact"
CONFLICT = "zz-tail-conflict"


@pytest.fixture
def proposals(project, monkeypatch):
    from orze.core import research_interfaces as api
    from orze.engine import cpu_proposals, native_cpu_action as native
    from orze.idea_lake import IdeaLake

    root, cfg, run = project
    results = root / "results"
    results.mkdir()
    cfg["action_domain"] = {"version": 1, "kind": "command", "config": {}}
    declared = request("from pathlib import Path; Path('answer').write_text('proposal tail observed')",
                       outputs={"answer": {"path": "answer", "max_bytes": 128}})
    base = {"kind": "Propose", "task_id": TASK, "reason": "one queued task; historical request review",
            "domain_request": declared}
    admitted = []
    lake = IdeaLake(root / "lake.db")
    try:
        for i in range(33):
            decision = {**base, "request_id": "request-" + str(i).zfill(2)}
            admitted.append(cpu_proposals.propose(lake, results, cfg, decision, expected_sources=()))
        admitted.append(cpu_proposals.propose(lake, results, cfg,
                        {**base, "request_id": EXACT}, expected_sources=()))
        changed = copy.deepcopy(declared)
        changed["payload"]["command"][-1] = "raise AssertionError('conflicting proposal must never run')"
        admitted.append(cpu_proposals.propose(lake, results, cfg,
                        {**base, "request_id": CONFLICT, "domain_request": changed}, expected_sources=()))
    finally:
        lake.close()
    assert len(admitted) == 35 and admitted[0]["status"] == "inserted"
    assert all(x["status"] == "already_present_exact" for x in admitted[1:34])
    assert admitted[-1]["status"] == "conflict"
    before = _database(root)
    assert len(before["cpu_proposal_requests"]) == 35 and len(before["ideas"]) == 1
    assert before["execution_attempts"] == [] and before["cpu_action_reservations"] == []
    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))
    cfg["action_policy"] = {"version": 2, "kind": "proposal_paging_review",
                            "idle": "wait", "wait_seconds": .01,
                            "evidence_page_size": 8, "proposal_page_size": 8, "config": {}}
    processes = []
    actual_prepare = native.prepare_supervised

    def capture(*args, **kwargs):
        process = actual_prepare(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(native, "prepare_supervised", capture)
    state = {"root": root, "results": results, "cfg": cfg, "run": run, "admissions": admitted,
             "before": before, "trace": [], "processes": processes, "outcomes": [],
             "metadata_evidence_prefixes": 0}
    try:
        yield state
    finally:
        receipts = []
        for process in processes:
            if process.poll() is None:
                process.stop(timeout=.5)
            receipts.append(process.closure_receipt())
        report = {key: state[key] for key in ("admissions", "before", "trace", "outcomes",
                                               "metadata_evidence_prefixes")}
        report.update(after=_database(root), actual_native_workers=len(processes),
                      actual_closure_receipts=receipts,
                      actual_cli_invocation_in_same_pytest_interpreter=True,
                      no_proposal_receipt_is_claimed_as_worker_or_execution_authority=True)
        path = root / "proposal-paging-review.json"
        path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        print("CPU_PROPOSAL_PAGING_REPORT=" + str(path))
        assert all(r["event"] == "TREE_CLOSED" and r["wait_proof"] == "ECHILD_WALL" for r in receipts)


def register(implementation):
    from orze.core import research_interfaces as api
    api.register_policy("proposal_paging_review", "proposal.paging.review.v1", implementation)


def capture(state, snapshot, budget):
    state["trace"].append({"snapshot": copy.deepcopy(snapshot), "budget": copy.deepcopy(budget),
                            "database_before_callback": _database(state["root"])})
    assert len(state["trace"]) < 20, "bounded test must not loop indefinitely"


def advance(snapshot):
    page = snapshot.get("proposal_page")
    if page is None:
        # Config/Bound v2 capability is an explicit baseline prerequisite.
        # The old phase terminates normally instead of unknown-API failure;
        # the positive business assertions expose its first-window blind spot.
        return {"kind": "Stop", "reason": "proposal_window_blind_spot", "wakeup": None}
    assert page["next_cursor"] is not None, "real tail receipt was not enumerated"
    return {"kind": "ReadProposals", "cursor": page["next_cursor"]}


def test_tail_exact_and_conflict_page_selects_then_once_executes_real_task(proposals, monkeypatch):
    from orze.engine import idea_ingress
    real_ingest = idea_ingress.ingest_ideas_source
    ingress_calls = []

    def ingest(*args, **kwargs):
        ingress_calls.append(len(proposals["trace"]))
        return real_ingest(*args, **kwargs)

    monkeypatch.setattr(idea_ingress, "ingest_ideas_source", ingest)

    class Policy:
        def __init__(self, declaration):
            self.selected = False

        def decide(self, snapshot, budget):
            capture(proposals, snapshot, budget)
            found = {r["request_id"]: r for r in snapshot["recorded_proposals"]["results"]}
            if not self.selected and EXACT in found and CONFLICT in found:
                self.selected = True
                decision = {"kind": "SelectProposals", "request_ids": [EXACT, CONFLICT]}
            elif self.selected:
                assert snapshot["proposal_page"]["mode"] == "selection"
                assert set(found) == {EXACT, CONFLICT}
                assert found[EXACT]["status"] == "already_present_exact"
                assert found[CONFLICT]["status"] == "conflict"
                decision = {"kind": "Execute", "task_id": TASK}
            else:
                decision = advance(snapshot)
            proposals["trace"][-1]["decision"] = copy.deepcopy(decision)
            return decision

    register(Policy)
    code = proposals["run"](once=True)
    proposals["outcomes"].append(code)
    assert code == 0
    after = _database(proposals["root"])
    assert len(proposals["processes"]) == 1
    assert len(after["execution_attempts"]) == 1
    assert [r["state"] for r in after["cpu_action_reservations"]] == ["SETTLED"]
    terminal = json.loads(after["execution_attempts"][0]["terminal_json"])
    assert terminal["outcome"] == "completed" and terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
    artifact = json.loads(after["research_artifacts"][0]["record_json"])
    from pathlib import Path
    assert Path(artifact["path"]).read_text() == "proposal tail observed"
    assert after["cpu_proposal_requests"] == proposals["before"]["cpu_proposal_requests"]
    assert after["cpu_action_decisions"] == []
    assert ingress_calls == [0]
    assert [t["decision"]["kind"] for t in proposals["trace"]] == ["ReadProposals"] * 4 + ["SelectProposals", "Execute"]
    for event in proposals["trace"]:
        assert event["database_before_callback"]["cpu_action_reservations"] == []
        assert event["database_before_callback"]["execution_attempts"] == []
        assert event["database_before_callback"]["cpu_action_decisions"] == []


def test_tail_conflict_drives_stop_without_retry_debit_or_worker(proposals):
    class Policy:
        def __init__(self, declaration):
            pass

        def decide(self, snapshot, budget):
            capture(proposals, snapshot, budget)
            found = {r["request_id"]: r for r in snapshot["recorded_proposals"]["results"]}
            decision = ({"kind": "Stop", "reason": "known_tail_conflict_no_retry", "wakeup": None}
                        if CONFLICT in found and found[CONFLICT]["status"] == "conflict"
                        else advance(snapshot))
            proposals["trace"][-1]["decision"] = copy.deepcopy(decision)
            return decision

    register(Policy)
    code = proposals["run"](once=False)
    proposals["outcomes"].append(code)
    assert code == 0
    after = _database(proposals["root"])
    assert proposals["trace"][-1]["decision"]["reason"] == "known_tail_conflict_no_retry"
    assert after["cpu_proposal_requests"] == proposals["before"]["cpu_proposal_requests"]
    assert after["execution_attempts"] == [] and after["cpu_action_reservations"] == []
    assert proposals["processes"] == []
    records = [json.loads(row["record_json"]) for row in after["cpu_action_decisions"]]
    assert len(records) == 1 and records[0]["kind"] == "Stop"
    assert records[0]["reason"] == "known_tail_conflict_no_retry"
    with sqlite3.connect(proposals["root"] / "lake.db") as conn:
        assert json.loads(conn.execute("SELECT stop_json FROM cpu_action_scopes").fetchone()[0])["reason"] == "known_tail_conflict_no_retry"


def test_real_proposal_cursor_is_rejected_on_evidence_channel(proposals):
    from orze.idea_lake import IdeaLake
    lake = IdeaLake(proposals["root"] / "lake.db")
    try:
        refs = _metadata_prefix(lake, proposals["results"])
    finally:
        lake.close()
    proposals["metadata_evidence_prefixes"] = len(refs)
    before = _database(proposals["root"])

    class Policy:
        def __init__(self, declaration):
            pass

        def decide(self, snapshot, budget):
            capture(proposals, snapshot, budget)
            proposal_cursor = snapshot["proposal_page"]["next_cursor"]
            evidence_cursor = snapshot["evidence_page"]["next_cursor"]
            assert isinstance(proposal_cursor, str) and isinstance(evidence_cursor, str)
            assert proposal_cursor != evidence_cursor
            return {"kind": "ReadEvidence", "cursor": proposal_cursor}

    register(Policy)
    code = proposals["run"](once=True)
    proposals["outcomes"].append(code)
    assert code == 75
    after = _database(proposals["root"])
    assert after["execution_attempts"] == before["execution_attempts"]
    assert after["cpu_proposal_requests"] == before["cpu_proposal_requests"]
    assert after["cpu_action_reservations"] == [] and after["cpu_action_decisions"] == []
    assert proposals["processes"] == []


def test_peer_write_before_read_proposals_refuses_stale_continuation(proposals):
    class Policy:
        def __init__(self, declaration):
            pass

        def decide(self, snapshot, budget):
            capture(proposals, snapshot, budget)
            cursor = snapshot["proposal_page"]["next_cursor"]
            assert isinstance(cursor, str)
            with sqlite3.connect(proposals["root"] / "lake.db") as peer:
                assert peer.execute("UPDATE ideas SET title=title || ' peer' WHERE idea_id=?", (TASK,)).rowcount == 1
            return {"kind": "ReadProposals", "cursor": cursor}

    register(Policy)
    code = proposals["run"](once=True)
    proposals["outcomes"].append(code)
    assert code == 75
    after = _database(proposals["root"])
    assert len(proposals["trace"]) == 1
    assert after["cpu_proposal_requests"] == proposals["before"]["cpu_proposal_requests"]
    assert after["execution_attempts"] == [] and after["cpu_action_reservations"] == []
    assert after["cpu_action_decisions"] == [] and proposals["processes"] == []
