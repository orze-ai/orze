"""Actual CLI paging -> selected source -> analysis/replica, not reader-only tests.

Thirty-three prefix entries use real SQLite/catalog/effect transactions but
are explicitly METADATA fixtures: no worker, claim, budget, or TREE proof is
fabricated. The tail source, analysis and replica use actual native workers.
GPU/provider/legacy prohibitions come from the unchanged project fixture.
Fresh CLI below means a new actual Orze/Lake/pager invocation in the same test
process, not an independently exec'd interpreter.
"""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import pytest

from test_cpu_product_loop import project
from test_cpu_domain_product import request, submit

SOURCE = "idea-zz-source"
ANALYSIS = "idea-zzzz-analysis"
PREFIX_COUNT = 33
ANALYSIS_PROGRAM = """import json,os
from pathlib import Path
fds=json.loads(os.environ['ORZE_ACTION_SOURCE_FDS'])
assert len(fds)==1
values=[int(os.read(fd,128)) for fd in fds.values()]
assert values==[17]
Path('analysis-started').write_text('real GO')
Path('result.json').write_text(json.dumps({'version':1,'observations':[
 {'name':'answer','values':{'value':sum(values)*2},
  'validation':{'status':'valid','reason_code':'explicit_arithmetic'},
  'comparison_scope':'paging-double-v1'}]}))
"""


def _database(root):
    names = ("ideas", "execution_attempts", "research_artifacts",
             "research_observations", "cpu_action_reservations",
             "cpu_action_decisions", "cpu_proposal_requests", "replication_requests")
    with sqlite3.connect(root / "lake.db") as conn:
        conn.row_factory = sqlite3.Row
        present = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {name: [dict(r) for r in conn.execute("SELECT * FROM main." + name + " ORDER BY rowid")]
                if name in present else [] for name in names}


def _metadata_prefix(lake, results):
    from orze.core.execution_attempts import create_attempt, mark_running, finish_attempt
    from orze.engine.execution_authority import execution_transaction
    from orze.engine.execution_catalog import bind_catalog
    refs = []
    for i in range(PREFIX_COUNT):
        task = "idea-00-metadata-%02d" % i
        folder = results / task
        folder.mkdir()
        with execution_transaction(lake, folder) as tx:
            bind_catalog(lake, folder, tx.lease)
            ref = create_attempt(tx.conn, task, "action", task + "-attempt", {})
            mark_running(tx.conn, ref)
            digest = tx.prepare(ref, {"operation": "paging_metadata_fixture_not_native"})
            finish_attempt(tx.conn, ref, {"outcome": "completed", "artifact_ids": [],
                "observation_ids": [], "effect_receipt_sha256": digest})
        refs.append(asdict(ref))
    return refs


@pytest.fixture
def history(project, monkeypatch):
    from orze.core import research_interfaces as api
    from orze.engine import native_cpu_action as native
    from orze.idea_lake import IdeaLake
    root, cfg, run = project
    cfg["execution"]["wall_budget_seconds"] = 12
    cfg["action_domain"] = {"version": 1, "kind": "command", "config": {}}
    processes = []
    real_prepare = native.prepare_supervised

    def capture(*args, **kwargs):
        process = real_prepare(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(native, "prepare_supervised", capture)
    submit(root, SOURCE, request("from pathlib import Path; Path('value').write_text('17')",
        outputs={"value": {"path": "value", "max_bytes": 128}}))
    assert run() == 0
    initial = _database(root)
    source = json.loads(initial["research_artifacts"][0]["record_json"])
    terminal = json.loads(initial["execution_attempts"][0]["terminal_json"])
    assert terminal["outcome"] == "completed" and terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
    assert initial["cpu_action_reservations"][0]["state"] == "SETTLED"
    lake = IdeaLake(root / "lake.db")
    try:
        prefixes = _metadata_prefix(lake, root / "results")
    finally:
        lake.close()
    monkeypatch.setattr(api, "_DOMAINS", dict(api._DOMAINS))
    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))

    class MixedDomain(api.CommandDomain):
        def prepare(self, declared, sources):
            if "result_output" in declared["payload"]:
                return api.JsonObservationDomain.prepare(self, declared, sources)
            return super().prepare(declared, sources)

        def interpret(self, prepared, envelope):
            if prepared["observation"] is None:
                return ()
            return api.JsonObservationDomain.interpret(self, prepared, envelope)

    api.register_domain("paging_product_mixed", "paging.product.mixed.v1", MixedDomain)
    cfg["action_domain"]["kind"] = "paging_product_mixed"
    cfg["action_policy"] = {"version": 2, "kind": "paging_product_policy", "idle": "wait",
                           "wait_seconds": .01, "evidence_page_size": 8, "config": {}}
    state = {"root": root, "cfg": cfg, "run": run, "source": source,
             "prefix_refs": prefixes, "processes": processes, "trace": [],
             "before": _database(root), "outcomes": []}
    try:
        yield state
    finally:
        closures = []
        for process in processes:
            if process.poll() is None:
                process.stop(timeout=.5)
            closures.append(process.closure_receipt())
        report = {"fixture": "33 metadata-only prefixes; actual CLI/native tail work separately counted",
            "source": source, "prefix_refs": prefixes, "trace": state["trace"],
            "outcomes": state["outcomes"], "before": state["before"], "after": _database(root),
            "actual_native_workers": len(processes), "actual_native_closures": closures,
            "fresh_cli_is_new_invocation_not_new_interpreter": True}
        (root / "paging-product-report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        print("CPU_EVIDENCE_PAGING_REPORT=" + str(root / "paging-product-report.json"))
        assert all(c["event"] == "TREE_CLOSED" and c["wait_proof"] == "ECHILD_WALL" for c in closures)


def _register(history, implementation):
    from orze.core import research_interfaces as api
    api.register_policy("paging_product_policy", "paging.product.policy.v1", implementation)


def _capture(history, snapshot, budget):
    history["trace"].append({"snapshot": copy.deepcopy(snapshot), "budget": copy.deepcopy(budget),
                             "database_before_callback": _database(history["root"])})
    if len(history["trace"]) > 40:
        raise AssertionError("bounded fixture must not busy-loop")


def _next(snapshot):
    meta = snapshot.get("evidence_page")
    if meta is None:
        # Product baseline has a usable v2 declaration but only its old first
        # window. Finish normally; positive business assertions expose the gap.
        return {"kind": "Stop", "reason": "unpaged_view_cannot_analyze", "wakeup": None}
    if meta["next_cursor"] is None:
        raise AssertionError("required real tail result missing at traversal end")
    return {"kind": "ReadEvidence", "cursor": meta["next_cursor"]}


def _propose(source_id):
    declared = request(ANALYSIS_PROGRAM, sources=[source_id], observation=True,
        outputs={"result": {"path": "result.json", "max_bytes": 4096}})
    declared["purpose"] = "analyze actual tail-page source through its sealed descriptor"
    return {"kind": "Propose", "request_id": "page-analysis-request", "task_id": ANALYSIS,
            "reason": "tail evidence calls for explicit analysis", "domain_request": declared}


def test_cli_pages_tail_source_selects_then_analyzes_and_replicates(history):
    class Policy:
        def __init__(self, declaration):
            self.stage = "source"

        def decide(self, snapshot, budget):
            _capture(history, snapshot, budget)
            view = snapshot["recorded_evidence"]
            if snapshot["queue"]:
                decision = {"kind": "Execute", "task_id": snapshot["queue"][0]["idea_id"]}
            elif self.stage == "source":
                source = next((r for r in view["results"] if r["ref"]["task_id"] == SOURCE), None)
                if source is None:
                    decision = _next(snapshot)
                else:
                    self.stage = "selected_source"
                    decision = {"kind": "SelectEvidence", "refs": [copy.deepcopy(source["ref"])]}
            elif self.stage == "selected_source":
                assert snapshot["evidence_page"]["mode"] == "selection"
                selected, = view["results"]
                assert selected["ref"]["task_id"] == SOURCE
                self.stage = "analysis"
                decision = _propose(selected["artifact_records"][0]["artifact_id"])
            elif self.stage == "analysis":
                result = next((r for r in view["results"] if r["ref"]["task_id"] == ANALYSIS), None)
                if result is None:
                    decision = _next(snapshot)
                else:
                    self.stage = "replica"
                    decision = {"kind": "Replicate", "request_id": "page-analysis-confirm",
                                "source_ref": copy.deepcopy(result["ref"]),
                                "reason": "explicit independent analysis occurrence"}
            else:
                replica = next((r for r in view["results"] if r["observation_records"]
                                and r["ref"]["task_id"] != ANALYSIS), None)
                decision = (_next(snapshot) if replica is None else
                            {"kind": "Stop", "reason": "tail_analysis_confirmed", "wakeup": None})
            history["trace"][-1]["decision"] = copy.deepcopy(decision)
            return decision

    _register(history, Policy)
    result = history["run"](once=False)
    history["outcomes"].append(result)
    assert result == 0
    after = _database(history["root"])
    assert len(history["processes"]) == 3
    assert len(after["execution_attempts"]) == PREFIX_COUNT + 3
    assert [r["state"] for r in after["cpu_action_reservations"]] == ["SETTLED"] * 3
    decision_kinds = [entry["decision"]["kind"] for entry in history["trace"]]
    assert decision_kinds.count("ReadEvidence") >= 4
    assert "SelectEvidence" in decision_kinds and "Propose" in decision_kinds and "Replicate" in decision_kinds
    assert history["trace"][-1]["decision"]["reason"] == "tail_analysis_confirmed"
    first_source = next(t for t in history["trace"] if any(
        r["ref"]["task_id"] == SOURCE for r in t["snapshot"]["recorded_evidence"]["results"]))
    assert first_source["snapshot"]["evidence_page"]["seen"] >= 34
    observations = [json.loads(r["record_json"]) for r in after["research_observations"]]
    assert len(observations) == 2 and len({o["evaluator"]["attempt_id"] for o in observations}) == 2
    assert all(o["values"] == {"value": 34} and o["validation"]["status"] == "valid" for o in observations)
    assert all(o["input_artifact_ids"] == [history["source"]["artifact_id"]] for o in observations)
    assert all(o["input_artifact_bindings"][history["source"]["artifact_id"]]["producer"] ==
               history["source"]["producer"] for o in observations)
    records = [json.loads(r["record_json"]) for r in after["replication_requests"]]
    assert len(records) == 1 and records[0]["source_ref"]["task_id"] == ANALYSIS
    assert records[0]["task_id"] != ANALYSIS
    # All read/selection callbacks before Propose retain the original one
    # actual reservation and 34 records; metadata fixtures carry no permits.
    for entry in history["trace"]:
        if entry["decision"]["kind"] == "Propose":
            break
        db = entry["database_before_callback"]
        assert len(db["cpu_action_reservations"]) == 1
        assert len(db["execution_attempts"]) == PREFIX_COUNT + 1
        assert db["cpu_action_decisions"] == history["before"]["cpu_action_decisions"]
        assert db["cpu_proposal_requests"] == []


@pytest.mark.parametrize("fault", ["forged_ref", "changed_source_bytes"])
def test_cli_paged_adversarial_source_cannot_create_reservation_or_worker(history, fault):
    class Policy:
        def __init__(self, declaration):
            self.selected = False
            self.source_id = None

        def decide(self, snapshot, budget):
            _capture(history, snapshot, budget)
            results = snapshot["recorded_evidence"]["results"]
            if not self.selected:
                source = next((r for r in results if r["ref"]["task_id"] == SOURCE), None)
                if source is None:
                    return _next(snapshot)
                self.source_id = source["artifact_records"][0]["artifact_id"]
                self.selected = True
                ref = copy.deepcopy(source["ref"])
                if fault == "forged_ref":
                    ref["generation"] += 1
                    source["ref"] = copy.deepcopy(ref)
                    source["artifact_records"][0]["content_sha256"] = "0" * 64
                return {"kind": "SelectEvidence", "refs": [ref]}
            if fault == "changed_source_bytes":
                assert snapshot["evidence_page"]["mode"] == "selection"
                assert results[0]["artifact_records"][0] == history["source"]
                Path(history["source"]["path"]).write_bytes(b"19")
            # Forged-ref selection may return explicit unavailable or be denied
            # directly. Neither can authorize this known real artifact ID.
            return _propose(self.source_id)

    _register(history, Policy)
    result = history["run"](once=False)
    history["outcomes"].append(result)
    assert result == 75
    after = _database(history["root"])
    assert after["cpu_action_reservations"] == history["before"]["cpu_action_reservations"]
    assert after["execution_attempts"] == history["before"]["execution_attempts"]
    assert after["research_artifacts"] == history["before"]["research_artifacts"]
    assert after["research_observations"] == []
    assert after["cpu_proposal_requests"] == []
    assert len(history["processes"]) == 1
    assert not (history["root"] / "results" / ANALYSIS / "claim.json").exists()


def test_fresh_cli_cannot_reuse_previous_invocation_evidence_cursor(history):
    invocations = []

    class Policy:
        def __init__(self, declaration):
            self.ordinal = len(invocations)
            invocations.append(None)

        def decide(self, snapshot, budget):
            _capture(history, snapshot, budget)
            if self.ordinal == 0:
                invocations[0] = snapshot["evidence_page"]["next_cursor"]
                return {"kind": "Pause", "reason": "save_untrusted_old_cursor", "wakeup": None}
            assert snapshot["evidence_page"]["next_cursor"] != invocations[0]
            return {"kind": "ReadEvidence", "cursor": invocations[0]}

    _register(history, Policy)
    first = history["run"](once=False)
    history["outcomes"].append(first)
    assert first == 0 and type(invocations[0]) is str
    after_first = _database(history["root"])
    second = history["run"](once=False)
    history["outcomes"].append(second)
    assert second == 75
    after_second = _database(history["root"])
    assert after_second["cpu_action_reservations"] == after_first["cpu_action_reservations"]
    assert after_second["execution_attempts"] == after_first["execution_attempts"]
    assert after_second["cpu_action_decisions"] == after_first["cpu_action_decisions"]
    assert len(history["processes"]) == 1
    with sqlite3.connect(history["root"] / "lake.db") as conn:
        assert conn.execute("SELECT stop_json FROM cpu_action_scopes").fetchone() == (None,)

