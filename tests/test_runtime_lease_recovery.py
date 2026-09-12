"""C3 new typed provenance and real historical terminal-settlement controls.

The small metadata matrix is deliberately synthetic and proves no TREE/GO
authority. The CLI controls use real native children, SQLite and effect files;
only the old-v1 control loads a pinned complete historical native module.
No old tests, deadlines, closure receipts or stored terminal rows are patched.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest
import yaml

from cpu_terminal_recovery_helpers import admit, run_cli, save_report
from orze.core import cpu_action_budget as budget
from orze.core.execution_attempts import create_attempt
from orze.engine.execution_authority import execution_transaction
from test_cpu_action_budget import context, _not_started

pytest_plugins = ["cpu_terminal_recovery_helpers"]


def _metadata():
    # Historical host/boot deliberately need not match the running machine.
    lease = {"schema": 1, "clock": "CLOCK_BOOTTIME", "hostname": "historical-host",
             "boot_id": "11111111-2222-3333-4444-555555555555", "issued_ns": 10, "deadline_ns": 20}
    ready = {"schema": 2, "protocol": "orze.linux_subreaper.v2", "identity": {},
             "nonce_sha256": "a" * 64, "command_sha256": "b" * 64,
             "worker": {"pid": 101, "start_ticks": 20},
             "supervisor": {"pid": 102, "start_ticks": 19}, "runtime_lease": deepcopy(lease)}
    bound = {"runtime_lease": lease, "supervision": ready,
             "process_supervision_protocol": "orze.linux_subreaper.v2", "process_pid": 101,
             "command_sha256": "b" * 64, "timeout_seconds": 1}
    closure = {"schema": 2, "event": "TREE_CLOSED", "binding": deepcopy(ready),
               "worker_returncode": 0, "stop_requested": False, "forced_cleanup": False,
               "reaped_children": 1, "wait_proof": "ECHILD_WALL",
               "lease_expired": False, "lease_observed_ns": 12}
    terminal = {"outcome": "completed", "reason_code": "cpu_action_completed", "return_code": 0,
                "process_tree": closure, "artifact_ids": [], "observation_ids": [],
                "lifecycle_phase": "action", "elapsed_wall_seconds": .01, "lifecycle": {},
                "effect_receipt_sha256": "c" * 64,
                "runtime_lease": {"schema": 1, "status": "authorized", "observed_ns": 15}}
    return bound, terminal


@pytest.mark.parametrize("kind", ["authorized", "supervisor_expired", "interpret_expired"])
def test_historical_metadata_classifies_expiry_without_current_host_or_clock(kind):
    bound, terminal = _metadata()
    if kind != "authorized":
        terminal.update(outcome="interrupted", reason_code="cpu_runtime_lease_expired")
        terminal["runtime_lease"].update(status="expired", observed_ns=20)
    if kind == "supervisor_expired":
        terminal["process_tree"].update(lease_expired=True, lease_observed_ns=20, stop_requested=True)
    before = deepcopy((bound, terminal))
    assert budget._terminal_runtime_lease(bound, terminal) == (
        "authorized" if kind == "authorized" else "expired")
    assert (bound, terminal) == before


@pytest.mark.parametrize("where,key,value", [
    ("lease", "schema", True), ("lease", "issued_ns", 10.0),
    ("lease", "deadline_ns", True), ("lease", "deadline_ns", 2**63),
    ("lease", "clock", "CLOCK_MONOTONIC"), ("lease", "unexpected", 1),
    ("ready", "schema", True), ("ready", "protocol", "orze.linux_subreaper.v1"),
    ("closure", "lease_expired", 1), ("closure", "lease_observed_ns", 12.0),
    ("closure", "lease_expired", True),
    ("sample", "schema", True), ("sample", "observed_ns", True),
    ("sample", "observed_ns", 11), ("sample", "observed_ns", 20),
    ("sample", "status", "expired"), ("sample", "unexpected", None),
    ("terminal", "unexpected", "even-if-an-effect-plan-repeats-this"),
    ("terminal", "elapsed_wall_seconds", True), ("terminal", "elapsed_wall_seconds", float("inf")),
    ("terminal", "lifecycle_phase", "training"), ("bound", "timeout_seconds", 1e-9),
])
def test_typed_metadata_rejects_invalid_or_inverted_provenance(where, key, value):
    bound, terminal = _metadata()
    target = {"lease": bound["runtime_lease"], "ready": bound["supervision"],
              "closure": terminal["process_tree"], "sample": terminal["runtime_lease"],
              "terminal": terminal, "bound": bound}[where]
    target[key] = value
    if where == "lease":
        # Match all copies so this tests the descriptor itself, not a trivial
        # mismatched-copy rejection. These remain non-authoritative metadata.
        bound["supervision"]["runtime_lease"] = deepcopy(target)
        terminal["process_tree"]["binding"] = deepcopy(bound["supervision"])
    with pytest.raises(budget.CpuBudgetHOLD):
        budget._terminal_runtime_lease(bound, terminal)


@pytest.mark.parametrize("field,value", [("outcome", "completed"), ("reason_code", "cpu_action_interrupted"),
                                          ("artifact_ids", ["artifact"]), ("observation_ids", ["observation"])])
def test_expiry_cannot_claim_normal_outcome_or_publish_results(field, value):
    bound, terminal = _metadata()
    terminal.update(outcome="interrupted", reason_code="cpu_runtime_lease_expired")
    terminal["runtime_lease"].update(status="expired", observed_ns=20)
    terminal[field] = value
    with pytest.raises(budget.CpuBudgetHOLD):
        budget._terminal_runtime_lease(bound, terminal)


def test_missing_sample_or_new_lease_with_old_protocol_is_not_legacy_compatibility():
    bound, terminal = _metadata()
    del terminal["runtime_lease"]
    with pytest.raises(budget.CpuBudgetHOLD):
        budget._terminal_runtime_lease(bound, terminal)
    bound, terminal = _metadata()
    bound["process_supervision_protocol"] = "orze.linux_subreaper.v1"
    with pytest.raises(budget.CpuBudgetHOLD):
        budget._terminal_runtime_lease(bound, terminal)


def test_historical_envelope_uses_the_same_exact_float_floor_as_lease_capture():
    bound, terminal = _metadata()
    bound["timeout_seconds"] = .3
    # float(.3) is just below 300,000,000 ns; its exact floor is 299,999,999.
    # Matching copies isolate the envelope predicate, not descriptor identity.
    bound["runtime_lease"]["deadline_ns"] = 300_000_010
    bound["supervision"]["runtime_lease"] = deepcopy(bound["runtime_lease"])
    terminal["process_tree"]["binding"] = deepcopy(bound["supervision"])
    with pytest.raises(budget.CpuBudgetHOLD, match="runtime_lease_exceeds_envelope"):
        budget._terminal_runtime_lease(bound, terminal)
    bound["runtime_lease"]["deadline_ns"] -= 1
    bound["supervision"]["runtime_lease"] = deepcopy(bound["runtime_lease"])
    terminal["process_tree"]["binding"] = deepcopy(bound["supervision"])
    assert budget._terminal_runtime_lease(bound, terminal) == "authorized"


@pytest.mark.parametrize("versioned", [False, True])
def test_genuine_no_exec_explicit_settlement_keeps_v1_but_refuses_new_lease(context, versioned):
    c = context
    permit = budget.reserve(c.lake, c.scope, "no-exec", 2)
    folder = c.results / "no-exec"
    folder.mkdir()
    binding = {"reservation_id": permit["reservation_id"]}
    if versioned:
        # Controlled invalid production combination, only for denial: this
        # fixture creates no process, and never fabricates TREE evidence.
        binding.update(runtime_lease=_metadata()[0]["runtime_lease"],
                       process_supervision_protocol="orze.linux_subreaper.v2")
    with execution_transaction(c.lake, folder) as tx:
        ref = create_attempt(tx.conn, "no-exec", "action", "never-executed", binding)
        tx.watch_attempt(ref)
    budget.bind(c.lake, permit, ref)
    terminal = _not_started(c, folder, ref)
    if versioned:
        with pytest.raises(budget.CpuBudgetHOLD, match="runtime_lease_not_started_unconfirmed"):
            budget.settle(c.lake, permit, ref, terminal)
    else:
        assert budget.settle(c.lake, permit, ref, terminal) == "settled"
    assert c.lake.conn.execute("SELECT state FROM cpu_action_reservations").fetchone()[0] == (
        "BOUND" if versioned else "SETTLED")
    assert budget.snapshot(c.lake, c.scope)["reserved_wall_seconds"] == 2


def _recover_twice(project, *, protocol):
    before = project["snapshots"][-1]
    original = before["database"]["execution_attempts"]
    assert len(original) == len(before["database"]["cpu_action_reservations"]) == 1
    row = original[0]
    bound, terminal = json.loads(row["binding_json"]), json.loads(row["terminal_json"])
    assert row["state"] == "TERMINAL" and row["hold_reason"] is None
    assert bound["process_supervision_protocol"] == protocol
    assert terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
    assert before["database"]["cpu_action_reservations"][0]["state"] == "BOUND"
    assert len(before["worker_events"]) == 1
    if protocol.endswith("v2"):
        lease = bound["runtime_lease"]
        assert bound["supervision"]["runtime_lease"] == lease
        assert terminal["runtime_lease"]["status"] == "authorized"
        assert lease["issued_ns"] <= terminal["process_tree"]["lease_observed_ns"] <= (
            terminal["runtime_lease"]["observed_ns"]) < lease["deadline_ns"]
    else:
        assert "runtime_lease" not in bound and "runtime_lease" not in terminal
        assert terminal["process_tree"]["schema"] == 1
    for label in ("fresh_recovery", "fresh_recovery_replay"):
        run_cli(project, label)
        after = project["snapshots"][-1]
        assert after["database"]["execution_attempts"] == original
        assert after["worker_events"] == before["worker_events"]
        assert after["files"] == before["files"]
        for table in ("ideas", "idea_state", "idea_stage_state", "idea_transitions",
                      "idea_stage_transitions", "research_artifacts", "research_observations"):
            assert after["database"][table] == before["database"][table]
        reservations = after["database"]["cpu_action_reservations"]
        assert len(reservations) == 1 and reservations[0]["state"] == "SETTLED"
        assert int(json.loads(reservations[0]["permit_json"])["reserved_nanoseconds"]) == 2_000_000_000
    births = [tuple(call["controller_binding"]["worker"][key] for key in ("pid", "start_ticks"))
              for call in project["calls"]]
    assert len(births) == len(set(births)) == 3


@pytest.mark.parametrize("exit_code", [0, 7])
def test_real_v2_confirmed_terminal_crash_recovers_metadata_only(make_crashed_project, exit_code):
    project = make_crashed_project(exit_code=exit_code)
    terminal = json.loads(project["snapshots"][-1]["database"]["execution_attempts"][0]["terminal_json"])
    assert terminal["outcome"] == ("completed" if exit_code == 0 else "failed")
    _recover_twice(project, protocol="orze.linux_subreaper.v2")


def test_real_historical_v1_terminal_recovers_without_upgrade_or_reexecution(tmp_path, request):
    root = tmp_path / "historical-project"
    root.mkdir()
    cfg = {"execution": {"version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 6},
           "results_dir": str(root / "results"), "idea_lake_db": str(root / "lake.db"),
           "ideas_file": str(root / "ideas.md"), "min_disk_gb": 0,
           "action_policy": {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": .05}}
    (root / "orze.yaml").write_text(yaml.safe_dump(cfg))
    project = {"root": root, "cfg": cfg, "calls": [], "snapshots": [], "admissions": []}
    try:
        admit(project, "idea-first", "historical-v1")
        run_cli(project, "historical_controller_crash", crash=True, expected=86,
                child_module="test_runtime_lease_recovery")
        _recover_twice(project, protocol="orze.linux_subreaper.v1")
    finally:
        reports = getattr(request.config, "_cpu_recovery_reports", [])
        reports.append({**save_report(project), "test": request.node.nodeid})
        request.config._cpu_recovery_reports = reports


def _legacy_child():
    # Exact complete old module, not a flags-off path in the new native code.
    # Only this fresh producer interpreter uses it; all recovery CLIs load
    # current modules. Hardware/provider guards are the existing helper's.
    source = Path(__file__).resolve().parents[1] / "docs/evidence/snapshots/2026-09-12-c3-before-native-cpu-action.py"
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "a3af30c67e83afb0cc8eb311c26b574a4193a91e24dedd8d3653d1317a38e92e"
    import orze.engine as engine
    module = types.ModuleType("orze.engine.native_cpu_action")
    module.__file__, module.__package__ = str(source), "orze.engine"
    sys.modules[module.__name__] = module
    engine.native_cpu_action = module
    exec(compile(raw, str(source), "exec"), module.__dict__)
    from cpu_terminal_recovery_helpers import _child_main
    return _child_main()


if __name__ == "__main__":
    raise SystemExit(_legacy_child())
