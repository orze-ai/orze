"""Independent whole-old-module differential audit of budget row readers.

Every row below is explicitly metadata-only. No native attempt, process TREE,
effect confirmation, public settlement or real consumption is fabricated.
The private reader layer is tested directly; deliberate malformed SQL rows do
not claim to have passed the public schema/ownership gates.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from orze.core import cpu_action_budget as current
from orze.idea_lake import IdeaLake

BASELINE = Path(__file__).resolve().parents[1] / (
    "docs/evidence/runs/2026-09-14-cost-equivalence/baseline/src/orze/core/cpu_action_budget.py")
BASE_SHA = "71bc89977cb582aa86a218003398431bb975ce11410f1df0d43b385ae4667169"

CASES = [
    "empty", "mixed_states", "huge_exact_integers", "other_scope_excluded",
    "noncanonical_permit", "duplicate_permit_key", "scope_hash_invalid",
    "permit_scope_changed", "sql_id_missing_target", "sql_ids_swapped",
    "sql_task_changed", "sql_slot_changed", "wrong_ref_task", "bool_ref_generation",
    "unknown_state", "reserved_has_ref", "bound_missing_ref",
    "settled_missing_terminal", "bound_has_terminal", "live_slot_conflict",
    "oversized_permit", "finite_budget_exceeded", "alias_existing_target",
]


def load_old():
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == BASE_SHA
    spec = importlib.util.spec_from_file_location("orze.core._independent_old_budget", BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(call):
    try:
        return {"accepted": True, "value": call()}
    except Exception as exc:
        return {"accepted": False, "exception": type(exc).__name__, "reason": str(exc)}


def _row(module, scope, task, *, state="SETTLED", slot=0, wall=2):
    identity = hashlib.sha256(task.encode()).hexdigest()[:48]
    permit = {"schema": 1, "budget_scope": deepcopy(scope), "reservation_id": identity,
              "task_id": task, "slot": slot, "wall_limit_seconds": wall,
              "reserved_nanoseconds": str(module._ns(wall, reservation=True))}
    ref = {"task_id": task, "phase": "action", "attempt_id": identity, "generation": 1}
    row = [identity, scope["results_dir"], task, slot, module._json(permit),
           None if state == "RESERVED" else module._json(ref), state,
           "0" * 64 if state == "SETTLED" else None]
    return row, permit


def metadata_case(root, case, old):
    results = root / "results"
    results.mkdir()
    other_results = root / "other"
    other_results.mkdir()
    lake = IdeaLake(root / "lake.db")
    declaration = {"version": 2, "resource": "cpu", "slots": 3, "wall_budget_seconds": None}
    if case == "finite_budget_exceeded":
        declaration = {**declaration, "version": 1, "wall_budget_seconds": 1}
    scope = current.initialize(lake, results, declaration)
    other = current.initialize(lake, other_results, {**declaration, "version": 2,
                                                   "wall_budget_seconds": None})
    a, pa = _row(old, scope, "row-a")
    b, pb = _row(old, scope, "row-b")
    rows, permits = [a, b], [pa, pb]
    if case == "empty":
        rows, permits = [], []
    elif case == "mixed_states":
        a, pa = _row(old, scope, "row-a", state="RESERVED", slot=0)
        b, pb = _row(old, scope, "row-b", state="BOUND", slot=1)
        c, pc = _row(old, scope, "row-c", state="SETTLED", slot=1)
        rows, permits = [a, b, c], [pa, pb, pc]
    elif case == "huge_exact_integers":
        a, pa = _row(old, scope, "row-a", wall=10**300)
        b, pb = _row(old, scope, "row-b", wall=10**300 + 1)
        rows, permits = [a, b], [pa, pb]
    elif case == "other_scope_excluded":
        foreign, pf = _row(old, other, "foreign")
        foreign[4] = "deliberately invalid foreign-scope metadata"
        rows.append(foreign)
    elif case == "noncanonical_permit":
        a[4] = json.dumps(pa, indent=1)
    elif case == "duplicate_permit_key":
        a[4] = '{"schema":1,' + old._json(pa)[1:]
    elif case == "scope_hash_invalid":
        value = deepcopy(pa); value["budget_scope"]["policy_sha256"] = "0" * 64
        a[4] = old._json(value)
    elif case == "permit_scope_changed":
        value = deepcopy(pa); value["budget_scope"] = other
        a[4] = old._json(value)
    elif case == "sql_id_missing_target":
        a[0] = "f" * 48
    elif case == "sql_ids_swapped":
        a[0], b[0] = b[0], a[0]
    elif case == "sql_task_changed":
        a[2] = "wrong-task"
    elif case == "sql_slot_changed":
        a[3] = 2
    elif case == "wrong_ref_task":
        ref = json.loads(a[5]); ref["task_id"] = "wrong-task"; a[5] = old._json(ref)
    elif case == "bool_ref_generation":
        ref = json.loads(a[5]); ref["generation"] = True; a[5] = old._json(ref)
    elif case == "unknown_state":
        a[6] = "INVALID"
    elif case == "reserved_has_ref":
        a[6], a[7] = "RESERVED", None
    elif case == "bound_missing_ref":
        a[5], a[6], a[7] = None, "BOUND", None
    elif case == "settled_missing_terminal":
        a[7] = None
    elif case == "bound_has_terminal":
        a[6] = "BOUND"
    elif case == "live_slot_conflict":
        a, pa = _row(old, scope, "row-a", state="RESERVED", slot=0)
        b, pb = _row(old, scope, "row-b", state="BOUND", slot=0)
        rows, permits = [a, b], [pa, pb]
        lake.conn.execute("DROP INDEX cpu_action_live_slot")
    elif case == "oversized_permit":
        a[4] += " " * 17000
    elif case == "alias_existing_target":
        # The old scan reads row-a's permit, then re-reads the valid row-b by
        # that permit's ID. Thus row-a's actual SQL identity/state/ref can hide.
        a[4], a[5], a[6], a[7] = b[4], "unparsed hidden row", "INVALID", None
    lake.conn.execute("PRAGMA ignore_check_constraints=ON")
    lake.conn.executemany("INSERT INTO cpu_action_reservations VALUES (?,?,?,?,?,?,?,?)", rows)
    lake.conn.commit()
    return lake, scope, permits


@pytest.mark.parametrize("case", CASES)
def test_whole_old_and_current_budget_readers_on_identical_metadata(tmp_path, case):
    old = load_old()
    lake, scope, permits = metadata_case(tmp_path, case, old)
    evidence = {"case": case, "baseline_sha256": BASE_SHA,
                "classification": "metadata-only private row-reader differential"}
    try:
        before = "\n".join(lake.conn.iterdump())
        evidence["rows"] = [list(r) for r in lake.conn.execute(
            "SELECT * FROM cpu_action_reservations ORDER BY reservation_id")]
        evidence["scope"] = scope
        old_total = result(lambda: old._totals(lake.conn, scope))
        new_total = result(lambda: current._totals(lake.conn, scope))
        evidence["totals"] = {"old": old_total, "new": new_total}
        points = []
        for permit in permits:
            prior = result(lambda: old._reservation(lake.conn, permit))
            candidate = result(lambda: current._reservation(lake.conn, permit))
            points.append({"reservation_id": permit["reservation_id"],
                           "old": prior, "new": candidate})
            assert prior["accepted"] == candidate["accepted"]
            if prior["accepted"]:
                assert prior["value"] == candidate["value"]
        evidence["point_reads"] = points
        assert "\n".join(lake.conn.iterdump()) == before
        if case == "alias_existing_target":
            assert old_total == {"accepted": True, "value": (4_000_000_000, {})}
            assert new_total["accepted"] is False
            evidence["classification"] = "actual old whole-module alias accepted; new SQL/permit-ID fence refuses"
        else:
            assert old_total["accepted"] == new_total["accepted"]
            if old_total["accepted"]:
                assert old_total["value"] == new_total["value"]
        if case == "huge_exact_integers":
            assert old_total["value"][0] == (2 * 10**300 + 1) * 1_000_000_000
        if case == "mixed_states":
            assert old_total["value"] == (6_000_000_000, {0: "row-a", 1: "row-b"})
    finally:
        lake.close()
        path = tmp_path / "budget-equivalence.json"
        path.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
        print("CPU_BUDGET_EQUIVALENCE_REPORT=" + str(path))

