"""Independent refusal checks over an actual worker/controller crash boundary.

Every project first executes the real CLI/native worker and exits its controller
at the existing settle entrance. Subsequent SQLite/file edits are deliberately
invalid negative inputs, never synthesized successful execution or TREE proof.
These are new recovery requirements, not extra historical product defects.
No test edits an old fixture, clears an owner, or launches an unowned process.
"""
from copy import deepcopy
import hashlib
import json
import sqlite3

import pytest

from cpu_terminal_recovery_helpers import admit, canonical, run_cli, snapshot

pytest_plugins = ["cpu_terminal_recovery_helpers"]

_BUSINESS_TABLES = (
    "ideas", "idea_state", "idea_transitions", "idea_stage_state",
    "idea_stage_transitions", "execution_attempts", "research_artifacts",
    "research_observations", "cpu_action_reservations",
)


def _attempt(project):
    rows = project["snapshots"][-1]["database"]["execution_attempts"]
    assert len(rows) == 1 and rows[0]["state"] == "TERMINAL"
    return deepcopy(rows[0])


def _edit_json(project, column, mutate):
    """Corrupt only the temporary completed occurrence, then close the writer."""
    assert column in ("binding_json", "terminal_json")
    row = _attempt(project)
    value = json.loads(row[column])
    mutate(value)
    with sqlite3.connect(project["root"] / "lake.db") as conn:
        conn.execute("UPDATE main.execution_attempts SET " + column + "=? WHERE attempt_id=?",
                     (canonical(value).decode(), row["attempt_id"]))


def _unchanged_business(before, after):
    # Recovery's own barrier/HOLD metadata may change; business evidence may not.
    for name in _BUSINESS_TABLES:
        assert after["database"][name] == before["database"][name], name
    assert after["files"] == before["files"]
    assert after["worker_events"] == before["worker_events"]
    assert len(after["database"]["execution_attempts"]) == 1
    assert after["database"]["cpu_action_reservations"][0]["state"] == "BOUND"
    statuses = {row["idea_id"]: row["status"] for row in after["database"]["ideas"]}
    assert statuses["idea-second"] == "queued"


def _restart_refuses(project, label):
    before = snapshot(project, label + ":tampered_input")
    call = run_cli(project, label, expected=75)
    after = project["snapshots"][-1]
    _unchanged_business(before, after)
    assert call["metadata"][-1]["exit_code"] == 75
    return before, after


@pytest.mark.parametrize("field", [
    "origin", "kind", "resource", "attempt_ref", "process_pid",
    "command_sha256", "timeout_seconds",
])
def test_recovery_refuses_changed_native_binding(make_crashed_project, field):
    project = make_crashed_project()
    admit(project, "idea-second", "second")

    def change(binding):
        replacements = {"origin": "legacy_import", "kind": "training", "resource": "gpu",
                        "command_sha256": "0" * 64, "timeout_seconds": 3}
        if field == "attempt_ref":
            # Equal under loose Python numeric comparison, not the full Ref contract.
            binding[field]["generation"] = float(binding[field]["generation"])
        elif field == "process_pid":
            binding[field] = float(binding[field])
        else:
            binding[field] = replacements[field]

    _edit_json(project, "binding_json", change)
    _restart_refuses(project, "binding_" + field)


@pytest.mark.parametrize("field", ["database", "claim_sha256", "config_sha256"])
def test_recovery_refuses_changed_captured_source(make_crashed_project, field):
    project = make_crashed_project()
    admit(project, "idea-second", "second")

    def change(binding):
        binding["source"][field] = (str(project["root"] / "foreign.db")
                                    if field == "database" else "0" * 64)

    _edit_json(project, "binding_json", change)
    _restart_refuses(project, "source_" + field)


def test_recovery_refuses_changed_current_lifecycle(make_crashed_project):
    project = make_crashed_project()
    admit(project, "idea-second", "second")
    # Not a valid lifecycle transition: a deliberate on-disk consistency fault.
    with sqlite3.connect(project["root"] / "lake.db") as conn:
        changed = conn.execute("UPDATE main.idea_stage_state SET current_state='PENDING' "
                               "WHERE idea_id='idea-first' AND stage='action'").rowcount
    assert changed == 1
    _restart_refuses(project, "stage_projection_changed")


def test_recovery_refuses_rehashed_non_cpu_prepared_plan(make_crashed_project):
    from orze.engine.attempt_effect_receipts import _scan

    project = make_crashed_project()
    admit(project, "idea-second", "second")
    row = _attempt(project)
    folder = project["root"] / "results" / row["task_id"] / "_execution_effects" / row["attempt_id"]
    prepared_path, committed_path = folder / "prepared.json", folder / "committed.json"
    prepared = json.loads(prepared_path.read_bytes())
    original_tree = deepcopy(prepared["plan"]["process_tree"])
    # Preserve the actual captured TREE; changing/resealing the operation is
    # adversarial test data, not a legitimate producer or a newly granted effect.
    prepared["plan"]["operation"] = "not_cpu_action_terminal"
    raw = canonical(prepared)
    digest = hashlib.sha256(raw).hexdigest()
    committed = json.loads(committed_path.read_bytes())
    committed["prepared_sha256"] = digest
    prepared_path.write_bytes(raw)
    committed_path.write_bytes(canonical(committed))
    _edit_json(project, "terminal_json", lambda value: value.update(effect_receipt_sha256=digest))
    assert json.loads(prepared_path.read_bytes())["plan"]["process_tree"] == original_tree
    # The negative is plan eligibility, not malformed JSON or a broken hash link.
    assert _scan(folder.parent.parent)[row["attempt_id"]] == (digest, True)
    _restart_refuses(project, "wrong_effect_operation")


def test_recovery_refuses_changed_catalog_route(make_crashed_project):
    project = make_crashed_project()
    admit(project, "idea-second", "second")
    catalog = project["root"] / "results" / "idea-first" / "_execution_catalog.json"
    value = json.loads(catalog.read_bytes())
    value["database"] = str(project["root"] / "not-the-main-lake.db")
    raw = canonical(value) + b"\n"
    catalog.write_bytes(raw)
    _restart_refuses(project, "catalog_route_changed")
    assert catalog.read_bytes() == raw


def test_recovery_refuses_changed_ready_member_identity(make_crashed_project):
    project = make_crashed_project()
    admit(project, "idea-second", "second")
    # Stored READY no longer agrees with the original actual closure. Never
    # replace the closure with a fabricated tree or rediscover a process by PID.
    _edit_json(project, "binding_json", lambda value: value["supervision"]["worker"].update(
        start_ticks=float(value["supervision"]["worker"]["start_ticks"])))
    _restart_refuses(project, "ready_identity_type_changed")


def test_recovery_rechecks_source_after_enumeration_before_settlement(make_crashed_project, monkeypatch):
    from orze.core import cpu_action_budget as budget
    from orze.idea_lake import IdeaLake

    project = make_crashed_project()
    admit(project, "idea-second", "second")
    lake = IdeaLake(project["root"] / "lake.db")
    boundary = []
    actual_settle = budget._settle

    def changed_source(lake_arg, permit, ref, evidence, **kwargs):
        assert lake_arg is lake and kwargs.get("recovery") is not None
        assert not lake_arg.conn.in_transaction
        # A real independent SQLite writer, after recovery selected the source
        # but before its actual guarded writer. All authority checks remain real.
        _edit_json(project, "binding_json", lambda value: value["source"].update(config_sha256="0" * 64))
        boundary.append(snapshot(project, "enumerated_source_changed"))
        return actual_settle(lake_arg, permit, ref, evidence, **kwargs)

    monkeypatch.setattr(budget, "_settle", changed_source)
    try:
        scope = budget.initialize(lake, project["root"] / "results", project["cfg"]["execution"])
        with pytest.raises(budget.CpuBudgetHOLD):
            budget.reconcile_confirmed_terminals(lake, scope)
        assert len(boundary) == 1
        _unchanged_business(boundary[0], snapshot(project, "failed_same_kernel"))
    finally:
        lake.close()
    # A fresh controller must observe the persistent barrier even though this
    # Python process will not service its callback or perform recovery for it.
    _restart_refuses(project, "restart_after_enumeration_fault")


@pytest.mark.parametrize("mode", ["ordinary_stop", "storage_hold", "masked_storage_hold"])
def test_preexisting_stop_never_authorizes_automatic_settlement(make_crashed_project, mode):
    from orze.core import cpu_action_budget as budget
    from orze.idea_lake import IdeaLake

    project = make_crashed_project()
    admit(project, "idea-second", "second")
    lake = IdeaLake(project["root"] / "lake.db")
    try:
        scope = budget.initialize(lake, project["root"] / "results", project["cfg"]["execution"])
        reason = "budget_storage_unconfirmed" if mode == "storage_hold" else "operator_stop"
        budget.record_decision(lake, scope, {"kind": "Stop", "reason": reason, "wakeup": None})
        stop_before = lake.conn.execute("SELECT stop_json FROM main.cpu_action_scopes").fetchone()[0]
        if mode == "masked_storage_hold":
            # Exercise the existing COALESCE behavior, not an invented Stop:
            # a later uncertainty latches in this process but leaves the old
            # Stop bytes. The fresh CLI has no access to this in-memory set.
            budget._latch(scope)
            assert lake.conn.execute("SELECT stop_json FROM main.cpu_action_scopes").fetchone()[0] == stop_before
    finally:
        lake.close()
    before, after = _restart_refuses(project, mode)
    assert before["database"]["cpu_action_scopes"] == after["database"]["cpu_action_scopes"]
    assert after["database"]["cpu_action_scopes"][0]["stop_json"] == stop_before


def test_recovery_refuses_unbound_replication_request_field(make_crashed_project):
    project = make_crashed_project()
    admit(project, "idea-second", "second")
    original = json.loads(_attempt(project)["binding_json"])
    assert "replication_request" not in original["source"]
    # A new canonical JSON field is not a durable replication request. The
    # actual unreplicated task, worker/tree proof and receipt files are unchanged.
    _edit_json(project, "binding_json", lambda value: value["source"].update(replication_request=True))
    _restart_refuses(project, "forged_replication_metadata")


def test_recovery_refuses_artifact_root_unrelated_to_published_records(make_crashed_project):
    from orze.core.artifact_contract import validate_artifact_publication_binding

    project = make_crashed_project()
    admit(project, "idea-second", "second")
    alternative = str(project["root"] / "another-artifact-root")
    changed = json.loads(_attempt(project)["binding_json"])["artifact_publication"]
    changed["root"] = alternative
    # This path is a valid declaration in isolation. The fault is its mismatch
    # with this occurrence's real published records, not a path syntax error.
    assert validate_artifact_publication_binding(changed)["root"] == alternative
    _edit_json(project, "binding_json", lambda value: value.update(artifact_publication=changed))
    _restart_refuses(project, "artifact_root_record_mismatch")
