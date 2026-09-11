"""New controller-action mechanisms; real CPU source, no training or fixer.

The producer uses real READY/GO and its confirmed terminal/effect receipts.
SQL/file fault injection below concerns publication, not simulated tree proof.
"""
from dataclasses import replace
import json
import os
import sys

import pytest

from orze.core.execution_attempts import AttemptRef, current_attempt
from orze.engine import accounting, scheduler
from orze.engine import pre_script_failure_report as report
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.native_pre_script import PreScriptHOLD, run_native_pre_script
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_pre_script_tree_completion import cpu_pre_script


@pytest.fixture
def failed(cpu_pre_script):
    c = cpu_pre_script
    assert scheduler.claim(c.idea, c.results, 4, lake=c.lake)
    c.result = run_native_pre_script(c.idea, 4, c.results, c.cfg, c.lake,
        [sys.executable, "-c", "raise SystemExit(7)"], 5, dict(os.environ))
    assert not c.result
    c.ref = c.result.attempt_ref
    c.claim = json.loads((c.folder / "claim.json").read_text())
    c.counters = {}
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
    assert c.lake.get_stage_state(c.idea, "training") == "NOT_STARTED"
    return c


def _call(c, ref=None):
    return report.report_pre_script_failure(c.lake, c.folder,
        c.ref if ref is None else ref, c.counters, c.cfg)


def _row(c):
    return current_attempt(c.lake.conn, c.idea, report.PHASE)


def _files(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def test_confirmed_failure_once_and_restart_projection_without_fixer(failed):
    c = failed
    assert c.ref.attempt_id != c.claim["attempt_id"]
    assert _call(c) == {"status": "reported", "failure_count_after": 1,
                        "repair_status": "pending_explicit_action"}
    assert c.lake.get_fsm_state(c.idea) == "FAILED"
    assert c.lake.get_stage_state(c.idea, "training") == "FAILED"
    assert c.counters == {c.idea: 1}
    row = _row(c)
    assert row["state"] == "TERMINAL"
    assert row["binding"]["source_attempt"]["attempt_id"] == c.ref.attempt_id
    assert row["terminal"]["claim_attempt_id"] == c.claim["attempt_id"]
    receipt = c.folder / "_compute_receipts" / c.claim["attempt_id"] / "terminal.json"
    value = json.loads(receipt.read_text())
    assert value["allocated_gpu_seconds"] == 0.0
    assert value["reason_code"] == "pre_script_failed" and value["phase"] == "admission"
    assert not receipt.with_name("start.json").exists()
    assert not (receipt.parent.parent / c.ref.attempt_id).exists()
    before, history = _files(c.folder), c.lake.get_fsm_history(c.idea)
    c.counters.clear()
    assert _call(c)["status"] == "duplicate"
    assert c.counters == {c.idea: 1}
    assert _files(c.folder) == before
    assert c.lake.get_fsm_history(c.idea) == history
    assert c.events == []  # Neither the phase's fixer nor a training launch.


@pytest.mark.parametrize("kind", ["none", "phase", "task", "generation"])
def test_explicit_reference_required_and_stale_delivery_is_inert(failed, kind):
    c = failed
    ref = {"none": None, "phase": replace(c.ref, phase="training"),
           "task": replace(c.ref, task_id="other-task"),
           "generation": replace(c.ref, generation=c.ref.generation + 1)}[kind]
    before = _files(c.folder)
    if kind == "generation":
        assert _call(c, ref) == {"status": "stale"}
    else:
        with pytest.raises(PreScriptHOLD):
            report.report_pre_script_failure(c.lake, c.folder, ref, c.counters, c.cfg)
    assert c.counters == {} and _row(c) is None
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
    assert _files(c.folder) == before


@pytest.mark.parametrize("after_report", [False, True])
def test_claim_replacement_cannot_report_or_replay_counter(failed, after_report):
    c = failed
    if after_report:
        _call(c)
        c.counters.clear()
    path = c.folder / "claim.json"
    value = json.loads(path.read_text())
    value["attempt_id"] = "different-claim"
    path.write_text(json.dumps(value))
    before = _files(c.folder)
    with pytest.raises(AttemptEffectInDoubt):
        _call(c)
    assert c.counters == {}
    assert {key: value for key, value in _files(c.folder).items()
            if not key.startswith("_attempt_effect.lock/")} == before


def test_missing_source_confirmation_does_not_create_report(failed):
    c = failed
    path = c.folder / "_execution_effects" / c.ref.attempt_id / "committed.json"
    path.unlink()
    with pytest.raises(AttemptEffectInDoubt):
        _call(c)
    assert _row(c) is None and c.counters == {}
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
    assert not (c.folder / "_compute_receipts").exists()


def test_existing_allocation_is_never_relabelled_zero(failed):
    c = failed
    path = c.folder / "_compute_receipts" / c.claim["attempt_id"] / "start.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"allocation":"already-started"}')
    before = path.read_bytes()
    with pytest.raises(AttemptEffectInDoubt):
        _call(c)
    assert path.read_bytes() == before and not path.with_name("terminal.json").exists()
    assert _row(c) is None and c.counters == {}


def test_wrong_existing_zero_reason_is_not_accepted_as_this_publication(failed):
    c = failed
    receipt = accounting.record_zero_gpu_outcome(c.idea, c.folder, 4,
        "rejected", "different_reason", phase="admission")
    with pytest.raises(AttemptEffectInDoubt):
        _call(c)
    path = c.folder / "_compute_receipts" / c.claim["attempt_id"] / "terminal.json"
    assert json.loads(path.read_text()) == receipt
    assert _row(c) is None and c.counters == {}
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
    assert (c.folder / "_attempt_effect.lock").exists()


def test_real_lifecycle_write_rejection_rolls_back_sql_and_retains_hold(failed):
    c = failed
    c.lake.conn.execute("CREATE TRIGGER reject_pre_failure BEFORE UPDATE ON idea_state "
                        "WHEN NEW.current_state='FAILED' BEGIN SELECT RAISE(IGNORE); END")
    c.lake.conn.commit()
    with pytest.raises(AttemptEffectInDoubt):
        _call(c)
    assert _row(c) is None and c.counters == {}
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
    assert (c.folder / "_attempt_effect.lock").exists()
    assert json.loads((c.folder / "metrics.json").read_text())["status"] == "FAILED"
    assert (c.folder / "_compute_receipts" / c.claim["attempt_id"] / "terminal.json").exists()
    with pytest.raises(AttemptEffectInDoubt):
        _call(c)
    assert c.events == []


def test_zero_receipt_partial_write_cannot_commit_failure(failed, monkeypatch):
    c = failed
    actual = accounting.record_zero_gpu_outcome
    def corrupt(*args, **kwargs):
        payload = actual(*args, **kwargs)
        path = c.folder / "_compute_receipts" / c.claim["attempt_id"] / "terminal.json"
        path.write_bytes(b'{"schema_version":')
        return payload
    monkeypatch.setattr(accounting, "record_zero_gpu_outcome", corrupt)
    with pytest.raises(AttemptEffectInDoubt):
        _call(c)
    assert _row(c) is None and c.counters == {}
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
    assert (c.folder / "_attempt_effect.lock").exists()


def test_duplicate_requires_exact_failed_fence_and_effect(failed):
    c = failed
    _call(c)
    c.counters.clear()
    assert c.lake.record_state_transition(c.idea, "FAILED", "QUEUED")
    before = _files(c.folder)
    with pytest.raises(AttemptEffectInDoubt):
        _call(c)
    assert c.counters == {}
    assert {key: value for key, value in _files(c.folder).items()
            if not key.startswith("_attempt_effect.lock/")} == before
    assert c.lake.get_fsm_state(c.idea) == "QUEUED"


def test_successful_source_never_authorizes_failure(cpu_pre_script):
    c = cpu_pre_script
    assert scheduler.claim(c.idea, c.results, 4, lake=c.lake)
    result = run_native_pre_script(c.idea, 4, c.results, c.cfg, c.lake,
        [sys.executable, "-c", "pass"], 5, dict(os.environ))
    assert result
    with pytest.raises(AttemptEffectInDoubt):
        report.report_pre_script_failure(c.lake, c.folder, result.attempt_ref, {}, c.cfg)
    assert _row(c) is None
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"


def test_changed_source_process_pid_refuses_before_any_report_effect(failed):
    c = failed
    source = current_attempt(c.lake.conn, c.idea, "pre_script")
    binding = dict(source["binding"])
    binding["process_pid"] += 1
    c.lake.conn.execute("UPDATE execution_attempts SET binding_json=? WHERE attempt_id=?",
        (json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
         c.ref.attempt_id))
    c.lake.conn.commit()
    with pytest.raises(AttemptEffectInDoubt):
        _call(c)
    assert not (c.folder / "metrics.json").exists()
    assert not (c.folder / "_compute_receipts").exists()
    assert _row(c) is None and c.counters == {}


@pytest.mark.parametrize("kind", ["unclosed", "malformed_terminal"])
def test_nonstale_invalid_source_is_always_a_hold_not_ordinary_failure(failed, kind):
    c = failed
    if kind == "unclosed":
        c.lake.conn.execute("UPDATE execution_attempts SET state='RUNNING', terminal_json=NULL "
                            "WHERE attempt_id=?", (c.ref.attempt_id,))
    else:
        row = current_attempt(c.lake.conn, c.idea, "pre_script")
        terminal = dict(row["terminal"])
        del terminal["reason_code"]
        c.lake.conn.execute("UPDATE execution_attempts SET terminal_json=? WHERE attempt_id=?",
            (json.dumps(terminal, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
             c.ref.attempt_id))
    c.lake.conn.commit()
    with pytest.raises(TerminationUnconfirmed):
        _call(c)
    assert c.counters == {} and _row(c) is None
    assert not (c.folder / "metrics.json").exists()
    assert not (c.folder / "_compute_receipts").exists()
