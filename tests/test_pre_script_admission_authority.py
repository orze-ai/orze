"""New public admission gates: real claim/SQLite, explicit protocol doubles.

These are new contract checks, not alleged failures in the preceding commit.
No GPU lease, training program, provider, or host process discovery is used.
"""
import json
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import launcher, native_posthoc, native_pre_script, posthoc_attempts
from orze.engine import process, training_attempts
from orze.engine.supervised_process import SupervisionUncertain
from orze.engine.termination_hold import TerminationUnconfirmed
from test_pre_script_supervision_binding import action


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), True, 0, -1, "60", 10 ** 400])
def test_public_native_budget_rejected_before_any_intent_or_prepare(action, budget):
    a = action
    a.cfg.update(pre_script="prepare.py", pre_timeout=budget)
    claim = (a.folder / "claim.json").read_bytes()
    with pytest.raises(TerminationUnconfirmed):
        process.run_pre_script(a.idea, 4, a.cfg, a.results, lake=a.lake)
    assert a.prepared == []
    assert current_attempt(a.lake.conn, a.idea, "pre_script") is None
    assert (a.folder / "claim.json").read_bytes() == claim
    assert not (a.folder / "_compute_receipts").exists()


def _pending(a, monkeypatch):
    def unknown(*args, **kwargs):
        raise SupervisionUncertain("explicit test-only unknown READY handoff")
    monkeypatch.setattr(process, "prepare_supervised", unknown)
    with pytest.raises(TerminationUnconfirmed):
        a.run()
    row = current_attempt(a.lake.conn, a.idea, "pre_script")
    assert row["state"] == "LAUNCHING"
    return row


@pytest.mark.parametrize("entry", ["removed_script", "missing_lake", "erased_routes",
    "launch", "training_begin", "posthoc_launch", "posthoc_begin"])
def test_pending_cpu_action_cannot_be_bypassed_by_configuration_or_direct_launch(
        action, monkeypatch, entry):
    a = action
    row = _pending(a, monkeypatch)
    forbidden = []
    monkeypatch.setattr(launcher, "_assert_campaign_evidence_authorized",
                        lambda *args: forbidden.append("campaign"))
    claim = json.loads((a.folder / "claim.json").read_bytes())
    handle = SimpleNamespace(idea_id=a.idea, gpu=4, process=None,
                             attempt_id=claim["attempt_id"])
    if entry == "erased_routes":
        (a.folder / "_execution_catalog.json").unlink()
        claim.pop("lifecycle_db")
        (a.folder / "claim.json").write_text(json.dumps(claim))
    with pytest.raises(TerminationUnconfirmed):
        if entry == "removed_script":
            process.run_pre_script(a.idea, 4, a.cfg, a.results, lake=a.lake)
        elif entry in ("missing_lake", "erased_routes"):
            process.run_pre_script(a.idea, 4, a.cfg, a.results)
        elif entry == "launch":
            launcher.launch(a.idea, 4, a.results, a.cfg, lake=a.lake)
        elif entry == "training_begin":
            training_attempts.begin(a.lake, handle, a.folder, cfg=a.cfg)
        elif entry == "posthoc_launch":
            native_posthoc.launch(a.idea, 4, a.results, a.cfg, kind="test",
                                  idea_cfg_path=a.folder / "idea_config.yaml", lake=a.lake)
        else:
            posthoc_attempts.begin(a.lake, handle, a.folder, launch_inputs={})
    assert current_attempt(a.lake.conn, a.idea, "pre_script") == row
    assert current_attempt(a.lake.conn, a.idea, "training") is None
    assert current_attempt(a.lake.conn, a.idea, "posthoc") is None
    assert forbidden == []
    assert a.prepared == []
    assert not (a.folder / "_compute_receipts").exists()
    assert not (a.folder / "metrics.json").exists()


@pytest.mark.parametrize("entry", ["launch", "training_begin", "posthoc_begin"])
def test_confirmed_failure_does_not_authorize_same_claim_training(action, entry):
    a = action
    a.code = 7
    result = a.run()
    assert not result
    claim = json.loads((a.folder / "claim.json").read_bytes())
    handle = SimpleNamespace(idea_id=a.idea, gpu=4, process=None,
                             attempt_id=claim["attempt_id"])
    with pytest.raises(TerminationUnconfirmed):
        if entry == "launch":
            launcher.launch(a.idea, 4, a.results, a.cfg, lake=a.lake)
        elif entry == "training_begin":
            training_attempts.begin(a.lake, handle, a.folder, cfg=a.cfg)
        else:
            posthoc_attempts.begin(a.lake, handle, a.folder, launch_inputs={})
    assert current_attempt(a.lake.conn, a.idea, "training") is None
    assert current_attempt(a.lake.conn, a.idea, "posthoc") is None
    assert len(a.prepared) == 1
    assert not (a.folder / "_compute_receipts").exists()


def test_closed_setup_allows_distinct_real_training_intent_without_faking_allocation(action):
    a = action
    result = a.run()
    pre = current_attempt(a.lake.conn, a.idea, "pre_script")
    claim = json.loads((a.folder / "claim.json").read_bytes())
    handle = SimpleNamespace(idea_id=a.idea, gpu=4, process=None,
                             attempt_id=claim["attempt_id"])
    ref = training_attempts.begin(a.lake, handle, a.folder, cfg=a.cfg)
    assert result
    assert ref.phase == "training" and result.attempt_ref.phase == "pre_script"
    assert ref.attempt_id == claim["attempt_id"] != result.attempt_ref.attempt_id
    assert current_attempt(a.lake.conn, a.idea, "training")["state"] == "LAUNCHING"
    assert current_attempt(a.lake.conn, a.idea, "pre_script") == pre
    assert a.lake.get_fsm_state(a.idea) == "CLAIMED"
    assert not (a.folder / "_compute_receipts").exists()


@pytest.mark.parametrize("changed", [{"timeout": 11}, {"env": {"LABEL": "changed"}}])
def test_same_claim_changed_launch_inputs_are_not_silently_reused(action, changed):
    a = action
    a.run()
    before = current_attempt(a.lake.conn, a.idea, "pre_script")
    with pytest.raises(TerminationUnconfirmed):
        a.run(**changed)
    assert len(a.prepared) == 1
    assert current_attempt(a.lake.conn, a.idea, "pre_script") == before


@pytest.mark.parametrize("boundary", ["before_go", "after_wait"])
def test_stored_worker_pid_remains_fenced_at_go_and_terminal(action, monkeypatch, boundary):
    a = action
    prepare = process.prepare_supervised
    def change_pid():
        row = current_attempt(a.lake.conn, a.idea, "pre_script")
        binding = dict(row["binding"], process_pid=row["binding"]["process_pid"] + 1)
        a.lake.conn.execute("UPDATE execution_attempts SET binding_json=? WHERE attempt_id=?",
            (json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
             row["attempt_id"]))
        a.lake.conn.commit()
    if boundary == "before_go":
        admission = native_pre_script._admission
        def gate(*args):
            admission(*args)
            row = current_attempt(a.lake.conn, a.idea, "pre_script")
            if row is not None and row["state"] == "RUNNING":
                change_pid()
        monkeypatch.setattr(native_pre_script, "_admission", gate)
    else:
        def prepared(*args, **kwargs):
            child = prepare(*args, **kwargs)
            wait = child.wait
            def changed_wait(*args, **kwargs):
                ret = wait(*args, **kwargs)
                change_pid()
                return ret
            child.wait = changed_wait
            return child
        monkeypatch.setattr(process, "prepare_supervised", prepared)
    with pytest.raises(TerminationUnconfirmed):
        a.run()
    assert current_attempt(a.lake.conn, a.idea, "pre_script")["state"] == "RUNNING"
    assert not (a.folder / "_execution_effects").exists()
    assert not (a.folder / "_compute_receipts").exists()
