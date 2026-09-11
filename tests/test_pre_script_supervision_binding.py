"""C2d2 new CPU-action mechanisms: real SQL, explicit OS-protocol doubles.

No missing-API failures count as baseline bugs; actual process trees are tested
separately. These doubles grant only declared test protocol facts, not OS proof.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import AttemptRef, create_attempt, current_attempt
from orze.engine import native_pre_script as native, pre_script_supervision as proof, process
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.execution_authority import execution_transaction
from orze.engine.failure import _reset_idea_for_retry
from orze.engine.scheduler import claim
from orze.idea_lake import IdeaLake
from supervision_fixture import SimulatedSupervisedProcess
from test_evaluation_supervision_binding import case as evaluation_case


@pytest.fixture
def action(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    cfg = {"results_dir": str(results), "idea_lake_db": str(tmp_path / "lake.db")}
    lake = IdeaLake(cfg["idea_lake_db"])
    idea = "idea-pre-proof"
    lake.insert(idea, "CPU hook", "seed: 1\n", "", status="queued")
    assert claim(idea, results, 4, lake)
    a = SimpleNamespace(lake=lake, cfg=cfg, results=results, folder=results / idea,
                        idea=idea, code=0, prepared=[], args=[])

    def prepare(cmd, *, identity, **kwargs):
        child = SimpleNamespace(pid=500001 + len(a.prepared), returncode=a.code)
        child.poll = lambda: child.returncode
        child.wait = lambda timeout=None: child.returncode
        supervised = SimulatedSupervisedProcess(child, identity, cmd)
        a.prepared.append(supervised)
        a.args.append((list(cmd), deepcopy(kwargs)))
        return supervised

    monkeypatch.setattr(process, "prepare_supervised", prepare, raising=False)
    a.run = lambda **extra: native.run_native_pre_script(a.idea, 4, results, cfg, lake,
        extra.get("cmd", ["python", "pre.py"]), extra.get("timeout", 10),
        extra.get("env", {"LABEL": "private-test-value"}))
    try:
        yield a
    finally:
        lake.close()


@pytest.mark.parametrize("code", [0, 7])
def test_same_claim_returns_exact_once_only_cpu_result_without_gpu_accounting(action, code):
    a = action
    a.code = code
    claim_before = (a.folder / "claim.json").read_bytes()
    result = a.run()
    row = current_attempt(a.lake.conn, a.idea, "pre_script")
    before = deepcopy(row)
    repeated = a.run()
    assert isinstance(result, native.PreScriptResult)
    assert bool(result) is (code == 0)
    assert repeated == result
    assert result.attempt_ref.phase == "pre_script"
    assert result.attempt_ref.attempt_id != json.loads(claim_before)["attempt_id"]
    assert len(a.prepared) == 1
    assert current_attempt(a.lake.conn, a.idea, "pre_script") == before
    assert (a.folder / "claim.json").read_bytes() == claim_before
    assert a.lake.get_fsm_state(a.idea) == "CLAIMED"
    assert not (a.folder / "_compute_receipts").exists()
    assert not (a.folder / "metrics.json").exists()
    assert "private-test-value" not in json.dumps(row)
    assert all(a.args[0][1]["env"][key] == "" for key in (
        "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    if code == 0:
        native.require_launch_ready(a.lake, a.folder, a.cfg)
    else:
        with pytest.raises(native.PreScriptHOLD):
            native.require_launch_ready(a.lake, a.folder, a.cfg)


def test_new_claim_allows_new_action_without_reusing_training_attempt_id(action):
    a = action
    a.code = 7
    old = a.run()
    _reset_idea_for_retry(a.folder)
    new_claim = json.loads((a.folder / "claim.json").read_text())["attempt_id"]
    a.code = 0
    new = a.run()
    assert not old and new
    assert new.attempt_ref.generation == old.attempt_ref.generation + 1
    assert new.attempt_ref.attempt_id not in (old.attempt_ref.attempt_id, new_claim)
    assert len(a.prepared) == 2


def test_same_claim_changed_command_cannot_replay_cached_action(action):
    a = action
    a.run()
    before = current_attempt(a.lake.conn, a.idea, "pre_script")
    with pytest.raises(native.PreScriptHOLD):
        a.run(cmd=["python", "replacement.py"])
    assert len(a.prepared) == 1
    assert current_attempt(a.lake.conn, a.idea, "pre_script") == before


@pytest.mark.parametrize("phase", ["training", "evaluation"])
def test_another_phase_launch_intent_blocks_pre_script_before_prepare(action, phase):
    a = action
    with execution_transaction(a.lake, a.folder) as tx:
        ref = create_attempt(tx.conn, a.idea, phase, "other-active", {"source": "fixture"})
        tx.watch_attempt(ref)
    with pytest.raises(native.PreScriptHOLD):
        a.run()
    assert a.prepared == []
    assert current_attempt(a.lake.conn, a.idea, "pre_script") is None


def test_changed_sql_terminal_cannot_return_cached_success(action):
    a = action
    a.code = 7
    a.run()
    row = current_attempt(a.lake.conn, a.idea, "pre_script")
    row["terminal"]["outcome"] = "completed"
    a.lake.conn.execute("UPDATE execution_attempts SET terminal_json=? WHERE attempt_id=?",
        (json.dumps(row["terminal"], ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")), row["attempt_id"]))
    a.lake.conn.commit()
    with pytest.raises(native.PreScriptHOLD):
        a.run()
    assert len(a.prepared) == 1


@pytest.mark.parametrize("fault", ["wrong_phase", "missing_closure", "stop_zero"])
def test_pre_script_needs_its_own_closed_protocol_and_stop_is_not_success(tmp_path, fault):
    handle, row, folder = evaluation_case(tmp_path)
    old = handle.attempt_ref
    handle.attempt_ref = AttemptRef(old.task_id, "pre_script", old.attempt_id, old.generation)
    handle.process._test_binding["identity"]["attempt_ref"] = asdict(handle.attempt_ref)
    handle.process._test_closure["binding"] = deepcopy(handle.process._test_binding)
    row["binding"]["supervision"] = deepcopy(handle.process._test_binding)
    if fault == "wrong_phase":
        handle.attempt_ref = old
    elif fault == "missing_closure":
        handle.process._test_closure = None
    else:
        handle.process._test_closure["stop_requested"] = True
        closure = proof.require_closed(handle, row, folder, 0)
        assert closure["worker_returncode"] == 0
        assert proof.failure_override(closure, ("completed", "old", ""))[0] == "failed"
        return
    with pytest.raises(AttemptEffectBusy):
        proof.require_closed(handle, row, folder, 0)
