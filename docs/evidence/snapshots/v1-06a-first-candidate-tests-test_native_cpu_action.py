"""V1-06A new CPU action mechanisms, not historical missing-API reds.

Real SQLite, sealed input, worker/supervisor and budget. No GPU/provider/host
discovery. Finally only closes handles returned by our actual prepare call.
"""
import json
from pathlib import Path
import sys
import time

import pytest
import yaml

from orze.core import cpu_action_budget as budget
from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.engine import native_cpu_action as native
from orze.engine.execution_authority import lifecycle_fence
from orze.engine.idea_ingress import _proposal_fields
from orze.engine.scheduler import claim
from orze.idea_lake import IdeaLake


@pytest.fixture
def context(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    scope = budget.initialize(lake, results, {
        "version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 20})
    handles = []
    real_prepare = native.prepare_supervised

    def prepare(*args, **kwargs):
        process = real_prepare(*args, **kwargs)
        handles.append(process)
        return process

    monkeypatch.setattr(native, "prepare_supervised", prepare)
    cfg = {"_project_root": str(tmp_path), "_orze_dir": str(tmp_path / ".orze")}

    def create(code, *, outputs=None, timeout=2):
        action = {"version": 1, "adapter": "command", "purpose": "test real CPU action",
                  "inputs": {"message": "sealed value"}, "command": [sys.executable, "-c", code],
                  "timeout_seconds": timeout, "outputs": outputs or {}}
        result = lake.insert("idea-cpu", "CPU action", yaml.safe_dump({
            "kind": "native_cpu_action", "action": action}), "", status="queued",
            kind="native_cpu_action", if_absent=True)
        assert result["status"] == "inserted"
        permit = budget.reserve(lake, scope, "idea-cpu", timeout)
        assert permit is not None
        assert claim("idea-cpu", results, None, lake, resource="cpu")
        return action, permit

    yield lake, results, scope, cfg, create, handles
    for process in handles:
        if process.poll() is None:
            process.stop(timeout=0.2)
        assert type(process.poll()) is int
    lake.close()


def _finish(handle, results, cfg, lake, permit):
    until = time.monotonic() + 6
    while time.monotonic() < until:
        terminal = native.harvest(handle, results, cfg, lake=lake, permit=permit)
        if terminal is not None:
            return terminal
        time.sleep(0.01)
    pytest.fail("owned CPU action did not reach closure")


def test_cpu_kind_admission_and_claim_do_not_create_training_stages(context):
    lake, results, scope, cfg, create, handles = context
    action, permit = create("pass")
    state = dict(lake.conn.execute("SELECT * FROM idea_state WHERE idea_id='idea-cpu'").fetchone())
    assert state["sop_type"] == "action"
    assert lifecycle_fence(lake, "idea-cpu", "action")["phase_state"] == "PENDING"
    assert [row[0] for row in lake.conn.execute("SELECT stage FROM idea_stage_state")] == ["action"]
    assert json.loads((results / "idea-cpu" / "claim.json").read_text())["gpu"] is None
    with pytest.raises(ValueError):
        lake.record_state_transition("idea-cpu", "CLAIMED", "IN_PROGRESS", sop_type="training")
    assert not handles


def test_kind_aware_dedup_preserves_cpu_and_legacy_domains(tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    try:
        for idea, kind in (("idea-train", "train"), ("idea-cpu", "native_cpu_action")):
            assert lake.insert(idea, idea, "seed: 1", "", status="queued", kind=kind,
                               if_absent=True)["status"] == "inserted"
        assert lake.insert("idea-cpu2", "copy", "seed: 1", "", status="queued",
            kind="native_cpu_action", if_absent=True)["status"] == "config_duplicate"
    finally:
        lake.close()


def test_proposal_kind_sources_must_agree():
    assert _proposal_fields({"raw": "**Kind**: native_cpu_action", "config": {}})["kind"] == "native_cpu_action"
    assert _proposal_fields({"config": {"kind": "native_cpu_action"}})["kind"] == "native_cpu_action"
    with pytest.raises(ValueError, match="kind_conflict"):
        _proposal_fields({"raw": "**Kind**: train", "config": {"kind": "native_cpu_action"}})


@pytest.mark.parametrize("code,success", [(0, True), (7, False)])
def test_actual_zero_artifact_action_closes_and_settles_without_gpu_accounting(context, code, success):
    lake, results, scope, cfg, create, handles = context
    action, permit = create("raise SystemExit(" + str(code) + ")")
    handle = native.launch("idea-cpu", results, cfg, lake=lake, action=action, permit=permit, admission=lambda: None)
    terminal = _finish(handle, results, cfg, lake, permit)
    assert terminal["outcome"] == ("completed" if success else "failed")
    assert terminal["return_code"] == code
    assert terminal["artifact_ids"] == terminal["observation_ids"] == []
    assert terminal["process_tree"]["binding"]["worker"]["pid"] == handle.process.pid
    assert handle.process.pid != handle.process.supervisor_pid
    assert lake.get_fsm_state("idea-cpu") == ("COMPLETE" if success else "FAILED")
    assert {r[0] for r in lake.conn.execute("SELECT stage FROM idea_stage_state")} == {"action"}
    assert current_attempt(lake.conn, "idea-cpu", "training") is None
    assert not (results / "idea-cpu" / "_compute_receipts").exists()
    assert budget.snapshot(lake, scope)["active_reservations"] == 0
    assert budget.snapshot(lake, scope)["reserved_wall_seconds"] == 2


def test_real_worker_reads_sealed_inputs_and_publishes_owned_artifact(context):
    lake, results, scope, cfg, create, handles = context
    code = """import os,json,fcntl
from pathlib import Path
fd=int(os.environ['ORZE_ACTION_INPUT_FD'])
data=json.loads(os.pread(fd,65536,0))
assert all(os.environ[k]=='' for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
assert fcntl.fcntl(fd,fcntl.F_GET_SEALS)&fcntl.F_SEAL_WRITE
Path('result.txt').write_text(data['message'])
"""
    action, permit = create(code, outputs={"result": {"path": "result.txt", "max_bytes": 100}})
    handle = native.launch("idea-cpu", results, cfg, lake=lake, action=action, permit=permit, admission=lambda: None)
    terminal = _finish(handle, results, cfg, lake, permit)
    records = artifacts_for_attempt(lake.conn, handle.attempt_ref)
    assert len(records) == len(terminal["artifact_ids"]) == 1
    assert records[0]["producer"]["phase"] == "action"
    assert Path(records[0]["path"]).read_text() == "sealed value"
    assert terminal["observation_ids"] == []


def test_leader_exit_does_not_publish_before_real_escaped_writer_closes(context):
    lake, results, scope, cfg, create, handles = context
    code = """import os,time
from pathlib import Path
if os.fork()==0:
 os.setsid()
 Path('writer-alive').write_text('ready')
 time.sleep(.35)
 Path('result.txt').write_text('closed')
 os._exit(0)
os._exit(0)
"""
    action, permit = create(code, outputs={"result": {"path": "result.txt", "max_bytes": 100}})
    handle = native.launch("idea-cpu", results, cfg, lake=lake, action=action, permit=permit, admission=lambda: None)
    work = results / "idea-cpu" / "_action_attempts" / handle.attempt_id / "work"
    until = time.monotonic() + 2
    while not (work / "writer-alive").exists() and time.monotonic() < until:
        time.sleep(.005)
    assert (work / "writer-alive").exists()
    assert native.harvest(handle, results, cfg, lake=lake, permit=permit) is None
    assert current_attempt(lake.conn, "idea-cpu", "action")["state"] == "RUNNING"
    assert budget.snapshot(lake, scope)["free_slots"] == 0
    assert _finish(handle, results, cfg, lake, permit)["process_tree"]["reaped_children"] >= 2


def test_real_stop_zero_is_interrupted_and_settles_after_policy_stop(context):
    lake, results, scope, cfg, create, handles = context
    code = """import signal,time
from pathlib import Path
signal.signal(signal.SIGTERM,lambda *_:exit(0))
Path('ready').write_text('ready')
while True: time.sleep(.01)
"""
    action, permit = create(code)
    handle = native.launch("idea-cpu", results, cfg, lake=lake, action=action, permit=permit, admission=lambda: None)
    work = results / "idea-cpu" / "_action_attempts" / handle.attempt_id / "work"
    until = time.monotonic() + 2
    while not (work / "ready").exists() and time.monotonic() < until:
        time.sleep(.005)
    assert (work / "ready").exists()
    budget.record_decision(lake, scope, {"kind": "Stop", "reason": "test", "wakeup": None})
    terminal = native.stop(handle, results, cfg, lake=lake, permit=permit)
    assert terminal["return_code"] == 0
    assert terminal["outcome"] == "interrupted"
    assert terminal["process_tree"]["stop_requested"] is True
    assert terminal["artifact_ids"] == []
    assert budget.snapshot(lake, scope)["free_slots"] == 1


def test_ready_admission_refusal_preserves_owner_without_go_or_budget_refund(context):
    lake, results, scope, cfg, create, handles = context
    action, permit = create("from pathlib import Path; Path('executed').write_text('bad')")

    def admission():
        if handles:
            raise RuntimeError("controlled READY admission refusal")

    with pytest.raises(native.CPUActionHOLD) as caught:
        native.launch("idea-cpu", results, cfg, lake=lake, action=action, permit=permit, admission=admission)
    handle = caught.value.cpu_action_handle
    assert handle is not None and handle.process is handles[0]
    assert handle.process.poll() == 0
    assert handle.process.closure_receipt()["stop_requested"] is True
    assert not list(results.rglob("executed"))
    assert current_attempt(lake.conn, "idea-cpu", "action")["state"] == "RUNNING"
    assert budget.snapshot(lake, scope)["free_slots"] == 0
