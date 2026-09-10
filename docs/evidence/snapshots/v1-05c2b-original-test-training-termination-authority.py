"""V1-02D1 public training-stop behavior, frozen before production changes.

The OS boundary models a dead process leader with unresolved descendants:
an integer poll result is not proof that the execution was fully reaped.
"""

import json
import time
from types import SimpleNamespace

import pytest

from orze.engine import launcher
from orze.engine.accounting import record_compute_start
from orze.engine.process import TrainingProcess
from orze.engine.scheduler import claim
from orze.idea_lake import IdeaLake


class ProcessBoundary:
    pid = 934771

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def running_case(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "idea_lake.db"))
    idea_id = "idea-stop-authority"
    lake.insert(idea_id, "Synthetic stop authority", "{}", "", status="queued")
    assert claim(idea_id, results, 0, lake=lake)
    assert lake.record_state_transition(idea_id, "CLAIMED", "IN_PROGRESS")
    idea_dir = results / idea_id
    metrics = idea_dir / "metrics.json"
    metrics.write_text('{"status":"IN_PROGRESS","samples_done":0}\n')
    log = idea_dir / "train_output.log"
    log.write_text("")
    proc = ProcessBoundary()
    attempt_id = json.loads((idea_dir / "claim.json").read_text())["attempt_id"]
    tp = TrainingProcess(
        idea_id=idea_id, gpu=0, process=proc, start_time=time.time() - 10,
        log_path=log, timeout=3600, attempt_id=attempt_id,
    )
    record_compute_start(tp, idea_dir)
    monkeypatch.setattr(launcher, "notify", lambda *args, **kwargs: None)
    cfg = {"stall_minutes": 0, "resume": {"enabled": False},
           "sops": {"failure_feedback": False}, "max_fix_attempts": 0}
    case = SimpleNamespace(
        results=results, lake=lake, idea_id=idea_id, idea_dir=idea_dir,
        metrics=metrics, metrics_before=metrics.read_bytes(), proc=proc, tp=tp,
        cfg=cfg, active={0: tp}, failures={}, fixes={},
    )
    yield case
    lake.close()


def _arm(case, reason):
    if reason == "timeout":
        case.tp.timeout = 1
    elif reason == "stall":
        case.cfg["stall_minutes"] = 1
        case.tp._stall_since = time.time() - 121
        case.tp._last_samples_done = 0
    else:
        (case.idea_dir / ".kill").write_text("operator request\n")


def _reaper(monkeypatch, case, confirmed):
    calls = []

    def reap(proc, idea_id, **kwargs):
        assert proc is case.proc
        assert idea_id == case.idea_id
        calls.append(idea_id)
        proc.returncode = -15
        return confirmed

    monkeypatch.setattr(launcher, "_terminate_and_reap", reap)
    return calls


def _poll(case):
    return launcher.check_active(
        case.active, case.results, case.cfg, case.failures,
        case.fixes, lake=case.lake,
    )


def _assert_held(case):
    assert list(case.idea_dir.glob("_compute_receipts/*/terminal.json")) == []
    assert not (case.idea_dir / "interruption.json").exists()
    assert case.metrics.read_bytes() == case.metrics_before
    assert case.lake.get_fsm_state(case.idea_id) == "IN_PROGRESS"
    assert case.lake.get(case.idea_id)["status"] == "running"
    assert case.active == {0: case.tp}
    assert case.failures == {}
    assert not (case.idea_dir / "failure_analysis.json").exists()


@pytest.mark.parametrize("reason", ["timeout", "stall", "admin_kill"])
def test_unconfirmed_stop_cannot_publish_failure_or_release_active(
        running_case, monkeypatch, reason):
    case = running_case
    _arm(case, reason)
    calls = _reaper(monkeypatch, case, False)
    finished = _poll(case)
    assert calls == [case.idea_id]
    _assert_held(case)
    assert finished == []
    if reason == "admin_kill":
        assert (case.idea_dir / ".kill").exists()


@pytest.mark.parametrize("leader_exit", [0, -15])
def test_later_integer_exit_does_not_bypass_an_unconfirmed_stop(
        running_case, monkeypatch, leader_exit):
    case = running_case
    _arm(case, "timeout")
    calls = _reaper(monkeypatch, case, False)
    first_finished = _poll(case)
    case.proc.returncode = leader_exit
    second_finished = _poll(case)
    assert calls  # A real stop was attempted before the second poll.
    _assert_held(case)
    assert first_finished == second_finished == []


def test_unconfirmed_stop_never_spends_executor_or_relaunch_budget(
        running_case, monkeypatch):
    case = running_case
    _arm(case, "timeout")
    _reaper(monkeypatch, case, False)
    external_calls = []

    def executor(*args, **kwargs):
        external_calls.append("executor")
        return True

    def relaunch(*args, **kwargs):
        external_calls.append("launch")
        return case.tp

    monkeypatch.setattr("orze.engine.failure._try_executor_fix", executor)
    monkeypatch.setattr(launcher, "launch", relaunch)
    assert _poll(case) == []
    assert external_calls == []
    _assert_held(case)


@pytest.mark.parametrize("reason", ["timeout", "stall", "admin_kill"])
def test_confirmed_stop_retains_normal_terminal_behavior(
        running_case, monkeypatch, reason):
    case = running_case
    _arm(case, reason)
    calls = _reaper(monkeypatch, case, True)
    assert _poll(case) == [(case.idea_id, 0)]
    assert calls == [case.idea_id]
    assert case.active == {}
    assert case.lake.get_fsm_state(case.idea_id) == "FAILED"
    assert json.loads(case.metrics.read_text())["status"] == "FAILED"
    paths = list(case.idea_dir.glob("_compute_receipts/*/terminal.json"))
    assert len(paths) == 1
    assert json.loads(paths[0].read_text())["outcome"] == "interrupted"
    assert (case.idea_dir / "interruption.json").exists()


@pytest.mark.parametrize("confirmed", [False, True])
def test_launch_initialization_failure_needs_confirmed_cleanup_before_terminal(
        tmp_path, monkeypatch, confirmed):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "idea_lake.db"))
    idea_id = "idea-init-stop"
    lake.insert(idea_id, "Initialization failure", "{}", "", status="queued")
    assert claim(idea_id, results, 0, lake=lake)
    idea_dir = results / idea_id
    train = tmp_path / "train.py"
    train.write_text("# Synthetic training boundary\n")
    base = tmp_path / "base.yaml"
    base.write_text("{}\n")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("")
    cfg = {"train_script": str(train), "base_config": str(base),
           "ideas_file": str(ideas), "python": "python3"}
    proc = ProcessBoundary()
    popen_calls = []
    reaper_calls = []

    def popen(*args, **kwargs):
        popen_calls.append(True)
        return proc

    def failed_identity(pid):
        assert pid == proc.pid
        raise OSError("synthetic post-Popen identity failure")

    def reap(process, stopped_id, **kwargs):
        assert process is proc and stopped_id == idea_id
        reaper_calls.append(True)
        proc.returncode = -15
        return confirmed

    monkeypatch.setattr(launcher, "_verify_gpu_free", lambda *a, **k: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher, "capture_process_identity", failed_identity)
    monkeypatch.setattr(launcher, "_terminate_and_reap", reap)
    try:
        with pytest.raises(Exception):
            launcher.launch(idea_id, 0, results, cfg, lake=lake)
        assert popen_calls == reaper_calls == [True]
        starts = list(idea_dir.glob("_compute_receipts/*/start.json"))
        terminals = list(idea_dir.glob("_compute_receipts/*/terminal.json"))
        assert len(starts) == 1
        if confirmed:
            assert len(terminals) == 1
            assert json.loads(terminals[0].read_text())["outcome"] == "failed"
        else:
            assert terminals == []
        assert lake.get_fsm_state(idea_id) == "CLAIMED"
        assert not (idea_dir / "metrics.json").exists()
        assert (idea_dir / "claim.json").exists()
    finally:
        lake.close()
