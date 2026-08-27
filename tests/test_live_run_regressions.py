"""Regressions discovered by an isolated real-project run."""

import json
import socket
import time
from pathlib import Path

import pytest

import orze
from orze.engine.orchestrator import Orze
from orze.engine.launcher import _write_failure, check_active, launch
from orze.engine.process import TrainingProcess
from orze.engine.scheduler import claim, get_unclaimed
from orze.engine.upgrade import UpgradeManager
from orze.core.config import _validate_config
from orze.idea_lake import IdeaLake
from orze.reporting.state import load_state


def test_source_checkout_version_wins_over_installed_metadata():
    pyproject = Path(orze.__file__).resolve().parents[2] / "pyproject.toml"
    assert pyproject.exists()
    assert orze.__version__ == "4.6.1"


def test_missing_sentinel_does_not_consume_pypi_advisory(tmp_path):
    mgr = UpgradeManager(tmp_path, {"auto_upgrade": True})
    mgr.pending = "99.0.0"  # an advisory populated by check_pypi()
    assert mgr.check_sentinel() is False
    assert mgr.pending == "99.0.0"

    orchestrator = Orze.__new__(Orze)
    orchestrator._upgrade_mgr = mgr
    orchestrator._pending_upgrade = "99.0.0"
    called = []
    orchestrator._do_auto_upgrade = lambda: called.append(True)
    orchestrator._check_upgrade_sentinel()
    assert called == []


def test_terminal_transition_atomically_updates_queue_status(tmp_path):
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert("idea-0001", "test", "{}", "", status="queued")
    assert lake.record_state_transition("idea-0001", "QUEUED", "CLAIMED")
    assert lake.record_state_transition("idea-0001", "CLAIMED", "IN_PROGRESS")
    assert lake.record_state_transition("idea-0001", "IN_PROGRESS", "COMPLETE")
    assert lake.get("idea-0001")["status"] == "completed"
    assert lake.get_fsm_state("idea-0001") == "COMPLETE"
    lake.close()


def test_corrupt_orchestrator_state_fails_closed(tmp_path):
    state_path = tmp_path / f".orze_state_{socket.gethostname()}.json"
    state_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Cannot load orchestrator state"):
        load_state(tmp_path)


def test_config_check_rejects_report_column_shorthand(tmp_path, monkeypatch):
    train = tmp_path / "train.py"
    train.write_text("# idea_config.yaml\n")
    monkeypatch.chdir(tmp_path)
    errors, _ = _validate_config({
        "train_script": "train.py",
        "report": {"columns": ["r2"]},
    })
    assert any("report.columns[0]" in error for error in errors)


@pytest.mark.parametrize(
    "target", [True, "1.0", float("nan"), float("inf"), float("-inf")]
)
def test_config_check_rejects_nonfinite_or_nonnumeric_target(
        tmp_path, monkeypatch, target):
    train = tmp_path / "train.py"
    train.write_text("# idea_config.yaml\n")
    monkeypatch.chdir(tmp_path)
    errors, _ = _validate_config({
        "train_script": "train.py",
        "report": {"target": target},
    })
    assert "report.target: must be a finite number or null" in errors


@pytest.mark.parametrize("target", [None, 0, -1.5, 2.0])
def test_config_check_accepts_finite_or_absent_target(
        tmp_path, monkeypatch, target):
    train = tmp_path / "train.py"
    train.write_text("# idea_config.yaml\n")
    monkeypatch.chdir(tmp_path)
    errors, _ = _validate_config({
        "train_script": "train.py",
        "report": {"target": target},
    })
    assert not any(error.startswith("report.target:") for error in errors)


def test_config_check_reports_boolean_notifications(tmp_path, monkeypatch):
    train = tmp_path / "train.py"
    train.write_text("# idea_config.yaml\n")
    monkeypatch.chdir(tmp_path)
    errors, _ = _validate_config({
        "train_script": "train.py",
        "notifications": False,
    })
    assert "notifications: must be a mapping" in errors


def test_malformed_metrics_cannot_complete_the_lifecycle(tmp_path):
    class FinishedProcess:
        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    results = tmp_path / "results"
    idea_dir = results / "idea-0001"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text("{broken", encoding="utf-8")
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert("idea-0001", "test", "{}", "", status="queued")
    assert lake.record_state_transition("idea-0001", "QUEUED", "CLAIMED")
    assert lake.record_state_transition("idea-0001", "CLAIMED", "IN_PROGRESS")
    tp = TrainingProcess(
        idea_id="idea-0001", gpu=0, process=FinishedProcess(),
        start_time=time.time(), log_path=idea_dir / "train_output.log",
        timeout=60,
    )

    assert check_active(
        {0: tp}, results, {"sops": {"failure_feedback": False}}, {}, lake=lake,
    ) == [("idea-0001", 0)]

    assert lake.get_fsm_state("idea-0001") == "FAILED"
    assert lake.get("idea-0001")["status"] == "failed"
    assert json.loads((idea_dir / "metrics.json").read_text())["status"] == "FAILED"
    assert list(idea_dir.glob("metrics.invalid.*.json"))
    lake.close()


def test_vram_contention_releases_claim_and_audits_requeue(tmp_path):
    class FinishedProcess:
        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    results = tmp_path / "results"
    idea_dir = results / "idea-vram"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(json.dumps({
        "status": "FAILED",
        "error": "insufficient_vram: available=100 required=1000",
    }), encoding="utf-8")
    (idea_dir / "claim.json").write_text(json.dumps({
        "claimed_by": socket.gethostname(), "pid": 999999999, "gpu": 4,
    }), encoding="utf-8")
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert("idea-vram", "vram", "{}", "", status="queued")
    assert lake.record_state_transition("idea-vram", "QUEUED", "CLAIMED")
    assert lake.record_state_transition(
        "idea-vram", "CLAIMED", "IN_PROGRESS")
    tp = TrainingProcess(
        idea_id="idea-vram", gpu=4, process=FinishedProcess(),
        start_time=time.time(), log_path=idea_dir / "train_output.log",
        timeout=60,
    )

    assert check_active(
        {4: tp}, results, {"sops": {"failure_feedback": False}}, {}, lake=lake,
    ) == [("idea-vram", 4)]

    assert lake.get_fsm_state("idea-vram") == "QUEUED"
    assert lake.get("idea-vram")["status"] == "queued"
    assert not (idea_dir / "claim.json").exists()
    assert list(idea_dir.glob("claim.retry.*.json"))
    assert not (idea_dir / "metrics.json").exists()
    assert get_unclaimed({
        "idea-vram": {"priority": "medium", "config": {}}
    }, results, lake=lake) == ["idea-vram"]
    lake.close()


def test_training_completion_accepts_catch_up_winning_the_transition(tmp_path):
    class FinishedProcess:
        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    results = tmp_path / "results"
    idea_dir = results / "idea-0001"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED", "training_loss": 1.0}),
        encoding="utf-8",
    )
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert("idea-0001", "test", "{}", "", status="queued")
    assert lake.record_state_transition("idea-0001", "QUEUED", "CLAIMED")
    assert lake.record_state_transition("idea-0001", "CLAIMED", "IN_PROGRESS")
    assert lake.record_state_transition(
        "idea-0001", "IN_PROGRESS", "COMPLETE", "catch_up_training_completed"
    )
    tp = TrainingProcess(
        idea_id="idea-0001", gpu=0, process=FinishedProcess(),
        start_time=time.time(), log_path=idea_dir / "train_output.log",
        timeout=60,
    )

    assert check_active(
        {0: tp}, results, {"sops": {"failure_feedback": False}}, {}, lake=lake,
    ) == [("idea-0001", 0)]
    assert lake.get_fsm_state("idea-0001") == "COMPLETE"
    assert len(lake.get_fsm_history("idea-0001")) == 3
    lake.close()


def test_direct_launch_without_lake_does_not_require_claim(tmp_path, monkeypatch):
    class RunningProcess:
        pid = 12345

        def poll(self):
            return None

    results = tmp_path / "results"
    idea_dir = results / "idea-direct"
    idea_dir.mkdir(parents=True)
    train = tmp_path / "train.py"
    train.write_text("# test", encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text("{}", encoding="utf-8")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "orze.engine.launcher.subprocess.Popen",
        lambda *args, **kwargs: RunningProcess(),
    )

    tp = launch("idea-direct", 0, results, {
        "train_script": str(train),
        "base_config": str(base),
        "ideas_file": str(ideas),
        "python": "python3",
    })

    assert tp.idea_id == "idea-direct"
    assert not (idea_dir / "claim.json").exists()
    tp.close_log()


def test_scheduler_does_not_hide_an_idea_by_learning_rate(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    ideas = {
        "idea-valid": {
            "priority": "high",
            "config": {"lr": 2e-5},
        },
    }

    assert get_unclaimed(ideas, results) == ["idea-valid"]


def test_prelaunch_failure_closes_claimed_fsm_state(tmp_path):
    results = tmp_path / "results"
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert("idea-invalid", "invalid", "{}", "", status="queued")
    assert claim("idea-invalid", results, gpu=0, lake=lake)
    assert lake.get_fsm_state("idea-invalid") == "CLAIMED"

    _write_failure(
        results / "idea-invalid",
        "method_validator_rejected: invalid",
        lake=lake,
        idea_id="idea-invalid",
    )

    assert lake.get_fsm_state("idea-invalid") == "FAILED"
    assert lake.get("idea-invalid")["status"] == "failed"
    assert lake.get_fsm_history("idea-invalid")[-1]["to_state"] == "FAILED"
    lake.close()
