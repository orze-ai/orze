"""One-idea managed runs must fail closed before any GPU observation."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import orze.cli as cli
from orze.core import managed_run
from orze.core.managed_run import (
    ManagedRunError,
    prepare_managed_idea_run,
    verify_managed_idea_outcome,
)
from orze.engine.phases import OrzePhaseMixin
from orze.engine.gpu_slots import _query_all_gpu_usage
from orze.engine.orchestrator import Orze
from orze.hardware.gpu import _query_gpu_details


@pytest.fixture
def managed_case(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    cfg = {
        "results_dir": str(results),
        "idea_lake_db": str(tmp_path / "idea_lake.db"),
        "controller_runtime": {"contract_version": 1},
        "launcher": {"paused": False},
        "gpu_scheduling": {"allowed_gpus": [4, 5, 6, 7]},
        "research_policy": {},
    }
    monkeypatch.setattr(
        "orze.core.config._validate_config", lambda value: ([], []))
    monkeypatch.setattr(
        "orze.service.runtime_contract.require_controller_runtime_contract",
        lambda contract: {"contract_ok": True, "errors": []},
    )
    monkeypatch.setattr(
        "orze.reporting.evidence.authoritative_idea_lifecycle",
        lambda path, ids: ({
            ids[0]: {"state": "QUEUED", "family": "architecture"},
        }, "authoritative_lifecycle_loaded"),
    )
    monkeypatch.setattr(
        "orze.core.decision_batches.validate_idea_decision_admission",
        lambda results_dir, config, idea_id: None,
    )
    return results, cfg


def test_managed_run_authorizes_one_exact_queued_idea(managed_case):
    _, cfg = managed_case

    report = prepare_managed_idea_run(cfg, "idea-managed", 4)

    assert report == {
        "schema_version": 1,
        "authorized": True,
        "idea_id": "idea-managed",
        "gpu": 4,
        "lifecycle_state": "QUEUED",
        "approach_family": "architecture",
    }


def test_managed_run_requires_exact_runtime_pin(managed_case):
    _, cfg = managed_case
    cfg["controller_runtime"] = None

    with pytest.raises(
            ManagedRunError, match="controller_runtime_pin_required"):
        prepare_managed_idea_run(cfg, "idea-managed", 4)


@pytest.mark.parametrize("sentinel", [
    ".orze_disabled", ".orze_stop_all", ".orze_shutdown",
])
def test_managed_run_honors_stop_state_before_gpu_telemetry(
        managed_case, sentinel, monkeypatch):
    results, cfg = managed_case
    (results / sentinel).write_text("stop\n", encoding="utf-8")
    monkeypatch.setattr(
        "orze.engine.launcher._assert_gpu_authorized",
        lambda *args: pytest.fail("stop state must reject before GPU scope"),
    )

    with pytest.raises(ManagedRunError, match="blocked_by_sentinel"):
        prepare_managed_idea_run(cfg, "idea-managed", 4)


def test_managed_run_rejects_other_physical_gpu(managed_case):
    _, cfg = managed_case

    with pytest.raises(ManagedRunError, match="outside_managed_scope:0"):
        prepare_managed_idea_run(cfg, "idea-managed", 0)


def test_vram_query_restricts_nvidia_smi_to_requested_gpu():
    completed = SimpleNamespace(
        stdout="4, 1024, 81920\n", returncode=0)
    with patch("orze.engine.gpu_slots.subprocess.run",
               return_value=completed) as run:
        assert _query_all_gpu_usage([4]) == {4: (1024, 81920)}

    command = run.call_args.args[0]
    assert "--id=4" in command
    assert "--id=0" not in command


def test_detailed_telemetry_restricts_nvidia_smi_to_requested_gpus():
    completed = SimpleNamespace(
        stdout=("4, NVIDIA H100, 0, 81920, 0, 30\n"
                "7, NVIDIA H100, 0, 81920, 0, 31\n"),
        returncode=0,
    )
    with patch("orze.hardware.gpu.subprocess.run",
               return_value=completed) as run:
        rows = _query_gpu_details([7, 4])

    assert [row["index"] for row in rows] == [4, 7]
    assert "--id=4,7" in run.call_args.args[0]


def test_managed_pid_receipt_never_overwrites_daemon_pid(tmp_path):
    daemon_pid = tmp_path / ".orze.pid"
    host_pid = tmp_path / ".orze.pid.test-host"
    daemon_pid.write_text("123", encoding="utf-8")
    host_pid.write_text("123", encoding="utf-8")
    runner = Orze.__new__(Orze)
    runner.cfg = {"_managed_idea_id": "idea-managed"}
    runner.results_dir = tmp_path

    runner._write_pid_file()

    assert runner._pid_file.name.startswith(
        ".orze.managed.idea-managed.")
    assert daemon_pid.read_text(encoding="utf-8") == "123"
    assert host_pid.read_text(encoding="utf-8") == "123"
    runner._remove_pid_file()
    assert not runner._pid_file.exists()
    assert daemon_pid.read_text(encoding="utf-8") == "123"
    assert host_pid.read_text(encoding="utf-8") == "123"


def test_managed_run_requires_authoritative_queued_state(
        managed_case, monkeypatch):
    _, cfg = managed_case
    monkeypatch.setattr(
        "orze.reporting.evidence.authoritative_idea_lifecycle",
        lambda path, ids: ({
            ids[0]: {"state": "COMPLETE", "family": "architecture"},
        }, "authoritative_lifecycle_loaded"),
    )

    with pytest.raises(ManagedRunError, match="idea_not_queued"):
        prepare_managed_idea_run(cfg, "idea-managed", 4)


def test_managed_run_rejects_existing_attempt(managed_case):
    results, cfg = managed_case
    idea_dir = results / "idea-managed"
    idea_dir.mkdir()
    (idea_dir / "claim.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ManagedRunError, match="idea_already_attempted"):
        prepare_managed_idea_run(cfg, "idea-managed", 4)


def test_managed_run_requires_current_decision_receipt(
        managed_case, monkeypatch):
    _, cfg = managed_case
    monkeypatch.setattr(
        "orze.core.decision_batches.validate_idea_decision_admission",
        lambda *args: "decision_contract_launch_admission_missing",
    )

    with pytest.raises(
            ManagedRunError, match="decision_contract_launch_admission_missing"):
        prepare_managed_idea_run(cfg, "idea-managed", 4)


def test_managed_queue_loader_never_syncs_or_expands_other_ideas(
        tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        managed_run, "prepare_managed_idea_run",
        lambda cfg, idea_id, gpu: calls.append((idea_id, gpu)),
    )
    monkeypatch.setattr(
        "orze.engine.phases.get_unclaimed",
        lambda ideas, results, skipped, lake: list(ideas),
    )

    class Lake:
        def get(self, idea_id):
            assert idea_id == "idea-managed"
            return {
                "idea_id": idea_id,
                "title": "selected",
                "priority": "high",
                "config": "learning_rate: 0.0001\n",
            }

    runner = SimpleNamespace(
        lake=Lake(), failure_counts={}, results_dir=tmp_path)
    cfg = {"_managed_idea_gpu": 4, "max_idea_failures": 0}

    ideas, unclaimed, skipped, raw = OrzePhaseMixin._sync_managed_idea(
        runner, cfg, "idea-managed")

    assert calls == [("idea-managed", 4)]
    assert list(ideas) == ["idea-managed"]
    assert unclaimed == ["idea-managed"]
    assert skipped == set()
    assert raw == {}


def test_managed_outcome_requires_completed_untainted_evidence(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_dir = results / "idea-managed"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        '{"status":"COMPLETED","tainted_leakage":false}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "orze.reporting.evidence.authoritative_idea_lifecycle",
        lambda path, ids: ({
            ids[0]: {"state": "COMPLETE", "family": "architecture"},
        }, "authoritative_lifecycle_loaded"),
    )

    report = verify_managed_idea_outcome({
        "results_dir": str(results),
        "idea_lake_db": str(tmp_path / "idea_lake.db"),
    }, "idea-managed")

    assert report["completed"] is True
    assert report["evaluation_required"] is False


@pytest.mark.parametrize("metrics,reason", [
    ('{"status":"FAILED"}', "training_not_completed"),
    ('{"status":"COMPLETED","tainted_leakage":true}', "tainted_leakage"),
])
def test_managed_outcome_never_reports_failed_or_tainted_run_as_success(
        tmp_path, monkeypatch, metrics, reason):
    results = tmp_path / "results"
    idea_dir = results / "idea-managed"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(metrics, encoding="utf-8")
    monkeypatch.setattr(
        "orze.reporting.evidence.authoritative_idea_lifecycle",
        lambda path, ids: ({
            ids[0]: {"state": "COMPLETE", "family": "architecture"},
        }, "authoritative_lifecycle_loaded"),
    )

    with pytest.raises(ManagedRunError, match=reason):
        verify_managed_idea_outcome({
            "results_dir": str(results),
        }, "idea-managed")


def test_managed_outcome_requires_evaluation_artifact_when_configured(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_dir = results / "idea-managed"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        '{"status":"COMPLETED"}', encoding="utf-8")
    monkeypatch.setattr(
        "orze.reporting.evidence.authoritative_idea_lifecycle",
        lambda path, ids: ({
            ids[0]: {"state": "COMPLETE", "family": "architecture"},
        }, "authoritative_lifecycle_loaded"),
    )

    with pytest.raises(ManagedRunError, match="evaluation_output_missing"):
        verify_managed_idea_outcome({
            "results_dir": str(results),
            "eval_script": "/eval.py",
            "eval_output": "eval.json",
        }, "idea-managed")


def test_managed_outcome_requires_declared_post_script_artifact(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_dir = results / "idea-managed"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        '{"status":"COMPLETED"}', encoding="utf-8")
    monkeypatch.setattr(
        "orze.reporting.evidence.authoritative_idea_lifecycle",
        lambda path, ids: ({
            ids[0]: {"state": "COMPLETE", "family": "architecture"},
        }, "authoritative_lifecycle_loaded"),
    )

    with pytest.raises(ManagedRunError, match="post_script_output_missing"):
        verify_managed_idea_outcome({
            "results_dir": str(results),
            "post_scripts": [{"script": "/post.py", "output": "post.json"}],
        }, "idea-managed")


def test_managed_outcome_requires_valid_model_lineage(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_dir = results / "idea-managed"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        '{"status":"COMPLETED"}', encoding="utf-8")
    monkeypatch.setattr(
        "orze.reporting.evidence.authoritative_idea_lifecycle",
        lambda path, ids: ({
            ids[0]: {"state": "COMPLETE", "family": "architecture"},
        }, "authoritative_lifecycle_loaded"),
    )
    from orze.core.model_lineage import ModelLineageError
    monkeypatch.setattr(
        "orze.core.model_lineage.validate_model_lineage_for_evaluation",
        lambda *args: (_ for _ in ()).throw(
            ModelLineageError("model_lineage_artifact_drift")),
    )

    with pytest.raises(ManagedRunError, match="model_lineage_artifact_drift"):
        verify_managed_idea_outcome({
            "results_dir": str(results),
            "model_lineage": {"enabled": True},
        }, "idea-managed")


def test_run_idea_cli_builds_one_gpu_once_runner(tmp_path, monkeypatch):
    observed = {}
    cfg = {"auto_upgrade": True}
    monkeypatch.setattr(
        "sys.argv", ["orze", "run-idea", "idea-managed", "--gpu", "4"])
    monkeypatch.setattr("orze.extensions._find_pro_key", lambda: "present")
    monkeypatch.setattr(cli, "load_project_config", lambda path: cfg)
    monkeypatch.setattr(
        managed_run, "prepare_managed_idea_run",
        lambda config, idea_id, gpu: {
            "idea_id": idea_id, "gpu": gpu, "authorized": True,
        },
    )
    monkeypatch.setattr(
        managed_run, "verify_managed_idea_outcome",
        lambda config, idea_id: {
            "idea_id": idea_id,
            "completed": True,
            "lifecycle_state": "COMPLETE",
        },
    )

    class Runner:
        def __init__(self, gpus, config, once):
            observed.update(gpus=gpus, config=dict(config), once=once)

        def run(self):
            observed["ran"] = True

    monkeypatch.setattr("orze.engine.orchestrator.Orze", Runner)

    assert cli.main() == 0
    assert observed["gpus"] == [4]
    assert observed["once"] is True
    assert observed["config"]["_managed_idea_id"] == "idea-managed"
    assert observed["config"]["auto_upgrade"] is False
    assert observed["config"]["max_fix_attempts"] == 0
    assert observed["config"]["notifications"]["enabled"] is False
    assert observed["ran"] is True


def test_run_idea_cli_returns_failure_when_terminal_proof_is_missing(
        monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["orze", "run-idea", "idea-managed", "--gpu", "4"])
    monkeypatch.setattr("orze.extensions._find_pro_key", lambda: "present")
    monkeypatch.setattr(cli, "load_project_config", lambda path: {})
    monkeypatch.setattr(
        managed_run, "prepare_managed_idea_run",
        lambda *args: {"idea_id": "idea-managed", "gpu": 4},
    )
    monkeypatch.setattr(
        managed_run, "verify_managed_idea_outcome",
        lambda *args: (_ for _ in ()).throw(
            ManagedRunError("managed_run_metrics_missing")),
    )

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            pass

    monkeypatch.setattr("orze.engine.orchestrator.Orze", Runner)

    assert cli.main() == 1
