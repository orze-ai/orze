"""One-idea managed runs must fail closed before any GPU observation."""

import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import orze.cli as cli
from orze.engine import lifecycle
from orze.core import managed_run
from orze.core.config import _validate_config
from orze.core.gpu_lease import GpuLeaseError
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


@pytest.mark.parametrize("key", [
    "require_data_separation",
    "require_model_lineage",
    "require_benchmark_contract",
    "require_explicit_untainted_metrics",
    "require_clean_training_access_log",
])
def test_managed_evidence_policy_requires_booleans(key):
    errors, _ = _validate_config({"managed_run": {key: "yes"}})

    assert f"managed_run.{key}: must be true or false" in errors


def test_managed_evidence_policy_rejects_unknown_or_nonmapping_policy():
    errors, _ = _validate_config({
        "managed_run": {"require_data_seperation": True},
    })
    assert "managed_run: unknown keys: require_data_seperation" in errors

    with pytest.raises(ManagedRunError, match="evidence_policy_invalid"):
        verify_managed_idea_outcome({"managed_run": None}, "idea-managed")


@pytest.mark.parametrize("key,reason", [
    ("require_data_separation", "requires data_separation.enabled: true"),
    ("require_model_lineage", "requires model_lineage.enabled: true"),
    ("require_benchmark_contract", "requires report.benchmark_contract"),
])
def test_managed_evidence_policy_rejects_missing_contracts(key, reason):
    errors, _ = _validate_config({"managed_run": {key: True}})

    assert any(reason in error for error in errors)


@pytest.mark.parametrize("policy_key,reason", [
    ("require_data_separation", "data_separation_required"),
    ("require_model_lineage", "model_lineage_required"),
    ("require_benchmark_contract", "benchmark_contract_required"),
])
def test_managed_admission_enforces_required_evidence_contract(
        managed_case, policy_key, reason):
    _, cfg = managed_case
    cfg["managed_run"] = {policy_key: True}

    with pytest.raises(ManagedRunError, match=reason):
        prepare_managed_idea_run(cfg, "idea-managed", 4)


@pytest.mark.parametrize("sentinel", [
    ".orze_disabled", ".orze_stop_all", ".orze_shutdown",
])
def test_managed_run_honors_stop_state_before_gpu_telemetry(
        managed_case, sentinel, monkeypatch):
    results, cfg = managed_case
    (results / sentinel).write_text("stop\n", encoding="utf-8")
    monkeypatch.setattr(
        "orze.core.config._validate_config",
        lambda *args: pytest.fail(
            "stop state must reject before config validation"),
    )
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


def test_disabled_orchestrator_exits_before_gpu_lease_or_startup(
        tmp_path, monkeypatch):
    calls = []

    class Lake:
        def close(self):
            calls.append("lake_close")

    runner = Orze.__new__(Orze)
    runner.gpu_ids = [4]
    runner._gpu_leases = None
    runner.lake = Lake()
    runner._write_pid_file = lambda: calls.append("pid_write")
    runner._remove_pid_file = lambda: calls.append("pid_remove")
    runner._check_disabled = lambda: True
    runner._run_leased = lambda: pytest.fail("disabled controller started")
    monkeypatch.setattr(
        "orze.engine.orchestrator.acquire_gpu_leases",
        lambda *args: pytest.fail("disabled controller acquired GPU leases"),
    )

    runner.run()

    assert calls == ["pid_write", "lake_close", "pid_remove"]


def test_orchestrator_rejects_external_gpu_compute_before_startup(
        tmp_path, monkeypatch):
    calls = []

    class Leases:
        def close(self):
            calls.append("lease_close")

    runner = Orze.__new__(Orze)
    runner.gpu_ids = [4, 5, 6, 7]
    runner._gpu_leases = None
    runner.lake = None
    runner._write_pid_file = lambda: calls.append("pid_write")
    runner._remove_pid_file = lambda: calls.append("pid_remove")
    runner._check_disabled = lambda: False
    runner._run_leased = lambda: pytest.fail("occupied scope started")
    monkeypatch.setattr(
        "orze.engine.orchestrator.acquire_gpu_leases",
        lambda ids: Leases(),
    )
    monkeypatch.setattr(
        "orze.engine.orchestrator.assert_gpu_scope_idle",
        lambda ids: (_ for _ in ()).throw(GpuLeaseError(
            "gpu_lease_external_compute_detected: physical_gpu=4"
        )),
    )

    with pytest.raises(GpuLeaseError, match="external_compute_detected"):
        runner.run()

    assert calls == ["pid_write", "lease_close", "pid_remove"]


def test_managed_orchestrator_skips_daemon_wide_hooks(
        tmp_path, monkeypatch):
    calls = []

    class Slots(dict):
        def gpu_ids_in_use(self):
            return set()

    class Lake:
        def close(self):
            calls.append("lake_close")

    class Healthy:
        retry_delay = 0

        def __init__(self, results_dir):
            calls.append("health_monitor")

        def check_before_write(self):
            return True

    runner = Orze.__new__(Orze)
    runner.cfg = {
        "_managed_idea_id": "idea-managed",
        "_managed_idea_gpu": 4,
        "ideas_file": str(tmp_path / "ideas.md"),
        "results_dir": str(tmp_path),
        "timeout": 60,
        "poll": 1,
        "roles": {},
        "notifications": {"enabled": False},
        "sealed_files": [],
        "cleanup": {},
        "min_disk_gb": 0,
        "report": {"primary_metric": "score"},
    }
    runner.gpu_ids = [4]
    runner.once = True
    runner.results_dir = tmp_path
    runner.slot_mgr = Slots()
    runner.active = runner.slot_mgr
    runner.active_evals = {}
    runner.active_roles = {}
    runner.pending_evals = []
    runner.running = True
    runner.iteration = 0
    runner.failure_counts = {}
    runner.fix_counts = {}
    runner.role_states = {}
    runner.notification_health = {}
    runner._hostname = "test-host"
    runner._instance_uuid = "test-instance"
    runner._leader_handle = None
    runner._auto_gpu_mode = False
    runner._stop_event = threading.Event()
    runner.lake = Lake()
    runner._write_pid_file = lambda: calls.append("pid_write")
    runner._remove_pid_file = lambda: calls.append("pid_remove")
    runner._check_disabled = lambda: False
    runner._check_stop_all = lambda: False
    runner._sync_managed_idea = lambda cfg, idea_id: (
        {idea_id: {"title": "selected", "config": {}}}, [], set(), {})
    runner._launch_evals = lambda *args: ([], [])
    runner._launch_training = lambda *args: [4]

    forbidden = (
        "_startup_checks", "_kill_orphans", "_check_auto_upgrade",
        "_check_upgrade_sentinel", "_check_cluster_versions",
        "_hot_reload_config", "_run_all_roles", "_rebuild_config_hashes",
        "_report_and_notify",
    )
    for name in forbidden:
        setattr(
            runner, name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"managed run called daemon-wide hook {_name}"),
        )

    monkeypatch.setattr("orze.engine.health.HealthMonitor", Healthy)
    monkeypatch.setattr(
        "orze.engine.orchestrator.check_disk_space", lambda *args: True)
    monkeypatch.setattr(
        "orze.engine.orchestrator.notify",
        lambda *args: pytest.fail("managed run sent a notification"))
    monkeypatch.setattr(
        "orze.engine.orchestrator.startup_canary",
        lambda *args: pytest.fail("managed run called startup canary"))
    monkeypatch.setattr(
        "orze.engine.orchestrator.parse_ideas",
        lambda *args: pytest.fail("managed run parsed the global queue"))
    monkeypatch.setattr(
        "orze.engine.orchestrator.update_report",
        lambda *args: pytest.fail("managed run updated the global report"))
    monkeypatch.setattr(
        "orze.engine.orchestrator.write_host_heartbeat",
        lambda *args: pytest.fail("managed run wrote a daemon heartbeat"))
    monkeypatch.setattr(
        "orze.engine.orchestrator.save_state",
        lambda *args: pytest.fail("managed run saved daemon state"))
    monkeypatch.setattr(
        "orze.engine.orchestrator.assert_gpu_scope_idle", lambda ids: None)

    runner.run()

    assert calls == ["pid_write", "health_monitor", "lake_close", "pid_remove"]


def test_managed_shutdown_forces_child_termination(monkeypatch, tmp_path):
    observed = {}
    runner = Orze.__new__(Orze)
    runner.cfg = {"_managed_idea_id": "idea-managed"}
    runner.results_dir = tmp_path
    runner.active = {4: object()}
    runner.active_evals = {}
    runner.active_roles = {}
    runner.iteration = 1
    runner.lake = None
    runner._hostname = "test-host"
    runner._instance_uuid = "test-instance"
    runner._pid_file = tmp_path / ".orze.managed.idea-managed.1.pid"
    runner._build_state_dict = lambda: {}
    monkeypatch.setattr(
        "orze.engine.orchestrator.graceful_shutdown",
        lambda *args, **kwargs: observed.update(kwargs),
    )

    runner._graceful_shutdown(kill_all=False)

    assert observed["managed"] is True
    assert observed["kill_all"] is True
    assert observed["pid_file_path"] == runner._pid_file


def test_managed_shutdown_does_not_mutate_daemon_state(
        monkeypatch, tmp_path):
    managed_pid = tmp_path / ".orze.managed.idea-managed.1.pid"
    managed_pid.write_text("1", encoding="utf-8")
    daemon_pid = tmp_path / ".orze.pid"
    daemon_pid.write_text("123", encoding="utf-8")
    closed = []
    lake = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(
        lifecycle, "write_shutdown_heartbeat",
        lambda *args: pytest.fail("managed shutdown wrote daemon heartbeat"))
    monkeypatch.setattr(
        lifecycle, "save_state",
        lambda *args: pytest.fail("managed shutdown saved daemon state"))
    monkeypatch.setattr(
        lifecycle, "notify",
        lambda *args: pytest.fail("managed shutdown sent notification"))

    lifecycle.graceful_shutdown(
        tmp_path, {}, {}, {}, {}, 1, {}, lake,
        "test-host", "test-instance", kill_all=True,
        managed=True, pid_file_path=managed_pid,
    )

    assert closed == [True]
    assert not managed_pid.exists()
    assert daemon_pid.read_text(encoding="utf-8") == "123"
    assert not (tmp_path / ".orze_shutdown").exists()


def test_python_module_propagates_managed_rejection_exit_code(tmp_path):
    config = tmp_path / "orze.yaml"
    config.write_text("controller_runtime: null\n", encoding="utf-8")
    source_root = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [
            sys.executable, "-m", "orze.cli", "run-idea", "idea-managed",
            "--gpu", "4", "-c", str(config),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 2
    assert "managed idea run rejected" in result.stdout


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


def test_managed_outcome_rejects_tainted_access_log_despite_false_metric(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_dir = results / "idea-managed"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        '{"status":"COMPLETED","tainted_leakage":false}',
        encoding="utf-8",
    )
    (idea_dir / "_access_log.tsv").write_text(
        "WATCH\t/evaluation/private\t/evaluation/private/sample.arrow\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "orze.reporting.evidence.authoritative_idea_lifecycle",
        lambda path, ids: ({
            ids[0]: {"state": "COMPLETE", "family": "architecture"},
        }, "authoritative_lifecycle_loaded"),
    )

    with pytest.raises(
            ManagedRunError, match="training_access_log_not_clean"):
        verify_managed_idea_outcome({
            "results_dir": str(results),
            "managed_run": {
                "require_explicit_untainted_metrics": True,
                "require_model_lineage": True,
                "require_clean_training_access_log": True,
            },
            "model_lineage": {"enabled": True},
        }, "idea-managed")


def test_clean_access_log_policy_requires_model_lineage():
    errors, _ = _validate_config({
        "managed_run": {"require_clean_training_access_log": True},
    })

    assert (
        "managed_run.require_clean_training_access_log: requires "
        "require_model_lineage: true"
    ) in errors


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


def test_managed_outcome_requires_explicit_untainted_marker(
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

    with pytest.raises(
            ManagedRunError, match="explicit_untainted_metrics_required"):
        verify_managed_idea_outcome({
            "results_dir": str(results),
            "managed_run": {
                "require_explicit_untainted_metrics": True,
            },
        }, "idea-managed")


def test_managed_outcome_revalidates_required_data_separation(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_dir = results / "idea-managed"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        '{"status":"COMPLETED","tainted_leakage":false}',
        encoding="utf-8")
    monkeypatch.setattr(
        "orze.reporting.evidence.authoritative_idea_lifecycle",
        lambda path, ids: ({
            ids[0]: {"state": "COMPLETE", "family": "architecture"},
        }, "authoritative_lifecycle_loaded"),
    )
    calls = []
    monkeypatch.setattr(
        "orze.core.data_separation.ensure_data_separation",
        lambda cfg: calls.append(cfg) or {"schema_version": 1},
    )
    cfg = {
        "results_dir": str(results),
        "managed_run": {"require_data_separation": True},
        "data_separation": {"enabled": True},
    }

    report = verify_managed_idea_outcome(cfg, "idea-managed")

    assert calls == [cfg]
    assert report["completed"] is True


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


def test_run_idea_cli_cleans_up_after_orchestrator_exception(
        monkeypatch, capsys):
    observed = {}
    monkeypatch.setattr(
        "sys.argv", ["orze", "run-idea", "idea-managed", "--gpu", "4"])
    monkeypatch.setattr("orze.extensions._find_pro_key", lambda: "present")
    monkeypatch.setattr(cli, "load_project_config", lambda path: {})
    monkeypatch.setattr(
        managed_run, "prepare_managed_idea_run",
        lambda *args: {"idea_id": "idea-managed", "gpu": 4},
    )

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            raise RuntimeError("sensitive detail must not be printed")

        def _graceful_shutdown(self, kill_all=False):
            observed["kill_all"] = kill_all

        def _remove_pid_file(self):
            pytest.fail("fallback cleanup should not be needed")

    monkeypatch.setattr("orze.engine.orchestrator.Orze", Runner)

    assert cli.main() == 1
    assert observed == {"kill_all": True}
    output = capsys.readouterr().out
    assert "RuntimeError" in output
    assert "sensitive detail" not in output


def test_run_idea_cli_reports_gpu_lease_contention_as_safe_rejection(
        monkeypatch, capsys):
    from orze.core.gpu_lease import GpuLeaseError

    observed = {}
    monkeypatch.setattr(
        "sys.argv", ["orze", "run-idea", "idea-managed", "--gpu", "4"])
    monkeypatch.setattr("orze.extensions._find_pro_key", lambda: "present")
    monkeypatch.setattr(cli, "load_project_config", lambda path: {})
    monkeypatch.setattr(
        managed_run, "prepare_managed_idea_run",
        lambda *args: {"idea_id": "idea-managed", "gpu": 4},
    )

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            raise GpuLeaseError(
                "gpu_lease_contended: physical_gpu=4")

        def _graceful_shutdown(self, kill_all=False):
            observed["kill_all"] = kill_all

        def _remove_pid_file(self):
            pytest.fail("fallback cleanup should not be needed")

    monkeypatch.setattr("orze.engine.orchestrator.Orze", Runner)

    assert cli.main() == 2
    assert observed == {"kill_all": True}
    output = capsys.readouterr().out
    assert "gpu_lease_contended: physical_gpu=4" in output
