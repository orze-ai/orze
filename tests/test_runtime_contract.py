"""Installed service policy and runtime identity must fail closed on drift."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orze.core import config as config_module
from orze.service import install
from orze.service.runtime_contract import (
    CONTROLLER_CONTRACT_VERSION,
    CONTRACT_VERSION,
    audit_controller_runtime_contract,
    audit_runtime_contract,
    capture_controller_runtime_contract,
    capture_runtime_packages,
)
from orze.service import runtime_contract
from orze.service import watchdog


def _fixture(tmp_path):
    workdir = tmp_path / "project"
    workdir.mkdir()
    results = workdir / "results"
    results.mkdir()
    config = workdir / "orze.yaml"
    config.write_text("results_dir: results\n", encoding="utf-8")
    python = sys.executable
    runtime = [{
        "name": "orze",
        "root": "/runtime/orze",
        "sha256": "a" * 64,
        "file_count": 10,
    }]
    svc_cfg = {
        "method": "systemd",
        "python": str(python),
        "workdir": str(workdir),
        "config_file": str(config),
        "results_dir": str(results),
        "runtime_contract_version": CONTRACT_VERSION,
        "runtime_packages": runtime,
    }
    properties = {
        "Restart": "no",
        "WorkingDirectory": str(workdir),
        "ExecStart": (
            f"{{ path={python} ; argv[]={python} -m orze.cli -c {config} ; }}"
        ),
        "ExecStartPre": (
            f"{{ path={python} ; argv[]={python} -m "
            "orze.service.runtime_contract --startup-check ; }}"
        ),
        "Environment": "",
        "EnvironmentFiles": "",
        "PassEnvironment": "",
        "ActiveState": "inactive",
        "UnitFileState": "disabled",
    }
    return svc_cfg, properties, runtime


def test_matching_effective_service_and_runtime_are_accepted(tmp_path):
    svc_cfg, properties, runtime = _fixture(tmp_path)

    report = audit_runtime_contract(
        svc_cfg, properties=properties, observed_packages=runtime)

    assert report == {
        "schema_version": 1,
        "contract_ok": True,
        "startup_allowed": True,
        "errors": [],
        "active_latches": [],
    }


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("Restart", "always", "systemd_restart_policy_drift"),
    ("WorkingDirectory", "/wrong", "systemd_workdir_drift"),
    ("ExecStart", "/bin/false", "systemd_exec_start_drift"),
    ("ExecStartPre", "", "systemd_exec_start_pre_missing"),
    ("Environment", "PYTHONPATH=/mutable", "systemd_pythonpath_override"),
    ("Environment", "ORZE_SUBSCRIPTION_LIMIT_ACTIONS=off",
     "systemd_shutdown_actions_disabled"),
    ("EnvironmentFiles", "/tmp/unpinned.env",
     "systemd_environment_files_unpinned"),
    ("PassEnvironment", "HOME PYTHONPATH", "systemd_pythonpath_override"),
])
def test_effective_systemd_drift_is_rejected(
        tmp_path, field, value, reason):
    svc_cfg, properties, runtime = _fixture(tmp_path)
    properties[field] = value

    report = audit_runtime_contract(
        svc_cfg, properties=properties, observed_packages=runtime)

    assert report["contract_ok"] is False
    assert report["startup_allowed"] is False
    assert reason in report["errors"]


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("root", "/different/orze", "runtime_package_root_drift:orze"),
    ("sha256", "b" * 64, "runtime_package_sha256_drift:orze"),
    ("file_count", 11, "runtime_package_file_count_drift:orze"),
])
def test_runtime_package_drift_is_rejected(tmp_path, field, value, reason):
    svc_cfg, properties, runtime = _fixture(tmp_path)
    observed = [dict(runtime[0], **{field: value})]

    report = audit_runtime_contract(
        svc_cfg, properties=properties, observed_packages=observed)

    assert reason in report["errors"]
    assert report["startup_allowed"] is False


def test_stop_latch_is_safe_only_when_systemd_unit_is_inactive_and_disabled(
        tmp_path):
    svc_cfg, properties, runtime = _fixture(tmp_path)
    (tmp_path / "project" / "results" / ".orze_disabled").write_text(
        "operator stop\n", encoding="utf-8")

    stopped = audit_runtime_contract(
        svc_cfg, properties=properties, observed_packages=runtime)
    assert stopped["contract_ok"] is True
    assert stopped["startup_allowed"] is False
    assert stopped["active_latches"] == [".orze_disabled"]

    properties["ActiveState"] = "active"
    properties["UnitFileState"] = "enabled"
    unsafe = audit_runtime_contract(
        svc_cfg, properties=properties, observed_packages=runtime)
    assert "latched_systemd_unit_active" in unsafe["errors"]
    assert "latched_systemd_unit_enabled" in unsafe["errors"]
    assert unsafe["contract_ok"] is False


def test_watchdog_refuses_launch_when_runtime_contract_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "orze.service.runtime_contract.audit_runtime_contract",
        lambda cfg: {
            "startup_allowed": False,
            "errors": ["systemd_restart_policy_drift"],
            "active_latches": [],
        },
    )
    monkeypatch.setattr(
        watchdog.subprocess, "run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(RuntimeError, match="systemd_restart_policy_drift"):
        watchdog._launch_orze({"method": "systemd"})

    assert calls == []


def test_service_install_refuses_stop_latch_before_writing_config(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    (results / ".orze_disabled").write_text("stop\n", encoding="utf-8")
    config = tmp_path / "orze.yaml"
    config.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        config_module, "load_project_config",
        lambda path: {"results_dir": str(results)},
    )
    monkeypatch.setattr(
        install, "_save_service_config",
        lambda *args, **kwargs: pytest.fail("must not write service config"),
    )

    with pytest.raises(RuntimeError, match="stop latch is present"):
        install.install(str(config), method="systemd")


def test_saved_service_config_pins_runtime_packages(tmp_path, monkeypatch):
    service_config = tmp_path / "service.json"
    runtime = [{
        "name": "orze", "root": "/runtime", "sha256": "a" * 64,
        "file_count": 1,
    }]
    monkeypatch.setattr(install, "SERVICE_CONFIG_PATH", service_config)
    monkeypatch.setattr(
        "orze.service.runtime_contract.capture_runtime_packages",
        lambda: runtime,
    )

    written = install._save_service_config(
        tmp_path / "orze.yaml", tmp_path / "results", tmp_path,
        "/usr/bin/python3", "systemd", 60, tmp_path / "orze.log",
    )

    assert written["runtime_contract_version"] == CONTRACT_VERSION
    assert written["runtime_packages"] == runtime
    assert json.loads(service_config.read_text())["runtime_packages"] == runtime


def test_real_orze_runtime_capture_is_nonempty_and_content_addressed():
    captured = capture_runtime_packages(names=("orze",))

    assert len(captured) == 1
    assert captured[0]["name"] == "orze"
    assert captured[0]["file_count"] > 0
    assert len(captured[0]["sha256"]) == 64


def test_matching_direct_controller_runtime_is_accepted(tmp_path):
    runtime = [{
        "name": "orze", "root": "/runtime/orze", "sha256": "a" * 64,
        "file_count": 10,
    }]
    contract = {
        "contract_version": CONTROLLER_CONTRACT_VERSION,
        "python": sys.executable,
        "packages": runtime,
    }

    report = audit_controller_runtime_contract(
        contract, observed_packages=runtime, observed_python=sys.executable)

    assert report == {"schema_version": 1, "contract_ok": True, "errors": []}


@pytest.mark.parametrize(("mutation", "reason"), [
    ({"python": "/missing/python"}, "controller_runtime_python_invalid"),
    ({"contract_version": 0},
     "controller_runtime_contract_version_missing_or_unsupported"),
])
def test_direct_controller_metadata_drift_is_rejected(mutation, reason):
    runtime = [{
        "name": "orze", "root": "/runtime/orze", "sha256": "a" * 64,
        "file_count": 10,
    }]
    contract = {
        "contract_version": CONTROLLER_CONTRACT_VERSION,
        "python": sys.executable,
        "packages": runtime,
        **mutation,
    }

    report = audit_controller_runtime_contract(
        contract, observed_packages=runtime, observed_python=sys.executable)

    assert reason in report["errors"]
    assert report["contract_ok"] is False


def test_capture_controller_contract_is_ready_for_project_config(monkeypatch):
    runtime = [{
        "name": "orze", "root": "/runtime/orze", "sha256": "a" * 64,
        "file_count": 10,
    }]
    monkeypatch.setattr(runtime_contract, "capture_runtime_packages",
                        lambda: runtime)

    captured = capture_controller_runtime_contract()

    assert captured["contract_version"] == CONTROLLER_CONTRACT_VERSION
    assert captured["python"] == str(Path(sys.executable).resolve())
    assert captured["packages"] == runtime


def test_capture_controller_cli_is_service_config_independent(
        monkeypatch, capsys):
    monkeypatch.setattr(
        runtime_contract, "capture_controller_runtime_contract",
        lambda: {"contract_version": 1, "python": "/python", "packages": []},
    )
    monkeypatch.setattr(
        runtime_contract, "load_service_config",
        lambda: pytest.fail("capture must not read installed service state"),
    )

    assert runtime_contract.main(["--capture-controller"]) == 0
    assert json.loads(capsys.readouterr().out)["python"] == "/python"


def test_controller_runtime_config_schema_rejects_ambiguous_pins():
    errors, _ = config_module._validate_config({
        "controller_runtime": {
            "contract_version": 1,
            "python": "",
            "packages": [{
                "name": "other",
                "root": "relative",
                "sha256": "NOT-A-HASH",
                "file_count": 0,
            }],
        },
    })

    assert any("controller_runtime.python" in error for error in errors)
    assert any(".name" in error for error in errors)
    assert any(".root" in error for error in errors)
    assert any(".sha256" in error for error in errors)
    assert any(".file_count" in error for error in errors)


def test_systemd_property_reader_treats_unset_optional_fields_as_empty(
        monkeypatch):
    required = (
        "Restart=no\n"
        "WorkingDirectory=/srv/project\n"
        "ExecStart=orze\n"
        "Environment=\n"
        "PassEnvironment=\n"
        "ActiveState=inactive\n"
        "UnitFileState=disabled\n"
    )

    def fake_run(args, **kwargs):
        if "show" in args:
            return SimpleNamespace(returncode=0, stdout=required, stderr="")
        return SimpleNamespace(
            returncode=0, stdout="[Service]\nRestart=no\n", stderr="")

    monkeypatch.setattr(runtime_contract.subprocess, "run", fake_run)

    properties = runtime_contract._systemd_properties()

    assert properties["ExecStartPre"] == ""
    assert properties["EnvironmentFiles"] == ""
    assert properties["_UnitText"].startswith("[Service]")


def test_environment_file_directive_is_rejected_even_when_property_is_absent(
        tmp_path):
    svc_cfg, properties, runtime = _fixture(tmp_path)
    properties["_UnitText"] = "[Service]\nEnvironmentFile=/tmp/unpinned\n"

    report = audit_runtime_contract(
        svc_cfg, properties=properties, observed_packages=runtime)

    assert "systemd_environment_files_unpinned" in report["errors"]
