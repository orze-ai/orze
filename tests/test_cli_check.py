"""The diagnostic config check must be side-effect-free and stop-aware."""

from pathlib import Path

import pytest

import orze.cli_setup as cli_setup
import orze.extensions as extensions


def _case(tmp_path: Path) -> dict:
    train = tmp_path / "train.py"
    ideas = tmp_path / "ideas.md"
    results = tmp_path / "results"
    train.write_text("# idea_config.yaml\n", encoding="utf-8")
    ideas.write_text("", encoding="utf-8")
    results.mkdir()
    return {
        "_config_path": str(tmp_path / "orze.yaml"),
        "train_script": str(train),
        "ideas_file": str(ideas),
        "results_dir": str(results),
        "roles": {},
    }


def _isolate_check(monkeypatch):
    monkeypatch.setattr(
        "orze.core.config._validate_config", lambda cfg: ([], []))
    monkeypatch.setattr(
        "orze.hardware.gpu.detect_all_gpus", lambda: [4, 5, 6, 7])
    monkeypatch.setattr(
        extensions, "inspect_pro_status",
        lambda: (False, "orze-pro not installed"))


def test_check_never_attempts_pro_auto_install(monkeypatch):
    import importlib.util

    attempted = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(
        extensions.importlib, "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)))
    monkeypatch.setattr(
        extensions, "_auto_install_pro", lambda: attempted.append(True))

    assert extensions.has_pro(auto_install=False) is False
    assert extensions.check_pro_status().startswith("orze-pro not installed")
    assert attempted == []


@pytest.mark.parametrize("control", ["sentinel", "config_pause", "flag"])
def test_check_reports_valid_but_not_runnable_when_stopped(
        tmp_path, monkeypatch, capsys, control):
    cfg = _case(tmp_path)
    Path(cfg["_config_path"]).write_text("{}\n", encoding="utf-8")
    if control == "sentinel":
        (Path(cfg["results_dir"]) / ".orze_disabled").touch()
    elif control == "config_pause":
        cfg["launcher"] = {"paused": True}
    else:
        (Path(cfg["results_dir"]) / "_launcher_paused.flag").touch()
    _isolate_check(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        cli_setup.do_check(cfg)

    assert exc.value.code == 2
    output = capsys.readouterr().out
    assert "Configuration valid; launch blocked" in output
    assert "Ready to run" not in output


def test_check_reports_ready_only_without_stop_controls(
        tmp_path, monkeypatch, capsys):
    cfg = _case(tmp_path)
    Path(cfg["_config_path"]).write_text("{}\n", encoding="utf-8")
    _isolate_check(monkeypatch)

    cli_setup.do_check(cfg)

    assert "Ready to run" in capsys.readouterr().out


def test_check_gpu_inventory_honors_configured_physical_scope(
        tmp_path, monkeypatch):
    cfg = _case(tmp_path)
    cfg["gpu_scheduling"] = {"allowed_gpus": [4, 5, 6, 7]}
    Path(cfg["_config_path"]).write_text("{}\n", encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        "orze.core.config._validate_config", lambda value: ([], []))
    monkeypatch.setattr(
        "orze.hardware.gpu.detect_all_gpus",
        lambda gpu_ids: observed.append(gpu_ids) or list(gpu_ids),
    )
    monkeypatch.setattr(
        extensions, "inspect_pro_status",
        lambda: (False, "orze-pro not installed"),
    )

    cli_setup.do_check(cfg)

    assert observed == [[4, 5, 6, 7]]


def test_relative_results_pause_flag_uses_one_results_prefix(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    (results / "_launcher_paused.flag").touch()

    assert cli_setup._runnable_blockers({"results_dir": "results"}) == [
        "launcher_pause_flag_present"]


def test_check_rejects_controller_runtime_drift(
        tmp_path, monkeypatch, capsys):
    cfg = _case(tmp_path)
    Path(cfg["_config_path"]).write_text("{}\n", encoding="utf-8")
    cfg["controller_runtime"] = {
        "contract_version": 1,
        "python": "/runtime/python",
        "packages": [],
    }
    _isolate_check(monkeypatch)
    monkeypatch.setattr(
        "orze.service.runtime_contract.audit_controller_runtime_contract",
        lambda contract: {
            "schema_version": 1,
            "contract_ok": False,
            "errors": ["runtime_package_sha256_drift:orze"],
        },
    )

    with pytest.raises(SystemExit) as exc:
        cli_setup.do_check(cfg)

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "runtime_package_sha256_drift:orze" in output
    assert "Ready to run" not in output
