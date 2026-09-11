"""New stop-profile metadata and CLI adapter requirements, not stop proof.

CLI operation results are explicitly typed transport doubles. Real controller
registration, ACK and pidfd-exit acceptance belong to product integration tests.
"""
import copy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from orze.core import config
from orze.core.controller_profile import (
    ControllerProfileError, controller_profile, profile_fingerprint, validate_profile_cli,
)


def supported(tmp_path):
    cfg = copy.deepcopy(config.DEFAULT_CONFIG)
    cfg.update(controller_control={"version": 1, "profile": "local_stop_v1"},
               telemetry=False, auto_upgrade=False, max_fix_attempts=0,
               metric_harvest={"llm_fallback": False},
               results_dir=str(tmp_path / "results"), idea_lake_db=str(tmp_path / "ideas.db"),
               _config_path=str(tmp_path / "orze.yaml"), _project_root=str(tmp_path),
               _orze_dir=str(tmp_path / ".orze"), _controller_workdir=str(tmp_path),
               auto_seal_eval=False)
    cfg["gpu_scheduling"]["allowed_gpus"] = [2, 4]
    return cfg


def test_default_off_does_not_revalidate_legacy_features():
    assert config.DEFAULT_CONFIG["controller_control"] is None
    assert controller_profile({"bot": {}, "telemetry": True}) is None
    assert profile_fingerprint({"controller_control": None, "unrelated": object()}) is None


@pytest.mark.parametrize("declaration", [True, {}, {"version": True, "profile": "local_stop_v1"},
    {"version": 2, "profile": "local_stop_v1"}, {"version": 1, "profile": "other"},
    {"version": 1, "profile": "local_stop_v1", "restart": True}])
def test_declaration_is_exact_and_not_boolean_aliased(tmp_path, declaration):
    cfg = supported(tmp_path)
    cfg["controller_control"] = declaration
    with pytest.raises(ControllerProfileError, match="declaration_invalid"):
        controller_profile(cfg)
    assert any("controller_profile_declaration_invalid" in value for value in config._validate_config(cfg)[0])


@pytest.mark.parametrize("key,value", [("telemetry", None), ("auto_upgrade", True),
    ("bot", {}), ("notifications", {"enabled": True}), ("retrospection", {"enabled": True}),
    ("cleanup", {"script": "unsafe.sh"}), ("metric_harvest", {}),
    ("gpu_scheduling", {"allowed_gpus": []}), ("max_fix_attempts", 1),
    ("fleet", ["other-host"]), ("substrate", {"elo_ranking_enabled": True})])
def test_unsupported_profile_features_are_not_silently_disabled(tmp_path, key, value):
    cfg = supported(tmp_path)
    cfg[key] = value
    before = copy.deepcopy(cfg)
    with pytest.raises(ControllerProfileError):
        controller_profile(cfg)
    assert cfg == before


def test_fingerprint_binds_public_config_paths_and_exact_gpu_scope(tmp_path):
    cfg = supported(tmp_path)
    expected = profile_fingerprint(cfg, [4, 2])
    assert type(expected) is str and len(expected) == 64
    cfg["_managed_gpu_ids"] = [2, 4]
    cfg["_untrusted_runtime_note"] = object()
    cfg["_controller_profile_fingerprint"] = "untrusted"
    assert profile_fingerprint(cfg) == expected
    for key, value in [("timeout", 19), ("_config_path", str(tmp_path / "other.yaml")),
                       ("_orze_dir", str(tmp_path / "other-control")),
                       ("_controller_workdir", str(tmp_path / "other-cwd"))]:
        changed = dict(cfg, **{key: value})
        assert profile_fingerprint(changed) != expected
    with pytest.raises(ControllerProfileError, match="gpu_scope_changed"):
        profile_fingerprint(cfg, [2])
    with pytest.raises(ControllerProfileError, match="explicit_gpus_required"):
        profile_fingerprint(cfg, [True, 4])
    cfg["public_non_json"] = float("nan")
    with pytest.raises(ControllerProfileError, match="json_invalid"):
        profile_fingerprint(cfg)


def test_real_loader_overwrites_untrusted_stamp_after_parsing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_load_dotenv", lambda path: 0)
    cfg = supported(tmp_path)
    cfg["_controller_profile_fingerprint"] = "forged"
    path = tmp_path / "orze.yaml"
    path.write_text(yaml.safe_dump(cfg))
    loaded = config.load_project_config(str(path))
    assert loaded["_controller_profile_fingerprint"] == profile_fingerprint(loaded)
    assert loaded["_controller_profile_fingerprint"] != "forged"
    assert loaded["_config_path"] == str(path)
    assert loaded["_controller_workdir"] == str(tmp_path)


def test_cli_gpu_and_runtime_overrides_cannot_ambiguate_observer_fingerprint(tmp_path):
    cfg = supported(tmp_path)
    assert validate_profile_cli(cfg, SimpleNamespace(command=None, gpus="4,2")) == cfg["controller_control"]
    for args in (SimpleNamespace(command=None, gpus="2"),
                 SimpleNamespace(command=None, timeout=99),
                 SimpleNamespace(command=None, results_dir="elsewhere")):
        with pytest.raises(ControllerProfileError):
            validate_profile_cli(cfg, args)
    assert validate_profile_cli(cfg, SimpleNamespace(command="stop", timeout=7)) is not None


@pytest.fixture
def cli_case(tmp_path, monkeypatch):
    from orze import cli
    import orze.extensions
    cfg = supported(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "setup_logging", lambda *a: None)
    monkeypatch.setattr(orze.extensions, "_find_pro_key", lambda: "test-only-key")
    monkeypatch.setattr(cli, "load_project_config", lambda path: cfg)
    monkeypatch.setattr(cli, "_require_controller_runtime", lambda cfg: None)
    monkeypatch.setattr(cli, "detect_all_gpus", Mock(side_effect=AssertionError("no GPU discovery")))
    module = ModuleType("orze.engine.controller_session")
    module.CompletedControllerStop = type("CompletedControllerStop", (), {})
    module.stop_controller = Mock(return_value=module.CompletedControllerStop())
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return SimpleNamespace(cli=cli, cfg=cfg, session=module)


@pytest.mark.parametrize("argv", [["start"], ["start", "--foreground"],
    ["--role-only", "worker"], ["run-idea", "idea-1", "--gpu", "2"]])
def test_unsupported_cli_execution_modes_refuse_before_launch(cli_case, monkeypatch, argv):
    c = cli_case
    monkeypatch.setattr(sys, "argv", ["orze", *argv])
    assert c.cli.main() == 2
    c.cli.detect_all_gpus.assert_not_called()
    c.session.stop_controller.assert_not_called()


@pytest.mark.parametrize("argv", [["stop", "--timeout", "7"], ["--stop", "--timeout", "7"]])
def test_only_completed_operation_type_returns_zero_and_stop_budget_does_not_mutate_config(cli_case, monkeypatch, argv):
    c = cli_case
    monkeypatch.setattr(sys, "argv", ["orze", *argv])
    assert c.cli.main() == 0
    c.session.stop_controller.assert_called_once_with(c.cfg, timeout=7)
    assert c.cfg["timeout"] == 3600
    c.cli.detect_all_gpus.assert_not_called()


@pytest.mark.parametrize("result", [True, None, SimpleNamespace(status="confirmed")])
def test_labels_booleans_and_none_are_not_completed_operations(cli_case, result):
    c = cli_case
    c.session.stop_controller.return_value = result
    assert c.cli._stop_controller_command(c.cfg, 7) == 75


def test_enabled_profile_skips_default_admin_and_uses_declared_gpus(cli_case, monkeypatch):
    c = cli_case
    fake = ModuleType("orze.engine.orchestrator")
    run = Mock()
    fake.Orze = Mock(return_value=SimpleNamespace(run=run))
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    import threading
    monkeypatch.setattr(threading, "Thread", Mock(side_effect=AssertionError("no background admin")))
    monkeypatch.setattr(sys, "argv", ["orze"])
    c.cli.main()
    fake.Orze.assert_called_once_with([2, 4], c.cfg, once=False)
    run.assert_called_once()
    c.cli.detect_all_gpus.assert_not_called()


def test_disabled_stop_keeps_legacy_request_only_contract(cli_case, monkeypatch):
    c = cli_case
    c.cfg["controller_control"] = None
    import orze.lifecycle
    stop = Mock()
    monkeypatch.setattr(orze.lifecycle, "do_stop", stop)
    assert c.cli._stop_controller_command(c.cfg, 7) == 75
    stop.assert_called_once_with(c.cfg, timeout=7)
    c.session.stop_controller.assert_not_called()
