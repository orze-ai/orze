"""Narrow profile entry omissions, using metadata and real CLI/OS-spy seams."""
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from test_controller_profile import cli_case, supported
from orze.core.controller_profile import ControllerProfileError, controller_profile


@pytest.mark.parametrize("name", ["_managed_idea_id", "_managed_idea_gpu"])
def test_managed_run_private_keys_do_not_bypass_profile_mode_gate(tmp_path, name):
    cfg = supported(tmp_path)
    cfg[name] = None
    with pytest.raises(ControllerProfileError, match="managed_mode_unsupported"):
        controller_profile(cfg)


@pytest.mark.parametrize("argv", [[], ["stop"]])
def test_profile_cli_loads_once_and_never_runs_first_time_star_process(cli_case, monkeypatch, argv):
    c = cli_case
    import orze.extensions
    monkeypatch.setattr(orze.extensions, "_find_pro_key", lambda: "")
    star = Mock()
    load = Mock(return_value=c.cfg)
    monkeypatch.setattr(c.cli, "maybe_star", star)
    monkeypatch.setattr(c.cli, "load_project_config", load)
    fake = ModuleType("orze.engine.orchestrator")
    fake.Orze = Mock(return_value=SimpleNamespace(run=Mock()))
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(sys, "argv", ["orze", *argv])
    c.cli.main()
    load.assert_called_once_with(None)
    star.assert_not_called()
    c.cli.detect_all_gpus.assert_not_called()
