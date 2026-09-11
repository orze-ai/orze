"""Actual two-file CLI routing and a private-entry result contract boundary.

Both configuration files use the real loader. Operation/constructor seams are
explicit doubles; these tests do not grant or prove controller handoff itself.
The legacy parameter isolates the parser behavior without any v2 API need.
"""
import copy
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from orze.core.control_outcome import StopOutcome
from test_controller_handoff_cli import handoff, invoke


@pytest.mark.parametrize("version", [None, 2], ids=["legacy", "v2"])
def test_restart_selected_configuration_is_identical_before_or_after_subcommand(handoff, monkeypatch, version):
    import orze.lifecycle
    c = handoff
    raw = copy.deepcopy(c.cfg)
    raw["controller_control"] = None if version is None else {"version": 2, "profile": "local_handoff_v1"}
    raw["timeout"] = 101
    c.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    raw["timeout"] = 202
    selected = c.path.with_name("selected.yaml")
    selected.write_text(yaml.safe_dump(raw), encoding="utf-8")
    legacy = Mock(return_value=StopOutcome("requested", "stop_requested"))
    monkeypatch.setattr(orze.lifecycle, "do_restart", legacy)
    service = legacy if version is None else c.module.restart_controller
    suffix = [] if version is None else ["--request-id", "selected-project"]
    result = 75 if version is None else 0
    assert invoke(monkeypatch, "restart", "-c", str(selected), *suffix) == result
    direct_cfg = service.call_args.args[0]
    assert direct_cfg["timeout"] == 202
    assert invoke(monkeypatch, "-c", str(selected), "restart", *suffix) == result
    global_cfg = service.call_args.args[0]
    assert global_cfg["timeout"] == 202
    assert global_cfg == direct_cfg
    assert Path(global_cfg["_config_path"]) == selected
    c.constructor.assert_not_called()


def test_private_entry_non_none_result_cannot_continue_to_start_gates(handoff, monkeypatch):
    c = handoff
    monkeypatch.setenv("ORZE_CONTROLLER_HANDOFF_FD", "3")
    c.module.prepare_successor_entry.return_value = False
    assert invoke(monkeypatch, "--no-admin") == 75
    c.module.prepare_successor_entry.assert_called_once()
    c.sentinel.assert_not_called()
    c.constructor.assert_not_called()
