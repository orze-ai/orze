"""Dispatch compatibility around the new early report-only branch.

These are narrow routing tests with explicit public-boundary substitutes;
test_report_only_cli_authority.py separately exercises actual report/DB work.
"""

import sys
from unittest.mock import Mock

import pytest

import orze.cli as cli
import orze.extensions as extensions
from orze.reporting import report_cli


@pytest.mark.parametrize("command", ["pro-status", "retry-eval"])
def test_explicit_subcommand_retains_priority_over_global_report_flag(
    monkeypatch, command,
):
    report = Mock(side_effect=AssertionError("report flag intercepted explicit command"))
    monkeypatch.setattr(report_cli, "run_report_only", report)
    monkeypatch.setattr(extensions, "_find_pro_key", lambda: "test-only-no-key-read")
    monkeypatch.setattr(cli, "maybe_star", Mock(side_effect=AssertionError("unexpected star")))
    if command == "pro-status":
        target = Mock()
        monkeypatch.setattr(cli, "pro_status", target)
        args = ["pro", "status"]
        expected = None
    else:
        target = Mock(return_value=17)
        monkeypatch.setattr(cli, "_run_retry_eval_subcommand", target)
        args = ["retry-eval", "idea-explicit"]
        expected = 17
    monkeypatch.setattr(sys, "argv", ["orze", "--report-only", *args])

    assert cli.main() == expected

    target.assert_called_once()
    report.assert_not_called()


@pytest.mark.parametrize("action", ["--stop", "--restart", "--admin"])
def test_report_only_rejects_conflicting_global_actions_before_side_effects(
    monkeypatch, action, capsys,
):
    tripwire = Mock(side_effect=AssertionError("conflicting flags reached runtime"))
    monkeypatch.setattr(extensions, "_find_pro_key", tripwire)
    monkeypatch.setattr(cli, "detect_all_gpus", tripwire)
    monkeypatch.setattr(report_cli, "load_project_config", tripwire)
    monkeypatch.setattr(sys, "argv", ["orze", "--report-only", action])

    assert cli.main() == 2

    assert "cannot be combined" in capsys.readouterr().err
    tripwire.assert_not_called()
