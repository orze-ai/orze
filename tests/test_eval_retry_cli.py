"""CLI-only contract for explicit evaluation retry admission.

Coordinator and existing-only lake opener are replaced at their public API
boundaries. These tests do not claim to validate their lifecycle transactions
or SQLite safety; independent integration tests exercise those implementations.
Before retry-eval exists, parser failures are missing capability, not a
behavioral red test of an existing retry mechanism.
"""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import orze.cli as cli
import orze.extensions as extensions


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "caller"
    outside.mkdir()
    config = root / "orze.yaml"
    values = {
        "results_dir": "results",
        "idea_lake_db": "state/authority.sqlite3",
        "eval_script": "evaluate.py",
        "eval_output": "assessment.json",
        "auto_seal_eval": False,
        "roles": {"operator": {"mode": "script", "script": "unused.py"}},
        "report": {
            "primary_metric": "arbitrary_score", "sort": "ascending",
            "columns": [{
                "key": "arbitrary_score", "source": "assessment.json:value",
            }],
        },
        "metric_validation": {"min_value": {"arbitrary_score": -3}},
        "custom_policy": {"zero": 0, "explicit": False},
    }
    config.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.chdir(outside)
    monkeypatch.setattr("orze.core.config._load_dotenv", lambda _path: 0)

    service = ModuleType("orze.engine.evaluation_retry")

    class EvaluationRetryError(ValueError):
        pass

    service.EvaluationRetryError = EvaluationRetryError
    service.request_evaluation_retry = Mock(return_value={
        "idea_id": "idea-retry", "request": "accepted", "cost": 0,
    })
    opener_module = ModuleType("orze.core.evaluation_retry_state")
    lake = SimpleNamespace(close=Mock())
    opener_module.open_existing_lake = Mock(return_value=lake)
    monkeypatch.setitem(sys.modules, service.__name__, service)
    monkeypatch.setitem(sys.modules, opener_module.__name__, opener_module)
    boundaries = []
    for owner, name in (
        (cli, "detect_all_gpus"), (cli, "maybe_star"),
        (extensions, "_find_pro_key"), (extensions, "_auto_install_pro"),
        (extensions, "get_extension"),
    ):
        boundary = Mock(side_effect=AssertionError("retry-eval crossed a launch/paid boundary"))
        monkeypatch.setattr(owner, name, boundary)
        boundaries.append(boundary)
    try:
        yield SimpleNamespace(
            root=root, outside=outside, config=config, values=values,
            service=service, opener=opener_module.open_existing_lake, lake=lake,
        )
    finally:
        for boundary in boundaries:
            boundary.assert_not_called()


def _run(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["orze", *arguments])
    return cli.main()


@pytest.mark.parametrize("config_position", ["before", "after", "both"])
def test_retry_eval_routes_complete_absolute_project_config_before_paid_or_gpu(
    project, monkeypatch, capsys, config_position,
):
    p = project
    if config_position == "before":
        arguments = ["-c", str(p.config), "retry-eval", "idea-retry"]
    elif config_position == "after":
        arguments = ["retry-eval", "idea-retry", "-c", str(p.config)]
    else:
        arguments = [
            "-c", str(p.outside / "not-this-project.yaml"), "retry-eval",
            "idea-retry", "-c", str(p.config),
        ]

    assert _run(monkeypatch, arguments) == 0

    p.opener.assert_called_once()
    assert Path(p.opener.call_args.args[0]) == p.root / "state" / "authority.sqlite3"
    p.service.request_evaluation_retry.assert_called_once()
    idea_id, results, cfg, lake = p.service.request_evaluation_retry.call_args.args
    assert idea_id == "idea-retry"
    assert results == p.root / "results"
    assert cfg["results_dir"] == str(results)
    assert cfg["idea_lake_db"] == str(p.root / "state" / "authority.sqlite3")
    assert cfg["_config_path"] == str(p.config)
    assert cfg["_project_root"] == str(p.root)
    assert cfg["report"]["primary_metric"] == "arbitrary_score"
    assert cfg["report"]["columns"] == p.values["report"]["columns"]
    assert cfg["metric_validation"] == p.values["metric_validation"]
    assert cfg["custom_policy"] == {"zero": 0, "explicit": False}
    assert "poll" in cfg  # Complete defaults, not a hand-built report subset.
    assert lake is p.lake
    p.lake.close.assert_called_once()
    assert Path.cwd() == p.outside
    assert json.loads(capsys.readouterr().out) == p.service.request_evaluation_retry.return_value


def test_retry_eval_uses_default_project_config_without_reinterpreting_the_result(
    project, monkeypatch, capsys,
):
    monkeypatch.chdir(project.root)

    assert _run(monkeypatch, ["retry-eval", "idea-retry"]) == 0

    cfg = project.service.request_evaluation_retry.call_args.args[2]
    assert cfg["_config_path"] == str(project.config)
    assert json.loads(capsys.readouterr().out)["cost"] == 0
    assert Path.cwd() == project.root


def test_retry_eval_keeps_explicit_absolute_results_and_lake_paths(project, monkeypatch):
    results = project.outside / "explicit-results"
    database = project.outside / "authority" / "explicit.sqlite3"
    project.values.update(results_dir=str(results), idea_lake_db=str(database))
    project.config.write_text(json.dumps(project.values), encoding="utf-8")

    assert _run(monkeypatch, ["retry-eval", "idea-retry", "-c", str(project.config)]) == 0

    assert Path(project.opener.call_args.args[0]) == database
    assert project.service.request_evaluation_retry.call_args.args[1] == results


def test_retry_eval_coordinator_rejection_returns_two_and_closes_lake(
    project, monkeypatch, capsys,
):
    project.service.request_evaluation_retry.side_effect = (
        project.service.EvaluationRetryError("training_not_completed")
    )

    assert _run(monkeypatch, ["retry-eval", "idea-retry", "-c", str(project.config)]) == 2

    assert "training_not_completed" in capsys.readouterr().out
    project.lake.close.assert_called_once()
    assert Path.cwd() == project.outside


@pytest.mark.parametrize("rejection", [ValueError("database_missing"), OSError("database_unreadable")])
def test_retry_eval_existing_database_rejection_cannot_fall_back_or_create_files(
    project, monkeypatch, capsys, rejection,
):
    project.opener.side_effect = rejection
    database = project.root / "state" / "authority.sqlite3"
    assert not database.exists()

    assert _run(monkeypatch, ["retry-eval", "idea-retry", "-c", str(project.config)]) == 2

    project.opener.assert_called_once()
    project.service.request_evaluation_retry.assert_not_called()
    project.lake.close.assert_not_called()
    assert str(rejection) in capsys.readouterr().out
    assert not database.exists()
    assert not database.parent.exists()
    assert Path.cwd() == project.outside


def test_retry_eval_missing_named_config_does_not_use_another_projects_defaults(
    project, monkeypatch, capsys,
):
    missing = project.outside / "absent.yaml"

    assert _run(monkeypatch, ["retry-eval", "idea-retry", "-c", str(missing)]) == 2

    project.opener.assert_not_called()
    project.service.request_evaluation_retry.assert_not_called()
    assert "config" in capsys.readouterr().out.lower()
    assert not missing.exists()
