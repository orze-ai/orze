"""New runner configuration handoff and procedure-selection mechanisms.

These are new-mechanism acceptances; the separately frozen registry test
records the old-release behavioral failure. Every main/step invocation is
real, with tiny project procedures and no external execution.
"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from orze.fsm import engine, runner


@pytest.fixture
def project(tmp_path, monkeypatch):
    caller = tmp_path / "caller"
    root = tmp_path / "selected"
    caller.mkdir()
    root.mkdir()
    results = root / "nested" / "results"
    results.mkdir(parents=True)
    cfg = {
        "results_dir": "nested/results", "ideas_file": "queue/ideas.md",
        "idea_lake_db": "authority/custom.db",
        "roles": {"disabled_fixture": {"enabled": False}},
        "idea_review": {"enabled": True},
        "full_policy_probe": {"custom": "must-survive", "sensitive": "SYNTHETIC_PRIVATE_VALUE"},
    }
    config = root / "orze.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    procedures = root / "procedures"
    procedures.mkdir()
    _procedure(procedures / "observer.yaml", "observer", guards=["observe_cfg"])
    calls = []
    monkeypatch.chdir(caller)
    monkeypatch.delenv("ORZE_CONFIG_PATH", raising=False)
    monkeypatch.setattr(engine, "_guards", {})
    monkeypatch.setattr(engine, "_actions", {})
    monkeypatch.setattr(engine, "_activity_log_path", None)
    monkeypatch.setattr(runner, "_find_pro_package", lambda: None)

    @engine.guard("observe_cfg")
    def observe(ctx):
        calls.append(deepcopy(ctx.extras.get("cfg")))
        return "configuration observed"

    def forbidden(*args, **kwargs):
        pytest.fail("FSM configuration test must not execute a subprocess")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    return SimpleNamespace(root=root, caller=caller, results=results, cfg=cfg,
                           config=config, procedures=procedures, calls=calls)


def _procedure(path, name, *, guards=None, maintain=None):
    path.write_text(yaml.safe_dump({
        "name": name, "initial": "waiting",
        "states": [{"name": "waiting", "maintain": maintain or [],
                    "transitions": ([{"to": "seen", "guards": guards}]
                                    if guards else [])},
                   {"name": "seen"}],
    }), encoding="utf-8")


def _main(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["orze-fsm", *map(str, arguments)])
    return runner.main()


@pytest.mark.parametrize("source", ["cli", "environment", "cwd"])
def test_real_main_loads_selected_full_policy_into_ephemeral_context(project, monkeypatch, source):
    p = project
    arguments = []
    if source == "cli":
        monkeypatch.setenv("ORZE_CONFIG_PATH", str(p.caller / "wrong-missing.yaml"))
        arguments = ["-c", p.config, "--results-dir", p.results / "."]
    elif source == "environment":
        monkeypatch.setenv("ORZE_CONFIG_PATH", str(p.config))
    else:
        monkeypatch.chdir(p.root)
    before = Path.cwd()

    _main(monkeypatch, arguments)

    assert len(p.calls) == 1
    cfg = p.calls[0]
    assert cfg["full_policy_probe"] == p.cfg["full_policy_probe"]
    assert cfg["_config_path"] == str(p.config)
    assert cfg["results_dir"] == str(p.results)
    assert cfg["ideas_file"] == str(p.root / "queue" / "ideas.md")
    assert cfg["idea_lake_db"] == str(p.root / "authority" / "custom.db")
    assert cfg["_orze_dir"] == str(p.root / "nested" / ".orze")
    assert Path.cwd() == before
    durable = (p.results / "_fsm_observer.json").read_text(encoding="utf-8")
    activity = (p.results / "_fsm_activity.jsonl").read_text(encoding="utf-8")
    assert json.loads(durable)["current"] == "seen"
    assert "SYNTHETIC_PRIVATE_VALUE" not in durable + activity
    assert "full_policy_probe" not in durable + activity
    assert "extras" not in json.loads(durable)
    assert not (p.root / "authority").exists()


def test_results_scope_conflict_rejects_before_plugins_or_procedure_actions(project, monkeypatch):
    other = project.caller / "other-results"
    other.mkdir()
    with pytest.raises(SystemExit) as raised:
        _main(monkeypatch, ["--config", project.config, "--results-dir", other])
    assert raised.value.code == 1
    assert project.calls == []
    assert not list(other.iterdir())
    assert not list(project.results.iterdir())


@pytest.mark.parametrize("selection", ["missing.yaml", ""])
def test_missing_explicit_config_cannot_fall_back_to_present_cwd_config(project, monkeypatch, selection):
    monkeypatch.chdir(project.root)
    with pytest.raises(SystemExit) as raised:
        _main(monkeypatch, ["--config", selection])
    assert raised.value.code == 1
    assert project.calls == []
    assert not list(project.results.iterdir())


def test_no_config_keeps_explicit_legacy_procedure_without_cfg_authority(project, monkeypatch):
    _main(monkeypatch, ["--results-dir", project.results,
                        "--procedures-dir", project.procedures])
    assert project.calls == [None]
    assert json.loads((project.results / "_fsm_observer.json").read_text())["current"] == "seen"


@pytest.mark.parametrize("fsm", [None, {"procedures": "quality_gate"},
                                 {"procedures": [True]}, {"procedures": ["quality_gate.yaml"]}])
def test_invalid_builtin_selection_fails_before_actions(project, monkeypatch, fsm):
    project.cfg["fsm"] = fsm
    project.config.write_text(yaml.safe_dump(project.cfg), encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        _main(monkeypatch, ["--config", project.config])
    assert raised.value.code == 1
    assert project.calls == []
    assert not list(project.results.iterdir())


@pytest.mark.parametrize("policy,expected", [
    ({}, ["activity_log"]),
    ({"idea_review": {"enabled": True}}, ["activity_log", "idea_verifier"]),
    ({"fsm": {"procedures": ["quality_gate", "idea_verifier"]}},
     ["activity_log", "quality_gate"]),
])
def test_actual_main_does_not_execute_unselected_bundled_maintain(
        project, monkeypatch, policy, expected):
    p = project
    pro = p.caller / "synthetic-pro-package"
    procedures = pro / "procedures"
    procedures.mkdir(parents=True)
    executed = []
    for name in ("activity_log", "idea_verifier", "quality_gate"):
        _procedure(procedures / f"{name}.yaml", name, maintain=[f"run_{name}"])
        engine.action(f"run_{name}")(lambda ctx, name=name: executed.append(name))
    # Restoring an old active state must not execute a disabled maintain action.
    old_quality = '{"current":"waiting","vars":{},"history":[]}'
    (p.results / "_fsm_quality_gate.json").write_text(old_quality, encoding="utf-8")
    p.cfg.pop("idea_review")
    p.cfg.update(policy)
    p.config.write_text(yaml.safe_dump(p.cfg), encoding="utf-8")
    monkeypatch.setattr(runner, "_find_pro_package", lambda: pro)

    _main(monkeypatch, ["--config", p.config])

    assert executed == expected
    assert len(p.calls) == 1  # Explicit project procedure was retained.
    if "quality_gate" not in expected:
        assert (p.results / "_fsm_quality_gate.json").read_text() == old_quality


def test_project_override_remains_explicit_even_when_builtin_not_selected(project, monkeypatch):
    p = project
    pro = p.caller / "synthetic-pro-package"
    procedures = pro / "procedures"
    procedures.mkdir(parents=True)
    _procedure(procedures / "quality_gate.yaml", "quality_gate", maintain=["builtin_danger"])
    _procedure(p.procedures / "quality_gate.yaml", "quality_gate", maintain=["project_action"])
    actions = []
    engine.action("builtin_danger")(lambda ctx: actions.append("builtin"))
    engine.action("project_action")(lambda ctx: actions.append("project"))
    monkeypatch.setattr(runner, "_find_pro_package", lambda: pro)

    _main(monkeypatch, ["--config", p.config])

    assert actions == ["project"]
