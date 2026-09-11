"""New public CPU domain configuration requirements, not historical defects."""
import copy

import pytest
import yaml

from orze.core.config import DEFAULT_CONFIG, _validate_config, load_project_config
from orze.core.cpu_execution import CPUExecutionError, cpu_execution


@pytest.mark.parametrize("domain", [
    {"version": True, "kind": "command", "config": {}},
    {"version": 1, "kind": "not_registered", "config": {}},
    {"version": 1, "kind": "command", "config": []},
    {"version": 1, "kind": "command", "config": {}, "extra": 1},
    {"version": 2, "kind": "command", "config": {}},
])
def test_existing_validator_rejects_invalid_explicit_domain(domain):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(train_script=None, base_config=None, ideas_file=".orze/ideas.md",
               execution={"version": 1, "resource": "cpu", "slots": 1,
                          "wall_budget_seconds": 5}, action_domain=domain)
    errors, _ = _validate_config(cfg)
    assert any("execution" in error for error in errors)


def test_domain_requires_explicit_cpu_resource():
    cfg = {"action_domain": {"version": 1, "kind": "command", "config": {}}}
    with pytest.raises(CPUExecutionError):
        cpu_execution(cfg)


def test_loaded_domain_declaration_cannot_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "orze.yaml"
    path.write_text(yaml.safe_dump({
        "execution": {"version": 1, "resource": "cpu", "slots": 1,
                      "wall_budget_seconds": 5},
        "action_domain": {"version": 1, "kind": "command", "config": {}},
        "results_dir": str(tmp_path / "results")}), encoding="utf-8")
    cfg = load_project_config(str(path))
    cfg["action_domain"]["kind"] = "json_observations"
    with pytest.raises(CPUExecutionError):
        cpu_execution(cfg)
