"""New opt-in evaluation IO requirements against the public config entry."""
from copy import deepcopy

import pytest

from orze.core.config import DEFAULT_CONFIG, _validate_config, load_project_config


def contract(**changes):
    value = {"version": 1, "adapter": "orze.json_observations.v1",
             "protocol_id": "local-correctness-v1", "inputs": ["candidate"],
             "output": {"path": "observations.json", "max_bytes": 65536}}
    value.update(changes)
    return value


@pytest.mark.parametrize("declaration", [
    False, [], contract(version=True), contract(version=2), contract(adapter="guess"),
    contract(protocol_id=""), contract(protocol_id="x" * 257), contract(inputs="candidate"),
    contract(inputs=["candidate", "candidate"]), contract(inputs=[True]),
    contract(inputs=[str(i) for i in range(33)]),
    contract(output={"path": "../escape", "max_bytes": 1}),
    contract(output={"path": "/tmp/escape", "max_bytes": 1}),
    contract(output={"path": "out.json", "max_bytes": True}),
    contract(output={"path": "out.json", "max_bytes": 0}),
    contract(output={"path": "out.json", "max_bytes": 1048577}),
    contract(auto_validate_science=True),
])
def test_invalid_observation_declaration_is_rejected(declaration):
    errors, _ = _validate_config({"observation_contract": declaration})
    assert any(message.startswith("observation_contract") for message in errors)


@pytest.mark.parametrize("companion", [
    {"report": {"benchmark_contract": {}}},
    {"evaluation_bundle": {"enabled": True}},
    {"model_lineage": {"enabled": True}},
    {"managed_run": {"require_explicit_untainted_metrics": True}},
])
def test_unwired_contract_combinations_are_not_silently_bypassed(companion):
    errors, _ = _validate_config({"observation_contract": contract(), **companion})
    assert any(message.startswith("observation_contract") for message in errors)


@pytest.mark.parametrize("declaration", [None, contract(), contract(inputs=[])])
def test_supported_observation_declarations_are_recognized_without_mutation(declaration):
    cfg = {"observation_contract": deepcopy(declaration)}
    errors, warnings = _validate_config(cfg)
    assert not any(message.startswith("observation_contract") for message in errors)
    assert not any("Unknown config key 'observation_contract'" in message for message in warnings)
    assert cfg["observation_contract"] == declaration


def test_observation_protocol_is_explicit_and_off_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "observation_contract" in DEFAULT_CONFIG
    assert DEFAULT_CONFIG["observation_contract"] is None
    path = tmp_path / "orze.yaml"
    path.write_text("observation_contract: null\n", encoding="utf-8")
    assert load_project_config(str(path))["observation_contract"] is None

