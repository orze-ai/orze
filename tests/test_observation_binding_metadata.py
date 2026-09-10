"""New generic metadata contracts; not execution or science qualification."""
from copy import deepcopy

import pytest

from orze.core.observation_contract import (
    get_observation_contract, validate_observation_publication_binding,
)
from test_observation_contract_config import contract


def binding(tmp_path):
    return {"adapter_id": "other.domain.adapter.v3", "protocol_fingerprint": "a" * 64,
            "spec_fingerprint": "b" * 64, "scope": str(tmp_path / "results"),
            "input_artifact_ids": ["artifact-a"]}


def test_declarations_are_detached_and_generic_binding_does_not_read_artifacts(tmp_path):
    cfg = {"observation_contract": contract()}
    selected = get_observation_contract(cfg)
    cfg["observation_contract"]["inputs"].append("late-input")
    cfg["observation_contract"]["output"]["max_bytes"] += 1
    assert selected == contract()
    original = binding(tmp_path)
    captured = validate_observation_publication_binding(original)
    original["input_artifact_ids"].append("late-artifact")
    assert captured == binding(tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert get_observation_contract(None) is None
    assert get_observation_contract({}) is None


@pytest.mark.parametrize("change", [
    {"protocol_fingerprint": True}, {"scope": "/tmp/../foreign"},
    {"input_artifact_ids": ["a", "a"]}, {"input_artifact_ids": [True]},
    {"adapter_id": ""}, {"extra": "not permitted"},
])
def test_changed_binding_metadata_is_rejected_without_side_effects(tmp_path, change):
    value = {**binding(tmp_path), **change}
    original = deepcopy(value)
    with pytest.raises(ValueError, match="observation_contract"):
        validate_observation_publication_binding(value)
    assert value == original
    assert list(tmp_path.iterdir()) == []


def test_zero_input_binding_is_explicit_not_an_inferred_artifact(tmp_path):
    value = binding(tmp_path)
    value["input_artifact_ids"] = []
    assert validate_observation_publication_binding(value) == value
    assert get_observation_contract({"observation_contract": contract(inputs=[])})["inputs"] == []
