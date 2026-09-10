"""New explicit artifact-publication declarations, not old artifact API bugs."""
from copy import deepcopy

import pytest

from orze.core.config import DEFAULT_CONFIG, _validate_config, load_project_config


def contract(path="output.bin", maximum=128):
    return {"version": 1, "outputs": {"result": {"path": path, "max_bytes": maximum}}}


@pytest.mark.parametrize("declaration", [
    False, [], {"version": True, "outputs": {}}, {"version": 2, "outputs": {}},
    {"version": 1, "outputs": []}, {"outputs": {}},
    contract("../other.bin"), contract("/tmp/other.bin"),
    contract(maximum=True), contract(maximum=0), contract(maximum=2**40 + 1),
    {**contract(), "infer_outputs": True},
    {"version": 1, "outputs": {str(i): {"path": f"{i}.bin", "max_bytes": 1} for i in range(33)}},
    {"version": 1, "outputs": {name: {"path": name, "max_bytes": 2**40} for name in ("a", "b")}},
])
def test_invalid_artifact_contract_is_not_silently_ignored(declaration):
    errors, _ = _validate_config({"artifact_contract": declaration})
    assert any(message.startswith("artifact_contract") for message in errors)


@pytest.mark.parametrize("declaration", [None, {"version": 1, "outputs": {}}, contract()])
def test_explicit_artifact_contract_is_recognized_without_mutation(declaration):
    cfg = {"artifact_contract": deepcopy(declaration)}
    errors, warnings = _validate_config(cfg)
    assert not any(message.startswith("artifact_contract") for message in errors)
    assert not any("Unknown config key 'artifact_contract'" in message for message in warnings)
    assert cfg["artifact_contract"] == declaration


def test_artifact_publication_is_opt_in_and_loader_keeps_the_declaration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "artifact_contract" in DEFAULT_CONFIG
    assert DEFAULT_CONFIG["artifact_contract"] is None
    path = tmp_path / "orze.yaml"
    path.write_text("artifact_contract:\n  version: 1\n  outputs: {}\n", encoding="utf-8")
    cfg = load_project_config(str(path))
    assert cfg["artifact_contract"] == {"version": 1, "outputs": {}}
    assert DEFAULT_CONFIG["artifact_contract"] is None
