"""V104B new configuration requirements, not historical provider-limit bugs."""
from copy import deepcopy

import pytest

from orze.core.config import DEFAULT_CONFIG, _validate_config, load_project_config


@pytest.mark.parametrize("declaration", [
    None, False, [], "unlimited", {"max_bytes": None}, {"max_bytes": False},
    {"max_bytes": 0}, {"max_bytes": -1}, {"max_bytes": 1.5},
    {"max_bytes": "131072"}, {"max_bytes": 2097153}, {"max_bytes": float("inf")},
    {"max_bytes": 8192, "max_tokens": 4000}, {1: 8192},
])
def test_validator_rejects_invalid_explicit_prompt_budget(declaration):
    errors, _ = _validate_config({"research_prompt": declaration})
    assert any(error.startswith("research_prompt") for error in errors)


@pytest.mark.parametrize("declaration", [{}, {"max_bytes": 1}, {"max_bytes": 2097152}])
def test_validator_accepts_explicit_bounded_prompt_budget(declaration):
    cfg = {"research_prompt": deepcopy(declaration)}
    errors, warnings = _validate_config(cfg)
    assert not any(error.startswith("research_prompt") for error in errors)
    assert not any("Unknown config key 'research_prompt'" in warning for warning in warnings)
    assert cfg["research_prompt"] == declaration


def test_default_prompt_budget_is_explicit_and_loader_preserves_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert DEFAULT_CONFIG.get("research_prompt") == {"max_bytes": 131072}
    path = tmp_path / "orze.yaml"
    path.write_text("research_prompt:\n  max_bytes: 9000\n", encoding="utf-8")
    cfg = load_project_config(str(path))
    assert cfg["research_prompt"] == {"max_bytes": 9000}
    assert DEFAULT_CONFIG["research_prompt"] == {"max_bytes": 131072}
