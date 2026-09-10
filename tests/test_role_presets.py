"""Explicit preset mechanism acceptance, not old-API absence evidence."""
import copy

import pytest
import yaml

from orze.core.config import DEFAULT_CONFIG, _validate_config, load_project_config
from orze.core.role_presets import configured_role_presets, role_preset_enabled


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "orze.yaml"

    def load(value):
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        return load_project_config(str(path))

    return load


def _all_credentials(monkeypatch):
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "TEST_ONLY_NOT_A_CREDENTIAL")


def test_default_presets_are_empty_and_config_loads_do_not_share_mutable_state(project, monkeypatch):
    _all_credentials(monkeypatch)
    first = project({})
    assert first["role_presets"] == [] and first["roles"] == {}
    first["role_presets"].append("environment_research")
    first["roles"]["temporary"] = {"enabled": False}
    second = project({})
    assert second["role_presets"] == [] and second["roles"] == {}
    assert DEFAULT_CONFIG["role_presets"] == [] and DEFAULT_CONFIG["roles"] == {}


def test_explicit_environment_preset_keeps_existing_three_backend_defaults(project, monkeypatch):
    _all_credentials(monkeypatch)
    cfg = project({"role_presets": ["environment_research"]})
    assert cfg["roles"] == {
        "research_gemini": {"mode": "research", "backend": "gemini", "model": "gemini-2.5-flash"},
        "research_openai": {"mode": "research", "backend": "openai", "model": "gpt-4o"},
        "research_anthropic": {"mode": "research", "backend": "anthropic"},
    }


@pytest.mark.parametrize("explicit", [
    {"enabled": False},
    {"mode": "research", "backend": "custom", "model": "chosen-model", "env": {"CUSTOM_FLAG": "keep"}},
])
def test_preset_preserves_explicit_role_as_a_whole_and_adds_other_missing_names(
        project, monkeypatch, explicit):
    _all_credentials(monkeypatch)
    before = copy.deepcopy(explicit)
    cfg = project({"role_presets": ["environment_research"], "roles": {
        "research_openai": explicit, "worker": {"mode": "script", "script": "explicit.py"},
    }})
    assert cfg["roles"]["research_openai"] == before
    assert cfg["roles"]["worker"] == {"mode": "script", "script": "explicit.py"}
    assert {"research_gemini", "research_anthropic"} <= set(cfg["roles"])
    assert explicit == before


def test_legacy_research_declaration_survives_explicit_preset_expansion(project, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "TEST_ONLY_NOT_A_CREDENTIAL")
    cfg = project({"role_presets": ["environment_research"],
                   "research": {"mode": "script", "script": "legacy-explicit.py"}})
    assert cfg["roles"]["research"] == {"mode": "script", "script": "legacy-explicit.py"}
    assert cfg["roles"]["research_openai"] == {
        "mode": "research", "backend": "openai", "model": "gpt-4o"}


def test_strategy_only_is_recognized_but_core_does_not_invent_pro_team(project, monkeypatch):
    _all_credentials(monkeypatch)
    cfg = project({"role_presets": ["strategy_team"]})
    assert cfg["roles"] == {}
    assert role_preset_enabled(cfg, "strategy_team") is True
    assert role_preset_enabled(cfg, "environment_research") is False
    errors, warnings = _validate_config(cfg)
    assert not any("role_presets" in item for item in errors)
    assert not any("Unknown config key 'role_presets'" in item for item in warnings)


def test_environment_preset_without_credentials_does_not_create_unrunnable_roles(project):
    cfg = project({"role_presets": ["environment_research"]})
    assert cfg["roles"] == {}


def test_preset_loaded_project_does_not_contaminate_next_project_or_defaults(project, monkeypatch):
    _all_credentials(monkeypatch)
    first = project({"role_presets": ["environment_research"]})
    first["roles"]["research_openai"]["model"] = "local-override"
    second = project({"roles": {}})
    third = project({"role_presets": ["environment_research"]})
    assert second["roles"] == {}
    assert third["roles"]["research_openai"]["model"] == "gpt-4o"


@pytest.mark.parametrize("invalid", [
    None, False, "environment_research", {"strategy_team": True}, [False],
    ["unknown"], ["ENVIRONMENT_RESEARCH"], ["strategy_team", "strategy_team"],
    ["environment_research", "unknown"],
])
def test_invalid_presets_reject_configuration_and_runtime_gate_fails_closed(
        project, monkeypatch, invalid):
    _all_credentials(monkeypatch)
    cfg = {"role_presets": invalid}
    before = copy.deepcopy(cfg)
    with pytest.raises(ValueError, match="role_presets"):
        project(cfg)
    with pytest.raises(ValueError, match="role_presets"):
        configured_role_presets(cfg)
    assert role_preset_enabled(cfg, "environment_research") is False
    assert role_preset_enabled(cfg, "strategy_team") is False
    errors, _ = _validate_config(cfg)
    assert any("role_presets" in item for item in errors)
    assert cfg == before


def test_shared_runtime_predicate_accepts_only_exact_names_and_valid_whole_list():
    cfg = {"role_presets": ["environment_research", "strategy_team"]}
    assert configured_role_presets(cfg) == ("environment_research", "strategy_team")
    assert role_preset_enabled(cfg, "strategy_team") is True
    assert role_preset_enabled(cfg, "unknown") is False
    assert role_preset_enabled({}, "strategy_team") is False
    assert role_preset_enabled(None, "strategy_team") is False
