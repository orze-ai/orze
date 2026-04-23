"""F5: launch-time nested-config validator."""
import pytest

from orze.engine.launcher import (
    validate_idea_config_no_nested,
    _NESTED_CONFIG_WHITELIST,
)


def test_flat_config_passes():
    cfg = {"learning_rate": 0.001, "epochs": 5, "model_name": "vjepa2"}
    assert validate_idea_config_no_nested(cfg) is None


def test_nested_dict_rejected():
    cfg = {"backbone": {"name": "vjepa2", "frozen": True}, "lr": 0.001}
    err = validate_idea_config_no_nested(cfg)
    assert err is not None
    assert "nested_config_not_allowed" in err
    assert "backbone" in err


def test_multiple_nested_dicts_listed():
    cfg = {"backbone": {"x": 1}, "data": {"y": 2}, "lr": 0.001}
    err = validate_idea_config_no_nested(cfg)
    assert "backbone" in err and "data" in err


def test_whitelisted_keys_allowed():
    cfg = {"ema": {"decay": 0.999}, "augmentation": {"flip": True},
           "lr": 0.001}
    assert validate_idea_config_no_nested(cfg) is None


def test_extra_whitelist_extends_default():
    cfg = {"my_subconfig": {"x": 1}, "lr": 0.001}
    assert validate_idea_config_no_nested(cfg) is not None
    assert validate_idea_config_no_nested(
        cfg, extra_whitelist=["my_subconfig"]) is None


def test_lists_are_allowed():
    cfg = {"layers": [64, 128, 256], "lr": 0.001}
    assert validate_idea_config_no_nested(cfg) is None


def test_non_dict_input_no_crash():
    assert validate_idea_config_no_nested(None) is None
    assert validate_idea_config_no_nested("not a dict") is None


def test_default_whitelist_minimal():
    """Document the whitelist — keep it small and intentional."""
    assert "ema" in _NESTED_CONFIG_WHITELIST
    assert "augmentation" in _NESTED_CONFIG_WHITELIST
    # Should NOT include common nested-config keys we explicitly want to reject:
    assert "backbone" not in _NESTED_CONFIG_WHITELIST
    assert "data" not in _NESTED_CONFIG_WHITELIST
    assert "model" not in _NESTED_CONFIG_WHITELIST
