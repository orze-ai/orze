"""The local profile does not materialize roles after freezing configuration."""
import pytest
from test_controller_profile import supported
from orze.core.controller_profile import ControllerProfileError, controller_profile


def test_dynamic_strategy_team_is_not_a_supported_frozen_profile(tmp_path):
    cfg = supported(tmp_path)
    cfg["role_presets"] = ["strategy_team"]
    with pytest.raises(ControllerProfileError, match="dynamic_roles_unsupported"):
        controller_profile(cfg)
