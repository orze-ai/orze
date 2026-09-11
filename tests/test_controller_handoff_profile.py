"""New v2 metadata requirements; no controller or process authority is mocked."""
import copy

import pytest
import yaml

from orze.core import config
from orze.core.controller_profile import (
    ControllerProfileError, controller_profile, profile_fingerprint,
)
from test_controller_profile import supported


def test_v2_is_explicit_detached_and_stamped_by_the_real_loader(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = supported(tmp_path)
    v1 = profile_fingerprint(cfg)
    cfg["controller_control"] = {"version": 2, "profile": "local_handoff_v1"}
    declaration = controller_profile(cfg)
    assert declaration == cfg["controller_control"]
    assert declaration is not cfg["controller_control"]
    assert profile_fingerprint(cfg) != v1
    path = tmp_path / "orze.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    loaded = config.load_project_config(str(path))
    assert loaded["controller_control"] == declaration
    assert loaded["_controller_profile_fingerprint"] == profile_fingerprint(loaded)
    assert config.DEFAULT_CONFIG["controller_control"] is None


@pytest.mark.parametrize("declaration", [
    {"version": 1, "profile": "local_handoff_v1"},
    {"version": 2, "profile": "local_stop_v1"},
    {"version": True, "profile": "local_handoff_v1"},
    {"version": 2, "profile": "local_handoff_v1", "upgrade": True},
])
def test_v2_does_not_coerce_or_upgrade_other_declarations(tmp_path, declaration):
    cfg = supported(tmp_path)
    cfg["controller_control"] = declaration
    with pytest.raises(ControllerProfileError, match="declaration_invalid"):
        controller_profile(cfg)


def test_v2_retains_strict_feature_and_gpu_fingerprint_boundaries(tmp_path):
    cfg = supported(tmp_path)
    cfg["controller_control"] = {"version": 2, "profile": "local_handoff_v1"}
    expected = profile_fingerprint(cfg)
    for key, value in (("telemetry", True), ("max_fix_attempts", 1),
                       ("containers", False), ("_managed_idea_id", None),
                       ("role_presets", ["strategy_team"])):
        changed = copy.deepcopy(cfg)
        changed[key] = value
        with pytest.raises(ControllerProfileError):
            controller_profile(changed)
    with pytest.raises(ControllerProfileError, match="gpu_scope_changed"):
        profile_fingerprint(cfg, [2])
    for key, value in (("timeout", 9), ("_env_ORZE_RESULTS_DIR", str(tmp_path / "other"))):
        assert profile_fingerprint(dict(cfg, **{key: value})) != expected
    assert profile_fingerprint(cfg, [4, 2]) == expected
