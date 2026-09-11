"""Derived launch routes are fixed inputs, not transient private metadata."""
import copy

import yaml

from test_controller_profile import supported
from orze.core import config
from orze.core.controller_profile import profile_fingerprint


def test_loaded_derived_routes_bind_fingerprint_before_redirect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_load_dotenv", lambda path: 0)
    path = tmp_path / "orze.yaml"
    path.write_text(yaml.safe_dump(supported(tmp_path)))
    cfg = config.load_project_config(str(path))
    baseline = profile_fingerprint(cfg)
    assert baseline == cfg["_controller_profile_fingerprint"]
    original_trigger = config.orze_path(cfg, "triggers", "_trigger_owned")
    assert original_trigger.parent == tmp_path / "results"
    for name in ("_env_ORZE_DIR", "_env_ORZE_RESULTS_DIR", "_env_ORZE_IDEAS_FILE",
                 "_env_ORZE_RULES_DIR", "_env_ORZE_METHODS_DIR",
                 "_env_ORZE_KNOWLEDGE_DIR", "_env_ORZE_FEEDBACK_DIR"):
        changed = copy.deepcopy(cfg)
        changed[name] = str(tmp_path / ("redirected-" + name))
        if name == "_env_ORZE_RESULTS_DIR":
            assert config.orze_path(changed, "triggers", "_trigger_owned").parent != original_trigger.parent
        assert profile_fingerprint(changed) != baseline, name
        missing = copy.deepcopy(cfg)
        del missing[name]
        assert profile_fingerprint(missing) != baseline, name
