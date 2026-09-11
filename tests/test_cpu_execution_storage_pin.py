"""New CPU draft pin boundary through actual loader and artifact metadata.

No command is launched. A changed configured artifact root is a real metadata
consumer difference, not evidence of a running worker or external path race.
"""
import pytest
import yaml

from orze.core.config import load_project_config
from orze.core.cpu_action_contract import artifact_binding
from orze.core.cpu_execution import CPUExecutionError, cpu_execution


def test_loaded_cpu_pin_rejects_changed_artifact_control_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "orze.yaml"
    path.write_text(yaml.safe_dump({
        "execution": {"version": 1, "resource": "cpu", "slots": 1,
                      "wall_budget_seconds": 20},
        "results_dir": str(tmp_path / "results"),
    }), encoding="utf-8")
    cfg = load_project_config(str(path))
    action = {"version": 1, "adapter": "command", "purpose": "Metadata only",
              "inputs": {}, "command": ["not-executed"], "timeout_seconds": 1,
              "outputs": {"result": {"path": "result.bin", "max_bytes": 8}}}
    folder = tmp_path / "results" / "action-a"
    before = artifact_binding(cfg, folder, action)
    assert cpu_execution(cfg)["resource"] == "cpu"
    cfg["_orze_dir"] = str(tmp_path / "different-control")
    after = artifact_binding(cfg, folder, action)
    assert before["root"] != after["root"]
    assert before["scope"] == after["scope"]
    assert before["spec_fingerprint"] == after["spec_fingerprint"]
    with pytest.raises(CPUExecutionError, match="loaded CPU configuration changed"):
        cpu_execution(cfg)
