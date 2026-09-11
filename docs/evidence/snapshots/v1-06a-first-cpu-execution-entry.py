"""Explicit CPU entry requirements; these are new product capabilities."""
import copy

import pytest
import yaml

from orze.core.config import DEFAULT_CONFIG, _validate_config, load_project_config


def declaration():
    return {"version": 1, "resource": "cpu", "slots": 1,
            "wall_budget_seconds": 20}


def test_cpu_config_does_not_require_training_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "orze.yaml"
    path.write_text(yaml.safe_dump({"execution": declaration(),
        "results_dir": str(tmp_path / "results")}), encoding="utf-8")
    cfg = load_project_config(str(path))
    errors, _ = _validate_config(cfg)
    assert errors == []


@pytest.mark.parametrize("update", [
    {"version": True}, {"resource": "gpu"}, {"slots": True},
    {"slots": 0}, {"wall_budget_seconds": float("nan")},
    {"unknown": 1},
])
def test_cpu_declaration_rejected_by_existing_validator(update):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(train_script=None, base_config=None,
               execution={**declaration(), **update})
    errors, _ = _validate_config(cfg)
    assert any("execution" in error for error in errors)


def test_cli_explicit_cpu_bypasses_gpu_inventory(tmp_path, monkeypatch):
    import orze.cli as cli
    observed = {}

    class Runner:
        def __init__(self, gpu_ids, cfg, once=False):
            observed.update(gpu_ids=gpu_ids, once=once)

        def run(self):
            observed["ran"] = True

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "orze.yaml"
    path.write_text(yaml.safe_dump({"execution": declaration(),
        "results_dir": str(tmp_path / "results")}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["orze", "-c", str(path), "--once", "--no-admin"])
    # Existing credential/star UX is not the behavior under test.
    monkeypatch.setattr("orze.extensions._find_pro_key", lambda: "fixture-present")
    monkeypatch.setattr(cli, "detect_all_gpus",
                        lambda: pytest.fail("CPU entry inventoried GPUs"))
    monkeypatch.setattr("orze.engine.orchestrator.Orze", Runner)
    cli.main()
    assert observed == {"gpu_ids": [], "once": True, "ran": True}


def test_empty_gpu_allowlist_does_not_infer_cpu_mode(monkeypatch):
    import orze.cli as cli
    seen = []
    monkeypatch.setattr("sys.argv", ["orze", "--once", "--no-admin"])
    monkeypatch.setattr("orze.extensions._find_pro_key", lambda: "fixture-present")
    monkeypatch.setattr(cli, "load_project_config", lambda _: {
        "gpu_scheduling": {"allowed_gpus": []}})
    monkeypatch.setattr(cli, "detect_all_gpus", lambda: seen.append("inventory") or [])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 1
    assert seen == ["inventory"]
