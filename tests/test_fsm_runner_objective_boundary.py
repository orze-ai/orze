"""An observer policy must not invent an objective from runtime defaults."""
import yaml

from orze.fsm import runner


def test_missing_declared_objective_stays_unconfigured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "orze.yaml"
    path.write_text(yaml.safe_dump({
        "results_dir": "results", "report": {"sort": "ascending"},
        "roles": {"disabled_fixture": {"enabled": False}},
    }), encoding="utf-8")

    cfg = runner._load_config(path)

    assert "primary_metric" not in cfg["report"]
    assert cfg["report"]["sort"] == "ascending"
