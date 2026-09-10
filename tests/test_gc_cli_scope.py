"""Actual GC CLI parsing/configuration, with the destructive consumer spied."""
import json
from pathlib import Path

import pytest
import yaml

from orze.agents import orze_gc


def _config(tmp_path, value):
    project = tmp_path / "project"
    project.mkdir()
    config = project / "orze.yaml"
    config.write_text(yaml.safe_dump(value), encoding="utf-8")
    caller = tmp_path / "caller"
    caller.mkdir()
    return project, config, caller


def _consumer(monkeypatch):
    observed = []
    def consume(**kwargs):
        observed.append(kwargs)
        return {"test_consumer": True}
    monkeypatch.setattr(orze_gc, "run_gc", consume)
    return observed


def test_config_paths_are_anchored_to_selected_project(tmp_path, monkeypatch, capsys):
    project, config, caller = _config(tmp_path, {
        "results_dir": "outputs", "idea_lake_db": "control/catalog.db",
        "gc": {"checkpoints_dir": "checkpoints", "archive_dir": "cold"},
    })
    monkeypatch.chdir(caller)
    monkeypatch.setattr("sys.argv", ["orze-gc", "-c", str(config)])
    observed = _consumer(monkeypatch)

    orze_gc.main()

    assert len(observed) == 1
    call = observed[0]
    assert call["results_dir"] == project / "outputs"
    assert call["checkpoints_dir"] == project / "checkpoints"
    assert call["archive_dir"] == project / "cold"
    assert call["lake_db_path"] == project / "control/catalog.db"
    assert call["cfg"]["_project_root"] == str(project)
    assert call["cfg"]["_config_path"] == str(config)
    assert json.loads(capsys.readouterr().out) == {"test_consumer": True}


def test_missing_selected_config_never_invokes_destructive_consumer(tmp_path, monkeypatch):
    missing = tmp_path / "absent.yaml"
    observed = _consumer(monkeypatch)
    monkeypatch.setattr("sys.argv", ["orze-gc", "-c", str(missing), "--gc-results"])

    with pytest.raises(SystemExit) as error:
        orze_gc.main()

    assert error.value.code == 2
    assert observed == []
    assert not missing.exists()


@pytest.mark.parametrize("raw", ["[unclosed", "false\n", "[]\n"])
def test_bad_selected_config_never_falls_back_to_destructive_defaults(
        tmp_path, monkeypatch, raw):
    config = tmp_path / "orze.yaml"
    config.write_text(raw, encoding="utf-8")
    observed = _consumer(monkeypatch)
    monkeypatch.setattr("sys.argv", ["orze-gc", "-c", str(config), "--gc-results"])

    with pytest.raises(SystemExit) as error:
        orze_gc.main()

    assert error.value.code == 2
    assert observed == []
    assert config.read_text(encoding="utf-8") == raw


def test_explicit_zero_overrides_are_not_replaced_by_config_values(tmp_path, monkeypatch):
    _, config, caller = _config(tmp_path, {
        "gc": {"keep_top": 50, "keep_recent": 20, "min_free_gb": 10},
    })
    monkeypatch.chdir(caller)
    monkeypatch.setattr("sys.argv", ["orze-gc", "-c", str(config),
                                      "--keep-top", "0", "--keep-recent", "0",
                                      "--min-free-gb", "0"])
    observed = _consumer(monkeypatch)

    orze_gc.main()

    assert len(observed) == 1
    assert {key: observed[0][key] for key in ("keep_top", "keep_recent", "min_free_gb")} == {
        "keep_top": 0, "keep_recent": 0, "min_free_gb": 0,
    }


def test_explicit_cli_paths_remain_relative_to_invocation_directory(tmp_path, monkeypatch):
    _, config, caller = _config(tmp_path, {"gc": {"checkpoints_dir": "configured"}})
    monkeypatch.chdir(caller)
    monkeypatch.setattr("sys.argv", ["orze-gc", "-c", str(config),
                                      "--checkpoints-dir", "override-checkpoints",
                                      "--archive-dir", "override-cold",
                                      "--lake-db", "override.db"])
    observed = _consumer(monkeypatch)

    orze_gc.main()

    assert len(observed) == 1
    assert Path(observed[0]["checkpoints_dir"]).absolute() == caller / "override-checkpoints"
    assert Path(observed[0]["archive_dir"]).absolute() == caller / "override-cold"
    assert Path(observed[0]["lake_db_path"]).absolute() == caller / "override.db"
