"""New CLI mechanism tests, not native replication execution evidence.

Use the real parser/config loader/existing-only SQLite opener. Only the new
control service is an explicit stub here; independent dispatch tests exercise
the actual service through launch. No provider, GPU or child is authorized.
"""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import orze.cli as cli
import orze.extensions as extensions
from orze.core import evaluation_retry_state
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "state").mkdir()
    db = root / "state" / "lake.db"
    with_lake = IdeaLake(db)
    with_lake.close()
    cfg = {"results_dir": "results", "idea_lake_db": "state/lake.db",
           "train_script": "train/main.py", "base_config": "cfg/base.yaml",
           "ideas_file": "empty.md", "python": "python3", "auto_seal_eval": False,
           "_orze_dir": "untrusted-input-value"}
    config = root / "orze.yaml"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    outside = tmp_path / "caller"
    outside.mkdir()
    monkeypatch.chdir(outside)
    service = Mock(return_value={"status": "replication_requested",
                                 "request_id": "repeat-1", "task_id": "idea-repeat"})
    module = ModuleType("orze.engine.replication")
    module.request_replication = service
    monkeypatch.setitem(sys.modules, module.__name__, module)
    boundaries = []
    for owner, key in ((cli, "detect_all_gpus"), (cli, "maybe_star"),
                       (extensions, "_find_pro_key"), (extensions, "_auto_install_pro"),
                       (extensions, "get_extension"), (subprocess, "Popen")):
        blocked = Mock(side_effect=AssertionError("control admission cannot use runtime boundary"))
        monkeypatch.setattr(owner, key, blocked)
        boundaries.append(blocked)
    yield SimpleNamespace(root=root, db=db, cfg=cfg, config=config,
                          outside=outside, service=service)
    for blocked in boundaries:
        blocked.assert_not_called()


def _args(p, monkeypatch, *extras):
    monkeypatch.setattr(sys, "argv", ["orze", "replicate", "idea-source",
                                     "--request-id", "repeat-1", "-c", str(p.config), *extras])


@pytest.mark.parametrize("position", ["global", "subcommand"])
def test_selected_project_paths_and_stable_key_reach_service_without_runtime(project, monkeypatch, capsys, position):
    p = project
    options = ["-c", str(p.config)]
    control = ["replicate", "idea-source", "--request-id", "repeat-1", "--reason", "repeat-check"]
    monkeypatch.setattr(sys, "argv", ["orze", *(options + control if position == "global" else control + options)])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == p.service.return_value
    source, results, cfg, lake = p.service.call_args.args
    assert source == "idea-source" and results == p.root / "results"
    assert p.service.call_args.kwargs == {"request_id": "repeat-1", "reason": "repeat-check"}
    assert Path(lake.db_path) == p.db
    for key, relative in (("train_script", "train/main.py"), ("base_config", "cfg/base.yaml"),
                          ("ideas_file", "empty.md"), ("idea_lake_db", "state/lake.db")):
        assert cfg[key] == str(p.root / relative)
    assert cfg["python"] == "python3"
    assert cfg["_orze_dir"] == str(p.root / ".orze")
    assert cfg["_config_path"] == str(p.config)
    assert cfg["_env_ORZE_RESULTS_DIR"] == str(results)
    assert Path.cwd() == p.outside
    with pytest.raises(sqlite3.ProgrammingError):
        lake.conn.execute("SELECT 1")


@pytest.mark.parametrize("kind", ["missing", "unrelated"])
def test_invalid_authority_does_not_create_bootstrap_or_call_service(project, monkeypatch, capsys, kind):
    p = project
    bad = p.root / ("absent/lake.db" if kind == "missing" else "state/unrelated.db")
    if kind == "unrelated":
        con = sqlite3.connect(bad)
        con.execute("CREATE TABLE unrelated(value TEXT)")
        con.commit()
        con.close()
    original = bad.read_bytes() if bad.exists() else None
    p.cfg["idea_lake_db"] = str(bad.relative_to(p.root))
    p.config.write_text(json.dumps(p.cfg), encoding="utf-8")
    before = set(p.root.rglob("*"))
    _args(p, monkeypatch)
    assert cli.main() == 2
    assert "evaluation_retry_database_" in json.loads(capsys.readouterr().out)["error"]
    assert set(p.root.rglob("*")) == before
    assert (bad.read_bytes() if bad.exists() else None) == original
    assert Path.cwd() == p.outside
    p.service.assert_not_called()


def test_missing_config_rejected_without_default_project_fallback(project, monkeypatch, capsys):
    p = project
    monkeypatch.setattr(sys, "argv", ["orze", "replicate", "idea-source", "--request-id", "repeat-1"])
    assert cli.main() == 2
    assert json.loads(capsys.readouterr().out) == {"error": "replication_config_missing"}
    assert list(p.outside.iterdir()) == []
    p.service.assert_not_called()


def test_idempotency_key_must_be_explicit(project, monkeypatch, capsys):
    p = project
    monkeypatch.setattr(sys, "argv", ["orze", "replicate", "idea-source", "-c", str(p.config)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "--request-id" in capsys.readouterr().err
    p.service.assert_not_called()


def test_runtime_pin_checked_before_opening_or_admission(project, monkeypatch, capsys):
    p = project
    p.cfg["controller_runtime"] = {"invalid": "declaration"}
    p.config.write_text(json.dumps(p.cfg), encoding="utf-8")
    opener = Mock(side_effect=AssertionError("runtime pin must precede database access"))
    monkeypatch.setattr(evaluation_retry_state, "open_existing_lake", opener)
    _args(p, monkeypatch)
    assert cli.main() == 2
    assert json.loads(capsys.readouterr().out)["error"]
    assert Path.cwd() == p.outside
    opener.assert_not_called()
    p.service.assert_not_called()


def test_service_rejection_is_single_json_without_retry(project, monkeypatch, capsys):
    p = project
    p.service.side_effect = ValueError("replication_source_not_confirmed")
    _args(p, monkeypatch)
    assert cli.main() == 2
    assert json.loads(capsys.readouterr().out) == {"error": "replication_source_not_confirmed"}
    assert p.service.call_count == 1
    assert Path.cwd() == p.outside


def test_close_uncertainty_cannot_print_success_or_reissue_request(project, monkeypatch, capsys):
    p = project
    close = IdeaLake.close

    def fail_after_close(lake):
        close(lake)
        raise sqlite3.OperationalError("synthetic close uncertainty")

    monkeypatch.setattr(IdeaLake, "close", fail_after_close)
    _args(p, monkeypatch)
    assert cli.main() == 2
    assert json.loads(capsys.readouterr().out) == {"error": "replication_database_close_unconfirmed"}
    assert p.service.call_count == 1
    assert Path.cwd() == p.outside
