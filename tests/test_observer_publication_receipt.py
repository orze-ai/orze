"""Observers may not report success after a silently lost publication.

Real CLI/report and admin producer calls use real SQLite and local artifacts.
Only atomic_write is fault-injected at the publication boundary, matching its
historical swallowed-ENOSPC behavior. The baseline replays cad5baa publisher
functions while retaining the new CLI; it is not a replay of the old CLI.
"""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import orze.cli as cli
import orze.extensions as extensions
from orze.core.config import load_project_config, orze_path
from orze.idea_lake import IdeaLake
from orze.reporting import leaderboard


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    folder = root / "results" / "idea-publication"
    folder.mkdir(parents=True)
    database = root / ".orze" / "idea_lake.db"
    database.parent.mkdir()
    lake = IdeaLake(database)
    try:
        lake.insert(folder.name, "Current observation", "seed: 5", "",
                    status="completed")
    finally:
        lake.close()
    (folder / "metrics.json").write_text(
        '{"status":"COMPLETED","score":900}', encoding="utf-8")
    (folder / "assessment.json").write_text('{"score":0}', encoding="utf-8")
    config = root / "orze.yaml"
    config.write_text(json.dumps({
        "results_dir": "results", "auto_seal_eval": False,
        "report": {
            "primary_metric": "score", "sort": "ascending",
            "columns": [{"key": "score", "source": "assessment.json:score"}],
        },
    }), encoding="utf-8")
    monkeypatch.chdir(root)
    cfg = load_project_config(str(config))
    for owner, name in (
        (extensions, "_find_pro_key"), (extensions, "get_extension"),
        (extensions, "_auto_install_pro"), (cli, "maybe_star"),
        (cli, "detect_all_gpus"), (subprocess, "Popen"), (subprocess, "run"),
    ):
        monkeypatch.setattr(owner, name, Mock(side_effect=AssertionError(
            f"observer must not call {name}")))
    return SimpleNamespace(root=root, results=folder.parent, folder=folder,
                           config=config, cfg=cfg, database=database)


def _drop_target_write(monkeypatch, target):
    real_write = leaderboard.atomic_write
    attempts = []

    def silent_no_write(path, content):
        if path == target:
            attempts.append(content)
            return None  # No exception: exactly the failure needing a receipt.
        return real_write(path, content)

    monkeypatch.setattr(leaderboard, "atomic_write", silent_no_write)
    return real_write, attempts


@pytest.mark.parametrize("existing", [False, True], ids=["absent", "stale"])
@pytest.mark.parametrize("output", ["_leaderboard.json", "report.md"], ids=["json", "markdown"])
def test_real_report_cli_rejects_silent_publication_loss_and_recovers(
    project, monkeypatch, capsys, existing, output,
):
    p = project
    target = p.results / output
    previous = b'{"top": [], "stale": true}' if output.endswith(".json") else b"# Stale report\n"
    if existing:
        target.write_bytes(previous)
    before_db = p.database.read_bytes()
    real_write, attempts = _drop_target_write(monkeypatch, target)
    monkeypatch.setattr(sys, "argv", [
        "orze", "--report-only", "-c", str(p.config),
    ])

    assert cli.main() == 2

    assert attempts
    assert "Report updated." not in capsys.readouterr().out
    assert target.read_bytes() == previous if existing else not target.exists()
    assert p.database.read_bytes() == before_db

    monkeypatch.setattr(leaderboard, "atomic_write", real_write)
    assert cli.main() == 0
    payload = json.loads((p.results / "_leaderboard.json").read_text(encoding="utf-8"))
    assert [(row["idea_id"], row["metric_value"]) for row in payload["top"]] == [
        (p.folder.name, 0),
    ]
    assert "| 1 | idea-publication |" in (p.results / "report.md").read_text(encoding="utf-8")
    outputs = [p.results / name for name in (
        "_leaderboard.json", "report.md", "_leaderboard_views.json",
    )]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in outputs}
    assert cli.main() == 0
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in outputs} == before


@pytest.mark.parametrize("existing", [False, True], ids=["absent", "stale"])
def test_real_admin_producer_rejects_silent_publication_loss_and_recovers(
    project, monkeypatch, existing,
):
    p = project
    target = orze_path(p.cfg, "state", "admin_cache.json")
    previous = b'{"queue": {"queue": []}, "stale": true}'
    if existing:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(previous)
    real_write, attempts = _drop_target_write(monkeypatch, target)
    before_db = p.database.read_bytes()

    with pytest.raises(OSError):
        leaderboard.write_admin_cache(p.results, {}, p.cfg)

    assert attempts
    assert target.read_bytes() == previous if existing else not target.exists()
    assert p.database.read_bytes() == before_db
    monkeypatch.setattr(leaderboard, "atomic_write", real_write)
    leaderboard.write_admin_cache(p.results, {}, p.cfg)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert [(row["idea_id"], row["status"]) for row in payload["queue"]["items"]] == [
        (p.folder.name, "completed"),
    ]
    assert payload["queue"]["counts"] == {"completed": 1}
    assert p.database.read_bytes() == before_db
