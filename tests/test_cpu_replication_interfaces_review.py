"""Independent CPU CLI selected-project boundary; no duplicate worker launch.

The source is produced by the actual foreground CPU loop. Replication is only
the real CLI/SQLite admission operation, invoked from a different caller cwd.
"""
import json
from pathlib import Path
import sqlite3
import sys

from test_cpu_product_loop import project, task


def test_cpu_replication_cli_uses_selected_project_without_repinning_to_caller_cwd(
        project, monkeypatch, capsys):
    import orze.cli as cli
    root, _, run = project
    task(root, outputs={}, program="pass")
    assert run() == 0
    with sqlite3.connect(root / "lake.db") as conn:
        assert conn.execute("SELECT state FROM cpu_action_reservations").fetchall() == [("SETTLED",)]
        assert conn.execute("SELECT state FROM execution_attempts").fetchall() == [("TERMINAL",)]
    caller = root / "other-caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["orze", "replicate", "idea-0001", "--request-id",
        "selected-project-repeat", "--reason", "explicit same specification",
        "-c", str(root / "orze.yaml")])
    result = cli.main()
    output = json.loads(capsys.readouterr().out)
    assert result == 0, output
    assert output["status"] == "created"
    assert Path.cwd() == caller
    with sqlite3.connect(root / "lake.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM ideas").fetchone() == (2,)
        assert conn.execute("SELECT COUNT(*) FROM replication_requests").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM execution_attempts").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM cpu_action_reservations").fetchone() == (1,)
    assert cli.main() == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay == {**output, "status": "already_requested"}
    assert Path.cwd() == caller
