"""Real CLI -> existing database -> evaluation-only retry coordinator contract.

No coordinator, opener, config loader, SQLite, or artifact operation is mocked.
Tripwires replace only process/GPU/paid-extension boundaries. All data is
synthetic and local; no holdout, live campaign, provider, or GPU is accessed.
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import orze.cli as cli
import orze.extensions as extensions
from orze.core.evaluation_retry_state import open_existing_lake
from orze.engine import evaluator
from orze.engine.evaluation_retry import pending_evaluation_retries
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    folder = root / "results" / "idea-cli-retry"
    folder.mkdir(parents=True)
    outside = tmp_path / "caller"
    outside.mkdir()
    database = root / "state" / "authority.sqlite3"
    database.parent.mkdir()
    config = root / "orze.yaml"
    cfg = {
        "results_dir": "results", "idea_lake_db": "state/authority.sqlite3",
        "eval_script": "evaluate.py", "eval_output": "assessment.json",
        "eval_checkpoint": "checkpoint.pt", "auto_seal_eval": False,
        "roles": {"operator": {"mode": "script", "script": "unused.py"}},
        "report": {
            "primary_metric": "score", "sort": "ascending",
            "columns": [{"key": "score", "source": "assessment.json:score"}],
        },
    }
    config.write_text(json.dumps(cfg), encoding="utf-8")
    (root / "evaluate.py").write_text("# Never executed in this test.\n", encoding="utf-8")
    protected = {
        "metrics.json": b'{"status":"COMPLETED","score":0}',
        "checkpoint.pt": b"completed-training-checkpoint\x00\xff",
        "claim.json": b'{"attempt_id":"original-training"}',
        "train_output.log": b"completed training, preserve this log\n",
    }
    for name, content in protected.items():
        (folder / name).write_bytes(content)
    failed_output = b'{"status":"FAILED","score":999}'
    (folder / "assessment.json").write_bytes(failed_output)
    (folder / "eval_output.log").write_text("failed evaluation\n", encoding="utf-8")
    lake = IdeaLake(database)
    try:
        lake.insert(folder.name, "Preserved experiment", "seed: 7", "Original notes", status="queued")
        assert lake.record_state_transition(folder.name, "QUEUED", "CLAIMED")
        assert lake.record_state_transition(folder.name, "CLAIMED", "IN_PROGRESS")
        assert lake.record_stage_transition(
            folder.name, "training", "IN_PROGRESS", "COMPLETE", "training_completed",
        )
        assert lake.record_stage_transition(
            folder.name, "evaluation", "PENDING", "IN_PROGRESS", "evaluation_launched",
        )
        assert lake.record_state_transition(folder.name, "IN_PROGRESS", "FAILED", "evaluation_failed")
        training = dict(lake.conn.execute(
            "SELECT * FROM idea_stage_state WHERE idea_id=? AND stage='training'",
            (folder.name,),
        ).fetchone())
        failure_id = lake.conn.execute(
            "SELECT MAX(id) FROM idea_transitions WHERE idea_id=?", (folder.name,),
        ).fetchone()[0]
    finally:
        lake.close()
    monkeypatch.chdir(outside)
    boundaries = []
    for owner, name in (
        (cli, "detect_all_gpus"), (cli, "maybe_star"),
        (extensions, "_find_pro_key"), (extensions, "_auto_install_pro"),
        (extensions, "get_extension"), (subprocess, "Popen"),
        (subprocess, "run"), (evaluator, "_verify_gpu_free"),
        (evaluator, "gpu_execution_lease"),
    ):
        boundary = Mock(side_effect=AssertionError("Admission must not run compute or paid extensions"))
        monkeypatch.setattr(owner, name, boundary)
        boundaries.append(boundary)
    try:
        yield SimpleNamespace(
            root=root, outside=outside, folder=folder, database=database,
            config=config, cfg=cfg, protected=protected, failed_output=failed_output,
            training=training, failure_id=failure_id,
        )
    finally:
        for boundary in boundaries:
            boundary.assert_not_called()


@pytest.mark.parametrize("position", ["global", "subcommand"])
def test_real_cli_admits_evaluation_only_retry_from_an_external_working_directory(
    project, monkeypatch, capsys, position,
):
    p = project
    options = ["-c", str(p.config)]
    arguments = ["retry-eval", p.folder.name]
    arguments = options + arguments if position == "global" else arguments + options
    monkeypatch.setattr(sys, "argv", ["orze", *arguments])

    assert cli.main() == 0

    # JSON must be the complete stdout document, not mixed with runtime logs.
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "evaluation_retry_pending"
    assert result["idea_id"] == p.folder.name
    assert result["retry_id"] == str(p.failure_id)
    assert Path.cwd() == p.outside
    reopened = open_existing_lake(p.database)
    try:
        assert reopened.get_fsm_state(p.folder.name) == "IN_PROGRESS"
        assert reopened.get_stage_state(p.folder.name, "training") == "COMPLETE"
        assert reopened.get_stage_state(p.folder.name, "evaluation") == "PENDING"
        assert pending_evaluation_retries(reopened) == [p.folder.name]
        assert reopened.get_queue() == []  # Never readmitted to training.
        assert dict(reopened.conn.execute(
            "SELECT * FROM idea_stage_state WHERE idea_id=? AND stage='training'",
            (p.folder.name,),
        ).fetchone()) == p.training
    finally:
        reopened.close()
    assert {name: (p.folder / name).read_bytes() for name in p.protected} == p.protected
    archive = p.folder / "_evaluation_retries" / result["retry_id"]
    assert (archive / "artifacts" / "assessment.json").read_bytes() == p.failed_output
    assert not (p.folder / "assessment.json").exists()


def test_real_cli_rejects_missing_and_wrong_database_without_creating_or_migrating(
    project, monkeypatch, capsys,
):
    p = project
    wrong = p.root / "state" / "unrelated.sqlite3"
    connection = sqlite3.connect(wrong)
    try:
        connection.execute("CREATE TABLE unrelated (payload TEXT)")
        connection.execute("INSERT INTO unrelated VALUES ('preserve')")
        connection.commit()
    finally:
        connection.close()
    wrong_bytes = wrong.read_bytes()
    original_bytes = p.database.read_bytes()
    targets = [p.root / "missing" / "absent.sqlite3", wrong]
    for target in targets:
        p.cfg["idea_lake_db"] = str(target.relative_to(p.root))
        p.config.write_text(json.dumps(p.cfg), encoding="utf-8")
        before = set(p.root.rglob("*"))
        monkeypatch.setattr(sys, "argv", [
            "orze", "retry-eval", p.folder.name, "-c", str(p.config),
        ])

        assert cli.main() == 2

        error = json.loads(capsys.readouterr().out)
        assert "evaluation_retry_database_" in error["error"]
        assert Path.cwd() == p.outside
        assert set(p.root.rglob("*")) == before
        assert wrong.read_bytes() == wrong_bytes
        assert p.database.read_bytes() == original_bytes
    assert not targets[0].exists()
    assert not targets[0].parent.exists()
    assert not (p.folder / "_evaluation_retries").exists()
    assert (p.folder / "assessment.json").read_bytes() == p.failed_output
    assert {name: (p.folder / name).read_bytes() for name in p.protected} == p.protected
