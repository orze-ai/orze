"""Independent CLI -> persistent queue -> native occurrence acceptance.

New B3 integration mechanisms, never missing-API historical reds. Configuration,
CLI, request service, SQLite, source sync, phase claim/launch and artifact
terminal publication remain real. Only external GPU/OS/plugin boundaries use
the established offline native fixture; no provider or worker is executed.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest
import yaml

from orze import cli
from orze.core.config import load_project_config
from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.engine import launcher, phases
from orze.engine.orchestrator import Orze
from test_artifact_snapshot_contract import project as artifact_project, _launch, _output, _poll
from test_native_training_caller_boundaries import case as native_case


@pytest.fixture
def completed(artifact_project, monkeypatch):
    c = artifact_project
    monkeypatch.chdir(c.results.parent)
    c.cfg.update({"_env_ORZE_RESULTS_DIR": str(c.results), "sealed_files": [],
                  "artifact_preflight": {"enabled": False}, "sweep": {},
                  "notifications": {"enabled": False}, "auto_seal_eval": False})
    c.config_path = c.results.parent / "orze.yaml"
    c.cfg["_config_path"] = str(c.config_path)
    c.config_path.write_text(yaml.safe_dump(c.cfg), encoding="utf-8")
    # The source must use the same complete project configuration as the CLI;
    # a partial unit fixture has different default execution inputs.
    c.cfg = load_project_config(str(c.config_path))
    c.cfg["_config_path"] = str(c.config_path)
    source, folder = _launch(c)
    _output(source, folder)
    assert _poll(c, source)[0] == [(c.idea, 0)]
    c.source = source
    c.original_row = deepcopy(c.lake.get(c.idea))
    c.original_attempt = deepcopy(current_attempt(c.lake.conn, c.idea, "training"))
    c.original_artifacts = artifacts_for_attempt(c.lake.conn, source.attempt_ref)
    c.original_owner_path = (Path(c.cfg["_orze_dir"]) / "state" /
                             "execution_identities" / (source.execution_identity + ".json"))
    c.original_owner = c.original_owner_path.read_bytes()
    instance = Orze.__new__(Orze)
    instance.cfg, instance.results_dir, instance.lake = c.cfg, c.results, c.lake
    instance.failure_counts, instance.fix_counts, instance.active_roles = {}, {}, {}
    instance.active, instance.active_evals, instance.gpu_ids = {}, {}, [0]
    c.runner = instance
    monkeypatch.setattr("orze.extensions.get_extension", lambda name: None)
    monkeypatch.setattr(phases, "get_gpu_memory_used", lambda gpu: 0)
    return c


def _request(c, monkeypatch, capsys, key, *, global_config=False):
    before = Path.cwd()
    outside = c.results.parent / "outside"
    outside.mkdir(exist_ok=True)
    tripwire = Mock(side_effect=AssertionError("admission must not launch/probe/install"))
    args = ["orze", "replicate", c.idea, "--request-id", key, "-c", str(c.config_path)]
    if global_config:
        args = ["orze", "-c", str(c.config_path), "replicate", c.idea, "--request-id", key]
    with monkeypatch.context() as scoped:
        scoped.chdir(outside)
        scoped.setattr(sys, "argv", args)
        scoped.setattr("orze.extensions._find_pro_key", tripwire)
        scoped.setattr(cli, "maybe_star", tripwire)
        scoped.setattr(cli, "detect_all_gpus", tripwire)
        scoped.setattr(launcher.subprocess, "Popen", tripwire)
        scoped.setattr(launcher.subprocess, "run", tripwire)
        capsys.readouterr()
        code = cli.main()
        output = capsys.readouterr().out
        assert code == 0, output
        value = json.loads(output)
        assert Path.cwd() == outside
    assert Path.cwd() == before
    tripwire.assert_not_called()
    assert value["request_id"] == key and value["task_id"] != c.idea
    assert Path(c.cfg["ideas_file"]).read_bytes() == b""
    return value


def _dispatch(c, task_id):
    ideas, unclaimed, _, raw = c.runner._sync_ideas(c.cfg)
    assert raw == {} and task_id in unclaimed and task_id in ideas
    assert ideas[task_id]["config"] == {"seed": 13}
    calls = len(c.popen_calls)
    # Execute the actual phase including claim, config materialization and
    # final launcher; selecting a task here does not substitute authorization.
    c.runner._launch_training([task_id], True, ideas)
    assert len(c.popen_calls) == calls + 1
    tp = c.runner.active[0]
    assert tp.idea_id == task_id
    c.handles.append(tp)
    assert current_attempt(c.lake.conn, task_id, "training")["state"] == "RUNNING"
    assert yaml.safe_load((c.results / task_id / "idea_config.yaml").read_text()) == {"seed": 13}
    return tp


def _source_unchanged(c):
    assert c.lake.get(c.idea) == c.original_row
    assert current_attempt(c.lake.conn, c.idea, "training") == c.original_attempt
    assert artifacts_for_attempt(c.lake.conn, c.source.attempt_ref) == c.original_artifacts
    assert c.original_owner_path.read_bytes() == c.original_owner


def test_two_cli_requests_dispatch_from_empty_inbox_without_config_or_identity_salt(
        completed, monkeypatch, capsys):
    c = completed
    a = _request(c, monkeypatch, capsys, "cli-repeat-a")
    b = _request(c, monkeypatch, capsys, "cli-repeat-b", global_config=True)
    assert a["task_id"] != b["task_id"]
    handles, occurrences = [], []
    for reply in (a, b):
        tp = _dispatch(c, reply["task_id"])
        handles.append(tp)
        assert tp.execution_identity == c.source.execution_identity
        _output(tp, c.results / tp.idea_id)
        events = launcher.check_active(c.runner.active, c.results, c.cfg, {}, lake=c.lake)
        assert events == [(tp.idea_id, 0)] and not c.runner.active
        records = artifacts_for_attempt(c.lake.conn, tp.attempt_ref)
        assert {r["spec_fingerprint"] for r in records} == {
            r["spec_fingerprint"] for r in c.original_artifacts}
        assert {r["content_sha256"] for r in records} == {
            r["content_sha256"] for r in c.original_artifacts}
        row = current_attempt(c.lake.conn, tp.idea_id, "training")
        assert set(row["terminal"]["artifact_ids"]) == {r["artifact_id"] for r in records}
        occurrences.append({r["artifact_id"] for r in records})
    assert len({c.source.attempt_id, *(tp.attempt_id for tp in handles)}) == 3
    assert occurrences[0].isdisjoint(occurrences[1])
    assert all(ids.isdisjoint({r["artifact_id"] for r in c.original_artifacts}) for ids in occurrences)
    _source_unchanged(c)


def test_cli_request_replay_cannot_reset_an_already_running_child(
        completed, monkeypatch, capsys):
    c = completed
    first = _request(c, monkeypatch, capsys, "cli-repeat-once")
    tp = _dispatch(c, first["task_id"])
    database = list(c.lake.conn.iterdump())
    claim = (c.results / tp.idea_id / "claim.json").read_bytes()
    before_calls = len(c.popen_calls)
    repeated = _request(c, monkeypatch, capsys, "cli-repeat-once", global_config=True)
    assert repeated == {**first, "status": "already_requested"}
    assert list(c.lake.conn.iterdump()) == database
    assert (c.results / tp.idea_id / "claim.json").read_bytes() == claim
    assert len(c.popen_calls) == before_calls and c.runner.active[0] is tp
    _, queued, _, _ = c.runner._sync_ideas(c.cfg)
    assert tp.idea_id not in queued
    _source_unchanged(c)
