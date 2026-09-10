"""C2d1 new immutable action-input mechanisms.

Real public/native action code, SQLite and receipts, with the existing explicit
Popen/supervision OS double. These are not real CPU or scientific-result tests.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import sys

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import evaluator
from orze.engine.native_post_script import run_native_post_script

from test_completion_event_authority import accepted
from test_stale_evaluation_completion import project


def _sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _completed_popen(p, captured):
    original = p.popen.side_effect

    def completed(cmd, **kwargs):
        captured.append((list(cmd), dict(kwargs["env"])))
        child = original(cmd, **kwargs)
        child.returncode = 0
        return child

    p.popen.side_effect = completed


@pytest.mark.parametrize("budget", [None, 37.5], ids=["default", "positive"])
def test_native_budget_and_actual_input_hashes_are_pinned_without_plaintext_environment(
        project, monkeypatch, budget):
    p = project
    folder, source = accepted(p)
    monkeypatch.setenv("ORZE_C2D_TEST_SECRET", "not-a-real-credential-c2d1")
    script = {"script": "unused-pinned-post.py"}
    if budget is not None:
        script["timeout"] = budget
    p.cfg["post_scripts"] = [script]
    captured = []
    _completed_popen(p, captured)

    evaluator.run_post_scripts(folder.name, 0, p.results, p.cfg,
                               lake=p.lake, source_event=source)

    assert len(captured) == 1
    command, environment = captured[0]
    row = current_attempt(p.lake.conn, folder.name, "post_script")
    assert row["state"] == "TERMINAL"
    bound = row["binding"]
    assert type(bound["timeout_seconds"]) is float
    assert bound["timeout_seconds"] == (3600.0 if budget is None else budget)
    assert bound["environment_sha256"] == _sha(environment)
    assert bound["command_sha256"] == _sha(command)
    assert bound["supervision"]["command_sha256"] == _sha(command)
    assert bound["supervision"]["identity"]["attempt_ref"]["phase"] == "post_script"
    assert "not-a-real-credential-c2d1" not in json.dumps(row)
    assert "ORZE_C2D_TEST_SECRET" not in json.dumps(row)
    assert row["terminal"]["process_tree"]["binding"] == bound["supervision"]
    assert row["terminal"]["process_tree"]["wait_proof"] == "ECHILD_WALL"


def test_admitted_command_and_environment_are_detached_before_waiting_for_lease(project, monkeypatch):
    p = project
    folder, source = accepted(p)
    command = [sys.executable, "unused-original-post.py"]
    environment = {"TEST_LABEL": "original", "TEST_SECRET": "private-fixture-value"}
    original_command, original_environment = list(command), dict(environment)
    captured = []
    _completed_popen(p, captured)

    @contextmanager
    def lease(*args, **kwargs):
        assert current_attempt(p.lake.conn, folder.name, "post_script")["state"] == "LAUNCHING"
        assert not p.lake.conn.in_transaction
        assert not (folder / "_attempt_effect.lock").exists()
        command[:] = [sys.executable, "unused-replaced-post.py"]
        environment["TEST_LABEL"] = "replaced"
        yield ()

    monkeypatch.setattr(evaluator, "gpu_execution_lease", lease)
    result = run_native_post_script(source, folder.name, 0, p.results, p.cfg, p.lake,
                                    command, 12.5, folder / "post.log", environment)

    assert result == 0
    assert captured == [(original_command, original_environment)]
    assert command != original_command and environment != original_environment
    row = current_attempt(p.lake.conn, folder.name, "post_script")
    assert row["binding"]["command_sha256"] == _sha(original_command)
    assert row["binding"]["environment_sha256"] == _sha(original_environment)
    assert row["binding"]["timeout_seconds"] == 12.5
    assert row["terminal"]["outcome"] == "completed"


def test_new_budget_or_environment_does_not_reexecute_a_historical_same_command_action(project):
    p = project
    folder, source = accepted(p)
    command = [sys.executable, "unused-deduplicated-post.py"]
    captured = []
    _completed_popen(p, captured)
    assert run_native_post_script(source, folder.name, 0, p.results, p.cfg, p.lake,
                                  command, 12.5, folder / "post.log", {"LABEL": "first"}) == 0
    before = deepcopy(current_attempt(p.lake.conn, folder.name, "post_script"))
    count = p.popen.call_count

    assert run_native_post_script(source, folder.name, 0, p.results, p.cfg, p.lake,
                                  command, 25.0, folder / "post.log", {"LABEL": "second"}) is None

    assert current_attempt(p.lake.conn, folder.name, "post_script") == before
    assert p.popen.call_count == count
    assert len(captured) == 1
    assert before["binding"]["timeout_seconds"] == 12.5
    assert before["binding"]["environment_sha256"] == _sha({"LABEL": "first"})
    p.reaper.assert_not_called()
