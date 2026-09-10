"""C2d1 draft row-identity regressions; real SQLite, explicit OS doubles.

Mutation timing covers both GO authorization and terminal acceptance. No actual
process-tree ownership or CPU execution is inferred from these protocol doubles.
"""
from contextlib import contextmanager
import json

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import evaluator
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from test_completion_event_authority import accepted
from test_stale_evaluation_completion import project


@pytest.mark.parametrize("boundary", ["before_go", "before_terminal"])
def test_changed_process_pid_cannot_authorize_go_or_terminal(project, monkeypatch, boundary):
    p = project
    folder, source = accepted(p)
    p.cfg["post_scripts"] = [{"script": "unused-identity-post.py", "timeout": 5}]
    original_popen, original_prepare = p.popen.side_effect, evaluator.prepare_supervised
    captured, mutations = [], []

    def completed(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        child.returncode = 0
        return child

    def mutate():
        row = current_attempt(p.lake.conn, folder.name, "post_script")
        assert row["state"] == "RUNNING"
        assert type(row["binding"]["process_pid"]) is int
        row["binding"]["process_pid"] = True
        p.lake.conn.execute("UPDATE main.execution_attempts SET binding_json=? WHERE attempt_id=?",
                            (json.dumps(row["binding"]), row["attempt_id"]))
        p.lake.conn.commit()
        mutations.append(row["attempt_id"])

    def prepare(*args, **kwargs):
        process = original_prepare(*args, **kwargs)
        captured.append(process)
        if boundary == "before_terminal":
            original_wait = process.wait

            def wait(timeout=None):
                result = original_wait(timeout=timeout)
                mutate()
                return result

            object.__setattr__(process, "wait", wait)
        return process

    @contextmanager
    def lease(*args, **kwargs):
        yield ()
        if boundary == "before_go":
            mutate()

    p.popen.side_effect = completed
    monkeypatch.setattr(evaluator, "prepare_supervised", prepare)
    monkeypatch.setattr(evaluator, "gpu_execution_lease", lease)
    error = None
    try:
        evaluator.run_post_scripts(folder.name, 0, p.results, p.cfg,
                                   lake=p.lake, source_event=source)
    except Exception as exc:
        error = exc

    assert len(captured) == 1 and len(mutations) == 1
    row = current_attempt(p.lake.conn, folder.name, "post_script")
    if boundary == "before_go":
        assert captured[0]._sim_started is False, "GO ignored the changed persistent worker PID"
    else:
        assert row["state"] == "RUNNING", "terminal accepted an inconsistent persistent worker PID"
    assert isinstance(error, AttemptEffectInDoubt)
    assert row["state"] == "RUNNING"
    assert not (folder / "_compute_receipts" / row["attempt_id"] / "terminal.json").exists()
    assert current_attempt(p.lake.conn, folder.name, "evaluation")["attempt_id"] == source.attempt_ref.attempt_id
    p.reaper.assert_not_called()
