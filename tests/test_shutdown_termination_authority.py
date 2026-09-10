"""Public shutdown must not equate a leader exit with stopped writers."""
import json
import time
from types import SimpleNamespace

import pytest

from orze.engine import lifecycle, process
from orze.engine.accounting import record_compute_start
from orze.idea_lake import IdeaLake


@pytest.mark.parametrize("confirmed", [False, True])
@pytest.mark.parametrize("entry,phase", [
    ("graceful_kill", "training"), ("graceful_kill", "evaluation"),
    ("graceful_detach", "evaluation"), ("atexit", "training"),
    ("atexit", "evaluation"),
])
def test_shutdown_requires_tree_stop(tmp_path, monkeypatch, entry, phase, confirmed):
    results = tmp_path / "results"
    folder = results / "idea-shutdown"
    folder.mkdir(parents=True)
    lake = IdeaLake(tmp_path / "ideas.db")
    lake.insert(folder.name, "shutdown", "{}", "", status="queued")
    assert lake.record_state_transition(folder.name, "QUEUED", "CLAIMED")
    assert lake.record_state_transition(folder.name, "CLAIMED", "IN_PROGRESS")
    if phase == "evaluation":
        assert lake.record_stage_transition(
            folder.name, "training", "IN_PROGRESS", "COMPLETE", "trained")
        assert lake.record_stage_transition(
            folder.name, "evaluation", "PENDING", "IN_PROGRESS", "evaluating")
    # No real process or signal: both code versions see an exited leader,
    # while the OS reaper boundary separately reports descendant uncertainty.
    child = SimpleNamespace(pid=999999991, returncode=-15,
                            poll=lambda: -15, wait=lambda **kwargs: -15)
    cls = process.TrainingProcess if phase == "training" else process.EvalProcess
    tracked = cls(folder.name, 4, child, time.time() - 1,
                  folder / "execution.log", 60, attempt_id="a" * 32)
    record_compute_start(tracked, folder, phase)
    active = {4: tracked} if phase == "training" else {}
    evals = {4: tracked} if phase == "evaluation" else {}
    monkeypatch.setattr(lifecycle, "_kill_pg", lambda *args: None)
    monkeypatch.setattr(process, "_terminate_and_reap", lambda *args, **kw: confirmed)
    monkeypatch.setattr(lifecycle, "save_state", lambda *args: None)
    monkeypatch.setattr(lifecycle, "notify", lambda *args: None)
    try:
        if entry == "atexit":
            lifecycle.atexit_cleanup(active, evals, {}, results)
        else:
            lifecycle.graceful_shutdown(
                results, {}, active, evals, {}, 1, {}, lake,
                "fixture", "fixture", kill_all=entry == "graceful_kill", managed=True)
        terminal = folder / "_compute_receipts" / tracked.attempt_id / "terminal.json"
        assert terminal.exists() is confirmed, "integer leader exit is not tree closure"
        reopened = IdeaLake(tmp_path / "ideas.db")
        try:
            assert reopened.get_fsm_state(folder.name) == "IN_PROGRESS"
            if phase == "evaluation":
                expected = "PENDING" if confirmed and entry != "atexit" else "IN_PROGRESS"
                assert reopened.get_stage_state(folder.name, "evaluation") == expected
        finally:
            reopened.close()
        if not confirmed:
            assert (active or evals).get(4) is tracked, "unconfirmed handle must not vanish"
        else:
            receipt = json.loads(terminal.read_text())
            assert receipt["outcome"] == "interrupted"
            assert receipt["return_code"] == -15
    finally:
        lake.close()
