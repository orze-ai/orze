"""D2 draft: shutdown must accept the exact native attempt, not an idea ID."""
import json

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import evaluator, lifecycle, process
from orze.engine.evaluation_retry import request_evaluation_retry
from orze.idea_lake import IdeaLake
from test_stale_evaluation_completion import project, _prepare, _launch, _exit, _files, _lifecycle


@pytest.mark.parametrize("entry", ["graceful", "atexit"])
@pytest.mark.parametrize("replaced", [False, True], ids=["current", "stale"])
def test_shutdown_closes_only_its_exact_native_execution(project, monkeypatch, entry, replaced):
    p = project
    folder = _prepare(p, "idea-native-shutdown")
    original = p.popen.side_effect
    stops = []

    def spawn(*args, **kwargs):
        proc = original(*args, **kwargs)
        proc.pid = 709120 + len(p.processes)
        return proc

    def reap(proc, *args, **kwargs):
        stops.append(proc)
        if proc.returncode is None:
            proc.returncode = -15
        return True

    p.popen.side_effect = spawn
    monkeypatch.setattr(process, "_terminate_and_reap", reap)
    old = _launch(p, folder.name)
    if replaced:
        _exit(p, old, 1)
        assert evaluator.check_active_evals({0: old}, p.results, p.cfg, lake=p.lake)
        request_evaluation_retry(folder.name, p.results, p.cfg, p.lake)
        newer = _launch(p, folder.name)
    files, state = _files(folder), _lifecycle(p.lake)
    db_path = p.lake.db_path
    active = {0: old}
    if entry == "atexit":
        lifecycle.atexit_cleanup({}, active, {}, p.results)
    else:
        lifecycle.graceful_shutdown(p.results, p.cfg, {}, active, {}, 1, {},
                                    p.lake, "fixture", "fixture", managed=True)
    peer = IdeaLake(db_path)
    try:
        if replaced:
            assert stops == []
            assert _files(folder) == files
            assert _lifecycle(peer) == state
            assert newer.process.returncode is None
            assert current_attempt(peer.conn, folder.name, "evaluation")["attempt_id"] == newer.attempt_id
        else:
            assert stops == [old.process]
            row = current_attempt(peer.conn, folder.name, "evaluation")
            assert row["state"] == "TERMINAL"
            assert row["terminal"]["outcome"] == "interrupted"
            receipt = json.loads((folder / "_compute_receipts" / old.attempt_id / "terminal.json").read_text())
            assert receipt["return_code"] == -15
            assert (folder / "_execution_effects" / old.attempt_id / "committed.json").is_file()
            assert active == {}
    finally:
        peer.close()
