"""D2 draft: copied tokens cannot substitute another observed process."""
from copy import copy

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import evaluator
from orze.engine.termination_hold import TerminationUnconfirmed
from test_stale_evaluation_completion import project, _prepare, _launch, _exit, _files, _lifecycle


@pytest.mark.parametrize("changed", ["process_pid", "physical_gpu"])
def test_substituted_process_is_rejected_before_poll_or_any_terminal_effect(project, changed):
    p = project
    folder = _prepare(p, "idea-process-binding")
    original = p.popen.side_effect

    def with_pid(*args, **kwargs):
        proc = original(*args, **kwargs)
        proc.pid = 736129
        return proc

    p.popen.side_effect = with_pid
    ep = _launch(p, folder.name)
    _exit(p, ep, 1)
    wrong = copy(ep)
    if changed == "process_pid":
        wrong.process = copy(ep.process)
        wrong.process.pid += 1
    else:
        wrong.gpu = ep.gpu + 1
    polls = []
    wrong.process.on_poll = lambda: polls.append(True)
    files, lifecycle = _files(folder), _lifecycle(p.lake)
    active = {0: wrong}
    try:
        finished = evaluator.check_active_evals(active, p.results, p.cfg, lake=p.lake)
    except TerminationUnconfirmed:
        finished = []
    assert polls == []
    assert finished == []
    assert active.get(0) is wrong
    assert _files(folder) == files
    assert _lifecycle(p.lake) == lifecycle
    assert current_attempt(p.lake.conn, folder.name, "evaluation")["state"] == "RUNNING"
