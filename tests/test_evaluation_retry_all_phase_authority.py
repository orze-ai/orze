"""D2 draft: evaluation-only retry cannot supersede another owned action."""
import pytest

from orze.core.execution_attempts import create_attempt, hold_attempt, mark_running
from orze.engine import evaluator
from orze.engine.evaluation_retry import EvaluationRetryError, request_evaluation_retry
from orze.engine.termination_hold import TerminationUnconfirmed
from test_stale_evaluation_completion import project, _prepare, _launch, _exit, _files, _lifecycle


@pytest.mark.parametrize("state", ["LAUNCHING", "RUNNING", "IN_DOUBT"])
def test_retry_preserves_owned_post_action_and_all_source_evidence(project, state):
    p = project
    folder = _prepare(p, "idea-post-action-retry")
    ep = _launch(p, folder.name)
    _exit(p, ep, 1)
    assert evaluator.check_active_evals({0: ep}, p.results, p.cfg, lake=p.lake)
    p.lake.conn.execute("BEGIN IMMEDIATE")
    ref = create_attempt(p.lake.conn, folder.name, "post_script", "post-action-1",
                         {"origin": "controller_action", "source_attempt_id": ep.attempt_id})
    if state == "RUNNING":
        mark_running(p.lake.conn, ref)
    elif state == "IN_DOUBT":
        hold_attempt(p.lake.conn, ref, "process_creation_unconfirmed")
    p.lake.conn.commit()
    files, lifecycle = _files(folder), _lifecycle(p.lake)
    before = [tuple(row) for row in p.lake.conn.execute("SELECT * FROM execution_attempts")]
    rejected = False
    try:
        request_evaluation_retry(folder.name, p.results, p.cfg, p.lake)
    except (EvaluationRetryError, TerminationUnconfirmed):
        rejected = True
    assert rejected
    assert _files(folder) == files
    assert _lifecycle(p.lake) == lifecycle
    assert [tuple(row) for row in p.lake.conn.execute("SELECT * FROM execution_attempts")] == before
