"""A bare optional-Lake call cannot downgrade a durable native launch intent."""
import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import evaluator
from orze.engine.termination_hold import TerminationUnconfirmed

from test_stale_evaluation_completion import project, _prepare, _files, _lifecycle


def test_missing_lake_cannot_relaunch_a_pre_popen_native_intent_after_restart(project):
    p = project
    folder = _prepare(p, "idea-launch-without-catalog")
    actual_popen = p.popen.side_effect

    def disappear_before_process_creation(*args, **kwargs):
        p.logs.append(kwargs["stdout"])
        raise KeyboardInterrupt("fixture controller loss at the Popen boundary")

    p.popen.side_effect = disappear_before_process_creation
    with pytest.raises(KeyboardInterrupt):
        evaluator.launch_eval(folder.name, 0, p.results, p.cfg, lake=p.lake)
    for handle in p.logs:
        handle.close()
    row = current_attempt(p.lake.conn, folder.name, "evaluation")
    assert row["state"] == "LAUNCHING"
    assert not (folder / "_compute_receipts" / row["attempt_id"] / "start.json").exists()
    before_files, before_state = _files(folder), _lifecycle(p.lake)
    p.popen.side_effect = actual_popen

    with pytest.raises(TerminationUnconfirmed):
        evaluator.launch_eval(folder.name, 0, p.results, p.cfg)

    assert p.popen.call_count == 1
    assert p.processes == []
    assert _files(folder) == before_files
    assert _lifecycle(p.lake) == before_state
