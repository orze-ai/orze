"""D2 draft: a bare failure writer must not bypass native ownership."""
import json

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import launcher
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_training_caller_boundaries import case, _launch


@pytest.mark.parametrize("with_lake", [True, False])
def test_bare_failure_writer_cannot_overwrite_native_running_task(case, with_lake):
    c = case
    tp = _launch(c)
    metrics = c.folder / "metrics.json"
    metrics.write_text('{"status":"IN_PROGRESS","step":12}')
    before = metrics.read_bytes()
    try:
        try:
            launcher._write_failure(c.folder, "late failure", lake=c.lake if with_lake else None,
                                    idea_id=c.idea, cfg=c.cfg)
        except TerminationUnconfirmed:
            pass
        assert metrics.read_bytes() == before
        assert c.lake.get_fsm_state(c.idea) == "IN_PROGRESS"
        assert current_attempt(c.lake.conn, c.idea, "training")["state"] == "RUNNING"
        assert not list(c.folder.glob("_compute_receipts/*/terminal.json"))
    finally:
        tp.close_log()
