"""A confirmed process stop does not authorize bare native receipt writes."""
import copy

import pytest

from orze.engine.resume import write_interruption_receipt, ResumeValidationError
from orze.engine.termination_hold import terminate_execution, TerminationUnconfirmed
from test_native_training_caller_boundaries import case, _launch
from test_stale_evaluation_completion import _files, _lifecycle


@pytest.mark.parametrize("strip_reference", [False, True])
def test_legacy_interruption_writer_cannot_close_native_execution(case, strip_reference):
    c = case
    tp = _launch(c)
    c.child.returncode = -15
    assert terminate_execution(tp, c.folder, phase="training", reaper=lambda *a, **kw: True) == -15
    caller = copy.copy(tp)
    if strip_reference:
        caller.attempt_ref = None
    before, states = _files(c.folder), _lifecycle(c.lake)
    try:
        with pytest.raises((ResumeValidationError, TerminationUnconfirmed)):
            write_interruption_receipt(caller, c.results, c.cfg, "timeout", "SIGTERM", -15)
        assert _files(c.folder) == before and _lifecycle(c.lake) == states
    finally:
        tp.close_log()
