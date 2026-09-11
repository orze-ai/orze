"""A missing optional path argument must not erase the configured native scope.

Draft integration regression; real persisted intent and explicit Popen spy.
This is not an alleged defect of the preceding pre-native implementation.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import process
from orze.engine.termination_hold import TerminationUnconfirmed
from test_pre_script_admission_authority import _pending
from test_pre_script_supervision_binding import action


@pytest.mark.parametrize("configured", [False, True])
def test_configured_scope_cannot_downgrade_when_optional_arguments_are_omitted(
        action, monkeypatch, configured):
    a = action
    row = _pending(a, monkeypatch)
    if configured:
        a.cfg["pre_script"] = "prepare.py"
    child = SimpleNamespace(returncode=0, communicate=lambda **kwargs: ("", ""))
    popen = Mock(return_value=child)
    monkeypatch.setattr(process.subprocess, "Popen", popen)
    with pytest.raises(TerminationUnconfirmed):
        process.run_pre_script(a.idea, 4, a.cfg)
    popen.assert_not_called()
    assert current_attempt(a.lake.conn, a.idea, "pre_script") == row
    assert a.prepared == []
    assert not (a.folder / "_compute_receipts").exists()
