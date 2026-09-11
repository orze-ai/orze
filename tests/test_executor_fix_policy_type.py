"""Actual fixer-created admission callback, controlled runner/OS boundary.

This does not execute a CPU worker or prove process cleanup. It verifies that
the real callback rejects an explicitly invalid current tool-policy flag even
when Python equality treats that value as equal to the earlier valid flag.
"""
import subprocess

import pytest

from orze.engine import failure
from orze.engine.termination_hold import TerminationUnconfirmed
from test_executor_fix_caller_authority import repair_case, _repair


def test_current_policy_must_remain_exact_true_before_go(repair_case, monkeypatch):
    c = repair_case

    def run(cmd, **kwargs):
        c.calls.append("runner")
        c.cfg["agent_tool_policy"]["enabled"] = 1
        kwargs["before_start"]()
        c.calls.append("callback_returned")
        return subprocess.CompletedProcess(cmd, 0, stdout="FIX_APPLIED\n", stderr="")

    monkeypatch.setattr(failure, "_run_bounded_executor", run)
    with pytest.raises(TerminationUnconfirmed):
        _repair(c)
    assert c.calls == ["runner"]
    assert c.fixes == {}
    assert list((c.results / "_fix_logs").glob("*.log")) == []
