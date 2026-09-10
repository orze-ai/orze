"""New native repair handoff contract; not an automatic provider worker."""
import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import launcher, failure
from test_native_training_caller_boundaries import case, _launch


@pytest.mark.parametrize("enabled", [False, True])
def test_native_failure_records_repair_demand_without_inline_provider(case, monkeypatch, enabled):
    c = case
    c.cfg["max_fix_attempts"] = 2 if enabled else 0
    called = []
    monkeypatch.setattr(failure, "_try_executor_fix", lambda *a, **k: called.append(True))
    tp = _launch(c)
    c.child.returncode = 1
    failures = {}
    try:
        finished = launcher.check_active({0: tp}, c.results, c.cfg, failures,
                                         fix_counts={}, lake=c.lake)
        assert len(finished) == 1 and finished[0].attempt_ref == tp.attempt_ref
        row = current_attempt(c.lake.conn, c.idea, "training")
        assert row["state"] == "TERMINAL" and row["terminal"]["outcome"] == "failed"
        assert row["terminal"]["repair_status"] == (
            "pending_explicit_action" if enabled else "not_requested")
        assert called == []
        assert failures == {c.idea: 1}
    finally:
        tp.close_log()
