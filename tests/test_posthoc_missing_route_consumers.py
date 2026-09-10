"""C2c draft regressions for actual consumers after in-memory routing loss.

Both prepare real native CPU attempts and retain the real configured catalog.
Shutdown asserts the early-return/fallback contract, not a downstream legacy
STOP. Report asserts rejection instead of legacy None, not a second write.
"""
import json

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import launcher, lifecycle
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.launch_failure_report import report_launch_failure
from orze.engine.shutdown_publication import handle_shutdown
from test_native_posthoc_tree_completion import (
    cpu_posthoc, cpu_training, native_case, _launch_posthoc, _alive,
)


def _lose_transient_routes(c, tp):
    assert c.cfg["idea_lake_db"] == str(c.lake.db_path)
    assert tp.attempt_ref.phase == "posthoc"
    (c.folder / "_execution_catalog.json").unlink()
    claim_path = c.folder / "claim.json"
    claim = json.loads(claim_path.read_bytes())
    assert claim.pop("lifecycle_db") == str(c.lake.db_path)
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    ref, tp.attempt_ref = tp.attempt_ref, None
    return ref


def test_shutdown_missing_route_does_not_authorize_legacy_fallback(cpu_posthoc, tmp_path):
    c = cpu_posthoc
    tp = _launch_posthoc(c, tmp_path, detached=True)
    assert _alive(c.daemon_pidfd) and tp.process.poll() is None
    before = current_attempt(c.lake.conn, c.idea, "posthoc")
    assert before["state"] == "RUNNING"
    saved_ref = _lose_transient_routes(c, tp)
    claim_before = (c.folder / "claim.json").read_bytes()
    try:
        result = handle_shutdown(tp, c.results, "posthoc", lifecycle._stop_for_shutdown,
                                 lake=None, cfg=c.cfg)

        assert result is False, "native history was returned as legacy shutdown fallback"
        assert _alive(c.daemon_pidfd), "the fallback-contract check must not signal this tree"
        assert current_attempt(c.lake.conn, c.idea, "posthoc") == before
        assert (c.folder / "claim.json").read_bytes() == claim_before
        assert not (c.folder / "metrics.json").exists()
        assert not (c.folder / "_execution_stops").exists()
        assert not (c.folder / "_compute_receipts" / tp.attempt_id / "terminal.json").exists()
    finally:
        tp.attempt_ref = saved_ref


def test_report_missing_route_does_not_return_legacy_none(cpu_posthoc, tmp_path):
    c = cpu_posthoc
    tp = _launch_posthoc(c, tmp_path, detached=False, code=7)
    launcher.check_active({0: tp}, c.results, c.cfg, {}, lake=c.lake)
    before = current_attempt(c.lake.conn, c.idea, "posthoc")
    assert before["state"] == "TERMINAL" and before["terminal"]["outcome"] == "failed"
    history_before = c.lake.get_fsm_history(c.idea)
    metrics_before = (c.folder / "metrics.json").read_bytes()
    saved_ref = _lose_transient_routes(c, tp)
    failures = {}
    try:
        # The delivered exception has lost its captured ref too. It is not
        # permission to report this completed execution as a new launch error.
        with pytest.raises(AttemptEffectBusy):
            report_launch_failure(None, c.folder, RuntimeError("lost native delivery"),
                                  failures, c.cfg)

        assert failures == {}
        assert current_attempt(c.lake.conn, c.idea, "posthoc") == before
        assert c.lake.get_fsm_history(c.idea) == history_before
        assert (c.folder / "metrics.json").read_bytes() == metrics_before
        assert current_attempt(c.lake.conn, c.idea, "launch_failure_report") is None
    finally:
        tp.attempt_ref = saved_ref
