"""D1 integration draft boundaries, not old-release API absence tests."""
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from orze.engine import evaluator
from orze.engine import termination_hold as stop
from test_evaluation_termination_authority import project


@pytest.mark.parametrize("exit_code", [-9, 0], ids=["failed_exit", "successful_exit"])
def test_monitor_obeys_durable_stop_even_when_its_own_handle_has_no_latch(project, exit_code):
    p = project
    ep = evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
    assert ep is not None
    # Another owner/observer has requested a stop. Do not mock the receipt
    # gate or fabricate its JSON; use the real durable stop protocol.
    other = SimpleNamespace(
        idea_id=ep.idea_id, gpu=ep.gpu, process=ep.process,
        attempt_id=ep.attempt_id, start_time=ep.start_time)

    def incomplete_reaper(proc, *args, **kwargs):
        proc.returncode = exit_code
        return False

    with pytest.raises(stop.TerminationUnconfirmed):
        stop.terminate_execution(other, p.folder, phase="evaluation", reaper=incomplete_reaper)
    assert not getattr(ep, "_termination_unconfirmed", False)
    if exit_code == 0:
        (p.folder / "assessment.json").write_text(
            '{"status":"COMPLETED","quality":0}', encoding="utf-8")
    output_before = {path.name: path.read_bytes() for path in p.folder.glob("*.json")}
    active = {0: ep}
    held = False
    try:
        evaluator.check_active_evals(active, p.results, p.cfg, lake=p.lake)
    except stop.TerminationUnconfirmed:
        held = True
    assert p.lake.get_fsm_state(p.idea_id) == "IN_PROGRESS"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "IN_PROGRESS"
    assert held
    assert active == {0: ep}
    assert list((p.folder / "_compute_receipts").glob("*/terminal.json")) == []
    assert {path.name: path.read_bytes() for path in p.folder.glob("*.json")} == output_before


def test_post_popen_lease_failure_and_log_close_error_cannot_mask_termination_hold(project, monkeypatch):
    p = project
    closed = []

    class LeaseExitFailure:
        def __enter__(self):
            return ()

        def __exit__(self, *args):
            raise OSError("test-only lease release failure")

    class CloseFailure:
        def __init__(self, handle):
            self.handle = handle
            self.failed_once = False

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def close(self):
            self.handle.close()
            if not self.failed_once:
                self.failed_once = True
                closed.append("close_failed_after_real_close")
                raise OSError("test-only log close failure")

    def log_open(path, *args, **kwargs):
        handle = builtins.open(path, *args, **kwargs)
        return CloseFailure(handle) if Path(path) == p.folder / "eval_output.log" else handle

    p.lease.side_effect = lambda *args, **kwargs: LeaseExitFailure()
    monkeypatch.setattr(evaluator, "open", log_open, raising=False)
    with pytest.raises(stop.TerminationUnconfirmed, match="^evaluation_termination_unconfirmed$"):
        evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
    p.popen.assert_called_once()
    p.reap.assert_called_once()
    assert closed == ["close_failed_after_real_close"]
    assert p.lake.get_fsm_state(p.idea_id) == "IN_PROGRESS"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "IN_PROGRESS"
    assert not (p.folder / "assessment.json").exists()
    assert not (p.folder / "_compute_receipts").exists()
    with pytest.raises(stop.TerminationUnconfirmed):
        stop.require_no_unconfirmed_stop(p.folder)
