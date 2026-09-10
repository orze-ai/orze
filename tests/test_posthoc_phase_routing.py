"""C2c draft routing regressions, using actual native CPU posthoc handles.

These do not infer OS ownership from a mutable flag or a raw PID. The first
case observes a real legacy terminal-writer call after an accepted completion;
the second observes a real wrongly routed STOP of an authenticated live child.
"""
import json

from orze.core.execution_attempts import AttemptRef, current_attempt
from orze.engine import accounting, launcher
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from test_native_posthoc_tree_completion import (
    cpu_posthoc, cpu_training, native_case, _launch_posthoc, _alive,
)


def test_closed_native_posthoc_without_ref_cannot_reenter_legacy_publication(cpu_posthoc, tmp_path, monkeypatch):
    c = cpu_posthoc
    tp = _launch_posthoc(c, tmp_path, detached=False)
    completed = launcher.check_active({0: tp}, c.results, c.cfg, {}, lake=c.lake)
    assert len(completed) == 1 and completed[0].attempt_ref.phase == "posthoc"
    before_row = current_attempt(c.lake.conn, c.idea, "posthoc")
    assert before_row["state"] == "TERMINAL"
    before_metrics = (c.folder / "metrics.json").read_bytes()
    before_compute = (c.folder / "_compute_receipts" / tp.attempt_id / "terminal.json").read_bytes()
    before_history = c.lake.get_fsm_history(c.idea)
    calls = []
    original_terminal = accounting.record_compute_terminal

    def observe_real_terminal(*args, **kwargs):
        calls.append((args, kwargs))
        return original_terminal(*args, **kwargs)

    monkeypatch.setattr(accounting, "record_compute_terminal", observe_real_terminal)
    tp.attempt_ref = None
    assert tp.is_posthoc is True
    active, failures, error = {0: tp}, {}, None
    try:
        launcher.check_active(active, c.results, c.cfg, failures, lake=c.lake)
    except Exception as exc:
        error = exc

    assert calls == [], "closed native history entered the actual legacy terminal writer"
    assert isinstance(error, AttemptEffectBusy)
    assert active == {0: tp} and failures == {}
    assert current_attempt(c.lake.conn, c.idea, "posthoc") == before_row
    assert c.lake.get_fsm_history(c.idea) == before_history
    assert (c.folder / "metrics.json").read_bytes() == before_metrics
    assert (c.folder / "_compute_receipts" / tp.attempt_id / "terminal.json").read_bytes() == before_compute


def test_active_posthoc_with_foreign_phase_ref_cannot_enter_legacy_timeout(cpu_posthoc, tmp_path):
    c = cpu_posthoc
    tp = _launch_posthoc(c, tmp_path, detached=True)
    original_ref = tp.attempt_ref
    assert original_ref.phase == "posthoc"
    assert _alive(c.daemon_pidfd) and tp.process.poll() is None
    before_row = current_attempt(c.lake.conn, c.idea, "posthoc")
    before_claim = (c.folder / "claim.json").read_bytes()
    assert before_row["state"] == "RUNNING"
    assert not (c.folder / "_execution_stops").exists()
    tp.attempt_ref = AttemptRef(c.idea, "evaluation", tp.attempt_id, original_ref.generation)
    tp.timeout = 0
    assert tp.is_posthoc is True
    active, failures, error = {0: tp}, {}, None
    try:
        try:
            launcher.check_active(active, c.results, c.cfg, failures, lake=c.lake)
        except Exception as exc:
            error = exc

        assert _alive(c.daemon_pidfd), "wrong-phase handle reached real legacy STOP before ownership validation"
        assert isinstance(error, AttemptEffectBusy)
        assert not (c.folder / "_execution_stops").exists()
        assert active == {0: tp} and failures == {}
        assert current_attempt(c.lake.conn, c.idea, "posthoc") == before_row
        assert (c.folder / "claim.json").read_bytes() == before_claim
        assert not (c.folder / "metrics.json").exists()
        assert not (c.folder / "_compute_receipts" / tp.attempt_id / "terminal.json").exists()
    finally:
        tp.attempt_ref = original_ref
