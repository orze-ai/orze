"""New C2c execution-budget pinning with real READY/GO and SQLite.

These are new contract checks, not historical API-absence regressions.
The adapter is tiny marker-writing CPU code; resource boundaries are simulated.
"""
import copy
import hashlib
import json

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import launcher
from orze.engine.attempt_effect_receipts import require_closed_effects
from orze.engine.termination_hold import require_no_unconfirmed_stop
from test_native_posthoc_tree_completion import cpu_posthoc, cpu_training, native_case
from test_posthoc_ready_go_boundary import _adapter, _remember


def _identity(c, timeout):
    value = {"schema": "orze.legacy_posthoc_execution.v1", "kind": "posthoc_eval",
             "configuration": c.adapter_config, "python": c.cfg["python"],
             "timeout_seconds": float(timeout)}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


@pytest.mark.parametrize("configured, expected", [(None, 3600.0), (37.5, 37.5)],
                         ids=["default-budget", "explicit-positive-budget"])
def test_posthoc_budget_is_pinned_before_real_ready_and_go(cpu_posthoc, tmp_path, monkeypatch, configured, expected):
    c = cpu_posthoc
    if configured is not None:
        c.cfg["posthoc_timeout"] = configured
    marker = _adapter(c, tmp_path)
    real_prepare = launcher.prepare_supervised
    handles, seen_ready, seen_go = [], [], []

    def prepare(*args, **kwargs):
        process = real_prepare(*args, **kwargs)
        _remember(c, handles, process)
        row = current_attempt(c.lake.conn, c.idea, "posthoc")
        assert row["state"] == "LAUNCHING"
        assert row["binding"]["launch_inputs"]["timeout_seconds"] == expected
        assert row["binding"]["launch_inputs"]["execution_identity"] == _identity(c, expected)
        assert not marker.exists()
        seen_ready.append(copy.deepcopy(row["binding"]["launch_inputs"]))
        actual_go = process.start

        def go():
            current = current_attempt(c.lake.conn, c.idea, "posthoc")
            assert current["state"] == "RUNNING"
            assert current["binding"]["launch_inputs"] == seen_ready[0]
            assert not marker.exists()
            seen_go.append(True)
            return actual_go()

        process.start = go
        return process

    monkeypatch.setattr(launcher, "prepare_supervised", prepare)
    try:
        tp = launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
        c.handles.append(tp)
        assert len(handles) == len(seen_ready) == len(seen_go) == 1
        assert tp.timeout == expected
        assert tp.execution_identity == _identity(c, expected)
        assert tp.execution_identity != _identity(c, expected + 1)
        assert tp.process.wait(timeout=5) == 0
        assert marker.read_text() == "executed"
        assert not (c.folder / "metrics.json").exists()
        active = {0: tp}
        assert launcher.check_active(active, c.results, c.cfg, {}, lake=c.lake) == [(c.idea, 0)]
        assert active == {}
        row = current_attempt(c.lake.conn, c.idea, "posthoc")
        assert row["binding"]["launch_inputs"] == seen_ready[0]
        assert row["terminal"]["outcome"] == "completed"
        assert json.loads((c.folder / "metrics.json").read_bytes())["score"] == 0
        require_closed_effects(c.folder)
    finally:
        for process in handles:
            process.stop(timeout=3)


def test_changed_posthoc_budget_after_ready_cannot_go_or_publish(cpu_posthoc, tmp_path, monkeypatch):
    c = cpu_posthoc
    c.cfg["posthoc_timeout"] = 97.5
    marker = _adapter(c, tmp_path)
    real_prepare = launcher.prepare_supervised
    handles, seen_go = [], []
    captured = {}

    def prepare(*args, **kwargs):
        process = real_prepare(*args, **kwargs)
        _remember(c, handles, process)
        row = current_attempt(c.lake.conn, c.idea, "posthoc")
        captured.update(copy.deepcopy(row["binding"]["launch_inputs"]))
        assert captured["timeout_seconds"] == 97.5
        assert captured["execution_identity"] == _identity(c, 97.5)
        assert not marker.exists()
        actual_go = process.start

        def go():
            seen_go.append(True)
            return actual_go()

        process.start = go
        c.cfg["posthoc_timeout"] = 193.0
        return process

    monkeypatch.setattr(launcher, "prepare_supervised", prepare)
    try:
        with pytest.raises(launcher.LaunchIntegrityError, match="posthoc_execution_inputs_changed"):
            launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
        assert len(handles) == 1 and seen_go == []
        process = handles[0]
        process.wait(timeout=5)
        closure = process.closure_receipt()
        assert closure["stop_requested"] is True and closure["wait_proof"] == "ECHILD_WALL"
        assert not marker.exists()
        row = current_attempt(c.lake.conn, c.idea, "posthoc")
        assert row["binding"]["launch_inputs"] == captured
        assert row["state"] == "RUNNING" and row["terminal"] is None
        assert not (c.folder / "metrics.json").exists()
        assert not list(c.folder.glob("_compute_receipts/*/terminal.json"))
        assert not list(c.folder.glob("_posthoc_attempts/*/work/*"))
        require_no_unconfirmed_stop(c.folder)
    finally:
        for process in handles:
            process.stop(timeout=3)
