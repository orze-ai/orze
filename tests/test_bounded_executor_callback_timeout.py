"""Actual READY callback refusal; exact own process cleanup, no provider."""
import os
import subprocess
import sys

import pytest

from orze.engine import bounded_executor as executor
from orze.engine.supervised_process import prepare_supervised


def test_callback_timeout_exception_still_stops_blocked_worker_before_hold(tmp_path):
    handles, stops = [], []
    marker = tmp_path / "must-not-run"
    def prepare(*args, **kwargs):
        child = prepare_supervised(*args, **kwargs)
        original = child.stop
        def stop(*a, **k):
            stops.append(True)
            return original(*a, **k)
        child.stop = stop
        handles.append(child)
        return child
    def reject():
        raise subprocess.TimeoutExpired(["synthetic-admission-check"], 1)
    try:
        with pytest.raises(executor.BoundedExecutorHOLD):
            executor.run_bounded_executor([sys.executable, "-c",
                "from pathlib import Path; Path('must-not-run').write_text('bad')"],
                timeout=3, env=dict(os.environ), cwd=tmp_path,
                prepare=prepare, before_start=reject)
        observed_stops = tuple(stops)
        assert observed_stops == (True,)
        assert type(handles[0].poll()) is int
        assert handles[0].closure_receipt()["stop_requested"] is True
        assert not marker.exists()
        executor.require_executor_scope_clear(tmp_path)
    finally:
        for handle in handles:
            if handle.poll() is None:
                handle.stop()
