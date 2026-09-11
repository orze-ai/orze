"""Actual native CPU closure retires heavy ownership, never unknown execution.

The weakly observed object is held only by a genuine admission callback. All
worker/supervisor/SQLite/effect/budget behavior is real; no process-table scan.
"""
import gc
from pathlib import Path
import time
import weakref

import pytest

from orze.core import cpu_action_budget as budget
from orze.core.execution_attempts import current_attempt
from orze.engine import native_cpu_action as native
from test_native_cpu_action import context, _finish


class Resource:
    def __init__(self):
        self.data = bytearray(65536)


def launch(context, code):
    lake, results, scope, cfg, create, processes = context
    action, permit = create(code)
    resource = Resource()
    observed = weakref.ref(resource)
    def admission(held=resource):
        assert len(held.data) == 65536
    handle = native.launch("idea-cpu", results, cfg, lake=lake, action=action,
                           permit=permit, admission=admission)
    return handle, permit, observed


@pytest.mark.parametrize("code,outcome", [("pass", "completed"), ("raise SystemExit(7)", "failed")])
def test_confirmed_terminal_and_settled_budget_release_heavy_callback(context, code, outcome):
    lake, results, scope, cfg, create, processes = context
    handle, permit, observed = launch(context, code)
    terminal = _finish(handle, results, cfg, lake, permit)
    assert terminal["outcome"] == outcome
    assert terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
    assert lake.conn.execute("SELECT state FROM cpu_action_reservations").fetchone()[0] == "SETTLED"
    gc.collect()
    assert observed() is None


def test_explicit_stop_zero_settled_interruption_can_release_heavy_callback(context):
    lake, results, scope, cfg, create, processes = context
    code = """import signal
from pathlib import Path
signal.signal(signal.SIGTERM, lambda *_: exit(0))
Path('ready').write_text('ready')
signal.pause()
"""
    handle, permit, observed = launch(context, code)
    ready = results / "idea-cpu" / "_action_attempts" / handle.attempt_id / "work" / "ready"
    until = time.monotonic() + 2
    while not ready.exists() and time.monotonic() < until:
        time.sleep(.005)
    assert ready.exists()
    terminal = native.stop(handle, results, cfg, lake=lake, permit=permit)
    assert terminal["outcome"] == "interrupted"
    assert terminal["return_code"] == 0
    assert terminal["process_tree"]["stop_requested"] is True
    assert lake.conn.execute("SELECT state FROM cpu_action_reservations").fetchone()[0] == "SETTLED"
    gc.collect()
    assert observed() is None


def test_running_owner_remains_strong_despite_collection(context):
    lake, results, scope, cfg, create, processes = context
    handle, permit, observed = launch(context, "import time; time.sleep(.3)")
    gc.collect()
    assert observed() is not None
    assert native._OWNERS[id(handle)].handle is handle
    assert current_attempt(lake.conn, "idea-cpu", "action")["state"] == "RUNNING"
    assert budget.snapshot(lake, scope)["active_reservations"] == 1
    _finish(handle, results, cfg, lake, permit)


def test_settlement_uncertainty_retains_strong_owner_and_does_not_refund(context, monkeypatch):
    lake, results, scope, cfg, create, processes = context
    handle, permit, observed = launch(context, "pass")
    calls = []
    def uncertain(*args, **kwargs):
        calls.append("settle")
        raise OSError("explicit budget settlement uncertainty")
    monkeypatch.setattr(budget, "settle", uncertain)
    with pytest.raises(native.CPUActionHOLD):
        _finish(handle, results, cfg, lake, permit)
    assert calls == ["settle"]
    assert current_attempt(lake.conn, "idea-cpu", "action")["state"] == "TERMINAL"
    assert lake.conn.execute("SELECT state FROM cpu_action_reservations").fetchone()[0] == "BOUND"
    gc.collect()
    assert observed() is not None
    assert native._OWNERS[id(handle)].held is True
    with pytest.raises(native.CPUActionHOLD):
        native.harvest(handle, results, cfg, lake=lake, permit=permit)
    assert calls == ["settle"]
