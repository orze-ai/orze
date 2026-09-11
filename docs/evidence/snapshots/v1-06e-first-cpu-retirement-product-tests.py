"""Real CLI lifetime proof plus exact invocation hook ownership controls.

No global registry is cleared to cause collection. Real CPU completion and
settlement are checked by the unchanged source-bound two-action product test.
The hook-unit controls substitute only the interpreter callback registry.
"""
import atexit
import gc
import signal
import sqlite3
import weakref

import pytest

from test_cpu_product_loop import project
from test_cpu_proposal_product import (
    test_policy_builds_experiment_then_analysis_from_new_artifact_ids as two_actions,
)


def test_real_two_action_cli_releases_closed_invocation_and_source_captures(project, monkeypatch):
    from orze.core import research_interfaces as api
    from orze.engine import cpu_phase, cpu_action_sources
    references = []
    original_initialize = cpu_phase.initialize
    original_prepare = api.prepare_domain_run
    original_sources = cpu_action_sources.capture_sources

    def initialize(engine, *args):
        result = original_initialize(engine, *args)
        references.extend([weakref.ref(engine), weakref.ref(engine._cpu_interfaces)])
        return result

    def prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        references.append(weakref.ref(result))
        return result

    def sources(*args, **kwargs):
        result = original_sources(*args, **kwargs)
        references.append(weakref.ref(result._owner))
        return result

    monkeypatch.setattr(cpu_phase, "initialize", initialize)
    monkeypatch.setattr(api, "prepare_domain_run", prepare)
    monkeypatch.setattr(cpu_action_sources, "capture_sources", sources)
    # Includes actual Propose/Execute/Propose(source IDs)/Execute/Stop, two
    # TERMINAL/completed and SETTLED records, and source-bound value 49.
    two_actions(project, monkeypatch)
    assert len(references) >= 6
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.fixture
def invocation(project, monkeypatch):
    from orze.engine.orchestrator import Orze
    from orze.engine import cpu_phase
    _, cfg, _ = project
    cfg["action_domain"] = {"version": 1, "kind": "command", "config": {}}
    callbacks, removed = [], []

    def register(callback):
        callbacks.append(callback)
        return callback

    def unregister(callback):
        removed.append(callback)
        callbacks[:] = [item for item in callbacks if item != callback]

    monkeypatch.setattr(atexit, "register", register)
    monkeypatch.setattr(atexit, "unregister", unregister)
    engine = Orze([], cfg, once=True)
    real_close = engine.lake.close
    try:
        yield engine, callbacks, removed
    finally:
        # This fixture admits no actions. Restore its injected close fault,
        # then close its real SQLite handle; do not clear execution owners.
        real_close()
        callbacks.clear()


def test_successful_close_releases_interfaces_even_if_caller_keeps_engine(invocation):
    from orze.engine import cpu_phase
    engine, callbacks, removed = invocation
    context = weakref.ref(engine._cpu_interfaces)
    cpu_phase.close(engine)
    assert engine._cpu_closed and not engine._cpu_handles
    with pytest.raises(sqlite3.ProgrammingError):
        engine.lake.conn.execute("SELECT 1")
    gc.collect()
    assert engine._cpu_interfaces is None
    assert engine._cpu_policy is None
    assert context() is None
    assert callbacks == []
    assert len(removed) == 1


def test_close_unregisters_only_its_unique_exit_callback(invocation):
    from orze.engine import cpu_phase
    engine, callbacks, removed = invocation
    owned = callbacks[0]
    # A separate caller can register the same bound cleanup method. Python's
    # unregister uses equality, so Orze needs its own unique wrapper token.
    unrelated = engine._atexit_cleanup
    atexit.register(unrelated)
    cpu_phase.close(engine)
    cpu_phase.close(engine)
    assert callbacks == [unrelated]
    assert removed == [owned]
    assert owned is not unrelated


def test_unconfirmed_close_preserves_captured_callbacks_and_interfaces(invocation, monkeypatch):
    from orze.core.cpu_execution import CPUExecutionError
    from orze.engine import cpu_phase
    engine, callbacks, removed = invocation
    context = engine._cpu_interfaces
    policy = engine._cpu_policy
    owned = callbacks[0]
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def uncertain():
        raise OSError("injected SQLite close uncertainty; no actions admitted")

    monkeypatch.setattr(engine.lake, "close", uncertain)
    with pytest.raises(CPUExecutionError, match="cleanup remains unconfirmed"):
        cpu_phase.close(engine)
    assert callbacks == [owned] and removed == []
    assert engine._cpu_interfaces is context and engine._cpu_policy is policy
    assert all(signal.getsignal(sig) is handler for sig, handler in handlers.items())
    # Existing close-attempt latch stays one-shot; this does not settle or
    # retry an uncertain action because this fixture never admitted one.
    cpu_phase.close(engine)
    assert callbacks == [owned] and removed == []
