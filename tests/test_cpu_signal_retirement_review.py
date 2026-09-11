"""Independent real signal ownership across two CPU Orze invocations.

Both controllers use private real SQLite handles and admit no actions. No
signal is sent. The existing CPU fixture rejects GPU and legacy entry points.
Original process handlers are restored even if a lifetime assertion fails.
"""
import copy
import gc
import signal
import sqlite3
import weakref

import pytest

from orze.engine import cpu_phase
from orze.engine.orchestrator import Orze
from test_cpu_product_loop import project


@pytest.mark.parametrize("order", ["non_lifo", "lifo"])
def test_closed_invocations_do_not_restore_a_retired_signal_owner(project, order):
    root, cfg, _ = project
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    first = second = None

    def configuration(name):
        value = copy.deepcopy(cfg)
        folder = root / name
        value.update(results_dir=str(folder / "results"),
                     idea_lake_db=str(folder / "lake.db"),
                     ideas_file=str(folder / "ideas.md"),
                     action_domain={"version": 1, "kind": "command", "config": {}})
        return value

    try:
        first = Orze([], configuration("first"), once=True)
        second = Orze([], configuration("second"), once=True)
        references = weakref.ref(first), weakref.ref(second)
        if order == "non_lifo":
            cpu_phase.close(first)
            # Capture by identity, without requiring a bound method or a new
            # wrapper API: a closing older owner must not replace newer hooks.
            newer = {sig: signal.getsignal(sig) for sig in original}
            assert all(handler is installed for handler, installed in (
                (newer[sig], second._cpu_signal_handlers[sig][1]) for sig in original))
            newer = None
            cpu_phase.close(second)
        else:
            cpu_phase.close(second)
            cpu_phase.close(first)
        assert first._cpu_closed and second._cpu_closed
        assert not first._cpu_handles and not second._cpu_handles
        with pytest.raises(sqlite3.ProgrammingError):
            first.lake.conn.execute("SELECT 1")
        with pytest.raises(sqlite3.ProgrammingError):
            second.lake.conn.execute("SELECT 1")
        first = second = None
        gc.collect()
        assert all(signal.getsignal(sig) is handler for sig, handler in original.items())
        assert all(reference() is None for reference in references)
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)
        if first is not None:
            cpu_phase.close(first)
        if second is not None:
            cpu_phase.close(second)
