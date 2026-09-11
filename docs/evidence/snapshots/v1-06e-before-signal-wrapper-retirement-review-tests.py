"""Independent retirement/cache boundaries with real supervised CPU work.

Collection is not closure authority. Only after the real effect and budget
settlement may a discarded handle become unreachable. Fake public handles,
wrong routes and modified permits confer no cached outcome permission. A
caller-held DomainRun/source capability remains usable after its consumer ends.
No GPU/provider, process adoption, host scan, or simulated closure proof.
"""
import copy
import gc
import os
import weakref

import pytest
import yaml

from orze.core import cpu_action_budget as budget
from orze.core import research_interfaces as api
from orze.engine import cpu_action_sources as sources
from orze.engine import native_cpu_action as native
from orze.engine.scheduler import claim
from test_cpu_action_sources import published
from test_cpu_domain_product import request
from test_native_cpu_action import context, _finish


def test_closed_settled_handle_has_no_terminal_cache_value_backreference(context):
    lake, results, scope, cfg, create, processes = context
    action, permit = create("pass")
    handle = native.launch("idea-cpu", results, cfg, lake=lake, action=action,
                           permit=permit, admission=lambda: None)
    terminal = _finish(handle, results, cfg, lake, permit)
    assert terminal["outcome"] == "completed"
    assert terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
    assert lake.conn.execute("SELECT state FROM cpu_action_reservations").fetchone()[0] == "SETTLED"
    reference = weakref.ref(handle)
    del handle
    gc.collect()
    # The cleanup fixture deliberately retains the actual process object.
    # That process or a cache value must not strongly retain the retired handle.
    assert reference() is None


def test_cached_terminal_is_detached_and_exact_handle_route_permit_bound(context):
    lake, results, scope, cfg, create, processes = context
    action, permit = create("pass")
    handle = native.launch("idea-cpu", results, cfg, lake=lake, action=action,
                           permit=permit, admission=lambda: None)
    terminal = _finish(handle, results, cfg, lake, permit)
    original = copy.deepcopy(terminal)
    terminal["outcome"] = "forged"
    terminal["process_tree"]["binding"]["worker"]["pid"] = -1
    assert native.harvest(handle, results, cfg, lake=lake, permit=permit) == original
    cached = native.stop(handle, results, cfg, lake=lake, permit=permit)
    cached["artifact_ids"].append("unpublished")
    assert native.stop(handle, results, cfg, lake=lake, permit=permit) == original
    forged = native.CPUActionHandle(handle.idea_id, handle.attempt_id,
                                    handle.attempt_ref, handle.process)
    with pytest.raises(native.CPUActionHOLD):
        native.harvest(forged, results, cfg, lake=lake, permit=permit)
    with pytest.raises(native.CPUActionHOLD):
        native.stop(handle, results.parent / "other-scope", cfg, lake=lake, permit=permit)
    changed = copy.deepcopy(permit)
    changed["reservation_id"] = "another-reservation"
    with pytest.raises(native.CPUActionHOLD):
        native.harvest(handle, results, cfg, lake=lake, permit=changed)
    assert native.harvest(handle, results, cfg, lake=lake, permit=permit) == original
    assert lake.conn.execute("SELECT count(*) FROM execution_attempts").fetchone()[0] == 1
    assert lake.conn.execute("SELECT state FROM cpu_action_reservations").fetchone()[0] == "SETTLED"


def test_retiring_one_native_consumer_preserves_external_run_and_shared_sources(published):
    lake, results, records, source_handles = published
    cfg = {"_project_root": str(results.parent), "_orze_dir": str(results.parent / ".orze"),
           "action_domain": {"version": 1, "kind": "command", "config": {}},
           "action_policy": {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": .05}}
    interface = api.capture_interfaces(cfg)
    prepared = sources.capture_sources(lake, results, [records[0]["artifact_id"]])
    declared = request("pass", sources=[records[0]["artifact_id"]])
    raw = yaml.safe_dump({"kind": "native_cpu_action", "domain_request": declared})
    first = api.prepare_domain_run(interface, raw, prepared)
    second = api.prepare_domain_run(interface, raw, prepared)
    assert api.domain_sources(first) is api.domain_sources(second) is prepared
    assert lake.insert("review-analysis", "review analysis", raw, "", status="queued",
                       kind="native_cpu_action", if_absent=True)["status"] == "inserted"
    scope = budget.initialize(lake, results, {
        "version": 1, "resource": "cpu", "slots": 2, "wall_budget_seconds": 20})
    permit = budget.reserve(lake, scope, "review-analysis", 2)
    assert permit is not None
    assert claim("review-analysis", results, None, lake, resource="cpu")
    handle = None
    try:
        handle = native.launch("review-analysis", results, cfg, lake=lake, action=first.action,
                               permit=permit, admission=lambda: None, domain_run=first)
        terminal = _finish(handle, results, cfg, lake, permit)
        assert terminal["outcome"] == "completed"
        assert budget.snapshot(lake, scope)["active_reservations"] == 0
        del handle
        handle = None
        gc.collect()
        # Caller ownership remains explicit, including another unexecuted run.
        # This does not claim that a single run can be concurrently launched twice.
        assert api.domain_sources(first) is api.domain_sources(second) is prepared
        assert api.interpret_domain_run(first, None) == ()
        assert api.interpret_domain_run(second, None) == ()
        sources.require_sources(lake, results, prepared)
        with sources.sealed_sources(api.domain_sources(second)) as (environment, descriptors):
            assert len(descriptors) == 1
            assert os.pread(descriptors[0], 1024, 0) == b"one"
    finally:
        if handle is not None and handle.process.poll() is None:
            handle.process.stop(timeout=.2)


def test_successful_close_does_not_replace_an_external_signal_handler(tmp_path, monkeypatch):
    import signal
    from orze.core.config import load_project_config
    from orze.engine import cpu_phase
    from orze.engine.orchestrator import Orze

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    path = tmp_path / "orze.yaml"
    path.write_text(yaml.safe_dump({
        "execution": {"version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 5},
        "results_dir": str(tmp_path / "results"), "ideas_file": str(tmp_path / "ideas.md"),
        "idea_lake_db": str(tmp_path / "lake.db"), "min_disk_gb": 0,
        "action_policy": {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": .05}}))
    monkeypatch.chdir(tmp_path)
    engine = None
    called = []

    def external_handler(signum, frame):
        called.append(signum)

    try:
        engine = Orze([], load_project_config(str(path)), once=True)
        assert signal.getsignal(signal.SIGTERM).__self__ is engine
        signal.signal(signal.SIGTERM, external_handler)
        cpu_phase.close(engine)
        assert signal.getsignal(signal.SIGTERM) is external_handler
        assert engine._cpu_closed
        assert called == []
    finally:
        if engine is not None and not engine._cpu_closed:
            cpu_phase.close(engine)
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def test_cached_terminal_cannot_rebind_process_using_handle_supplied_weakref(context):
    lake, results, scope, cfg, create, processes = context
    action, permit = create("pass")
    handle = native.launch("idea-cpu", results, cfg, lake=lake, action=action,
                           permit=permit, admission=lambda: None)
    terminal = _finish(handle, results, cfg, lake, permit)
    assert terminal["outcome"] == "completed"
    assert lake.conn.execute("SELECT state FROM cpu_action_reservations").fetchone()[0] == "SETTLED"

    class ForeignProcess:
        pass

    actual = handle.process
    supplied = getattr(handle, "_retired_process", None)
    foreign = ForeignProcess()
    try:
        handle.process = foreign
        handle._retired_process = weakref.ref(foreign)
        with pytest.raises(native.CPUActionHOLD):
            native.harvest(handle, results, cfg, lake=lake, permit=permit)
    finally:
        handle.process = actual
        if supplied is None:
            del handle._retired_process
        else:
            handle._retired_process = supplied
    assert native.harvest(handle, results, cfg, lake=lake, permit=permit) == terminal
