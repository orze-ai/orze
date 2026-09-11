"""Lifetime requirements for actual captured Domain/Policy/source interfaces.

The two source publications use the existing real CPU fixture and own-process
cleanup. GC is only measured after references are dropped; it does not authorize
execution, settlement, closure, or forgetting an unresolved native owner.
"""
import copy
import gc
import hashlib
import sys
import weakref

import pytest
import yaml

from orze.core import research_interfaces as api
from orze.engine import cpu_action_sources as sources
from orze.idea_lake import IdeaLake
from test_cpu_action_sources import published


@pytest.fixture
def selection(monkeypatch):
    # Isolate trusted registration, not the production capture registries.
    monkeypatch.setattr(api, "_DOMAINS", dict(api._DOMAINS))
    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))
    references = {}

    class Domain(api.CommandDomain):
        def __init__(self, config):
            super().__init__(config)
            references["domain"] = weakref.ref(self)

    class Policy:
        def __init__(self, declaration):
            references["policy"] = weakref.ref(self)

        def decide(self, snapshot, allowance):
            return {"kind": "Wait", "reason": "lifetime control", "wakeup": 2}

    api.register_domain("lifetime_domain", "lifetime.domain.v1", Domain)
    api.register_policy("lifetime_policy", "lifetime.policy.v1", Policy)
    cfg = {
        "action_domain": {"version": 1, "kind": "lifetime_domain", "config": {}},
        "action_policy": {"version": 1, "kind": "lifetime_policy", "idle": "wait",
                          "wait_seconds": 0.05},
    }
    return cfg, references


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "empty-results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "empty.db")
    try:
        yield lake, results
    finally:
        lake.close()


def prepare(selection, project, artifact_ids=()):
    cfg, _ = selection
    lake, results = project
    context = api.capture_interfaces(cfg)
    captured = sources.capture_sources(lake, results, list(artifact_ids))
    request = {
        "version": 1, "purpose": "captured lifetime", "inputs": {},
        "timeout_seconds": 1, "outputs": {}, "input_artifact_ids": list(artifact_ids),
        "payload": {"command": [sys.executable, "-c", "pass"]},
    }
    raw = yaml.safe_dump({"kind": "native_cpu_action", "domain_request": request})
    run = api.prepare_domain_run(context, raw, captured)
    return context, captured, run, raw


@pytest.mark.parametrize("callback_cycle", [False, True])
def test_unused_context_is_not_a_global_strong_root(selection, callback_cycle):
    cfg, references = selection
    context = api.capture_interfaces(cfg)
    if callback_cycle:
        references["domain"]().captured = context
        references["policy"]().captured = context
    reference = weakref.ref(context)
    identity = id(context)
    del context
    gc.collect()
    assert reference() is None
    assert references["domain"]() is None
    assert references["policy"]() is None
    assert identity not in api._CONTEXTS


def test_unused_run_releases_real_source_bytes_even_with_callback_cycle(selection, published):
    lake, results, records, _ = published
    context, captured, run, raw = prepare(
        selection, (lake, results), [record["artifact_id"] for record in records])
    assert sum(len(item.content) for item in captured._owner.items) == 6
    selection[1]["domain"]().captured = run
    references = (weakref.ref(run), weakref.ref(context), weakref.ref(captured._owner))
    identities = id(run), id(context)
    del run, context, captured
    gc.collect()
    assert all(reference() is None for reference in references)
    assert identities[0] not in api._RUNS
    assert identities[1] not in api._CONTEXTS


def test_external_run_keeps_context_sources_and_interpreter_valid(selection, project):
    context, captured, run, raw = prepare(selection, project)
    references = weakref.ref(context), weakref.ref(captured._owner)
    del context, captured
    gc.collect()
    assert all(reference() is not None for reference in references)
    assert api.require_domain_run(
        run, raw_config_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        action=run.action) == api.domain_run_metadata(run)
    sources.require_sources(*project, api.domain_sources(run))
    assert api.interpret_domain_run(run, None) == ()


def test_bound_policy_is_an_external_consumer(selection):
    context = api.capture_interfaces(selection[0])
    reference = weakref.ref(context)
    policy = api.BoundPolicy(context)
    del context
    gc.collect()
    assert reference() is not None
    assert policy.decide({"queue": [], "now": 1, "active": False}, {}) == {
        "kind": "Wait", "reason": "lifetime control", "wakeup": 2}


@pytest.mark.parametrize("kind", ["context", "run"])
def test_live_handle_copies_and_lookalikes_still_fail_closed(selection, project, kind):
    context, captured, run, raw = prepare(selection, project)
    if kind == "context":
        original, constructor, check = context, api.InterfaceContext, api.BoundPolicy
    else:
        original, constructor, check = run, api.DomainRun, api.domain_run_metadata
    for forged in (copy.copy(original), constructor(), {}):
        with pytest.raises(api.ResearchInterfaceError, match="unavailable"):
            check(forged)
    assert api.interpret_domain_run(run, None) == ()


@pytest.mark.parametrize("kind", ["domain", "policy"])
def test_live_registration_replacement_still_refuses_run(selection, project, kind):
    context, captured, run, raw = prepare(selection, project)
    registry = api._DOMAINS if kind == "domain" else api._POLICIES
    key = "lifetime_" + kind
    old = registry[key]
    registry[key] = (old[0], old[1])
    gc.collect()
    with pytest.raises(api.ResearchInterfaceError, match="registration changed"):
        api.domain_run_metadata(run)
    assert context is not None and captured is not None


@pytest.mark.parametrize("kind", ["context", "run"])
@pytest.mark.parametrize("fault", ["remove", "replace"])
def test_private_state_changes_cannot_rebind_live_authority(selection, project, kind, fault):
    # New state-storage mechanism: not a historical missing-API counterexample.
    context, captured, run, raw = prepare(selection, project)
    handle = context if kind == "context" else run
    original = handle._state
    try:
        if fault == "remove":
            object.__delattr__(handle, "_state")
        else:
            object.__setattr__(handle, "_state", dict(original))
        gc.collect()
        with pytest.raises(api.ResearchInterfaceError, match="unavailable"):
            api.domain_run_metadata(run)
    finally:
        object.__setattr__(handle, "_state", original)
    assert api.interpret_domain_run(run, None) == ()
