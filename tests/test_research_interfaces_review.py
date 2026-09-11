"""Independent interface mechanisms with a real Lake and source capture.

No worker is launched. Empty source captures are real process-local owners;
registry replacement and callback data mutation are explicit protocol faults,
not an OS sandbox test or a claim about arbitrary hostile trusted Python.
"""
import copy
import hashlib
import sys

import pytest
import yaml

from orze.core import cpu_action_budget as budget
from orze.core import research_interfaces as api
from orze.engine.cpu_action_sources import capture_sources
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path, monkeypatch):
    # Isolate real registrations, retaining built-ins and the actual API code.
    for name in ("_DOMAINS", "_POLICIES", "_CONTEXTS", "_RUNS"):
        monkeypatch.setattr(api, name, dict(getattr(api, name)))
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    scope = budget.initialize(lake, results, {
        "version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 5})
    request = {"version": 1, "purpose": "review captured interface authority",
        "inputs": {"data": [0, -1]}, "timeout_seconds": 1, "outputs": {},
        "input_artifact_ids": [], "payload": {"command": [sys.executable, "-c", "pass"]}}
    cfg = {"action_domain": {"version": 1, "kind": "review_domain", "config": {}},
        "action_policy": {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": 0.05}}
    try:
        yield lake, scope, results, request, cfg
    finally:
        lake.close()


def raw(request):
    return yaml.safe_dump({"kind": "native_cpu_action", "domain_request": request})


def test_prepare_cannot_rewrite_captured_execution_envelope(project):
    lake, scope, results, request, cfg = project

    class Domain(api.CommandDomain):
        def prepare(self, received, sources):
            received["timeout_seconds"] = 99
            return super().prepare(received, sources)

    api.register_domain("review_domain", "review.domain.v1", Domain)
    context = api.capture_interfaces(cfg)
    sources = capture_sources(lake, results, [])
    before = budget.snapshot(lake, scope)
    changes, owners = lake.conn.total_changes, len(api._RUNS)
    with pytest.raises(api.ResearchInterfaceError, match="execution bound"):
        api.prepare_domain_run(context, raw(request), sources)
    assert request["timeout_seconds"] == 1
    assert len(api._RUNS) == owners
    assert budget.snapshot(lake, scope) == before
    assert lake.conn.total_changes == changes


@pytest.mark.parametrize("stage", ["prepare", "interpret"])
def test_registration_replacement_inside_callback_cannot_return_authority(project, stage):
    lake, scope, results, request, cfg = project

    class Domain(api.CommandDomain):
        def replace(self):
            api._DOMAINS["review_domain"] = ("review.replacement.v1", api.CommandDomain)

        def prepare(self, received, sources):
            value = super().prepare(received, sources)
            if stage == "prepare":
                self.replace()
            return value

        def interpret(self, prepared, envelope):
            self.replace()
            return ()

    api.register_domain("review_domain", "review.domain.v1", Domain)
    context = api.capture_interfaces(cfg)
    sources = capture_sources(lake, results, [])
    before = budget.snapshot(lake, scope)
    changes, owners = lake.conn.total_changes, len(api._RUNS)
    if stage == "prepare":
        with pytest.raises(api.ResearchInterfaceError, match="registration changed"):
            api.prepare_domain_run(context, raw(request), sources)
        assert len(api._RUNS) == owners
    else:
        run = api.prepare_domain_run(context, raw(request), sources)
        with pytest.raises(api.ResearchInterfaceError, match="registration changed"):
            api.interpret_domain_run(run, None)
        with pytest.raises(api.ResearchInterfaceError, match="registration changed"):
            api.domain_run_metadata(run)
    assert budget.snapshot(lake, scope) == before
    assert lake.conn.total_changes == changes


def test_policy_input_mutation_cannot_expand_execute_selection(project):
    lake, scope, _, _, cfg = project
    api.register_domain("review_domain", "review.domain.v1", api.CommandDomain)
    called = []

    class Policy:
        def __init__(self, declaration):
            self.declaration = declaration

        def decide(self, received, allowance):
            called.append((received, allowance))
            received["queue"].append({"idea_id": "unlisted"})
            allowance["remaining_wall_seconds"] = 999
            return {"kind": "Execute", "task_id": "unlisted"}

    api.register_policy("review_policy", "review.policy.v1", Policy)
    cfg["action_policy"]["kind"] = "review_policy"
    policy = api.BoundPolicy(api.capture_interfaces(cfg))
    snapshot = {"queue": [{"idea_id": "listed"}], "now": 10, "active": False}
    allowance = budget.snapshot(lake, scope)
    before_snapshot, before_budget = copy.deepcopy(snapshot), copy.deepcopy(allowance)
    changes = lake.conn.total_changes
    with pytest.raises(api.ResearchInterfaceError, match="captured queue"):
        policy.decide(snapshot, allowance)
    assert len(called) == 1
    assert snapshot == before_snapshot
    assert allowance == before_budget
    assert budget.snapshot(lake, scope) == before_budget
    assert lake.conn.total_changes == changes


def test_run_copies_and_metadata_are_not_execution_capabilities(project):
    lake, _, results, request, cfg = project
    api.register_domain("review_domain", "review.domain.v1", api.CommandDomain)
    context = api.capture_interfaces(cfg)
    sources = capture_sources(lake, results, [])
    encoded = raw(request)
    run = api.prepare_domain_run(context, encoded, sources)
    action, metadata = run.action, api.domain_run_metadata(run)
    action["timeout_seconds"] = 99
    metadata["action_sha256"] = "f" * 64
    assert run.action["timeout_seconds"] == 1
    assert api.domain_run_metadata(run)["action_sha256"] != metadata["action_sha256"]
    assert api.require_domain_run(run, raw_config_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                                  action=run.action)["observation"] is None
    for forged in (copy.copy(run), api.DomainRun(), metadata):
        with pytest.raises(api.ResearchInterfaceError, match="run unavailable"):
            api.require_domain_run(forged, raw_config_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                                   action=run.action)
    assert api.interpret_domain_run(run, None) == ()
