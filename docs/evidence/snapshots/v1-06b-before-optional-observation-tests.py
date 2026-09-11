"""New Domain-to-native mechanisms: actual CPU/SQLite, no GPU or providers."""
import json
import os
from pathlib import Path
import sys
import time

import pytest
import yaml

from orze.core import cpu_action_budget as budget
from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.core.research_observations import observations_for_attempt
from orze.core.research_interfaces import capture_interfaces, prepare_domain_run, DomainRun
from orze.engine import native_cpu_action as native
from orze.engine.cpu_action_sources import capture_sources
from orze.engine.scheduler import claim
from orze.idea_lake import IdeaLake


def finish(handle, results, cfg, lake, permit):
    until = time.monotonic() + 6
    while time.monotonic() < until:
        terminal = native.harvest(handle, results, cfg, lake=lake, permit=permit)
        if terminal is not None:
            return terminal
        time.sleep(0.01)
    pytest.fail("captured CPU tree did not close")


@pytest.fixture
def domain_context(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    scope = budget.initialize(lake, results, {
        "version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 30})
    cfg = {"_project_root": str(tmp_path), "_orze_dir": str(tmp_path / ".orze")}
    handles = []
    real = native.prepare_supervised

    def prepared(*args, **kwargs):
        process = real(*args, **kwargs)
        handles.append(process)
        return process

    monkeypatch.setattr(native, "prepare_supervised", prepared)

    def create(code, *, domain="json_observations", input_ids=(), outputs=None):
        outputs = ({"measurement": {"path": "result.json", "max_bytes": 4096}}
                   if outputs is None else outputs)
        request = {"version": 1, "purpose": "inspect data", "inputs": {},
            "timeout_seconds": 2, "outputs": outputs, "input_artifact_ids": list(input_ids),
            "payload": {"command": [sys.executable, "-c", code]}}
        if domain == "json_observations":
            request["payload"].update(specification={"subject": "explicit target"},
                protocol={"id": "test-protocol"}, result_output="measurement")
        raw = yaml.safe_dump({"kind": "native_cpu_action", "domain_request": request})
        assert lake.insert("idea-domain", "Domain task", raw, "", status="queued",
                           kind="native_cpu_action", if_absent=True)["status"] == "inserted"
        interfaces = capture_interfaces({**cfg, "action_domain": {
            "version": 1, "kind": domain, "config": {}}})
        sources = capture_sources(lake, results, list(input_ids))
        run = prepare_domain_run(interfaces, raw, sources)
        permit = budget.reserve(lake, scope, "idea-domain", 2)
        assert permit is not None
        assert claim("idea-domain", results, None, lake, resource="cpu")
        return run, permit

    yield lake, results, scope, cfg, create, handles
    for process in handles:
        if process.poll() is None:
            process.stop(timeout=0.2)
        assert type(process.poll()) is int
    lake.close()


def launch(context, run, permit):
    lake, results, _, cfg, _, _ = context
    return native.launch("idea-domain", results, cfg, lake=lake, action=run.action,
                         permit=permit, admission=lambda: None, domain_run=run)


def test_real_command_domain_explicit_zero_without_measurement(domain_context):
    lake, results, scope, cfg, create, handles = domain_context
    run, permit = create("pass", domain="command", outputs={})
    handle = launch(domain_context, run, permit)
    terminal = finish(handle, results, cfg, lake, permit)
    assert terminal["outcome"] == "completed"
    assert terminal["artifact_ids"] == terminal["observation_ids"] == []
    assert observations_for_attempt(lake.conn, handle.attempt_ref) == []
    assert budget.snapshot(lake, scope)["active_reservations"] == 0


@pytest.mark.parametrize("count", [0, 2])
def test_real_json_domain_publishes_explicit_zero_or_multiple(domain_context, count):
    lake, results, scope, cfg, create, handles = domain_context
    claims = [{"name": "value-" + str(i), "values": {"score": -i},
        "validation": {"status": "valid" if i == 0 else "invalid", "reason_code": "reported"},
        "comparison_scope": "protocol-v1"} for i in range(count)]
    envelope = json.dumps({"version": 1, "observations": claims})
    code = "from pathlib import Path;Path('result.json').write_text(" + repr(envelope) + ")"
    run, permit = create(code)
    handle = launch(domain_context, run, permit)
    terminal = finish(handle, results, cfg, lake, permit)
    observations = observations_for_attempt(lake.conn, handle.attempt_ref)
    artifacts = artifacts_for_attempt(lake.conn, handle.attempt_ref)
    assert terminal["outcome"] == "completed"
    assert len(observations) == len(terminal["observation_ids"]) == count
    assert len(artifacts) == 1
    assert [record["values"] for record in observations] == [claim["values"] for claim in claims]
    assert all(record["schema"] == 2 and record["evaluator"]["phase"] == "action" for record in observations)
    assert budget.snapshot(lake, scope)["active_reservations"] == 0


@pytest.mark.parametrize("body", [None, '{"version":1,"observations":[],"observations":[]}', '{}',
                                  '{"version":1,"observations":[NaN]}'])
def test_missing_or_malformed_result_never_becomes_empty_observation(domain_context, body):
    lake, results, scope, cfg, create, handles = domain_context
    code = "pass" if body is None else "from pathlib import Path;Path('result.json').write_text(" + repr(body) + ")"
    run, permit = create(code)
    handle = launch(domain_context, run, permit)
    with pytest.raises(native.CPUActionHOLD):
        finish(handle, results, cfg, lake, permit)
    assert current_attempt(lake.conn, "idea-domain", "action")["state"] == "RUNNING"
    assert observations_for_attempt(lake.conn, handle.attempt_ref) == []
    assert budget.snapshot(lake, scope)["active_reservations"] == 1


def test_real_source_fd_keeps_original_spec_and_current_evaluator_distinct(domain_context):
    lake, results, scope, cfg, create, handles = domain_context
    action = {"version": 1, "adapter": "command", "purpose": "produce source", "inputs": {},
        "timeout_seconds": 2, "command": [sys.executable, "-c", "from pathlib import Path;Path('data').write_text('abc')"],
        "outputs": {"data": {"path": "data", "max_bytes": 100}}}
    assert lake.insert("idea-source", "Source", yaml.safe_dump({"kind": "native_cpu_action", "action": action}),
        "", status="queued", kind="native_cpu_action", if_absent=True)["status"] == "inserted"
    permit0 = budget.reserve(lake, scope, "idea-source", 2)
    assert claim("idea-source", results, None, lake, resource="cpu")
    source = native.launch("idea-source", results, cfg, lake=lake, action=action,
                           permit=permit0, admission=lambda: None)
    source_terminal = finish(source, results, cfg, lake, permit0)
    source_record = artifacts_for_attempt(lake.conn, source.attempt_ref)[0]
    code = """import os,json,fcntl
from pathlib import Path
fds=json.loads(os.environ['ORZE_ACTION_SOURCE_FDS'])
assert len(fds)==1
fd=next(iter(fds.values()))
assert fcntl.fcntl(fd,fcntl.F_GETFL)&os.O_ACCMODE==os.O_RDONLY
assert os.read(fd,10)==b'abc'
assert fcntl.fcntl(fd,fcntl.F_GET_SEALS)&fcntl.F_SEAL_WRITE
Path('result.json').write_text(json.dumps({'version':1,'observations':[{'name':'size','values':{'length':3},'validation':{'status':'valid','reason_code':'byte_count'},'comparison_scope':None}]}))
"""
    run, permit = create(code, input_ids=source_terminal["artifact_ids"])
    handle = launch(domain_context, run, permit)
    terminal = finish(handle, results, cfg, lake, permit)
    observed = observations_for_attempt(lake.conn, handle.attempt_ref)[0]
    assert terminal["outcome"] == "completed"
    assert observed["values"] == {"length": 3}
    assert observed["spec_fingerprint"] != source_record["spec_fingerprint"]
    assert observed["input_artifact_bindings"][source_record["artifact_id"]]["producer"] == source_record["producer"]
    assert source_record["producer"] != observed["evaluator"]


def test_raw_task_mutation_at_actual_ready_refuses_go(domain_context, monkeypatch):
    lake, results, scope, cfg, create, handles = domain_context
    run, permit = create("pass", domain="command", outputs={})
    real = native.prepare_supervised

    def changed(*args, **kwargs):
        process = real(*args, **kwargs)
        lake.conn.execute("UPDATE ideas SET config=config || '\n# changed' WHERE idea_id='idea-domain'")
        lake.conn.commit()
        return process

    monkeypatch.setattr(native, "prepare_supervised", changed)
    with pytest.raises(native.CPUActionHOLD):
        launch(domain_context, run, permit)
    assert len(handles) == 1 and not handles[0]._started
    assert current_attempt(lake.conn, "idea-domain", "action")["state"] == "LAUNCHING"
    assert budget.snapshot(lake, scope)["active_reservations"] == 1


def test_forged_domain_handle_cannot_select_interpreter(domain_context):
    lake, results, scope, cfg, create, handles = domain_context
    run, permit = create("pass", domain="command", outputs={})
    with pytest.raises(native.CPUActionHOLD):
        native.launch("idea-domain", results, cfg, lake=lake, action=run.action,
            permit=permit, admission=lambda: None, domain_run=DomainRun())
    assert handles == []
    assert current_attempt(lake.conn, "idea-domain", "action") is None


def test_actual_terminal_trigger_cannot_rewrite_observation_before_settlement(domain_context):
    lake, results, scope, cfg, create, handles = domain_context
    envelope = {"version": 1, "observations": [{"name": "metric", "values": {"score": 1},
        "validation": {"status": "valid", "reason_code": "reported"}, "comparison_scope": None}]}
    run, permit = create("from pathlib import Path;Path('result.json').write_text(" + repr(json.dumps(envelope)) + ")")
    handle = launch(domain_context, run, permit)
    lake.conn.execute("""CREATE TRIGGER rewrite_domain_observation AFTER UPDATE ON execution_attempts
        WHEN NEW.state='TERMINAL' BEGIN
        UPDATE research_observations SET record_json=json_set(record_json,'$.values.score',99)
        WHERE evaluator_attempt_id=NEW.attempt_id;
        END""")
    lake.conn.commit()
    with pytest.raises(native.CPUActionHOLD):
        finish(handle, results, cfg, lake, permit)
    assert current_attempt(lake.conn, "idea-domain", "action")["state"] == "RUNNING"
    assert observations_for_attempt(lake.conn, handle.attempt_ref) == []
    assert budget.snapshot(lake, scope)["active_reservations"] == 1
