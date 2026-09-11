"""Real invocation/source captures and replaceable local interface contracts."""
import hashlib
import sys

import pytest
import yaml

from orze.idea_lake import IdeaLake
from orze.core import research_interfaces as api
from orze.engine.cpu_action_sources import capture_sources


@pytest.fixture
def captured(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "lake.db"))
    cfg = {"action_domain": {"version": 1, "kind": "command", "config": {}},
           "action_policy": {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": 0.05}}
    request = {"version": 1, "purpose": "verify bounded generic data",
               "inputs": {"values": [0, None, False, -1]}, "timeout_seconds": 1,
               "outputs": {}, "input_artifact_ids": [],
               "payload": {"command": [sys.executable, "-c", "pass"]}}
    try:
        yield lake, results, cfg, request
    finally:
        lake.close()


def test_domain_preparation_handles_bounded_arrays_scalars_and_none(captured):
    lake, results, cfg, request = captured
    context = api.capture_interfaces(cfg)
    sources = capture_sources(lake, results, [])
    raw = yaml.safe_dump({"kind": "native_cpu_action", "domain_request": request})
    run = api.prepare_domain_run(context, raw, sources)
    assert run.action["purpose"] == request["purpose"]
    assert run.action["inputs"] == request["inputs"]
    assert api.interpret_domain_run(run, None) == ()
    assert api.require_domain_run(run, raw_config_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                                  action=run.action)["observation"] is None
