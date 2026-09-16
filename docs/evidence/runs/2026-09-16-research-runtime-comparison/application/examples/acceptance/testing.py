"""Test-only plugin: one shared four-project real CLI epoch per pytest session.

No authority/executor/Lake operation is mocked. Only fresh trusted registration,
argv/cwd and forbidden GPU/legacy entry points are isolated. Domain unit tests'
separate workers are not counted as these sixteen native action occurrences.
"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import signal
import sqlite3
import sys
import time

import pytest
import yaml


def _database(root):
    names = ("ideas", "execution_attempts", "research_artifacts", "research_observations",
             "cpu_action_reservations", "cpu_action_decisions", "cpu_action_scopes",
             "cpu_proposal_requests", "replication_requests")
    with sqlite3.connect((root / "lake.db").as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return {name: [dict(row) for row in conn.execute("SELECT * FROM " + name + " ORDER BY rowid")]
                for name in names}


def _invoke(root, config_path):
    from orze.core import research_interfaces as api
    import orze.cli as cli
    import orze.engine.orchestrator as orchestrator
    import orze.engine.gpu_slots as slots
    from .__main__ import main

    def forbidden(*args, **kwargs):
        raise AssertionError("acceptance CLI attempted GPU/legacy/provider integration")

    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    output, error = io.StringIO(), io.StringIO()
    started = time.monotonic()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.chdir(root)
            patch.setattr(sys, "argv", ["orze", "-c", str(config_path)])
            # Equivalent to fresh application registration in a new interpreter;
            # never replace an executing invocation's Domain/Policy.
            patch.setattr(api, "_DOMAINS", dict(api._DOMAINS))
            patch.setattr(api, "_POLICIES", dict(api._POLICIES))
            patch.setattr(cli, "detect_all_gpus", forbidden)
            patch.setattr(slots, "GpuSlotManager", forbidden)
            patch.setattr(orchestrator, "acquire_gpu_leases", forbidden)
            patch.setattr(orchestrator, "startup_canary", forbidden)
            patch.setattr(orchestrator.Orze, "_kill_orphans", forbidden)
            patch.setattr("orze.extensions.has_pro", forbidden)
            patch.setattr("orze.extensions._find_pro_key", forbidden)
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
                code = main()
        assert all(signal.getsignal(sig) is handler for sig, handler in original.items())
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)
    finished = time.monotonic()
    trace = [json.loads(line[len("ACCEPTANCE_DECISION="):])
             for line in output.getvalue().splitlines() if line.startswith("ACCEPTANCE_DECISION=")]
    return {"exit_code": code, "trace": trace, "stdout": output.getvalue(),
            "stderr": error.getvalue(), "started_monotonic": started,
            "finished_monotonic": finished, "wall_seconds": finished - started}


def run_scenario(root, domain, dataset):
    from .common import digest
    cfg = {"execution": {"version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 10},
        "results_dir": str(root / "results"), "idea_lake_db": str(root / "lake.db"),
        "ideas_file": str(root / "ideas.md"), "min_disk_gb": 0,
        "action_domain": {"version": 1, "kind": "acceptance_" + domain, "config": {"dataset": dataset}},
        "action_policy": {"version": 1, "kind": "acceptance", "idle": "stop",
                          "wait_seconds": .05, "config": {}}}
    config_path = root / "orze.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run = _invoke(root, config_path)
    assert run["exit_code"] == 0, (domain, run["exit_code"], run["stderr"][-2000:])
    database = _database(root)
    artifacts = [json.loads(row["record_json"]) for row in database["research_artifacts"]]
    observations = [json.loads(row["record_json"]) for row in database["research_observations"]]
    envelopes = {}
    for artifact in artifacts:
        path = Path(artifact["path"])
        assert path.is_relative_to(root), "acceptance cannot read outside its private project"
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == artifact["content_sha256"]
        envelopes[artifact["producer"]["task_id"]] = json.loads(raw)
    rerun = _invoke(root, config_path)
    assert rerun["exit_code"] == 75, "persistent Stop must refuse a fresh CLI invocation"
    assert _database(root) == database, "stopped restart changed authoritative research rows"
    assert all(item["snapshot_sha256"] == digest(item["snapshot"]) for item in run["trace"])
    run.update(root=root, cfg=copy.deepcopy(cfg), dataset_sha256=digest(dataset),
               database=database, artifacts=artifacts, observations=observations,
               result_envelopes=envelopes, stopped_rerun=rerun)
    return run


@pytest.fixture(scope="session")
def acceptance_runs(tmp_path_factory, request):
    from . import sorting, compression
    runs = {}
    for domain, module in (("sorting", sorting), ("compression", compression)):
        for variant, dataset in (("default", module.DEFAULT_DATASET),
                                 ("counterfactual", module.COUNTERFACTUAL_DATASET)):
            root = tmp_path_factory.mktemp("acceptance-" + domain + "-" + variant)
            name = domain + "_" + variant
            run = run_scenario(root, domain, copy.deepcopy(dataset))
            # Preserve the actual completed CLI output, trace, timings and rows;
            # never reconstruct a trace from database state or file timestamps.
            report = {**run, "root": str(root)}
            raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            report_path = root / "acceptance-report.json"
            report_path.write_bytes(raw)
            reports = getattr(request.config, "_acceptance_reports", None)
            if reports is None:
                reports = request.config._acceptance_reports = []
            reports.append({"name": name, "path": str(report_path),
                            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
            runs[name] = run
    return runs


def pytest_sessionfinish(session, exitstatus):
    reports = getattr(session.config, "_acceptance_reports", [])
    if reports:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line("ACCEPTANCE_REPORTS=" + json.dumps(reports, sort_keys=True))
