"""Test-only genuine-interpreter holdout workflow on existing public APIs.

Task declarations here are explicit acceptance inputs, not Policy-generated
research proposals. The outer supervisor owns only each test controller; every
research worker is still launched by the unchanged CLI/Orze native CPU path.
"""
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import tempfile
import time
from unittest.mock import patch

import pytest
import yaml

INSTANCE_PATH = Path(__file__).with_name("instance.json")
TASKS = {
    "baseline_producer": "idea-baseline-producer",
    "baseline_v1": "idea-baseline-v1",
    "challenger_producer": "idea-challenger-producer",
    "failed_v1": "idea-challenger-v1-failed",
    "recovered_v1": "idea-challenger-v1-recovered",
    "challenger_v2": "idea-challenger-v2",
}
TABLES = (
    "ideas", "idea_state", "idea_transitions", "idea_stage_state", "idea_stage_transitions",
    "execution_attempts", "research_artifacts", "research_observations",
    "cpu_action_reservations", "cpu_action_decisions", "cpu_action_scopes",
    "cpu_proposal_requests", "replication_requests",
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _database(root):
    path = root / "lake.db"
    if not path.exists():
        return {name: [] for name in TABLES}
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        existing = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        return {name: [dict(row) for row in connection.execute(
            "SELECT * FROM " + name + " ORDER BY rowid")] if name in existing else []
            for name in TABLES}


def _snapshot(run, label):
    # Admission and CLI execution may share a human-readable step name.
    # References must identify the exact snapshot, not that non-unique name.
    value = {"label": str(len(run["snapshots"])) + ":" + label,
             "database": _database(run["root"])}
    run["snapshots"].append(value)
    return value


def _admit(run, label, task_id, request):
    from orze.idea_lake import IdeaLake
    before = _snapshot(run, label + ":before")
    raw = _canonical({"kind": "native_cpu_action", "domain_request": request}).decode()
    lake = IdeaLake(str(run["root"] / "lake.db"))
    try:
        outcome = lake.insert(task_id, task_id, raw, "",
            status="queued", kind="native_cpu_action", if_absent=True)
    finally:
        lake.close()
    after = _snapshot(run, label + ":after")
    record = {"label": label, "task_id": task_id, "domain_request": copy.deepcopy(request),
              "raw_config": raw, "outcome": outcome,
              "before_snapshot": before["label"], "after_snapshot": after["label"]}
    run["admissions"].append(record)
    return outcome


def _invoke(run, label, arguments, *, fault=None):
    from orze.engine.supervised_process import prepare_supervised, SupervisionUncertain
    root = run["root"]
    repository = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=str(repository / "src") + os.pathsep + str(repository))
    env.pop("ORZE_HOLDOUT_EVALUATOR_FAULT", None)
    if fault is not None:
        assert fault == "partial_exit_71"
        env["ORZE_HOLDOUT_EVALUATOR_FAULT"] = fault
    command = [sys.executable, "-m", "examples.holdout.testing", *arguments]
    before = _snapshot(run, label + ":before")
    process = None
    started = time.monotonic()
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as error:
        try:
            try:
                process = prepare_supervised(command, identity={
                    "scope": str(root), "holdout_test_controller": label},
                    cwd=str(root), env=env, stdout=output, stderr=error)
            except SupervisionUncertain as exc:
                process = exc.process
                raise
            process.start()
            code = process.wait(timeout=30)
            closure = process.closure_receipt()
            binding = process.binding
        finally:
            if process is not None and process.poll() is None:
                process.stop(timeout=10)
        finished = time.monotonic()
        output.seek(0)
        error.seek(0)
        stdout, stderr = output.read().decode("utf-8"), error.read().decode("utf-8")
    metadata = [json.loads(line.split("=", 1)[1]) for line in stdout.splitlines()
                if line.startswith("HOLDOUT_CLI_META=")]
    assert len(metadata) == 2 and [item["event"] for item in metadata] == ["start", "finish"], (
        label, code, stdout[-1000:], stderr[-3000:])
    assert metadata[0]["pid"] == metadata[1]["pid"] == binding["worker"]["pid"]
    assert metadata[0]["start_ticks"] == metadata[1]["start_ticks"] == binding["worker"]["start_ticks"]
    assert code == metadata[1]["exit_code"] == 0, (label, code, stderr[-3000:])
    assert closure["event"] == "TREE_CLOSED" and closure["wait_proof"] == "ECHILD_WALL"
    assert closure["worker_returncode"] == code
    after = _snapshot(run, label + ":after")
    record = {"label": label, "command": command, "cwd": str(root), "exit_code": code,
              "stdout": stdout, "stderr": stderr, "metadata": metadata,
              "started_monotonic": started, "finished_monotonic": finished,
              "wall_seconds": finished - started, "fault_injection": fault,
              "controller_supervision": {"binding": binding, "closure": closure},
              "before_snapshot": before["label"], "after_snapshot": after["label"]}
    run["calls"].append(record)
    return record


def _execute(run, label, *, fault=None):
    return _invoke(run, label, ["-c", str(run["root"] / "orze.yaml"), "--once"], fault=fault)


def _new_run(root):
    instance = json.loads(INSTANCE_PATH.read_bytes())
    cfg = {"execution": {"version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 40},
           "results_dir": str(root / "results"), "idea_lake_db": str(root / "lake.db"),
           "ideas_file": str(root / "ideas.md"), "min_disk_gb": 0,
           "action_domain": {"version": 1, "kind": "schedule_holdout", "config": {"instance": instance}},
           "action_policy": {"version": 1, "kind": "queue", "idle": "wait",
                             "wait_seconds": .05, "config": {}}}
    (root / "orze.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return {"root": root, "cfg": cfg, "calls": [], "admissions": [], "snapshots": []}


def _artifact(run, task, name="candidate"):
    rows = [json.loads(row["record_json"]) for row in _database(run["root"])["research_artifacts"]]
    selected = [row for row in rows if row["producer"]["task_id"] == task and row["logical_name"] == name]
    assert len(selected) == 1
    return selected[0]


def _finish(run):
    run["database"] = _database(run["root"])
    run["artifacts"] = [json.loads(row["record_json"]) for row in run["database"]["research_artifacts"]]
    run["observations"] = [json.loads(row["record_json"]) for row in run["database"]["research_observations"]]
    contents, envelopes = {}, {}
    for artifact in run["artifacts"]:
        path = Path(artifact["path"])
        assert path.is_relative_to(run["root"])
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == artifact["content_sha256"]
        contents[artifact["artifact_id"]] = raw.decode("utf-8")
        if artifact["logical_name"] == "evaluation":
            envelopes[artifact["producer"]["task_id"]] = json.loads(raw)
    run.update(artifact_contents=contents, result_envelopes=envelopes)
    return run


def workflow(root):
    from .scheduling import make_request, PROTOCOLS
    run = _new_run(root)
    for name, candidate in (("baseline_producer", "baseline"), ("challenger_producer", "challenger")):
        request = make_request("produce", candidate=candidate)
        assert _admit(run, name, TASKS[name], request)["status"] == "inserted"
        _execute(run, name)
        if name == "baseline_producer":
            source = _artifact(run, TASKS[name])
            request = make_request("evaluate", protocol=PROTOCOLS[0], source_id=source["artifact_id"])
            assert _admit(run, "baseline_v1", TASKS["baseline_v1"], request)["status"] == "inserted"
            _execute(run, "baseline_v1")
    source = _artifact(run, TASKS["challenger_producer"])
    request = make_request("evaluate", protocol=PROTOCOLS[0], source_id=source["artifact_id"])
    assert _admit(run, "failed_v1", TASKS["failed_v1"], request)["status"] == "inserted"
    _execute(run, "failed_v1", fault="partial_exit_71")
    failed_row = next(row for row in _database(root)["execution_attempts"]
                      if row["task_id"] == TASKS["failed_v1"])
    binding = json.loads(failed_row["binding_json"])
    partial_path = Path(binding["work_dir"]) / "evaluation.json"
    assert partial_path.is_relative_to(root)
    partial = partial_path.read_bytes()
    run["partial_output"] = {"path": str(partial_path), "utf8": partial.decode(),
                             "sha256": hashlib.sha256(partial).hexdigest(), "bytes": len(partial)}
    assert _admit(run, "failed_same_id_replay", TASKS["failed_v1"], request)["status"] == "already_present_exact"
    assert _admit(run, "recovered_v1", TASKS["recovered_v1"], request)["status"] == "inserted"
    _execute(run, "recovered_v1")
    request_v2 = make_request("evaluate", protocol=PROTOCOLS[1], source_id=source["artifact_id"])
    assert _admit(run, "challenger_v2", TASKS["challenger_v2"], request_v2)["status"] == "inserted"
    _execute(run, "challenger_v2")
    replicate_args = ["replicate", TASKS["recovered_v1"], "-c", str(root / "orze.yaml"),
                      "--request-id", "holdout-confirm-v1", "--reason", "explicit holdout evaluator repeat"]
    _invoke(run, "replicate_admit", replicate_args)
    _execute(run, "execute_replica")
    _invoke(run, "replicate_replay", replicate_args)
    for index in range(3):
        _execute(run, "idle_" + str(index))
    return _finish(run)


def boundary_run(root, condition):
    from .scheduling import make_request, PROTOCOLS
    run = _new_run(root)
    instance_id = run["cfg"]["action_domain"]["config"]["instance"]["instance_id"]
    rows = {
        "boundary": [{"job_id": "a", "start": 0}, {"job_id": "b", "start": 3}, {"job_id": "e", "start": 6}],
        "duplicate_id": [{"job_id": "a", "start": 0}, {"job_id": "a", "start": 0}],
        "missing_prerequisite": [{"job_id": "a", "start": 0}, {"job_id": "e", "start": 6}],
        "overload": [{"job_id": "a", "start": 0}, {"job_id": "b", "start": 0}],
    }
    raw = ('{"instance_id":' if condition == "malformed" else
           _canonical({"instance_id": instance_id, "schedule": rows[condition]}).decode())
    request = make_request("produce", artifact_utf8=raw)
    assert _admit(run, "producer", "idea-fixture-producer", request)["status"] == "inserted"
    _execute(run, "producer")
    source = _artifact(run, "idea-fixture-producer")
    request = make_request("evaluate", protocol=PROTOCOLS[0], source_id=source["artifact_id"])
    assert _admit(run, "evaluator", "idea-fixture-evaluator", request)["status"] == "inserted"
    _execute(run, "evaluator")
    run["condition"] = condition
    return _finish(run)


@pytest.fixture(scope="session")
def holdout_runs(tmp_path_factory, request):
    runs = {}
    for name in ("workflow", "boundary", "duplicate_id", "missing_prerequisite", "overload", "malformed"):
        root = tmp_path_factory.mktemp("holdout-" + name)
        run = workflow(root) if name == "workflow" else boundary_run(root, name)
        raw = (json.dumps({**run, "root": str(root)}, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        path = root / "holdout-report.json"
        path.write_bytes(raw)
        reports = getattr(request.config, "_holdout_reports", None)
        if reports is None:
            reports = request.config._holdout_reports = []
        reports.append({"name": name, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        runs[name] = run
    return runs


def pytest_sessionfinish(session, exitstatus):
    reports = getattr(session.config, "_holdout_reports", [])
    if reports:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line("HOLDOUT_REPORTS=" + json.dumps(reports, sort_keys=True))


def _child_main():
    import orze.cli as cli
    import orze.engine.orchestrator as orchestrator
    import orze.engine.gpu_slots as slots
    from .__main__ import main

    def forbidden(*args, **kwargs):
        raise AssertionError("holdout attempted GPU/provider/legacy repair integration")

    def metadata(event, **extra):
        # Only inspect this freshly spawned controller's own birth identity.
        stat = Path("/proc/self/stat").read_text()
        ticks = int(stat[stat.rfind(")") + 2:].split()[19])
        return {"event": event, "pid": os.getpid(), "start_ticks": ticks,
                "python": sys.executable, "monotonic": time.monotonic(), **extra}

    sys.argv = ["orze", *sys.argv[1:]]
    print("HOLDOUT_CLI_META=" + json.dumps(metadata("start"), sort_keys=True), flush=True)
    with contextlib.ExitStack() as stack:
        for target in ("orze.extensions.has_pro", "orze.extensions._find_pro_key",
                       "orze.engine.failure._try_executor_fix"):
            stack.enter_context(patch(target, forbidden))
        stack.enter_context(patch.object(cli, "detect_all_gpus", forbidden))
        stack.enter_context(patch.object(slots, "GpuSlotManager", forbidden))
        stack.enter_context(patch.object(orchestrator, "acquire_gpu_leases", forbidden))
        stack.enter_context(patch.object(orchestrator, "startup_canary", forbidden))
        stack.enter_context(patch.object(orchestrator.Orze, "_kill_orphans", forbidden))
        code = main()
    print("HOLDOUT_CLI_META=" + json.dumps(metadata("finish", exit_code=code), sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(_child_main())
