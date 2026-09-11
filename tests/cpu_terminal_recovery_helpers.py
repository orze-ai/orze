"""Owned fresh-controller CPU terminal-recovery acceptance fixtures.

This is test instrumentation, not a replacement execution or recovery engine.
No database writer is substituted. A deliberate os._exit happens only after
the actual native terminal transaction and effect guard have completed.
"""
import contextlib
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from unittest.mock import patch

import pytest
import yaml

TABLES = ("ideas", "idea_state", "idea_transitions", "idea_stage_state",
          "idea_stage_transitions", "execution_attempts", "research_artifacts",
          "research_observations", "cpu_action_scopes", "cpu_action_reservations",
          "cpu_action_decisions", "cpu_action_recovery")
WORKER = """import json, os
from pathlib import Path
fd = int(os.environ['ORZE_ACTION_INPUT_FD'])
data = json.loads(os.pread(fd, 65536, 0))
stat = Path('/proc/self/stat').read_text()
birth = int(stat[stat.rfind(')') + 2:].split()[19])
event = json.dumps({'tag': data['tag'], 'pid': os.getpid(), 'start_ticks': birth}, sort_keys=True)
with open(data['audit_path'], 'a', encoding='utf-8') as stream:
    stream.write(event + '\\n')
    stream.flush()
    os.fsync(stream.fileno())
Path('answer.json').write_text(json.dumps({'tag': data['tag']}), encoding='utf-8')
raise SystemExit(data['exit_code'])
"""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def database(root):
    path = root / "lake.db"
    if not path.exists():
        return {name: [] for name in TABLES}
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {name: [dict(row) for row in conn.execute("SELECT * FROM " + name + " ORDER BY rowid")]
                if name in names else [] for name in TABLES}


def snapshot(project, label):
    db = database(project["root"])
    files = {}
    for row in db["execution_attempts"]:
        base = project["root"] / "results" / row["task_id"] / "_execution_effects" / row["attempt_id"]
        for name in ("prepared.json", "committed.json"):
            path = base / name
            if path.exists():
                raw = path.read_bytes()
                files[str(path)] = {"utf8": raw.decode(), "bytes": len(raw),
                                    "sha256": hashlib.sha256(raw).hexdigest()}
    for row in db["research_artifacts"]:
        artifact = json.loads(row["record_json"])
        path = Path(artifact["path"])
        assert path.is_relative_to(project["root"])
        if path.exists():
            raw = path.read_bytes()
            files[str(path)] = {"utf8": raw.decode(), "bytes": len(raw),
                                "sha256": hashlib.sha256(raw).hexdigest()}
    audit = project["root"] / "worker-events.jsonl"
    events = [json.loads(line) for line in audit.read_text().splitlines()] if audit.exists() else []
    result = {"label": str(len(project["snapshots"])) + ":" + label,
              "database": db, "files": files, "worker_events": events}
    project["snapshots"].append(result)
    return result


def save_report(project):
    raw = (json.dumps({**project, "root": str(project["root"])}, sort_keys=True,
                      indent=2, allow_nan=False) + "\n").encode()
    path = project["root"] / "recovery-report.json"
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def admit(project, task_id, tag, *, exit_code=0):
    from orze.idea_lake import IdeaLake
    action = {"version": 1, "adapter": "command", "purpose": "CPU terminal recovery control",
              "inputs": {"tag": tag, "audit_path": str(project["root"] / "worker-events.jsonl"),
                         "exit_code": exit_code},
              "command": [sys.executable, "-c", WORKER], "timeout_seconds": 2,
              "outputs": {"answer": {"path": "answer.json", "max_bytes": 128}}}
    raw = canonical({"kind": "native_cpu_action", "action": action}).decode()
    before = snapshot(project, "admit:" + task_id + ":before")
    lake = IdeaLake(project["root"] / "lake.db")
    try:
        result = lake.insert(task_id, task_id, raw, "", status="queued",
                             kind="native_cpu_action", if_absent=True)
    finally:
        lake.close()
    after = snapshot(project, "admit:" + task_id + ":after")
    project["admissions"].append({"task_id": task_id, "config": raw, "result": result,
                                  "before": before["label"], "after": after["label"]})
    save_report(project)
    assert result["status"] == "inserted"
    return result


def run_cli(project, label, *, crash=False, expected=0, child_module="cpu_terminal_recovery_helpers"):
    from orze.engine.supervised_process import prepare_supervised, SupervisionUncertain
    repository = Path(__file__).resolve().parents[1]
    root = project["root"]
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(str(p) for p in (repository / "src", repository / "tests", repository)))
    command = [sys.executable, "-m", child_module]
    if crash:
        command.append("--crash-before-settle")
    command.extend(["-c", str(root / "orze.yaml"), "--once"])
    before = snapshot(project, label + ":before")
    process = None
    started = time.monotonic()
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            try:
                process = prepare_supervised(command, identity={
                    "scope": str(root), "terminal_recovery_controller": label},
                    cwd=str(root), env=env, stdout=out, stderr=err)
            except SupervisionUncertain as exc:
                process = exc.process
                raise
            process.start()
            code = process.wait(timeout=30)
            binding, closure = process.binding, process.closure_receipt()
        finally:
            if process is not None and process.poll() is None:
                process.stop(timeout=10)
        finished = time.monotonic()
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read().decode(), err.read().decode()
    after = snapshot(project, label + ":after")
    metadata = [json.loads(line.split("=", 1)[1]) for line in stdout.splitlines()
                if line.startswith("CPU_RECOVERY_META=")]
    record = {"label": label, "command": command, "cwd": str(root), "exit_code": code,
              "stdout": stdout, "stderr": stderr, "metadata": metadata,
              "controller_binding": binding, "controller_closure": closure,
              "started_monotonic": started, "finished_monotonic": finished,
              "wall_seconds": finished - started, "before": before["label"], "after": after["label"]}
    project["calls"].append(record)
    save_report(project)
    assert code == expected, (label, code, stdout[-1000:], stderr[-4000:])
    assert closure["event"] == "TREE_CLOSED" and closure["wait_proof"] == "ECHILD_WALL"
    assert closure["worker_returncode"] == code and closure["forced_cleanup"] is False
    assert closure["stop_requested"] is False
    assert len(metadata) == 2 and metadata[0]["event"] == "start", (label, stdout, stderr)
    assert metadata[1]["event"] == ("crash_before_settle" if crash else "finish")
    for item in metadata:
        assert {k: item[k] for k in ("pid", "start_ticks")} == binding["worker"]
    return record


def new_crashed_project(root, *, exit_code=0, slots=1):
    root.mkdir()
    cfg = {"execution": {"version": 1, "resource": "cpu", "slots": slots, "wall_budget_seconds": 6},
           "results_dir": str(root / "results"), "idea_lake_db": str(root / "lake.db"),
           "ideas_file": str(root / "ideas.md"), "min_disk_gb": 0,
           "action_policy": {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": .05}}
    (root / "orze.yaml").write_text(yaml.safe_dump(cfg))
    project = {"root": root, "cfg": cfg, "calls": [], "snapshots": [], "admissions": []}
    admit(project, "idea-first", "first", exit_code=exit_code)
    run_cli(project, "first_controller_crash", crash=True, expected=86)
    return project


@pytest.fixture
def make_crashed_project(tmp_path, request):
    projects = []

    def make(*, exit_code=0, slots=1):
        project = new_crashed_project(tmp_path / ("project-" + str(len(projects))), exit_code=exit_code, slots=slots)
        projects.append(project)
        return project

    yield make
    reports = getattr(request.config, "_cpu_recovery_reports", None)
    if reports is None:
        reports = request.config._cpu_recovery_reports = []
    for project in projects:
        reports.append({**save_report(project), "test": request.node.nodeid})


def pytest_sessionfinish(session, exitstatus):
    reports = getattr(session.config, "_cpu_recovery_reports", [])
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reports and reporter is not None:
        reporter.write_line("CPU_RECOVERY_REPORTS=" + json.dumps(reports, sort_keys=True))


def _child_main():
    import orze.cli as cli
    import orze.engine.orchestrator as engine
    import orze.engine.gpu_slots as slots
    from orze.core import cpu_action_budget as budget
    from orze.core.execution_attempts import require_current
    from orze.engine.attempt_effect_receipts import _scan

    def metadata(event, **extra):
        stat = Path("/proc/self/stat").read_text()
        return {"event": event, "pid": os.getpid(),
                "start_ticks": int(stat[stat.rfind(")") + 2:].split()[19]),
                "monotonic": time.monotonic(), **extra}

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU recovery invoked GPU/provider/legacy repair path")

    def crash_at_settle(lake, permit, ref, terminal):
        assert not lake.conn.in_transaction
        row = require_current(lake.conn, ref, states=("TERMINAL",))
        assert canonical(row["terminal"]) == canonical(terminal)
        assert row["binding"]["reservation_id"] == permit["reservation_id"]
        folder = Path(permit["budget_scope"]["results_dir"]) / ref.task_id
        assert not (folder / "_attempt_effect.lock").exists()
        digest = terminal["effect_receipt_sha256"]
        assert _scan(folder).get(ref.attempt_id) == (digest, True)
        reservation = lake.conn.execute("SELECT state,ref_json,terminal_sha256 FROM cpu_action_reservations WHERE reservation_id=?",
                                         (permit["reservation_id"],)).fetchone()
        assert tuple(reservation) == ("BOUND", canonical(asdict(ref)).decode(), None)
        closure = terminal["process_tree"]
        assert closure["event"] == "TREE_CLOSED" and closure["wait_proof"] == "ECHILD_WALL"
        assert closure["forced_cleanup"] is False and closure["stop_requested"] is False
        record = metadata("crash_before_settle", ref=asdict(ref), terminal=terminal, permit=permit,
                          current_row=row, transaction_open=False, effect_confirmed=True,
                          effect_guard_absent=True, deliberate_exit_code=86)
        print("CPU_RECOVERY_META=" + json.dumps(record, sort_keys=True), flush=True)
        os._exit(86)

    arguments = sys.argv[1:]
    crash = "--crash-before-settle" in arguments
    sys.argv = ["orze", *[a for a in arguments if a != "--crash-before-settle"]]
    print("CPU_RECOVERY_META=" + json.dumps(metadata("start"), sort_keys=True), flush=True)
    with contextlib.ExitStack() as stack:
        for target in ("orze.extensions.has_pro", "orze.extensions._find_pro_key",
                       "orze.engine.failure._try_executor_fix"):
            stack.enter_context(patch(target, forbidden))
        for obj, name in ((cli, "detect_all_gpus"), (slots, "GpuSlotManager"),
                          (engine, "acquire_gpu_leases"), (engine, "startup_canary"),
                          (engine.Orze, "_kill_orphans")):
            stack.enter_context(patch.object(obj, name, forbidden))
        if crash:
            stack.enter_context(patch.object(budget, "settle", crash_at_settle))
        result = cli.main()
    code = 0 if result is None else result
    print("CPU_RECOVERY_META=" + json.dumps(metadata("finish", exit_code=code), sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(_child_main())
