"""Isolated C3 real restart diagnostics, deliberately outside pytest collection.

Uses the unchanged recovery helper's actual CLI/Orze/supervisor runner. Only
private child callbacks crash after real effect work. This is instrumentation,
not a replacement runtime, closure, database, or recovery implementation.
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
import traceback
from unittest.mock import patch

import yaml
import cpu_terminal_recovery_helpers as helpers


MODULE = "docs.evidence.snapshots.c3_terminal_restart_diagnostic"


def _guards(project):
    result = {}
    for name in ("results/idea-first/_attempt_effect.lock",
                 "results/_cpu_terminal_recovery.lock"):
        directory = project["root"] / name
        contents = {}
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    raw = path.read_bytes()
                    contents[str(path.relative_to(project["root"]))] = {
                        "sha256": hashlib.sha256(raw).hexdigest(), "utf8": raw.decode()}
        result[name] = {"exists": directory.exists(), "files": contents}
    return result


def _snapshot(project, label):
    result = helpers.snapshot(project, label)
    result["guards"] = _guards(project)
    helpers.save_report(project)
    return result


def _metadata(event, **extra):
    raw = Path("/proc/self/stat").read_text()
    return {"event": event, "pid": os.getpid(),
            "start_ticks": int(raw[raw.rfind(")") + 2:].split()[19]),
            "monotonic": time.monotonic(), **extra}


def _child():
    from orze.core import cpu_action_budget as budget
    from orze.core.execution_attempts import require_current
    from orze.engine import attempt_effect_receipts as effects
    args = sys.argv[1:]
    root = Path(args[args.index("-c") + 1]).parent
    mode = json.loads((root / "diagnostic.json").read_text())["mode"]
    crash = "--crash-before-settle" in args
    # The unchanged helper's built-in hook assumes non-stopped closure. This
    # diagnostic installs its own narrow hook and reuses all original CLI
    # safety tripwires, start/finish metadata, and actual product lifecycle.
    sys.argv = [sys.argv[0], *[arg for arg in args if arg != "--crash-before-settle"]]

    def crash_at_settle(lake, permit, ref, terminal):
        assert not lake.conn.in_transaction
        row = require_current(lake.conn, ref, states=("TERMINAL",))
        assert helpers.canonical(row["terminal"]) == helpers.canonical(terminal)
        folder = root / "results" / ref.task_id
        assert not (folder / "_attempt_effect.lock").exists()
        assert effects._scan(folder).get(ref.attempt_id) == (terminal["effect_receipt_sha256"], True)
        actual = lake.conn.execute("SELECT state,ref_json,terminal_sha256 FROM main.cpu_action_reservations "
                                  "WHERE reservation_id=?", (permit["reservation_id"],)).fetchone()
        assert tuple(actual) == ("BOUND", helpers.canonical(asdict(ref)).decode(), None)
        closure = terminal["process_tree"]
        assert terminal["outcome"] == "interrupted" and terminal["return_code"] == 0
        assert terminal["reason_code"] == "cpu_runtime_lease_expired"
        assert terminal["runtime_lease"]["status"] == "expired"
        assert closure["event"] == "TREE_CLOSED" and closure["wait_proof"] == "ECHILD_WALL"
        assert closure["lease_expired"] is True and closure["stop_requested"] is True
        assert terminal["artifact_ids"] == terminal["observation_ids"] == []
        print("CPU_RECOVERY_META=" + json.dumps(_metadata(
            "crash_before_settle", boundary="confirmed_expired_terminal_before_settle",
            ref=asdict(ref), current_row=row, permit=permit, terminal=terminal,
            effect_confirmed=True, effect_guard_absent=True, deliberate_exit_code=86),
            sort_keys=True), flush=True)
        os._exit(86)

    original_confirm = effects.confirm_effect

    def crash_after_real_confirm(lease, ref, digest):
        result = original_confirm(lease, ref, digest)
        folder = root / "results" / ref.task_id
        assert lease.idea_dir == folder
        assert (folder / "_attempt_effect.lock").is_dir()
        assert effects._scan(folder).get(ref.attempt_id) == (digest, True)
        conn = sqlite3.connect((root / "lake.db").as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            row = require_current(conn, ref, states=("TERMINAL",))
            reservation = dict(conn.execute("SELECT * FROM main.cpu_action_reservations").fetchone())
        finally:
            conn.close()
        assert reservation["state"] == "BOUND" and reservation["terminal_sha256"] is None
        terminal = row["terminal"]
        assert terminal["outcome"] == "completed" and terminal["runtime_lease"]["status"] == "authorized"
        assert terminal["effect_receipt_sha256"] == digest
        observed = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        assert observed < row["binding"]["runtime_lease"]["deadline_ns"]
        print("CPU_RECOVERY_META=" + json.dumps(_metadata(
            "crash_before_settle", boundary="real_confirm_written_before_coordinator_postconfirm_gate",
            ref=asdict(ref), current_row=row, terminal=terminal,
            reservation=reservation, observed_ns=observed, effect_confirmed=True,
            effect_guard_absent=False, deliberate_exit_code=86), sort_keys=True), flush=True)
        os._exit(86)
        return result  # unreachable; documents that the real callback never returned

    with contextlib.ExitStack() as stack:
        if crash and mode == "expired":
            stack.enter_context(patch.object(budget, "settle", crash_at_settle))
        elif crash and mode == "confirm":
            stack.enter_context(patch.object(effects, "confirm_effect", crash_after_real_confirm))
        return helpers._child_main()


def _project(base, mode):
    root = base / mode
    root.mkdir()
    cfg = {"execution": {"version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 6},
           "results_dir": str(root / "results"), "idea_lake_db": str(root / "lake.db"),
           "ideas_file": str(root / "ideas.md"), "min_disk_gb": 0,
           "action_policy": {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": .05}}
    if mode == "expired":
        cfg["cpu_runtime_lease"] = {"version": 1, "ttl_seconds": 1}
    (root / "orze.yaml").write_text(yaml.safe_dump(cfg))
    (root / "diagnostic.json").write_text(json.dumps({"mode": mode}))
    project = {"root": root, "cfg": cfg, "calls": [], "snapshots": [], "admissions": []}
    body = helpers.WORKER
    if mode == "expired":
        body = body.replace("raise SystemExit(data['exit_code'])", """import signal,time
signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
while True: time.sleep(.01)
""")
    # Public IdeaLake insertion and action shape are unchanged; only this
    # diagnostic's real CPU worker body differs, with the actual audit write.
    with patch.object(helpers, "WORKER", body):
        helpers.admit(project, "idea-first", "first")
    return project


def _no_new_work(before, after):
    for name in ("execution_attempts", "research_artifacts", "research_observations",
                 "ideas", "idea_state", "idea_transitions", "idea_stage_state",
                 "idea_stage_transitions"):
        assert after["database"][name] == before["database"][name], name
    assert after["files"] == before["files"]
    assert after["worker_events"] == before["worker_events"]
    assert len(after["worker_events"]) == 1


def _run_case(project, mode):
    helpers.run_cli(project, "first_crash", crash=True, expected=86, child_module=MODULE)
    crashed = _snapshot(project, "crash_with_guards")
    assert len(crashed["database"]["execution_attempts"]) == 1
    original = crashed["database"]["execution_attempts"][0]
    terminal = json.loads(original["terminal_json"])
    reservation = crashed["database"]["cpu_action_reservations"][0]
    assert original["state"] == "TERMINAL" and reservation["state"] == "BOUND"
    permit = json.loads(reservation["permit_json"])
    assert permit["reserved_nanoseconds"] == "2000000000"
    expected_code = 0 if mode == "expired" else 75
    if mode == "expired":
        assert terminal["runtime_lease"]["status"] == "expired"
        assert terminal["outcome"] == "interrupted" and terminal["return_code"] == 0
        assert not crashed["guards"]["results/idea-first/_attempt_effect.lock"]["exists"]
    else:
        assert crashed["guards"]["results/idea-first/_attempt_effect.lock"]["exists"]
        deadline = json.loads(original["binding_json"])["runtime_lease"]["deadline_ns"]
        # The controller has actually exited. Wait past its bound deadline,
        # so the next refusal cannot be explained by a still-current lease.
        remaining = (deadline - time.clock_gettime_ns(time.CLOCK_BOOTTIME)) / 1e9
        if remaining > 0:
            time.sleep(remaining + .02)
    helpers.run_cli(project, "fresh_restart", expected=expected_code, child_module=MODULE)
    recovered = _snapshot(project, "restart_with_guards")
    _no_new_work(crashed, recovered)
    new = recovered["database"]["cpu_action_reservations"][0]
    assert json.loads(new["permit_json"])["reserved_nanoseconds"] == permit["reserved_nanoseconds"]
    assert new["state"] == ("SETTLED" if mode == "expired" else "BOUND")
    if mode == "expired":
        assert new["terminal_sha256"] == hashlib.sha256(helpers.canonical(terminal)).hexdigest()
    else:
        assert new == reservation
        assert recovered["guards"]["results/idea-first/_attempt_effect.lock"] == crashed["guards"]["results/idea-first/_attempt_effect.lock"]
    helpers.run_cli(project, "second_fresh_restart", expected=expected_code, child_module=MODULE)
    repeated = _snapshot(project, "second_restart_with_guards")
    _no_new_work(recovered, repeated)
    assert repeated["database"]["cpu_action_reservations"] == recovered["database"]["cpu_action_reservations"]
    assert repeated["guards"]["results/idea-first/_attempt_effect.lock"] == recovered["guards"]["results/idea-first/_attempt_effect.lock"]
    births = [tuple(call["controller_binding"]["worker"][k] for k in ("pid", "start_ticks"))
              for call in project["calls"]]
    assert len(set(births)) == 3


def main():
    base = Path(tempfile.mkdtemp(prefix="orze-c3-terminal-restart-"))
    results = []
    for mode in ("expired", "confirm"):
        project = None
        try:
            project = _project(base, mode)
            _run_case(project, mode)
            outcome = {"mode": mode, "passed": True}
        except BaseException:
            outcome = {"mode": mode, "passed": False, "traceback": traceback.format_exc()}
        finally:
            if project is not None:
                _snapshot(project, "final_diagnostic_state")
                outcome["report"] = helpers.save_report(project)
        results.append(outcome)
    print("C3_TERMINAL_RESTART_DIAGNOSTIC=" + json.dumps({"root": str(base), "cases": results}, sort_keys=True), flush=True)
    return int(any(not item["passed"] for item in results))


if __name__ == "__main__":
    raise SystemExit(_child() if "-c" in sys.argv else main())
