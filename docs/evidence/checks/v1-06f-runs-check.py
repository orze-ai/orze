#!/usr/bin/env python3
"""Offline consistency checks for captured real V1-06F product invocations.

No execution authority is obtained; paths in reports are never opened.
The invocation logs and frozen tests, not this checker alone, witness execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

if not __debug__:
    raise SystemExit("assertions required; do not run this evidence checker with -O")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def equal(left, right):
    assert encoded(left) == encoded(right), (left, right)


def pairs(items):
    out = {}
    for key, value in items:
        assert key not in out, key
        out[key] = value
    return out


def decode(raw):
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def birth(value):
    assert set(value) == {"pid", "start_ticks"}
    assert all(type(v) is int and v > 0 for v in value.values())
    return value["pid"], value["start_ticks"]


def closure(value, binding, code):
    assert value["event"] == "TREE_CLOSED" and value["wait_proof"] == "ECHILD_WALL"
    equal(value["schema"], 1)
    equal(value["worker_returncode"], code)
    equal(value["forced_cleanup"], False)
    equal(value["stop_requested"], False)
    assert type(value["reaped_children"]) is int and value["reaped_children"] >= 1
    equal(value["binding"], binding)
    birth(binding["worker"])
    birth(binding["supervisor"])
    assert binding["protocol"] == "orze.linux_subreaper.v1"


def one(report, mode):
    equal(sorted(report), ["admissions", "calls", "cfg", "root", "snapshots"])
    calls, snapshots = report["calls"], report["snapshots"]
    equal([c["label"] for c in calls],
          ["first_controller_crash", "restart"] + (["second_restart_idle"] if mode == "green" else []))
    by_label = {s["label"]: s for s in snapshots}
    assert len(by_label) == len(snapshots)
    for index, shot in enumerate(snapshots):
        assert shot["label"].startswith(str(index) + ":")
        for entry in shot["files"].values():
            raw = entry["utf8"].encode()
            equal(len(raw), entry["bytes"])
            assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    controllers = []
    previous_end = 0
    for i, call in enumerate(calls):
        equal(call["exit_code"], 86 if i == 0 else 0)
        assert call["cwd"] == report["root"]
        assert call["before"] in by_label and call["after"] in by_label
        assert "--once" in call["command"]
        assert ("--crash-before-settle" in call["command"]) == (i == 0)
        assert call["started_monotonic"] >= previous_end
        assert call["finished_monotonic"] > call["started_monotonic"]
        assert abs(call["wall_seconds"] - (call["finished_monotonic"] - call["started_monotonic"])) < 1e-9
        previous_end = call["finished_monotonic"]
        closure(call["controller_closure"], call["controller_binding"], call["exit_code"])
        controllers.append(birth(call["controller_binding"]["worker"]))
        markers = [decode(line.split("=", 1)[1]) for line in call["stdout"].splitlines()
                   if line.startswith("CPU_RECOVERY_META=")]
        equal(markers, call["metadata"])
        equal(len(markers), 2)
        equal([m["event"] for m in markers],
              ["start", "crash_before_settle" if i == 0 else "finish"])
        for marker in markers:
            equal({k: marker[k] for k in ("pid", "start_ticks")}, call["controller_binding"]["worker"])
            assert call["started_monotonic"] <= marker["monotonic"] <= call["finished_monotonic"]
    assert len(set(controllers)) == len(controllers)
    crash = calls[0]["metadata"][1]
    for name in ("effect_confirmed", "effect_guard_absent"):
        equal(crash[name], True)
    equal(crash["transaction_open"], False)
    equal(crash["deliberate_exit_code"], 86)
    before = by_label[calls[0]["after"]]
    db = before["database"]
    equal(len(db["execution_attempts"]), 1)
    equal(len(db["cpu_action_reservations"]), 1)
    original = db["execution_attempts"][0]
    terminal, binding = decode(original["terminal_json"]), decode(original["binding_json"])
    ref = {k: original[k] for k in ("task_id", "phase", "generation", "attempt_id")}
    equal(crash["ref"], ref)
    equal(crash["terminal"], terminal)
    equal(binding["attempt_ref"], ref)
    equal(crash["current_row"]["binding"], binding)
    equal(original["state"], "TERMINAL")
    equal(original["hold_reason"], None)
    equal(terminal["return_code"], 0 if terminal["outcome"] == "completed" else 7)
    first_code = terminal["return_code"]
    closure(terminal["process_tree"], binding["supervision"], first_code)
    equal(len(before["worker_events"]), 1)
    equal({k: before["worker_events"][0][k] for k in ("pid", "start_ticks")},
          binding["supervision"]["worker"])
    equal(before["worker_events"][0]["tag"], "first")
    reservation = db["cpu_action_reservations"][0]
    equal(reservation["state"], "BOUND")
    equal(reservation["terminal_sha256"], None)
    equal(decode(reservation["ref_json"]), ref)
    equal(decode(reservation["permit_json"]), crash["permit"])
    equal(binding["reservation_id"], reservation["reservation_id"])
    folder = report["root"] + "/results/idea-first/_execution_effects/" + ref["attempt_id"]
    prepared_entry = before["files"][folder + "/prepared.json"]
    prepared, committed = decode(prepared_entry["utf8"]), decode(before["files"][folder + "/committed.json"]["utf8"])
    equal(terminal["effect_receipt_sha256"], prepared_entry["sha256"])
    equal(committed["prepared_sha256"], prepared_entry["sha256"])
    equal(prepared["plan"], {"operation": "cpu_action_terminal",
                            **{k: v for k, v in terminal.items()
                               if k not in ("lifecycle", "effect_receipt_sha256")}})
    equal(len(db["research_artifacts"]), int(first_code == 0))
    equal(db["research_observations"], [])
    equal([a["task_id"] for a in report["admissions"]], ["idea-first", "idea-second"])
    for admission in report["admissions"]:
        equal(admission["result"]["status"], "inserted")
    after = by_label[calls[1]["after"]]
    final = after["database"]
    equal([r for r in final["execution_attempts"] if r["task_id"] == "idea-first"], [original])
    equal({path: after["files"][path] for path in before["files"]}, before["files"])
    equal(after["worker_events"][0], before["worker_events"][0])
    statuses = {r["idea_id"]: r["status"] for r in final["ideas"]}
    equal(statuses["idea-first"], "completed" if first_code == 0 else "failed")
    if mode == "baseline":
        equal(statuses["idea-second"], "queued")
        equal(final["cpu_action_reservations"], db["cpu_action_reservations"])
        equal(final["execution_attempts"], db["execution_attempts"])
        equal(after["worker_events"], before["worker_events"])
    else:
        equal(statuses["idea-second"], "completed")
        equal(len(final["execution_attempts"]), 2)
        equal(len(final["cpu_action_reservations"]), 2)
        assert all(r["state"] == "SETTLED" for r in final["cpu_action_reservations"])
        for row in final["cpu_action_reservations"]:
            attempt = next(a for a in final["execution_attempts"]
                           if decode(row["ref_json"]) == {k: a[k] for k in ref})
            assert row["terminal_sha256"] == hashlib.sha256(encoded(decode(attempt["terminal_json"]))).hexdigest()
        equal([e["tag"] for e in after["worker_events"]], ["first", "second"])
        equal(len({birth({k: event[k] for k in ("pid", "start_ticks")})
                   for event in after["worker_events"]}), 2)
        equal(sum(int(decode(r["permit_json"])["reserved_nanoseconds"])
                  for r in final["cpu_action_reservations"]), 4000000000)
        recovery = final["cpu_action_recovery"]
        equal(len(recovery), 1)
        equal(recovery[0]["state"], "COMPLETE")
        summary = decode(recovery[0]["summary_json"])
        equal(summary["settled"], [reservation["reservation_id"]])
        equal(summary["already_settled"], [])
        equal(summary["examined"], 1)
        idle = by_label[calls[2]["after"]]
        for name in ("execution_attempts", "cpu_action_reservations", "research_artifacts",
                     "research_observations", "ideas", "idea_state", "idea_transitions",
                     "idea_stage_state", "idea_stage_transitions", "cpu_action_scopes"):
            equal(idle["database"][name], final[name])
        equal(idle["files"], after["files"])
        equal(idle["worker_events"], after["worker_events"])
        equal(idle["database"]["cpu_action_recovery"][0]["state"], "COMPLETE")
    return {"root": report["root"], "worker_exit_code": first_code,
            "fresh_cli_invocations": len(calls), "native_actions": len(after["worker_events"]),
            "wall_seconds": sum(c["wall_seconds"] for c in calls),
            "controller_births": controllers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "green"), required=True)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    results = [one(decode(path.read_bytes()), args.mode) for path in args.reports]
    assert len({r["root"] for r in results}) == len(results)
    print(json.dumps({"schema": 1, "mode": args.mode, "reports": results}, sort_keys=True))


if __name__ == "__main__":
    main()
