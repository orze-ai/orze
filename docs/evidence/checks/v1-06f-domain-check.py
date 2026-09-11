#!/usr/bin/env python3
"""Offline consistency check of the two V1-06F Domain recovery archives.

Stdlib only; never imports product/tests, opens paths embedded in a report,
queries live processes, writes a database, or grants execution authority.
Closure/event records are compared, not independently attested. The frozen
test's transparent callback guards witness the recovery-only environment;
archives contain the final event list, not per-call environment snapshots.
This is not a scientific re-evaluation of arbitrary scheduling inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

if sys.flags.optimize:
    raise SystemExit("assertions required; do not run this checker with -O")

REF = ("task_id", "phase", "generation", "attempt_id")
LABELS = ("domain-producer", "domain-evaluator-crash", "domain-recover-only",
          "domain-recover-idle-again")
BUSINESS = ("ideas", "idea_state", "idea_transitions", "idea_stage_state",
            "idea_stage_transitions", "execution_attempts", "research_artifacts",
            "research_observations", "cpu_action_scopes")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def equal(actual, expected):
    assert encoded(actual) == encoded(expected), (actual, expected)


def pairs(items):
    result = {}
    for key, value in items:
        assert key not in result, key
        result[key] = value
    return result


def decode(value):
    def invalid(token):
        raise ValueError(token)
    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)


def digest(value, length=64):
    assert type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % length, value)
    return value


def birth(value):
    assert set(value) == {"pid", "start_ticks"}
    assert all(type(v) is int and v > 0 for v in value.values())
    return value["pid"], value["start_ticks"]


def ref_of(row):
    ref = {key: row[key] for key in REF}
    equal(ref["generation"], 1)
    equal(ref["phase"], "action")
    assert type(ref["attempt_id"]) is str and ref["attempt_id"]
    return ref


def closure(value, binding, code):
    equal(value["schema"], 1)
    equal(value["event"], "TREE_CLOSED")
    equal(value["wait_proof"], "ECHILD_WALL")
    equal(value["worker_returncode"], code)
    equal(value["stop_requested"], False)
    equal(value["forced_cleanup"], False)
    assert type(value["reaped_children"]) is int and value["reaped_children"] >= 1
    equal(value["binding"], binding)
    equal(binding["schema"], 1)
    equal(binding["protocol"], "orze.linux_subreaper.v1")
    digest(binding["nonce_sha256"])
    digest(binding["command_sha256"])
    assert birth(binding["worker"]) != birth(binding["supervisor"])


def parsed_row(row):
    return {**{k: v for k, v in row.items() if k not in ("binding_json", "terminal_json")},
            "binding": decode(row["binding_json"]), "terminal": decode(row["terminal_json"])}


def native(shot, row, root):
    ref = ref_of(row)
    equal(row["state"], "TERMINAL")
    equal(row["hold_reason"], None)
    binding, terminal = decode(row["binding_json"]), decode(row["terminal_json"])
    folder = root + "/results/" + ref["task_id"]
    equal(binding["attempt_ref"], ref)
    equal(binding["attempt_id"], ref["attempt_id"])
    equal(binding["scope"], folder)
    equal(binding["origin"], "native_cpu_action")
    equal(binding["kind"], "native_cpu_action")
    equal(binding["resource"], "cpu")
    equal(binding["source"]["database"], root + "/lake.db")
    equal(binding["supervision"]["identity"], {"attempt_ref": ref, "scope": folder})
    equal(binding["process_pid"], binding["supervision"]["worker"]["pid"])
    equal(binding["command_sha256"], binding["supervision"]["command_sha256"])
    equal(binding["lifecycle"], terminal["lifecycle"])
    equal(binding["lifecycle_phase"], "action")
    equal(terminal["lifecycle_phase"], "action")
    equal(terminal["outcome"], "completed")
    equal(terminal["return_code"], 0)
    equal(terminal["reason_code"], "cpu_action_completed")
    for key in ("global_state", "phase_state"):
        equal(terminal["lifecycle"][key], "COMPLETE")
    equal(terminal["lifecycle"]["legacy_status"], "completed")
    closure(terminal["process_tree"], binding["supervision"], 0)
    base = folder + "/_execution_effects/" + ref["attempt_id"]
    prepared_file = shot["files"][base + "/prepared.json"]
    prepared = decode(prepared_file["utf8"])
    committed = decode(shot["files"][base + "/committed.json"]["utf8"])
    identity = {"schema_version": 1, **ref}
    plan = {"operation": "cpu_action_terminal", **{k: v for k, v in terminal.items()
            if k not in ("lifecycle", "effect_receipt_sha256")}}
    equal(prepared, {**identity, "event": "effect_prepared", "plan": plan})
    equal(committed, {**identity, "event": "effect_committed",
                      "prepared_sha256": prepared_file["sha256"]})
    equal(terminal["effect_receipt_sha256"], prepared_file["sha256"])
    return ref, binding, terminal


def one(path):
    raw = path.read_bytes()
    report = decode(raw)
    equal(sorted(report), ["admissions", "calls", "cfg", "domain_events", "root", "snapshots"])
    root, cfg = report["root"], report["cfg"]
    assert type(root) is str and root.startswith("/")
    scope = root + "/results"
    equal(cfg["results_dir"], scope)
    equal(cfg["idea_lake_db"], root + "/lake.db")
    equal(cfg["execution"], {"version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 6})
    calls, shots = report["calls"], report["snapshots"]
    equal([c["label"] for c in calls], list(LABELS))
    by_label = {s["label"]: s for s in shots}
    equal(len(by_label), len(shots))
    equal(len(shots), 12)
    for index, shot in enumerate(shots):
        assert shot["label"].startswith(str(index) + ":")
        equal(shot["worker_events"], [])  # Domain native events have a separate log.
        for entry in shot["files"].values():
            content = entry["utf8"].encode()
            equal(entry["bytes"], len(content))
            equal(entry["sha256"], sha(content))
    births, nonces = [], []
    previous_end = 0
    for index, call in enumerate(calls):
        code = 86 if index == 1 else 0
        equal(call["exit_code"], code)
        equal(call["cwd"], root)
        expected_args = ["-m", "test_cpu_terminal_recovery_domain"]
        if index == 1:
            expected_args.append("--crash-before-settle")
        expected_args.extend(["-c", root + "/orze.yaml", "--once"])
        equal(call["command"][1:], expected_args)
        assert call["before"] in by_label and call["after"] in by_label
        equal(call["controller_binding"]["identity"],
              {"scope": root, "terminal_recovery_controller": LABELS[index]})
        closure(call["controller_closure"], call["controller_binding"], code)
        births.append(birth(call["controller_binding"]["worker"]))
        nonces.append(call["controller_binding"]["nonce_sha256"])
        assert previous_end <= call["started_monotonic"] < call["finished_monotonic"]
        equal(call["wall_seconds"], call["finished_monotonic"] - call["started_monotonic"])
        previous_end = call["finished_monotonic"]
        markers = [decode(line.split("=", 1)[1]) for line in call["stdout"].splitlines()
                   if line.startswith("CPU_RECOVERY_META=")]
        equal(markers, call["metadata"])
        equal([m["event"] for m in markers], ["start", "crash_before_settle" if index == 1 else "finish"])
        for marker in markers:
            equal({k: marker[k] for k in ("pid", "start_ticks")}, call["controller_binding"]["worker"])
            assert call["started_monotonic"] <= marker["monotonic"] <= call["finished_monotonic"]
        if index != 1:
            equal(markers[1]["exit_code"], 0)
    equal(len(set(births)), 4)
    equal(len(set(nonces)), 4)
    produced = by_label[calls[0]["after"]]
    before = by_label[calls[1]["after"]]
    after = by_label[calls[2]["after"]]
    idle = by_label[calls[3]["after"]]
    db = before["database"]
    equal(len(produced["database"]["execution_attempts"]), 1)
    equal(len(produced["database"]["research_artifacts"]), 1)
    equal(produced["database"]["research_observations"], [])
    equal(len(db["execution_attempts"]), 2)
    equal(len(db["research_artifacts"]), 2)
    equal(len(db["research_observations"]), 1)
    rows = {r["task_id"]: r for r in db["execution_attempts"]}
    equal(sorted(rows), ["idea-evaluator", "idea-producer"])
    native_data = {task: native(before, row, root) for task, row in rows.items()}
    producer_ref, producer_binding, producer_terminal = native_data["idea-producer"]
    evaluator_ref, evaluator_binding, evaluator_terminal = native_data["idea-evaluator"]
    equal(produced["database"]["execution_attempts"], [rows["idea-producer"]])
    events = report["domain_events"]
    equal([e["event"] for e in events], ["native_ready", "interpret", "native_ready", "interpret"])
    equal(events[0], {"event": "native_ready", "binding": producer_binding["supervision"]})
    equal(events[1], {"event": "interpret", "operation": "produce"})
    equal(events[2], {"event": "native_ready", "binding": evaluator_binding["supervision"]})
    equal(events[3], {"event": "interpret", "operation": "evaluate"})
    native_births = [birth(events[i]["binding"]["worker"]) for i in (0, 2)]
    equal(len(set(births + native_births)), 6)
    artifacts = {}
    for row in db["research_artifacts"]:
        record = decode(row["record_json"])
        task = record["producer"]["task_id"]
        ref, binding, terminal = native_data[task]
        equal(record["producer"], ref)
        equal({k: row["producer_" + k] for k in REF}, ref)
        equal(record["schema"], 1)
        equal(record["artifact_id"], row["artifact_id"])
        equal(record["logical_name"], row["logical_name"])
        equal(record["scope"], scope)
        publication = binding["artifact_publication"]
        equal(publication["scope"], scope)
        equal(record["spec_fingerprint"], binding["action_sha256"])
        equal(publication["spec_fingerprint"], record["spec_fingerprint"])
        equal(record["path"], publication["root"] + "/" + record["artifact_id"] + "/content")
        content = before["files"][record["path"]]
        equal(record["content_sha256"], content["sha256"])
        equal(record["size_bytes"], content["bytes"])
        assert record["size_bytes"] <= publication["contract"]["outputs"][record["logical_name"]]["max_bytes"]
        equal(terminal["artifact_ids"], [record["artifact_id"]])
        assert task not in artifacts
        artifacts[task] = record
    source, result = artifacts["idea-producer"], artifacts["idea-evaluator"]
    source_snapshot = evaluator_binding["domain_run"]["source_snapshot"]
    budget_scope = decode(db["cpu_action_scopes"][0]["binding_json"])
    equal(source_snapshot, {"schema": 1, "scope": scope, "database": root + "/lake.db",
          "database_identity": budget_scope["database_identity"],
          "scope_identity": budget_scope["directory_identity"],
          "inputs": [{"artifact": source, "source_sha256": sha(encoded(parsed_row(rows["idea-producer"]))),
                      "effect_sha256": producer_terminal["effect_receipt_sha256"]}]})
    equal(producer_binding["domain_run"]["source_snapshot"], {**source_snapshot, "inputs": []})
    equal([a["task_id"] for a in report["admissions"]], ["idea-producer", "idea-evaluator"])
    for admission in report["admissions"]:
        equal(admission["result"]["status"], "inserted")
        equal(admission["result"]["idea_id"], admission["task_id"])
        assert admission["before"] in by_label and admission["after"] in by_label
        config = decode(admission["config"])
        equal(config["kind"], "native_cpu_action")
        binding = native_data[admission["task_id"]][1]
        equal(binding["source"]["config_sha256"], sha(encoded(config)))
        equal(binding["domain_run"]["request_sha256"], sha(encoded(config)))
    request = decode(report["admissions"][1]["config"])["domain_request"]
    protocol = request["payload"]["protocol"]
    equal(request["input_artifact_ids"], [source["artifact_id"]])
    assert protocol in ("schedule-feasibility-v1", "schedule-feasibility-v2")
    observation_row = db["research_observations"][0]
    obs = decode(observation_row["record_json"])
    equal(obs["schema"], 2)
    equal(obs["evaluator"], evaluator_ref)
    equal({k: observation_row["evaluator_" + k] for k in REF}, evaluator_ref)
    equal(obs["observation_id"], observation_row["observation_id"])
    equal(obs["name"], observation_row["name"])
    equal(obs["name"], "scheduled_value")
    equal(obs["result_artifact_ids"], [result["artifact_id"]])
    equal(obs["input_artifact_ids"], [source["artifact_id"]])
    provenance = {source["artifact_id"]: {k: source[k] for k in ("producer", "spec_fingerprint", "content_sha256")}}
    equal(obs["input_artifact_bindings"], provenance)
    publication = evaluator_binding["observation_publication"]
    equal(publication["version"], 2)
    for key in ("scope", "adapter_id", "protocol_fingerprint", "spec_fingerprint", "input_artifact_ids", "input_artifact_bindings"):
        equal(obs[key], publication[key])
    equal(obs["scope"], scope)
    equal(evaluator_binding["domain_run"]["observation"],
          {**{k: obs[k] for k in ("adapter_id", "protocol_fingerprint", "spec_fingerprint")}, "result_output": "evaluation"})
    equal(producer_terminal["observation_ids"], [])
    equal(evaluator_terminal["observation_ids"], [obs["observation_id"]])
    envelope = decode(before["files"][result["path"]]["utf8"])
    equal(envelope["version"], 1)
    equal(envelope["protocol"], protocol)
    equal(envelope["source_artifact_ids"], [source["artifact_id"]])
    equal(envelope["instance_sha256"], sha(encoded(cfg["action_domain"]["config"]["instance"])))
    equal(obs["values"]["instance_sha256"], envelope["instance_sha256"])
    equal(obs["values"]["direction"], "maximize")
    expected = {"status": "valid", "reason_code": "feasible"} if protocol.endswith("v1") else {
        "status": "invalid", "reason_code": "capacity_overload"}
    equal(obs["validation"], expected)
    equal({k: envelope["verdict"][k] for k in expected}, expected)
    if expected["status"] == "valid":
        equal(obs["values"]["value"], envelope["verdict"]["scheduled_value"])
        equal(obs["values"]["value"], 30)
    else:
        assert "value" not in obs["values"] and "scheduled_value" not in envelope["verdict"]
    crash = calls[1]["metadata"][1]
    for key in ("effect_confirmed", "effect_guard_absent"):
        equal(crash[key], True)
    equal(crash["transaction_open"], False)
    equal(crash["deliberate_exit_code"], 86)
    equal(crash["ref"], evaluator_ref)
    equal(crash["terminal"], evaluator_terminal)
    equal(crash["current_row"], parsed_row(rows["idea-evaluator"]))
    old_reservations = {r["task_id"]: r for r in db["cpu_action_reservations"]}
    equal(sorted(old_reservations), sorted(rows))
    equal({k: r["state"] for k, r in old_reservations.items()}, {"idea-producer": "SETTLED", "idea-evaluator": "BOUND"})
    equal(old_reservations["idea-evaluator"]["terminal_sha256"], None)
    equal(crash["permit"], decode(old_reservations["idea-evaluator"]["permit_json"]))
    equal(crash["permit"]["budget_scope"], budget_scope)
    equal(budget_scope["schema"], 1)
    equal(budget_scope["results_dir"], scope)
    equal(budget_scope["database"], root + "/lake.db")
    equal(budget_scope["declaration"], cfg["execution"])
    digest(budget_scope["policy_sha256"])
    for key in ("database_identity", "directory_identity"):
        assert len(budget_scope[key]) == 2 and all(type(n) is int and n > 0 for n in budget_scope[key])
    equal(db["cpu_action_scopes"], [{"binding_json": encoded(budget_scope).decode(), "scope": scope, "stop_json": None}])
    for shot in (after, idle):
        for table in BUSINESS:
            equal(shot["database"][table], db[table])
        equal(shot["files"], before["files"])
        new_reservations = {r["task_id"]: r for r in shot["database"]["cpu_action_reservations"]}
        equal(sorted(new_reservations), sorted(old_reservations))
        for task, old in old_reservations.items():
            ref, binding, terminal = native_data[task]
            new = new_reservations[task]
            equal(new, {**old, "state": "SETTLED", "terminal_sha256": sha(encoded(terminal))})
            equal(decode(new["ref_json"]), ref)
            permit = decode(new["permit_json"])
            equal(permit["reservation_id"], new["reservation_id"])
            equal(permit["reservation_id"], binding["reservation_id"])
            equal(permit["budget_scope"], budget_scope)
            equal(permit["task_id"], task)
            equal(permit["reserved_nanoseconds"], "2000000000")
            equal(permit["wall_limit_seconds"], 2)
            equal(new["scope"], scope)
        recovery = shot["database"]["cpu_action_recovery"]
        equal(len(recovery), 1)
        equal(sorted(recovery[0]), ["binding_json", "nonce", "scope", "state", "summary_json"])
        equal(recovery[0]["scope"], scope)
        equal(decode(recovery[0]["binding_json"]), budget_scope)
        digest(recovery[0]["nonce"], 48)
        equal(recovery[0]["state"], "COMPLETE")
        equal(decode(recovery[0]["summary_json"]), {"schema": 1, "examined": 1,
              "settled": [old_reservations["idea-evaluator"]["reservation_id"]],
              "already_settled": [], "retained": []})
    equal(after["database"]["cpu_action_recovery"], idle["database"]["cpu_action_recovery"])
    equal(after["database"]["cpu_action_reservations"], idle["database"]["cpu_action_reservations"])
    for index, shot in enumerate((after, idle), 1):
        decisions = shot["database"]["cpu_action_decisions"]
        equal(len(decisions), index)
        for decision in decisions:
            equal(decision["scope"], scope)
            value = decode(decision["record_json"])
            equal(value["decision_id"], decision["decision_id"])
            equal(value["kind"], "Wait")
            equal(value["reason"], "queue_empty")
    return {"path": str(path), "sha256": sha(raw), "bytes": len(raw), "root": root,
            "protocol": protocol, "validation": expected, "fresh_cli_invocations": 4,
            "controller_births": births, "native_actions": 2, "native_births": native_births,
            "attempt_refs": [producer_ref, evaluator_ref], "artifacts": 2, "observations": 1,
            "charged_nanoseconds": 4000000000, "recovery_only_cli_invocations": 2,
            "recovery_nonce": after["database"]["cpu_action_recovery"][0]["nonce"],
            "original_terminal_and_bytes_unchanged": True,
            "cli_wall_seconds": sum(c["wall_seconds"] for c in calls)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs=2, type=Path)
    args = parser.parse_args()
    results = [one(path) for path in args.reports]
    equal(sorted(r["protocol"] for r in results), ["schedule-feasibility-v1", "schedule-feasibility-v2"])
    equal(len({r["root"] for r in results}), 2)
    equal(len({tuple(b) for r in results for b in r["controller_births"] + r["native_births"]}), 12)
    equal(len({r["recovery_nonce"] for r in results}), 2)
    print(json.dumps({"schema": 1, "status": "passed", "scope": "offline_record_consistency_only",
                      "reports": results, "native_actions": 4, "fresh_cli_invocations": 8,
                      "charged_nanoseconds": 8000000000}, sort_keys=True))


if __name__ == "__main__":
    main()
