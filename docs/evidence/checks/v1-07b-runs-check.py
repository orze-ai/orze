#!/usr/bin/env python3
"""Check six archived holdout runs without importing Core, Domain, or tests.

Only the adjacent independent stdlib domain verifier supplies scientific
verdicts. Captured process/effect metadata is checked for consistency, not
promoted into fresh kernel, filesystem, restart or execution authority.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import sys


if sys.flags.optimize:
    raise SystemExit("v1-07b-runs-check requires assertions; optimized Python is refused")

NAMES = {"workflow", "boundary", "duplicate_id", "missing_prerequisite", "overload", "malformed"}
TABLES = {"ideas", "idea_state", "idea_transitions", "idea_stage_state", "idea_stage_transitions",
          "execution_attempts", "research_artifacts", "research_observations",
          "cpu_action_reservations", "cpu_action_decisions", "cpu_action_scopes",
          "cpu_proposal_requests", "replication_requests"}
REF_FIELDS = ("task_id", "phase", "attempt_id", "generation")
TASKS = {"baseline_producer": "idea-baseline-producer", "baseline_v1": "idea-baseline-v1",
         "challenger_producer": "idea-challenger-producer", "failed_v1": "idea-challenger-v1-failed",
         "recovered_v1": "idea-challenger-v1-recovered", "challenger_v2": "idea-challenger-v2"}
INSTANCE_SHA256 = "7105921fabefdfce632b0ef1c0afb7a04521d4d0a662e602f6825e29cca5e50b"
AUTHORITY_TABLES = TABLES - {"cpu_action_decisions"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def same(left, right):
    assert canonical(left) == canonical(right)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            assert key not in result, "duplicate JSON key"
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def sha(value):
    assert type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)


def seconds(value):
    assert type(value) in (int, float) and math.isfinite(value) and value >= 0
    return value


def one(rows):
    assert len(rows) == 1
    return rows[0]


def index(rows, key, count=None):
    assert type(rows) is list and len(rows) <= 1000
    result = {row[key]: row for row in rows}
    assert len(result) == len(rows), "duplicate identity: " + key
    if count is not None:
        assert len(result) == count
    return result


def reference(value):
    assert type(value) is dict and set(value) == set(REF_FIELDS)
    assert value["phase"] == "action" and type(value["generation"]) is int and value["generation"] > 0
    for key in REF_FIELDS[:3]:
        assert type(value[key]) is str and value[key]
    return value


def decode_attempt(row):
    value = {key: row[key] for key in (*REF_FIELDS, "state", "hold_reason")}
    value.update(binding=strict_json(row["binding_json"]), terminal=strict_json(row["terminal_json"]))
    return value


def tree(binding, closure, identity, code):
    assert type(binding["schema"]) is int and binding["schema"] == 1
    assert binding["protocol"] == "orze.linux_subreaper.v1"
    for key in ("command_sha256", "nonce_sha256"):
        sha(binding[key])
    for role in ("worker", "supervisor"):
        assert set(binding[role]) == {"pid", "start_ticks"}
        assert all(type(v) is int and v > 0 for v in binding[role].values())
    same(binding["identity"], identity)
    same(closure["binding"], binding)
    assert type(closure["schema"]) is int and closure["schema"] == 1
    assert closure["event"] == "TREE_CLOSED" and closure["wait_proof"] == "ECHILD_WALL"
    assert type(closure["worker_returncode"]) is int and closure["worker_returncode"] == code
    assert closure["stop_requested"] is False and closure["forced_cleanup"] is False
    assert type(closure["reaped_children"]) is int and closure["reaped_children"] >= 1


def cli_result(call):
    results = []
    for line in call["stdout"].splitlines():
        try:
            value = strict_json(line)
        except (ValueError, AssertionError):
            continue
        if type(value) is dict and set(value) == {"status", "request_id", "task_id"}:
            results.append(value)
    return one(results)


def check_run(path, verdict_checker):
    raw = path.read_bytes()
    assert len(raw) <= 16 * 1024 * 1024
    run = strict_json(raw)
    name = run.get("condition", "workflow")
    assert name in NAMES
    count = 7 if name == "workflow" else 2
    root = Path(run["root"])
    assert root.is_absolute()
    scope, cfg, db = str(root / "results"), run["cfg"], run["database"]
    assert set(db) == TABLES
    same(cfg["execution"], {"version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 40})
    same(cfg["action_policy"], {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": .05, "config": {}})
    assert cfg["results_dir"] == scope and cfg["idea_lake_db"] == str(root / "lake.db")
    assert cfg["ideas_file"] == str(root / "ideas.md")
    assert cfg["action_domain"]["kind"] == "schedule_holdout"
    instance = cfg["action_domain"]["config"]["instance"]
    assert digest(instance) == INSTANCE_SHA256
    assert db["cpu_proposal_requests"] == []
    ideas = index(db["ideas"], "idea_id", count)
    attempts = index(db["execution_attempts"], "task_id", count)
    reservations = index(db["cpu_action_reservations"], "task_id", count)
    artifacts = index(run["artifacts"], "artifact_id", 6 if name == "workflow" else 2)
    observations = index(run["observations"], "observation_id", 4 if name == "workflow" else 1)
    same(list(artifacts.values()), [strict_json(row["record_json"]) for row in db["research_artifacts"]])
    same(list(observations.values()), [strict_json(row["record_json"]) for row in db["research_observations"]])
    assert set(run["artifact_contents"]) == set(artifacts)
    snapshots = index(run["snapshots"], "label")
    assert list(snapshots) == [str(i) + ":" + label.split(":", 1)[1] for i, label in enumerate(snapshots)]
    positions = {label: i for i, label in enumerate(snapshots)}
    same(run["snapshots"][-1]["database"], db)

    def before_after(record):
        before, after = [record[key] for key in ("before_snapshot", "after_snapshot")]
        assert before in snapshots and after in snapshots and positions[after] == positions[before] + 1
        return snapshots[before]["database"], snapshots[after]["database"]

    for snapshot in snapshots.values():
        captured = snapshot["database"]
        assert set(captured) == TABLES
        for row in captured["execution_attempts"]:
            if row["state"] == "TERMINAL":
                same(row, attempts[row["task_id"]])
        for table, key, final in (("research_artifacts", "artifact_id", artifacts),
                                  ("research_observations", "observation_id", observations)):
            for row in captured[table]:
                decoded = strict_json(row["record_json"])
                same(decoded, final[decoded[key]])

    admissions = index(run["admissions"], "label", 7 if name == "workflow" else 2)
    for admission in admissions.values():
        before, after = before_after(admission)
        same(strict_json(admission["raw_config"]), {"kind": "native_cpu_action", "domain_request": admission["domain_request"]})
        same(ideas[admission["task_id"]]["config"], admission["raw_config"])
        for table in ("execution_attempts", "cpu_action_reservations", "research_artifacts", "research_observations"):
            same(before[table], after[table])
        expected = "already_present_exact" if admission["label"] == "failed_same_id_replay" else "inserted"
        assert admission["outcome"]["status"] == expected and admission["outcome"]["idea_id"] == admission["task_id"]
        if expected == "already_present_exact":
            same(before, after)
        else:
            assert len(after["ideas"]) == len(before["ideas"]) + 1
            assert one([r for r in after["ideas"] if r["idea_id"] == admission["task_id"]])["status"] == "queued"

    calls = index(run["calls"], "label", 12 if name == "workflow" else 2)
    expected_labels = (["baseline_producer", "baseline_v1", "challenger_producer", "failed_v1",
        "recovered_v1", "challenger_v2", "replicate_admit", "execute_replica", "replicate_replay",
        "idle_0", "idle_1", "idle_2"] if name == "workflow" else ["producer", "evaluator"])
    assert list(calls) == expected_labels
    identities, native_call, previous_finish = [], {}, 0
    for label, call in calls.items():
        started, finished = seconds(call["started_monotonic"]), seconds(call["finished_monotonic"])
        assert previous_finish <= started < finished
        previous_finish = finished
        assert math.isclose(seconds(call["wall_seconds"]), finished - started, abs_tol=1e-9)
        assert call["cwd"] == str(root) and type(call["exit_code"]) is int and call["exit_code"] == 0
        command = call["command"]
        assert command[1:3] == ["-m", "examples.holdout.testing"]
        metadata = [strict_json(line.split("=", 1)[1]) for line in call["stdout"].splitlines()
                    if line.startswith("HOLDOUT_CLI_META=")]
        same(metadata, call["metadata"])
        assert len(metadata) == 2 and [m["event"] for m in metadata] == ["start", "finish"]
        assert started <= metadata[0]["monotonic"] <= metadata[1]["monotonic"] <= finished
        binding, closure = [call["controller_supervision"][key] for key in ("binding", "closure")]
        tree(binding, closure, {"scope": str(root), "holdout_test_controller": label}, 0)
        assert binding["command_sha256"] == digest(command)
        for item in metadata:
            same({key: item[key] for key in ("pid", "start_ticks")}, binding["worker"])
            assert item["python"] == command[0]
        assert type(metadata[1]["exit_code"]) is int and metadata[1]["exit_code"] == 0
        identities.append((metadata[0]["pid"], metadata[0]["start_ticks"]))
        before, after = before_after(call)
        old = {row["attempt_id"] for row in before["execution_attempts"]}
        added = [row for row in after["execution_attempts"] if row["attempt_id"] not in old]
        execution = label not in ("replicate_admit", "replicate_replay") and not label.startswith("idle_")
        assert len(added) == int(execution)
        assert call["fault_injection"] == ("partial_exit_71" if label == "failed_v1" else None)
        if execution:
            assert command[3:] == ["-c", str(root / "orze.yaml"), "--once"]
            native_call[added[0]["task_id"]] = call
        else:
            for table in ("execution_attempts", "cpu_action_reservations", "research_artifacts", "research_observations"):
                same(before[table], after[table])
        if label.startswith("idle_"):
            assert command[3:] == ["-c", str(root / "orze.yaml"), "--once"]
            for table in AUTHORITY_TABLES:
                same(before[table], after[table])
            old_decisions = {r["decision_id"] for r in before["cpu_action_decisions"]}
            added_decisions = [strict_json(r["record_json"]) for r in after["cpu_action_decisions"]
                              if r["decision_id"] not in old_decisions]
            decision = one(added_decisions)
            assert decision["kind"] == "Wait" and decision["reason"] == "queue_empty"
    assert len(set(identities)) == len(identities)
    assert set(native_call) == set(attempts)

    bindings, terminals, by_artifact, by_observation = {}, {}, {}, {}
    for artifact in artifacts.values():
        ref = reference(artifact["producer"])
        assert ref["task_id"] not in by_artifact
        by_artifact[ref["task_id"]] = artifact
        sha(artifact["artifact_id"])
        content = run["artifact_contents"][artifact["artifact_id"]].encode("utf-8")
        assert len(content) == artifact["size_bytes"] <= 16384
        assert hashlib.sha256(content).hexdigest() == artifact["content_sha256"]
        assert artifact["scope"] == scope
    for observation in observations.values():
        ref = reference(observation["evaluator"])
        assert ref["task_id"] not in by_observation
        by_observation[ref["task_id"]] = observation
        assert observation["schema"] == 2 and type(observation["schema"]) is int
        assert observation["observation_id"] == digest({"schema": "orze.cpu_observation.v2", "evaluator": ref,
                                                        "scope": scope, "name": "scheduled_value"})
    native_wall, reserved = 0, 0
    for task, row in attempts.items():
        ref = reference({key: row[key] for key in REF_FIELDS})
        binding, terminal = strict_json(row["binding_json"]), strict_json(row["terminal_json"])
        bindings[task], terminals[task] = binding, terminal
        failed = name == "workflow" and task == TASKS["failed_v1"]
        assert row["state"] == "TERMINAL" and row["hold_reason"] is None
        assert ideas[task]["kind"] == "native_cpu_action" and ideas[task]["status"] == ("failed" if failed else "completed")
        assert binding["origin"] == binding["kind"] == "native_cpu_action" and binding["resource"] == "cpu"
        same(binding["attempt_ref"], ref)
        assert binding["scope"] == str(Path(scope) / task) and binding["timeout_seconds"] == 2
        assert binding["source"]["config_sha256"] == hashlib.sha256(ideas[task]["config"].encode()).hexdigest()
        assert terminal["outcome"] == ("failed" if failed else "completed")
        assert type(terminal["return_code"]) is int and terminal["return_code"] == (71 if failed else 0)
        tree(binding["supervision"], terminal["process_tree"], {"attempt_ref": ref, "scope": binding["scope"]}, terminal["return_code"])
        assert type(binding["process_pid"]) is int and binding["process_pid"] == binding["supervision"]["worker"]["pid"]
        sha(terminal["effect_receipt_sha256"])
        reservation, permit = reservations[task], strict_json(reservations[task]["permit_json"])
        assert reservation["state"] == "SETTLED" and reservation["terminal_sha256"] == digest(terminal)
        same(strict_json(reservation["ref_json"]), ref)
        assert permit["task_id"] == task and permit["reservation_id"] == binding["reservation_id"] == reservation["reservation_id"]
        assert permit["wall_limit_seconds"] == 2 and permit["reserved_nanoseconds"] == "2000000000"
        same(permit["budget_scope"]["declaration"], cfg["execution"])
        reserved += permit["wall_limit_seconds"]
        native_wall += seconds(terminal["elapsed_wall_seconds"])
        expected_state = "FAILED" if failed else "COMPLETE"
        transitions = [r for r in db["idea_transitions"] if r["idea_id"] == task]
        assert [(r["from_state"], r["to_state"]) for r in transitions] == [
            ("UNKNOWN", "QUEUED"), ("QUEUED", "CLAIMED"), ("CLAIMED", "IN_PROGRESS"), ("IN_PROGRESS", expected_state)]
        assert all(r["sop_type"] == "action" for r in transitions)
        assert transitions[1]["pid"] == native_call[task]["metadata"][0]["pid"]
        assert transitions[2]["pid"] == transitions[3]["pid"] == binding["process_pid"]
        stages = [r for r in db["idea_stage_transitions"] if r["idea_id"] == task]
        assert [(r["from_state"], r["to_state"]) for r in stages] == [
            ("NOT_STARTED", "PENDING"), ("PENDING", "IN_PROGRESS"), ("IN_PROGRESS", expected_state)]
        assert all(r["stage"] == "action" for r in stages)
        assert one([r for r in db["idea_state"] if r["idea_id"] == task])["current_state"] == expected_state
        if failed:
            assert terminal["artifact_ids"] == terminal["observation_ids"] == []
            assert task not in by_artifact and task not in by_observation
        else:
            artifact = by_artifact[task]
            same(artifact["producer"], ref)
            assert terminal["artifact_ids"] == [artifact["artifact_id"]]
            assert artifact["spec_fingerprint"] == binding["action_sha256"]
            assert artifact["path"] == str(Path(binding["artifact_publication"]["root"]) / artifact["artifact_id"] / "content")
            observation = by_observation.get(task)
            assert terminal["observation_ids"] == ([] if observation is None else [observation["observation_id"]])
            if observation is None:
                assert artifact["logical_name"] == "candidate" and binding["domain_run"]["observation"] is None
            else:
                same(observation["evaluator"], ref)
                assert observation["result_artifact_ids"] == [artifact["artifact_id"]] and artifact["logical_name"] == "evaluation"

    assert set(run["result_envelopes"]) == set(by_observation)
    verdicts = []
    for task, observation in by_observation.items():
        artifact, envelope = by_artifact[task], run["result_envelopes"][task]
        assert canonical(envelope) == run["artifact_contents"][artifact["artifact_id"]].encode()
        request = strict_json(ideas[task]["config"])["domain_request"]
        source = artifacts[one(request["input_artifact_ids"])]
        assert source["logical_name"] == "candidate"
        protocol = request["payload"]["protocol"]
        assert envelope["protocol"] == protocol and envelope["instance_sha256"] == INSTANCE_SHA256
        assert envelope["source_artifact_ids"] == [source["artifact_id"]]
        verdict = verdict_checker(instance, run["artifact_contents"][source["artifact_id"]].encode(), protocol, envelope["verdict"])
        same(observation["validation"], {k: verdict[k] for k in ("status", "reason_code")})
        expected_values = {"direction": "maximize", "instance_id": instance["instance_id"], "instance_sha256": INSTANCE_SHA256}
        if verdict["status"] == "valid":
            expected_values["value"] = verdict["scheduled_value"]
        same(observation["values"], expected_values)
        assert observation["name"] == "scheduled_value" and observation["scope"] == scope
        expected_protocol = digest({"schema": "scheduling.protocol.v1", "protocol": protocol})
        assert observation["protocol_fingerprint"] == expected_protocol
        assert observation["comparison_scope"] == digest({"schema": "scheduling.comparison.v1", "instance_sha256": INSTANCE_SHA256,
                                                          "protocol_fingerprint": expected_protocol})
        assert observation["input_artifact_ids"] == [source["artifact_id"]]
        same(observation["input_artifact_bindings"], {source["artifact_id"]: {k: source[k] for k in
                                                                               ("producer", "spec_fingerprint", "content_sha256")}})
        seconds(envelope["worker_cpu_seconds"])
        seconds(envelope["worker_wall_seconds"])
        verdicts.append({"task_id": task, "source_artifact_id": source["artifact_id"], "protocol": protocol, "verdict": verdict})

    if name == "workflow":
        failed, recovered, v2 = [TASKS[key] for key in ("failed_v1", "recovered_v1", "challenger_v2")]
        same(ideas[failed]["config"], ideas[recovered]["config"])
        for key in ("action_sha256", "inputs_sha256", "command_sha256", "domain_run"):
            same(bindings[failed][key], bindings[recovered][key])
        before_recovery, _ = before_after(calls["recovered_v1"])
        same(one([r for r in before_recovery["execution_attempts"] if r["task_id"] == failed]), attempts[failed])
        assert one([r for r in before_recovery["cpu_action_reservations"] if r["task_id"] == failed])["state"] == "SETTLED"
        partial = run["partial_output"]
        partial_bytes = partial["utf8"].encode()
        assert partial["path"] == str(Path(bindings[failed]["work_dir"]) / "evaluation.json")
        assert partial["bytes"] == len(partial_bytes) > 0 and partial["sha256"] == hashlib.sha256(partial_bytes).hexdigest()
        try:
            strict_json(partial_bytes)
        except ValueError:
            pass
        else:
            raise AssertionError("partial output must be incomplete JSON")
        assert partial["sha256"] not in {a["content_sha256"] for a in artifacts.values()}
        v1o, v2o = by_observation[recovered], by_observation[v2]
        assert v1o["input_artifact_ids"] == v2o["input_artifact_ids"] == [by_artifact[TASKS["challenger_producer"]]["artifact_id"]]
        assert v1o["validation"]["status"] == "valid" and v2o["validation"] == {"status": "invalid", "reason_code": "capacity_overload"}
        assert v1o["protocol_fingerprint"] != v2o["protocol_fingerprint"] and v1o["comparison_scope"] != v2o["comparison_scope"]
        requests = [copy.deepcopy(strict_json(ideas[t]["config"])["domain_request"]) for t in (recovered, v2)]
        requests[1]["payload"]["protocol"] = requests[0]["payload"]["protocol"]
        same(*requests)
        assert by_observation[TASKS["baseline_v1"]]["values"]["value"] == 0 < v1o["values"]["value"]
        replica_row = one(db["replication_requests"])
        replica = strict_json(replica_row["record_json"])
        assert replica["schema"] == 2 and replica["adapter"] == "native_cpu_action"
        assert replica["request_sha256"] == digest({k: v for k, v in replica.items() if k != "request_sha256"})
        same(replica["source_ref"], {k: attempts[recovered][k] for k in REF_FIELDS})
        target = replica["task_id"]
        assert target == replica_row["task_id"] and target not in TASKS.values()
        assert replica["request_id"] == replica_row["request_id"] == "holdout-confirm-v1"
        assert replica["reason"] == "explicit holdout evaluator repeat"
        assert replica["scope"] == scope and replica["database"] == str(root / "lake.db")
        assert replica["source_row_sha256"] == digest(decode_attempt(attempts[recovered]))
        assert replica["source_terminal_sha256"] == digest(terminals[recovered])
        assert replica["source_config_sha256"] == hashlib.sha256(ideas[recovered]["config"].encode()).hexdigest()
        assert replica["artifact_records_sha256"] == digest({"records": [by_artifact[recovered]]})
        assert replica["observation_records_sha256"] == digest({"records": [by_observation[recovered]]})
        assert replica["domain_run_sha256"] == digest({"domain_run": bindings[recovered]["domain_run"]})
        assert replica["artifact_binding_sha256"] == digest(bindings[recovered]["artifact_publication"])
        assert replica["action_sha256"] == replica["spec_fingerprint"] == bindings[recovered]["action_sha256"] == bindings[target]["action_sha256"]
        same(ideas[recovered]["config"], ideas[target]["config"])
        for key in ("values", "validation", "input_artifact_ids", "input_artifact_bindings", "protocol_fingerprint", "comparison_scope"):
            same(by_observation[recovered][key], by_observation[target][key])
        result = {"status": "created", "request_id": replica["request_id"], "task_id": target}
        same(cli_result(calls["replicate_admit"]), result)
        same(cli_result(calls["replicate_replay"]), {**result, "status": "already_requested"})
        same(*before_after(calls["replicate_replay"]))
        assert native_call[target]["label"] == "execute_replica"
        for label in ("replicate_admit", "replicate_replay"):
            assert calls[label]["command"][3:] == ["replicate", recovered, "-c", str(root / "orze.yaml"),
                "--request-id", replica["request_id"], "--reason", replica["reason"]]
        assert [(strict_json(r["record_json"])["kind"], strict_json(r["record_json"])["reason"])
                for r in db["cpu_action_decisions"]] == [("Wait", "queue_empty")] * 3
    else:
        expected = {"boundary": ("valid", "feasible"), "duplicate_id": ("invalid", "duplicate_job"),
            "missing_prerequisite": ("invalid", "missing_prerequisite"), "overload": ("invalid", "capacity_overload"),
            "malformed": ("invalid", "candidate_json_invalid")}[name]
        assert tuple(verdicts[0]["verdict"][k] for k in ("status", "reason_code")) == expected
        assert db["replication_requests"] == db["cpu_action_decisions"] == []
        if name == "boundary":
            source_id = verdicts[0]["source_artifact_id"]
            candidate = strict_json(run["artifact_contents"][source_id])
            starts = {item["job_id"]: item["start"] for item in candidate["schedule"]}
            jobs = {item["id"]: item for item in instance["jobs"]}
            assert any(starts[j] + jobs[j]["duration"] == jobs[j]["deadline"] for j in starts)
            assert any(starts[left] + jobs[left]["duration"] == starts[right]
                       and jobs[left]["demand"] == jobs[right]["demand"] == instance["capacity"]
                       for left in starts for right in starts if left != right)
    budget_row = one(db["cpu_action_scopes"])
    assert budget_row["scope"] == scope and budget_row["stop_json"] is None
    same(strict_json(budget_row["binding_json"])["declaration"], cfg["execution"])
    return {"name": name, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
        "native_actions": count, "reserved_wall_seconds": reserved, "fresh_cli_invocations": len(calls),
        "controller_identities": identities, "attempt_ids": [row["attempt_id"] for row in attempts.values()],
        "artifact_ids": list(artifacts), "observation_ids": list(observations), "verdicts": verdicts,
        "timings": {"outer_controller_wall_seconds": sum(c["wall_seconds"] for c in calls.values()),
            "native_terminal_elapsed_wall_seconds": native_wall,
            "evaluator_body_cpu_seconds": sum(v["worker_cpu_seconds"] for v in run["result_envelopes"].values()),
            "evaluator_body_wall_seconds": sum(v["worker_wall_seconds"] for v in run["result_envelopes"].values())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path, help="Exactly six complete archived run JSON paths")
    parser.add_argument("--prefix", help="Read the six named reports from the adjacent evidence/runs directory")
    args = parser.parse_args()
    if args.prefix is not None:
        if args.reports or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.prefix):
            parser.error("use a safe prefix or six explicit paths, not both")
        directory = Path(__file__).resolve().parents[1] / "runs"
        paths = [directory / (args.prefix + "-" + name + ".json") for name in sorted(NAMES)]
    else:
        if len(args.reports) != 6:
            parser.error("provide --prefix or exactly six report paths")
        paths = [p.resolve() for p in args.reports]
    assert len(set(paths)) == 6
    independent = Path(__file__).with_name("v1-07b-domain-check.py")
    module = runpy.run_path(str(independent), run_name="holdout_independent_domain_checker")
    results = [check_run(path, module["verify_result"]) for path in paths]
    assert {r["name"] for r in results} == NAMES and len({r["name"] for r in results}) == 6
    for field in ("attempt_ids", "artifact_ids", "observation_ids", "controller_identities"):
        values = [tuple(v) if type(v) is list else v for r in results for v in r[field]]
        assert len(set(values)) == len(values), field
    assert sum(r["native_actions"] for r in results) == 17
    assert sum(r["reserved_wall_seconds"] for r in results) == 34
    assert sum(r["fresh_cli_invocations"] for r in results) == 22
    print(json.dumps({"status": "passed", "classification": "read-only archived consistency plus independent domain verdicts",
        "independent_domain_checker_sha256": hashlib.sha256(independent.read_bytes()).hexdigest(),
        "reports": results, "native_actions": 17, "reserved_wall_seconds": 34, "fresh_cli_invocations": 22,
        "additional_workers_started": 0,
        "limits": ["Captured closure/effect metadata is not fresh kernel or filesystem authority.",
                   "Partial bytes are checked from the archive, not the original temporary work directory.",
                   "Recovery is an explicit new evaluation after confirmed failure, not adoption of an unknown attempt.",
                   "Same value and distinct IDs do not establish statistical independence or equivalence.",
                   "Outer controller wall includes supervisor/interpreter setup; evaluator timing excludes final output write. No overhead or efficiency claim."]}, sort_keys=True))


if __name__ == "__main__":
    main()
