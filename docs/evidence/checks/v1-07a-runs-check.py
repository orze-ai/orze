"""Read archived V1-07A reports; never import the product or start a worker.

This verifies the archive's internal evidence links, not current filesystem
authority, an external effect receipt, kernel state, or statistical independence.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys


if sys.flags.optimize:
    raise SystemExit("v1-07a-runs-check requires assertions; optimized Python is refused")


TABLES = {"ideas", "execution_attempts", "research_artifacts", "research_observations",
          "cpu_action_reservations", "cpu_action_decisions", "cpu_action_scopes",
          "cpu_proposal_requests", "replication_requests"}
REF_FIELDS = ("task_id", "phase", "attempt_id", "generation")
TASKS = {name: "idea-" + name for name in ("baseline", "challenger", "unchecked", "analysis")}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


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
    assert type(value) is str and re.fullmatch("[0-9a-f]{64}", value)


def seconds(value):
    assert type(value) in (int, float) and math.isfinite(value) and value >= 0
    return value


def reference(value):
    assert type(value) is dict and set(value) == set(REF_FIELDS)
    assert value["phase"] == "action" and type(value["generation"]) is int and value["generation"] > 0
    assert all(type(value[key]) is str and value[key] for key in REF_FIELDS[:3])
    return value


def index(rows, key, expected):
    assert type(rows) is list and len(rows) == expected
    result = {row[key]: row for row in rows}
    assert len(result) == expected
    return result


def decode_container(raw):
    assert 8 <= len(raw) <= 4104
    magic, size, payload = raw[:4], int.from_bytes(raw[4:8], "big"), raw[8:]
    assert size <= 2048
    if magic == b"RAW1":
        assert len(payload) == size
        return payload
    if magic == b"HEX1":
        assert len(payload) == 2 * size and re.fullmatch(b"(?:[0-9a-f]{2})*", payload)
        return bytes.fromhex(payload.decode("ascii"))
    assert magic == b"RLE1" and len(payload) % 2 == 0
    decoded = bytearray()
    for offset in range(0, len(payload), 2):
        count, value = payload[offset:offset + 2]
        assert count > 0 and len(decoded) + count <= size
        decoded.extend(bytes((value,)) * count)
    assert len(decoded) == size
    return bytes(decoded)


def domain_result(domain, dataset, envelope):
    cost, details = envelope["cost"], envelope["details"]
    assert type(cost) is int and cost >= 0
    if domain == "sorting":
        outputs, trace = details["outputs"], details["trace"]
        assert len(outputs) == len(dataset) and cost == len(trace)
        for event in trace:
            assert type(event) is list and len(event) == 5
            case, left, right, relation, answer = event
            assert all(type(value) is int for value in (case, left, right))
            assert 0 <= case < len(dataset) and relation in (">", "<") and type(answer) is bool
            assert answer == (left > right if relation == ">" else left < right)
        assert all(type(row) is list and all(type(value) is int for value in row) for row in outputs)
        return all(Counter(row) == Counter(output) and output == sorted(row)
                   for row, output in zip(dataset, outputs))
    raw = bytes.fromhex(details["container_hex"])
    assert cost == len(raw)
    try:
        return decode_container(raw) == bytes.fromhex(dataset)
    except AssertionError:
        return False


def check_run(path, domain, variant):
    raw = path.read_bytes()
    assert len(raw) <= 4 * 1024 * 1024
    run = strict_json(raw)
    count = 5 if variant == "default" else 3
    db, cfg, trace = run["database"], run["cfg"], run["trace"]
    assert set(db) == TABLES and type(run["exit_code"]) is int and run["exit_code"] == 0
    same(cfg["execution"], {"version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 10})
    same(cfg["action_policy"], {"version": 1, "kind": "acceptance", "idle": "stop", "wait_seconds": .05, "config": {}})
    assert cfg["action_domain"]["kind"] == "acceptance_" + domain
    dataset = cfg["action_domain"]["config"]["dataset"]
    assert run["dataset_sha256"] == digest(dataset)
    scope = str(Path(run["root"]) / "results")
    assert cfg["results_dir"] == scope and cfg["idea_lake_db"] == str(Path(run["root"]) / "lake.db")
    ideas = index(db["ideas"], "idea_id", count)
    attempts = index(db["execution_attempts"], "task_id", count)
    assert len({row["attempt_id"] for row in attempts.values()}) == count
    artifacts = index(run["artifacts"], "artifact_id", count)
    observations = index(run["observations"], "observation_id", count)
    same(list(artifacts.values()), [strict_json(row["record_json"]) for row in db["research_artifacts"]])
    same(list(observations.values()), [strict_json(row["record_json"]) for row in db["research_observations"]])
    by_artifact = {}
    by_observation = {}
    for artifact in artifacts.values():
        ref = reference(artifact["producer"])
        assert ref["task_id"] not in by_artifact
        by_artifact[ref["task_id"]] = artifact
    for observation in observations.values():
        ref = reference(observation["evaluator"])
        assert ref["task_id"] not in by_observation
        by_observation[ref["task_id"]] = observation
    reservations = index(db["cpu_action_reservations"], "task_id", count)
    native_wall = 0
    for task, row in attempts.items():
        ref = reference({key: row[key] for key in REF_FIELDS})
        assert row["state"] == "TERMINAL" and row["hold_reason"] is None
        assert ideas[task]["kind"] == "native_cpu_action" and ideas[task]["status"] == "completed"
        binding, terminal = strict_json(row["binding_json"]), strict_json(row["terminal_json"])
        same(binding["attempt_ref"], ref)
        assert binding["origin"] == binding["kind"] == "native_cpu_action" and binding["resource"] == "cpu"
        assert binding["source"]["config_sha256"] == hashlib.sha256(ideas[task]["config"].encode()).hexdigest()
        assert terminal["outcome"] == "completed" and type(terminal["return_code"]) is int and terminal["return_code"] == 0
        tree = terminal["process_tree"]
        assert tree["event"] == "TREE_CLOSED" and tree["wait_proof"] == "ECHILD_WALL"
        assert type(tree["schema"]) is int and tree["schema"] == 1
        assert type(tree["worker_returncode"]) is int and tree["worker_returncode"] == 0
        assert tree["stop_requested"] is False and tree["forced_cleanup"] is False
        assert type(tree["reaped_children"]) is int and tree["reaped_children"] >= 1
        same(tree["binding"], binding["supervision"])
        same(tree["binding"]["identity"], {"attempt_ref": ref, "scope": str(Path(scope) / task)})
        assert type(binding["process_pid"]) is int and binding["process_pid"] == tree["binding"]["worker"]["pid"]
        sha(terminal["effect_receipt_sha256"])
        artifact, observation = by_artifact[task], by_observation[task]
        same(artifact["producer"], ref)
        same(observation["evaluator"], ref)
        assert terminal["artifact_ids"] == [artifact["artifact_id"]]
        assert terminal["observation_ids"] == [observation["observation_id"]]
        assert observation["result_artifact_ids"] == [artifact["artifact_id"]]
        assert artifact["scope"] == observation["scope"] == scope
        assert artifact["spec_fingerprint"] == binding["action_sha256"]
        envelope = run["result_envelopes"][task]
        assert hashlib.sha256(canonical(envelope)).hexdigest() == artifact["content_sha256"]
        assert len(canonical(envelope)) == artifact["size_bytes"]
        assert envelope["dataset_sha256"] == run["dataset_sha256"]
        same(observation["values"], {key: envelope[key] for key in ("cost", "candidate", "dataset_sha256", "operation", "worker_cpu_seconds", "worker_wall_seconds")})
        valid = domain_result(domain, dataset, envelope)
        status = observation["validation"]["status"]
        if status != "unknown":
            assert status == ("valid" if valid else "invalid")
        reservation = reservations[task]
        permit = strict_json(reservation["permit_json"])
        assert reservation["state"] == "SETTLED" and permit["wall_limit_seconds"] == 2
        assert permit["reserved_nanoseconds"] == "2000000000" and permit["task_id"] == task
        same(strict_json(reservation["ref_json"]), ref)
        assert reservation["terminal_sha256"] == digest(terminal)
        assert reservation["reservation_id"] == binding["reservation_id"] == permit["reservation_id"]
        same(permit["budget_scope"]["declaration"], cfg["execution"])
        native_wall += seconds(terminal["elapsed_wall_seconds"])

    expected_kinds = (["Propose", "Execute"] * (count - 1)) + ["Replicate", "Execute", "Stop"]
    assert [entry["decision"]["kind"] for entry in trace] == expected_kinds
    stdout_trace = [strict_json(line[len("ACCEPTANCE_DECISION="):]) for line in run["stdout"].splitlines()
                    if line.startswith("ACCEPTANCE_DECISION=")]
    same(stdout_trace, trace)
    started, finished = seconds(run["started_monotonic"]), seconds(run["finished_monotonic"])
    assert started < finished and math.isclose(run["wall_seconds"], finished - started, abs_tol=1e-9)
    first_visible = {}
    last_tick = started
    for entry in trace:
        assert last_tick <= entry["monotonic"] <= finished
        last_tick = entry["monotonic"]
        snapshot = entry["snapshot"]
        assert entry["snapshot_sha256"] == digest(snapshot)
        view = snapshot["recorded_evidence"]
        assert view["unavailable"] == [] and view["more_available"] is False
        for result in view["results"]:
            task = result["ref"]["task_id"]
            same(result["ref"], {key: attempts[task][key] for key in REF_FIELDS})
            assert result["outcome"] == "completed"
            same(result["artifact_records"], [by_artifact[task]])
            same(result["observation_records"], [by_observation[task]])
            first_visible.setdefault(task, entry["monotonic"] - started)
    assert set(first_visible) == set(attempts)
    assert trace[0]["snapshot"]["recorded_evidence"]["results"] == []
    final = trace[-1]
    stop = {"kind": "Stop", "reason": "confirmed_selection", "wakeup": None}
    same(final["decision"], stop)
    assert final["budget"]["active_reservations"] == 0 and final["budget"]["remaining_wall_seconds"] == 10 - 2 * count
    assert final["budget"]["reserved_wall_seconds"] == 2 * count
    assert len(db["cpu_action_scopes"]) == 1
    same(strict_json(db["cpu_action_scopes"][0]["stop_json"]), stop)

    replica_row = db["replication_requests"]
    assert len(replica_row) == 1
    replica = strict_json(replica_row[0]["record_json"])
    assert replica["request_sha256"] == digest({key: value for key, value in replica.items() if key != "request_sha256"})
    source = TASKS["baseline" if variant == "default" else "challenger"]
    ref = {key: attempts[source][key] for key in REF_FIELDS}
    same(replica["source_ref"], ref)
    selected = next(entry["decision"] for entry in trace if entry["decision"]["kind"] == "Replicate")
    same(selected["source_ref"], ref)
    assert selected["request_id"] == replica["request_id"] == replica_row[0]["request_id"]
    assert selected["reason"] == replica["reason"]
    target = replica["task_id"]
    assert target == replica_row[0]["task_id"] and target != source and target not in TASKS.values()
    assert ideas[source]["config"] == ideas[target]["config"]
    assert by_artifact[source]["spec_fingerprint"] == by_artifact[target]["spec_fingerprint"] == replica["action_sha256"]
    source_binding = strict_json(attempts[source]["binding_json"])
    source_terminal = strict_json(attempts[source]["terminal_json"])
    decoded_row = {key: attempts[source][key] for key in (*REF_FIELDS, "state", "hold_reason")}
    decoded_row.update(binding=source_binding, terminal=source_terminal)
    assert replica["source_row_sha256"] == digest(decoded_row)
    assert replica["source_terminal_sha256"] == digest(source_terminal)
    assert replica["source_config_sha256"] == hashlib.sha256(ideas[source]["config"].encode()).hexdigest()
    assert replica["artifact_records_sha256"] == digest({"records": [by_artifact[source]]})
    assert replica["observation_records_sha256"] == digest({"records": [by_observation[source]]})
    assert replica["domain_run_sha256"] == digest({"domain_run": source_binding["domain_run"]})
    assert by_observation[target]["validation"]["status"] == "valid"
    assert by_observation[target]["values"]["cost"] == by_observation[source]["values"]["cost"]

    proposals = [strict_json(row["record_json"]) for row in db["cpu_proposal_requests"]]
    assert len(proposals) == count - 1
    same([record["decision"] for record in proposals], [entry["decision"] for entry in trace if entry["decision"]["kind"] == "Propose"])
    for record in proposals:
        assert record["record_sha256"] == digest({key: value for key, value in record.items() if key != "record_sha256"})
        assert record["outcome"]["status"] == "inserted"
        same(strict_json(ideas[record["task_id"]]["config"]), {"kind": "native_cpu_action", "domain_request": record["decision"]["domain_request"]})
    durable_decisions = [strict_json(row["record_json"]) for row in db["cpu_action_decisions"]]
    assert [(item["kind"], item["reason"]) for item in durable_decisions] == (
        [("Wait", "proposal_inserted")] * (count - 1)
        + [("Wait", "replication_created"), ("Stop", "confirmed_selection")])
    if variant == "default":
        base, bad, unknown, analysis = [by_observation[TASKS[key]] for key in ("baseline", "challenger", "unchecked", "analysis")]
        assert [item["validation"]["status"] for item in (base, bad, unknown, analysis)] == ["valid", "invalid", "unknown", "valid"]
        costs = (9, 29) if domain == "sorting" else (137, 266)
        assert base["values"]["cost"] == costs[0] and unknown["values"]["cost"] == analysis["values"]["cost"] == costs[1]
        assert bad["values"]["cost"] < costs[0] < costs[1]
        ids = [by_artifact[TASKS[key]]["artifact_id"] for key in ("baseline", "challenger", "unchecked")]
        assert analysis["input_artifact_ids"] == ids
        same(analysis["input_artifact_bindings"], {identity: {key: artifacts[identity][key] for key in ("producer", "spec_fingerprint", "content_sha256")} for identity in ids})
        analysis_proposal = next(record for record in proposals if record["task_id"] == TASKS["analysis"])
        assert analysis_proposal["decision"]["domain_request"]["input_artifact_ids"] == ids
        assert analysis["comparison_scope"] == base["comparison_scope"] and analysis["protocol_fingerprint"] == base["protocol_fingerprint"]
        assert analysis["observation_id"] != unknown["observation_id"] and analysis["evaluator"] != unknown["evaluator"]
    else:
        assert set(ideas).isdisjoint({TASKS["unchecked"], TASKS["analysis"]})
        assert by_observation[TASKS["challenger"]]["validation"]["status"] == "valid"
        assert by_observation[TASKS["challenger"]]["values"]["cost"] < by_observation[TASKS["baseline"]]["values"]["cost"]

    rerun = run["stopped_rerun"]
    assert rerun["exit_code"] == 75 and rerun["trace"] == []
    assert "ACCEPTANCE_DECISION=" not in rerun["stdout"]
    assert seconds(rerun["started_monotonic"]) >= finished
    assert math.isclose(rerun["wall_seconds"], rerun["finished_monotonic"] - rerun["started_monotonic"], abs_tol=1e-9)
    timings = {"worker_cpu_seconds": sum(seconds(item["values"]["worker_cpu_seconds"]) for item in observations.values()),
        "worker_wall_seconds": sum(seconds(item["values"]["worker_wall_seconds"]) for item in observations.values()),
        "native_terminal_elapsed_wall_seconds": native_wall, "cli_wall_seconds": run["wall_seconds"],
        "first_policy_visible_seconds": first_visible}
    return run, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
        "actions": count, "reserved_wall_seconds": 2 * count, "control_steps": len(trace),
        "selected_ref": ref, "replica_ref": {key: attempts[target][key] for key in REF_FIELDS}, "timings": timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="v1-07a-target")
    args = parser.parse_args()
    assert re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", args.prefix)
    directory = Path(__file__).resolve().parent.parent / "runs"
    runs, summaries = {}, []
    for domain in ("sorting", "compression"):
        for variant in ("default", "counterfactual"):
            name = domain + "_" + variant
            run, summary = check_run(directory / (args.prefix + "-" + name + ".json"), domain, variant)
            runs[name] = run
            summaries.append(summary)
    for domain in ("sorting", "compression"):
        default, counter = [copy.deepcopy(runs[domain + "_" + variant]["cfg"]) for variant in ("default", "counterfactual")]
        for cfg in (default, counter):
            for key in ("results_dir", "idea_lake_db", "ideas_file"):
                cfg.pop(key)
        left = default["action_domain"]["config"].pop("dataset")
        right = counter["action_domain"]["config"].pop("dataset")
        same(default, counter)
        if domain == "sorting":
            same(left, [[], [5], [1, 1], [0, 1, 2, 2, 4, 3, 5, 6]])
            same(right, [[], [5], [1, 1], [0, 1, 2, 2, 3, 4, 5, 6]])
        else:
            assert bytes.fromhex(left) == b"A" * 64 + b"\0" + b"B" * 64
            assert bytes.fromhex(right) == b"A" * 64 + b"B" * 64
    assert len({row["attempt_id"] for run in runs.values() for row in run["database"]["execution_attempts"]}) == 16
    for field, key in (("artifacts", "artifact_id"), ("observations", "observation_id")):
        assert len({row[key] for run in runs.values() for row in run[field]}) == 16
    print(json.dumps({"status": "passed", "classification": "read-only archive consistency, not execution authority",
        "reports": summaries, "native_actions": 16, "reserved_wall_seconds": 32, "cli_invocations": 8,
        "sticky_stop_refusals": 4, "additional_workers_started": 0,
        "timing_limit": "first_policy_visible is first recorded decision visibility, not exact publication time; wall residual is not pure overhead",
        "authority_limit": "No archived effect SHA/tree record is promoted into a fresh kernel or filesystem proof; stopped rerun database equality is checked by the source fixture, whose second DB copy is not archived."}, sort_keys=True))


if __name__ == "__main__":
    main()
