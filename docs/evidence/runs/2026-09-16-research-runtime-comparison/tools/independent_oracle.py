"""Read-only independent recomputation; stdlib only, no Orze/example imports.

Consumes an already completed formal round. Does not run workers, policies,
driver.qualify, driver.summarize, or modify the source/evidence directories.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import subprocess
import zipfile


TABLES = ("ideas", "execution_attempts", "research_artifacts", "research_observations",
          "cpu_action_reservations", "cpu_action_decisions", "cpu_action_scopes",
          "cpu_proposal_requests", "replication_requests")
REF = ("task_id", "phase", "attempt_id", "generation")
METRICS = ("native_actions", "reserved_seconds", "analysis_actions", "cli_wall_seconds",
           "first_valid_consumed_seconds", "confirmed_selection_seconds",
           "worker_cpu_seconds", "worker_wall_seconds", "native_elapsed_seconds")
STOP = {"kind": "Stop", "reason": "confirmed_selection", "wakeup": None}
DATA = {
    ("sorting", "default"): [[], [5], [1, 1], [0, 1, 2, 2, 4, 3, 5, 6]],
    ("sorting", "counterfactual"): [[], [5], [1, 1], [0, 1, 2, 2, 3, 4, 5, 6]],
    ("compression", "default"): (b"A" * 64 + b"\0" + b"B" * 64).hex(),
    ("compression", "counterfactual"): (b"A" * 64 + b"B" * 64).hex(),
}
PROTOCOL = {
    "sorting": {"name": "sorting_comparisons", "version": 1,
                "metric": "comparator_calls_including_false"},
    "compression": "container-raw1-rle1-hex1-be32-v1",
}


def require(condition, label):
    if not condition:
        raise ValueError(label)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(canonical(value))


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))


def read(path):
    return parse(Path(path).read_bytes())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def near(left, right):
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def one(items, label):
    require(len(items) == 1, label)
    return items[0]


def independent_scientific_check(domain, dataset, envelope):
    """Finite oracle, independent of either policy/Domain implementation."""
    details, candidate = envelope["details"], envelope["candidate"]
    if domain == "sorting":
        outputs = details["outputs"]
        valid = (len(outputs) == len(dataset)
                 and all(Counter(out) == Counter(inp) and out == sorted(out)
                         for inp, out in zip(dataset, outputs)))
        if candidate == "baseline":
            # Independent saved-key insertion count, versus worker swaps.
            calls = 0
            for original in dataset:
                values = list(original)
                for index in range(1, len(values)):
                    key, cursor = values[index], index - 1
                    while cursor >= 0:
                        calls += 1
                        if values[cursor] <= key:
                            break
                        values[cursor + 1] = values[cursor]
                        cursor -= 1
                    values[cursor + 1] = key
        elif candidate == "challenger":
            calls = 0
        elif candidate == "unchecked":
            calls = sum(len(row) * (len(row) - 1) // 2 for row in dataset)
        else:
            raise ValueError("unknown sorting candidate")
        require(envelope["cost"] == calls == len(details["trace"]), "sorting cost/trace")
        for case, left, right, relation, answer in details["trace"]:
            require(type(case) is int and 0 <= case < len(dataset), "trace case")
            require(type(answer) is bool and relation in ("<", ">"), "trace relation")
            require(answer == (left < right if relation == "<" else left > right), "trace comparison")
        return valid
    raw = bytes.fromhex(details["container_hex"])
    require(envelope["cost"] == len(raw), "container-byte objective")
    if len(raw) < 8:
        return False
    magic, size, body = raw[:4], int.from_bytes(raw[4:8], "big"), raw[8:]
    try:
        if magic == b"RAW1":
            decoded = body
        elif magic == b"HEX1":
            decoded = bytes.fromhex(body.decode("ascii"))
            require(decoded.hex().encode() == body, "noncanonical hex")
        elif magic == b"RLE1":
            if len(body) % 2 or any(body[i] == 0 for i in range(0, len(body), 2)):
                return False
            decoded = b"".join(bytes([body[i + 1]]) * body[i] for i in range(0, len(body), 2))
        else:
            return False
    except (ValueError, UnicodeError):
        return False
    return len(decoded) == size and decoded == bytes.fromhex(dataset)


def check_run(root, record, spec, manifest):
    domain, variant, arm = (spec[key] for key in ("domain", "variant", "arm"))
    require(record["root"] == str(root) and all(record[k] == spec[k] for k in ("domain", "variant", "arm")), "run identity")
    require(record["exit_code"] == 0 and record["error"] is None, "failed/unknown CLI")
    require(record.get("cleanup_error") is None, "cleanup unknown")
    dataset = DATA[domain, variant]
    cfg = record["config"]
    require(cfg["execution"] == {"version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 10}, "budget config")
    require(cfg["action_domain"] == {"version": 1, "kind": "acceptance_" + domain, "config": {"dataset": dataset}}, "dataset config")
    protocol = digest({"schema": "acceptance.protocol.v1", "protocol": PROTOCOL[domain]})
    require(manifest["datasets"][domain + "_" + variant] == digest(dataset), "manifest dataset")
    require(cfg["action_policy"] == {"version": 1, "kind": "acceptance" if arm == "A" else "dominance_pruning",
        "idle": "stop", "wait_seconds": .05, "config": {} if arm == "A" else {"validation_only_protocols": [protocol]}}, "policy declaration")
    require(record["command"][0] == manifest["python"] and record["command"][1] == "-I", "installed interpreter")
    require(record["command"][-2:] == ["-c", str(root / "orze.yaml")], "actual config argv")
    ready, closed = record["controller_binding"], record["controller_closure"]
    require(read(root / "controller-ready.json") == ready, "actual READY file")
    require(ready["identity"] == {"research_efficiency": arm, "scope": str(root)}, "controller identity")
    require(ready["command_sha256"] == digest(record["command"]), "controller command SHA")
    require(closed["binding"] == ready and closed["event"] == "TREE_CLOSED" and closed["wait_proof"] == "ECHILD_WALL", "controller closure")
    require(closed["worker_returncode"] == 0 and closed["stop_requested"] is False and closed["forced_cleanup"] is False, "clean controller exit")
    stdout, stderr = (root / "stdout.log").read_text(), (root / "stderr.log").read_text()
    require(stdout == record["stdout"] and stderr == record["stderr"], "actual stdout/stderr")
    trace = [parse(line.split("=", 1)[1]) for line in stdout.splitlines() if line.startswith("ACCEPTANCE_DECISION=")]
    pruning = [parse(line.split("=", 1)[1]) for line in stdout.splitlines() if line.startswith("PRUNING_DECISION=")]
    require(trace == record["trace"] and pruning == record["pruning"], "trace parsing")
    require(trace and trace[-1]["decision"] == STOP, "confirmed policy Stop")
    start, end = record["started_monotonic"], record["finished_monotonic"]
    require(all(finite(v) for v in (start, end, record["wall_seconds"])) and near(end - start, record["wall_seconds"]), "CLI timing")
    require([t["monotonic"] for t in trace] == sorted(t["monotonic"] for t in trace), "trace time ordering")
    require(all(start <= t["monotonic"] <= end and t["snapshot_sha256"] == digest(t["snapshot"]) for t in trace), "trace clock/hash")
    with sqlite3.connect((root / "lake.db").as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        db = {table: [dict(r) for r in conn.execute("SELECT * FROM " + table + " ORDER BY rowid")] for table in TABLES}
    require(db == record["database"], "actual final SQLite differs from saved snapshot")
    attempts, reservations = db["execution_attempts"], db["cpu_action_reservations"]
    require(len(attempts) == len(reservations) == len(db["ideas"]) and len(attempts) > 0, "attempt count")
    require(len({r["attempt_id"] for r in attempts}) == len(attempts), "unique attempts")
    require(all(r["status"] == "completed" for r in db["ideas"]), "actual task status")
    artifacts = {r["artifact_id"]: parse(r["record_json"]) for r in db["research_artifacts"]}
    observations = {r["observation_id"]: parse(r["record_json"]) for r in db["research_observations"]}
    require(len(artifacts) == len(observations) == len(attempts), "publication cardinality")
    require(len(artifacts) == len(db["research_artifacts"]) and len(observations) == len(db["research_observations"]), "unique publications")
    by_ref, binding_by_ref, used_reservations, envelopes = {}, {}, set(), {}
    prepared_count = 0
    for row in attempts:
        ref = {k: row[k] for k in REF}
        require(ref["phase"] == "action" and type(ref["generation"]) is int and ref["generation"] > 0, "exact native action Ref")
        require(row["state"] == "TERMINAL" and row["hold_reason"] is None, "terminal state")
        binding, terminal = parse(row["binding_json"]), parse(row["terminal_json"])
        folder = root / "results" / ref["task_id"]
        require(binding["attempt_ref"] == ref and binding["scope"] == str(folder), "binding full Ref/scope")
        require(binding["origin"] == binding["kind"] == "native_cpu_action" and binding["resource"] == "cpu", "native kind")
        require(terminal["outcome"] == "completed" and terminal["return_code"] == 0 and finite(terminal["elapsed_wall_seconds"]), "native outcome")
        tree = terminal["process_tree"]
        require(tree["binding"] == binding["supervision"] and tree["wait_proof"] == "ECHILD_WALL" and tree["event"] == "TREE_CLOSED", "actual native TREE")
        require(tree["binding"]["identity"] == {"attempt_ref": ref, "scope": str(folder)}, "TREE identity")
        require(tree["worker_returncode"] == 0 and tree["stop_requested"] is False and tree["forced_cleanup"] is False and tree["lease_expired"] is False, "unexpected interrupted native")
        lease = binding["runtime_lease"]
        require(lease == tree["binding"]["runtime_lease"] and lease["deadline_ns"] - lease["issued_ns"] == 2_000_000_000, "same lease descriptor")
        require(terminal["runtime_lease"]["status"] == "authorized" and lease["issued_ns"] <= tree["lease_observed_ns"] <= terminal["runtime_lease"]["observed_ns"] < lease["deadline_ns"], "authorized lease chronology")
        reservation = one([r for r in reservations if r["reservation_id"] == binding["reservation_id"]], "one reservation")
        permit = parse(reservation["permit_json"])
        require(reservation["state"] == "SETTLED" and parse(reservation["ref_json"]) == ref and reservation["terminal_sha256"] == digest(terminal), "settlement linkage")
        require(permit["task_id"] == ref["task_id"] and permit["reservation_id"] == reservation["reservation_id"] and permit["wall_limit_seconds"] == binding["timeout_seconds"] == 2 and permit["reserved_nanoseconds"] == "2000000000", "charge/actual bound")
        require(permit["budget_scope"]["declaration"] == cfg["execution"] and permit["budget_scope"]["database"] == str(root / "lake.db"), "permit actual route")
        used_reservations.add(reservation["reservation_id"])
        effect_dir = folder / "_execution_effects" / ref["attempt_id"]
        prepared_raw = (effect_dir / "prepared.json").read_bytes()
        prepared, committed = parse(prepared_raw), read(effect_dir / "committed.json")
        identity = {"schema_version": 1, **ref}
        require({k: prepared[k] for k in identity} == identity and prepared["event"] == "effect_prepared", "prepared identity")
        plan = {k: v for k, v in terminal.items() if k not in ("effect_receipt_sha256", "lifecycle")}
        require(prepared["plan"] == {"operation": "cpu_action_terminal", **plan}, "exact prepared terminal plan")
        require(sha(prepared_raw) == terminal["effect_receipt_sha256"] and committed == {**identity, "event": "effect_committed", "prepared_sha256": sha(prepared_raw)}, "exact committed receipt")
        # The .source-lock file permanently names the no-age namespace; only
        # the actual .lock owner directory represents an unreleased guard.
        require(not (folder / "_attempt_effect.lock").exists(), "retained uncertain effect guard")
        prepared_count += 1
        observation = one([o for o in observations.values() if o["evaluator"] == ref], "one per-attempt observation")
        artifact = one([a for a in artifacts.values() if a["producer"] == ref], "one per-attempt artifact")
        require(terminal["artifact_ids"] == [artifact["artifact_id"]] and terminal["observation_ids"] == [observation["observation_id"]], "terminal publication sets")
        require(observation["result_artifact_ids"] == [artifact["artifact_id"]], "observation actual result")
        path = Path(artifact["path"])
        require(path.is_relative_to(root), "private artifact path")
        raw = path.read_bytes()
        require(raw == record["artifact_bytes"][artifact["artifact_id"]].encode() and len(raw) == artifact["size_bytes"] and sha(raw) == artifact["content_sha256"], "actual artifact bytes")
        envelope = parse(raw)
        require(envelope["dataset_sha256"] == digest(dataset) and envelope["source_artifact_ids"] == observation["input_artifact_ids"], "envelope source/data")
        require(observation["protocol_fingerprint"] == protocol and observation["comparison_scope"] == digest({"schema": "acceptance.comparison.v1", "protocol": protocol, "dataset_sha256": digest(dataset)}), "declared comparable observation")
        for key in ("cost", "candidate", "operation", "dataset_sha256", "worker_cpu_seconds", "worker_wall_seconds"):
            require(envelope[key] == observation["values"][key], "envelope/value agreement:" + key)
        require(all(finite(envelope[k]) for k in ("worker_cpu_seconds", "worker_wall_seconds")), "finite worker timing")
        valid = independent_scientific_check(domain, dataset, envelope)
        status = observation["validation"]["status"]
        require(status in ("valid", "invalid", "unknown"), "scientific status")
        if status != "unknown":
            require(valid == (status == "valid"), "independent scientific oracle disagrees")
        elif not (envelope["candidate"] == "unchecked" and envelope["operation"] == "measure"):
            raise ValueError("unexpected unknown result")
        by_ref[digest(ref)] = (row, terminal)
        binding_by_ref[digest(ref)] = binding
        envelopes[artifact["artifact_id"]] = envelope
    require(len(used_reservations) == len(reservations), "reservation bijection")
    final = trace[-1]
    evidence = final["snapshot"]["recorded_evidence"]
    require(evidence["unavailable"] == [] and evidence["more_available"] is False, "complete evidence view")
    results = evidence["results"]
    require({digest(r["ref"]) for r in results} == set(by_ref) and len(results) == len(by_ref), "exact final Ref universe")
    for entry in trace:
        for result in entry["snapshot"]["recorded_evidence"]["results"]:
            require(digest(result["ref"]) in by_ref, "trace phantom Ref")
            for o in result["observation_records"]:
                require(observations.get(o["observation_id"]) == o, "trace changed stored observation")
            for a in result["artifact_records"]:
                require(artifacts.get(a["artifact_id"]) == a, "trace changed stored artifact")
    for result in results:
        terminal = by_ref[digest(result["ref"])][1]
        require(result["outcome"] == terminal["outcome"] and sorted(o["observation_id"] for o in result["observation_records"]) == sorted(terminal["observation_ids"]) and sorted(a["artifact_id"] for a in result["artifact_records"]) == sorted(terminal["artifact_ids"]), "final trace full publication sets")
    replica = parse(one(db["replication_requests"], "one replication request")["record_json"])
    source = one([r for r in results if r["ref"] == replica["source_ref"]], "replica source exact Ref")
    confirmation = one([r for r in results if r["ref"]["task_id"] == replica["task_id"]], "replica target")
    left, right = source["observation_records"][0], confirmation["observation_records"][0]
    require(source["ref"] != confirmation["ref"] and source["ref"]["attempt_id"] != confirmation["ref"]["attempt_id"], "distinct confirmation attempt")
    require(left["validation"]["status"] == right["validation"]["status"] == "valid", "both selected observations valid")
    for key in ("candidate", "cost", "operation", "dataset_sha256"):
        require(left["values"][key] == right["values"][key], "replica equal quality:" + key)
    require(left["protocol_fingerprint"] == right["protocol_fingerprint"] and left["comparison_scope"] == right["comparison_scope"], "replica comparability")
    signature = lambda r: sorted((a["logical_name"], a["spec_fingerprint"]) for a in r["artifact_records"])
    require(signature(source) == signature(confirmation), "replica action signature")
    require(any(t["decision"].get("kind") == "Replicate" and t["decision"]["source_ref"] == replica["source_ref"] and t["decision"]["request_id"] == replica["request_id"] for t in trace), "actual replication decision")
    require(replica["source_terminal_sha256"] == digest(by_ref[digest(source["ref"])][1]), "replica source terminal hash")
    require(replica["artifact_records_sha256"] == digest({"records": source["artifact_records"]}) and replica["observation_records_sha256"] == digest({"records": source["observation_records"]}), "replica source set hash")
    unknown_tasks = {o["evaluator"]["task_id"] for o in observations.values() if o["validation"]["status"] == "unknown"}
    for audit in pruning:
        matching = [t for t in trace if t["snapshot_sha256"] == audit["snapshot_sha256"] and t["decision"] == audit["decision"]]
        require(matching and audit["unknown_ref"]["task_id"] in unknown_tasks and audit["unknown_cost"] > audit["incumbent_cost"] and audit["validation_only_protocol"] == protocol, "honest strict pruning trace")
        require(all(r["ref"]["task_id"] != "idea-analysis" for r in matching[0]["snapshot"]["recorded_evidence"]["results"]), "phantom analysis in pruning snapshot")
    require((not pruning) if arm == "A" or variant == "counterfactual" else bool(pruning), "pruning event arm membership")
    require(parse(one(db["cpu_action_scopes"], "one scope")["stop_json"]) == STOP, "durable policy stop")
    charged = sum(int(parse(r["permit_json"])["reserved_nanoseconds"]) for r in reservations) / 1e9
    require(charged == 2 * len(attempts) and final["budget"]["remaining_wall_seconds"] == 10 - charged and final["budget"]["active_reservations"] == 0, "budget totals")
    first_valid = next(t["monotonic"] for t in trace if any(o["validation"]["status"] == "valid" for r in t["snapshot"]["recorded_evidence"]["results"] for o in r["observation_records"]))
    metrics = {"native_actions": len(attempts), "reserved_seconds": charged,
        "analysis_actions": sum(o["values"]["operation"] == "analyze" for o in observations.values()),
        "cli_wall_seconds": end - start, "first_valid_consumed_seconds": first_valid - start,
        "confirmed_selection_seconds": final["monotonic"] - start,
        "worker_cpu_seconds": sum(o["values"]["worker_cpu_seconds"] for o in observations.values()),
        "worker_wall_seconds": sum(o["values"]["worker_wall_seconds"] for o in observations.values()),
        "native_elapsed_seconds": sum(t[1]["elapsed_wall_seconds"] for t in by_ref.values())}
    require(all(near(value, record["metrics"][key]) for key, value in metrics.items()), "saved per-run metrics disagree")
    selection = {"candidate": left["values"]["candidate"], "operation": left["values"]["operation"], "cost": left["values"]["cost"], "protocol": protocol, "scope": left["comparison_scope"], "dataset_sha256": digest(dataset), "signature": signature(source)}
    return {"metrics": metrics, "selection": selection, "statuses": dict(Counter(o["validation"]["status"] for o in observations.values())), "controller_birth": ready["worker"], "native_attempts": [r["attempt_id"] for r in attempts], "prepared_and_committed_verified": prepared_count, "sources": len(artifacts)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--application", type=Path, required=True)
    args = p.parse_args()
    root, app = args.root.absolute(), args.application.absolute()
    manifest, summary = read(root / "manifest.json"), read(root / "summary.json")
    require(manifest["development_smoke"] is False, "not a formal round")
    expected = [{"repetition": rep, "domain": domain, "variant": variant, "arm": arm}
                for rep in range(6) for domain in ("sorting", "compression")
                for variant in ("default", "counterfactual")
                for arm in (("A", "B") if rep % 2 == 0 else ("B", "A"))]
    require(manifest["schedule"] == expected and len(summary["records"]) <= 48, "frozen schedule")
    require(manifest["outer_timeout_seconds"] == 30 and manifest["global_timeout_seconds"] == 900, "time budgets")
    require(manifest["source_before"] == summary["source_after"] and summary["source_unchanged"] is True, "recorded source epoch")
    for path, expected_hash in manifest["source_before"].items():
        actual = Path(path) if Path(path).is_absolute() else app / path
        require(sha(actual.read_bytes()) == expected_hash, "current source differs:" + path)
        if not Path(path).is_absolute():
            blob = subprocess.check_output(["git", "show", manifest["application_commit"] + ":" + path], cwd=app)
            require(sha(blob) == expected_hash, "fixed application Git blob differs:" + path)
    wheel = Path(manifest["core_wheel"]["path"])
    require(sha(wheel.read_bytes()) == manifest["core_wheel"]["sha256"], "original wheel SHA")
    package = Path(manifest["core_import"]).parent
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if n.startswith("orze/") and not n.endswith("/")]
        require({"orze/" + str(f.relative_to(package)) for f in package.rglob("*.py")} == {n for n in names if n.endswith(".py")}, "installed Python member set")
        require(all((package.parent / n).read_bytes() == archive.read(n) for n in names), "installed bytes vs wheel")
    require(len(list(root.glob("*/run.json"))) == len(summary["records"]), "unindexed run file")
    verified, failures, raw_refs = [], [], []
    for ordinal, pointer in enumerate(summary["records"]):
        spec = expected[ordinal]
        require(all(pointer[k] == spec[k] for k in spec), "pointer/schedule ordering")
        folder = root / (str(ordinal).zfill(2) + "-" + spec["domain"] + "-" + spec["variant"] + "-" + spec["arm"])
        path = folder / "run.json"
        require(str(path) == pointer["path"] and path.stat().st_size == pointer["bytes"] and sha(path.read_bytes()) == pointer["sha256"], "run pointer bytes/hash")
        record = read(path)
        raw_refs.append({"path": str(path), "sha256": sha(path.read_bytes())})
        try:
            result = check_run(folder, record, spec, manifest)
            require(record["quality_passed"] is True, "driver quality flag differs")
            verified.append({**spec, **result})
        except Exception as exc:
            failures.append({**spec, "path": str(path), "exit_code": record.get("exit_code"), "type": type(exc).__name__, "reason": str(exc)})
    groups, pairs = {}, []
    for domain in ("sorting", "compression"):
        for variant in ("default", "counterfactual"):
            matched = []
            for rep in range(6):
                rows = [r for r in verified if (r["domain"], r["variant"], r["repetition"]) == (domain, variant, rep)]
                arms = {r["arm"]: r for r in rows}
                equal = len(rows) == 2 and set(arms) == {"A", "B"} and arms["A"]["selection"] == arms["B"]["selection"]
                pair = {"repetition": rep, "domain": domain, "variant": variant, "quality_passed": equal}
                if equal:
                    pair["metrics"] = {k: {"A": arms["A"]["metrics"][k], "B": arms["B"]["metrics"][k], "B_minus_A": arms["B"]["metrics"][k] - arms["A"]["metrics"][k]} for k in METRICS}
                    pair["expected_actions_passed"] = (arms["A"]["metrics"]["native_actions"], arms["B"]["metrics"]["native_actions"]) == ((5, 4) if variant == "default" else (3, 3))
                    matched.append(pair)
                pairs.append(pair)
            name = domain + "_" + variant
            item = {"planned_pairs": 6, "quality_passed_pairs": len(matched)}
            if len(matched) == 6:
                item["median_paired_differences"] = {k: statistics.median(row["metrics"][k]["B_minus_A"] for row in matched) for k in METRICS}
                item["median_by_arm"] = {arm: {k: statistics.median(row["metrics"][k][arm] for row in matched) for k in METRICS} for arm in ("A", "B")}
                for key in METRICS:
                    require(near(item["median_paired_differences"][key], summary["groups"][name]["median_paired_differences"][key]), "paired median differs")
                    require(all(near(item["median_by_arm"][arm][key], summary["groups"][name]["median_by_arm"][arm][key]) for arm in ("A", "B")), "arm median differs")
            groups[name] = item
    require(len({(r["controller_birth"]["pid"], r["controller_birth"]["start_ticks"]) for r in verified}) == len(verified), "fresh controller birth reused")
    all_attempts = [a for r in verified for a in r["native_attempts"]]
    require(len(all_attempts) == len(set(all_attempts)), "cross-run attempt reuse")
    statuses = Counter()
    for r in verified:
        statuses.update(r["statuses"])
    result = {"schema": 1, "root": str(root), "application_commit": manifest["application_commit"],
        "checker_sha256": sha(Path(__file__).read_bytes()), "manifest_sha256": sha((root / "manifest.json").read_bytes()),
        "summary_sha256": sha((root / "summary.json").read_bytes()), "source_files": len(manifest["source_before"]),
        "source_and_installed_wheel_bytes_verified": True,
        "counts": {"planned_cli": 48, "actual_recorded_cli": len(summary["records"]), "unexecuted_cli": 48 - len(summary["records"]), "independent_quality_passed_cli": len(verified), "failed_checks": len(failures), "native_attempts_verified": len(all_attempts), "prepared_committed_pairs_verified": sum(r["prepared_and_committed_verified"] for r in verified), "observation_statuses": dict(statuses)},
        "groups": groups, "failures": failures, "raw_reports": raw_refs,
        "all_quality_passed": len(verified) == 48 and all(p["quality_passed"] for p in pairs),
        "action_hypothesis_passed": all(p.get("expected_actions_passed") is True for p in pairs),
        "limits": ["Independent stdlib reader/oracles; no driver/Domain/Core helper imported or worker run.", "Artifact bytes, closed final SQLite, receipt plans and captured TREE bindings are checked; this is not retrospective live pidfd observation or an independent measurement of elapsed time.", "This public finite task experiment is not an unseen holdout, independent scientific discoveries, LLM/provider/GPU benefit, or universal optimality.", "Unknown observations stay unknown; fewer analyses means less negative-result coverage."]}
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result["all_quality_passed"] and result["action_hypothesis_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
