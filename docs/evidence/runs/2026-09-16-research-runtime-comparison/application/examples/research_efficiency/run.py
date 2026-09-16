"""Preregistered paired CPU campaigns using installed Orze and real owned CLIs.

Run with the isolated core-only venv Python. Application registration is the
only injected code; no production function, license or hardware result is mocked.
"""
import argparse
import copy
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import subprocess
import sys
import time
import zipfile

APPLICATION = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPLICATION))

from examples.acceptance import compression, sorting
from examples.acceptance.common import digest
from examples.acceptance.policy import TASKS, _action_signature, _comparable, _observation

TABLES = ("ideas", "execution_attempts", "research_artifacts", "research_observations",
          "cpu_action_reservations", "cpu_action_decisions", "cpu_action_scopes",
          "cpu_proposal_requests", "replication_requests")


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def database(root):
    path = root / "lake.db"
    if not path.exists():
        return {name: [] for name in TABLES}
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {name: [dict(r) for r in conn.execute("SELECT * FROM " + name + " ORDER BY rowid")]
                if name in names else [] for name in TABLES}


def qualify(record):
    """Quality gates derive from real rows, published observations and trace."""
    if record["exit_code"] != 0 or record["error"] is not None:
        raise ValueError("CLI did not complete normally")
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    started, finished, elapsed = (record[k] for k in (
        "started_monotonic", "finished_monotonic", "wall_seconds"))
    if (not all(finite(v) for v in (started, finished, elapsed))
            or finished < started or abs((finished - started) - elapsed) > 1e-8):
        raise ValueError("inconsistent CLI clocks")
    closure = record["controller_closure"]
    if (not closure or closure["event"] != "TREE_CLOSED"
            or closure["wait_proof"] != "ECHILD_WALL"
            or closure["worker_returncode"] != 0
            or closure["binding"] != record["controller_binding"]
            or closure["stop_requested"] or closure["forced_cleanup"]):
        raise ValueError("controller tree closure unconfirmed")
    db, trace = record["database"], record["trace"]
    if not trace or trace[-1]["decision"] != {
            "kind": "Stop", "reason": "confirmed_selection", "wakeup": None}:
        raise ValueError("policy did not confirm its selection")
    if any(t["snapshot_sha256"] != digest(t["snapshot"]) for t in trace):
        raise ValueError("trace snapshot altered")
    clocks = [t["monotonic"] for t in trace]
    if (not all(finite(v) and started <= v <= finished for v in clocks)
            or clocks != sorted(clocks)):
        raise ValueError("trace clocks outside the actual ordered invocation")
    final = trace[-1]
    if (final["snapshot"]["recorded_evidence"]["unavailable"]
            or final["snapshot"]["recorded_evidence"]["more_available"]):
        raise ValueError("incomplete terminal evidence")
    attempts, reservations = db["execution_attempts"], db["cpu_action_reservations"]
    if (not attempts or len(attempts) != len(reservations)
            or len({r["attempt_id"] for r in attempts}) != len(attempts)
            or any(r["state"] != "TERMINAL" for r in attempts)
            or any(r["state"] != "SETTLED" for r in reservations)
            or any(r["status"] != "completed" for r in db["ideas"])):
        raise ValueError("unclosed attempts or reservations")
    terminals = [json.loads(r["terminal_json"]) for r in attempts]
    if any(t["outcome"] != "completed" or not t["effect_receipt_sha256"]
           or t["process_tree"]["wait_proof"] != "ECHILD_WALL"
           or t["process_tree"]["stop_requested"] or t["process_tree"]["forced_cleanup"]
           for t in terminals):
        raise ValueError("native terminal not completed and closed")
    attempts_by_ref = {}
    reservations_by_id = {r["reservation_id"]: r for r in reservations}
    if len(reservations_by_id) != len(reservations):
        raise ValueError("duplicate reservation identity")
    for attempt, terminal in zip(attempts, terminals):
        ref = {key: attempt[key] for key in ("task_id", "attempt_id", "generation", "phase")}
        binding = json.loads(attempt["binding_json"])
        reservation = reservations_by_id.get(binding["reservation_id"])
        if (reservation is None or digest(json.loads(reservation["ref_json"])) != digest(ref)
                or reservation["task_id"] != ref["task_id"]
                or reservation["terminal_sha256"] != digest(terminal)
                or digest(binding["attempt_ref"]) != digest(ref)
                or terminal["process_tree"]["binding"] != binding["supervision"]
                or digest(binding["supervision"]["identity"]["attempt_ref"]) != digest(ref)
                or not finite(terminal["elapsed_wall_seconds"])):
            raise ValueError("native closure or settled reservation mismatches full attempt")
        permit = json.loads(reservation["permit_json"])
        if (permit["reservation_id"] != reservation["reservation_id"]
                or permit["task_id"] != ref["task_id"]):
            raise ValueError("permit differs from its reservation")
        attempts_by_ref[digest(ref)] = (attempt, terminal)
    if len(db["replication_requests"]) != 1:
        raise ValueError("not exactly one real replica request")
    replica = json.loads(db["replication_requests"][0]["record_json"])
    results = final["snapshot"]["recorded_evidence"]["results"]
    by_task = {r["ref"]["task_id"]: r for r in results}
    if len(by_task) != len(results) or len(results) != len(attempts):
        raise ValueError("final trace does not contain the exact attempt universe")
    selected = by_task[replica["source_ref"]["task_id"]]
    confirmation = by_task[replica["task_id"]]
    left, right = _observation(selected), _observation(confirmation)
    if (selected["ref"] != replica["source_ref"]
            or selected["ref"]["attempt_id"] == confirmation["ref"]["attempt_id"]
            or left["validation"]["status"] != "valid"
            or right["validation"]["status"] != "valid"
            or not _comparable(left, right) or left["values"]["cost"] != right["values"]["cost"]
            or _action_signature(selected) != _action_signature(confirmation)):
        raise ValueError("selected result lacks a valid independent-attempt confirmation")
    observations = [json.loads(r["record_json"]) for r in db["research_observations"]]
    artifacts = [json.loads(r["record_json"]) for r in db["research_artifacts"]]
    recorded = {o["observation_id"]: o for o in observations}
    published_artifacts = {a["artifact_id"]: a for a in artifacts}
    if len(recorded) != len(observations) or len(published_artifacts) != len(artifacts):
        raise ValueError("duplicate published identity")
    for result in results:
        matched = attempts_by_ref.get(digest(result["ref"]))
        if matched is None or result["outcome"] != matched[1]["outcome"]:
            raise ValueError("trace full Ref does not match actual attempt")
        if (sorted(o["observation_id"] for o in result["observation_records"])
                != sorted(matched[1]["observation_ids"])
                or sorted(a["artifact_id"] for a in result["artifact_records"])
                != sorted(matched[1]["artifact_ids"])):
            raise ValueError("trace publication identities differ from terminal")
        for observation in result["observation_records"]:
            if (recorded.get(observation["observation_id"]) != observation
                    or digest(observation["evaluator"]) != digest(result["ref"])):
                raise ValueError("trace observation differs from published row")
            if not all(finite(observation["values"][k]) for k in (
                    "worker_cpu_seconds", "worker_wall_seconds")):
                raise ValueError("invalid worker timing")
        for artifact in result["artifact_records"]:
            if (published_artifacts.get(artifact["artifact_id"]) != artifact
                    or digest(artifact["producer"]) != digest(result["ref"])):
                raise ValueError("trace artifact differs from published producer")
    if len(artifacts) != len(attempts) or len(observations) != len(attempts):
        raise ValueError("finite application publication count mismatch")
    for artifact in artifacts:
        raw = record["artifact_bytes"][artifact["artifact_id"]].encode()
        if hashlib.sha256(raw).hexdigest() != artifact["content_sha256"]:
            raise ValueError("actual artifact hash mismatch")
    charged_ns = sum(int(json.loads(r["permit_json"])["reserved_nanoseconds"])
                     for r in reservations)
    if charged_ns != len(attempts) * 2_000_000_000:
        raise ValueError("actual action budget differs from preregistration")
    if (final["budget"]["active_reservations"] != 0
            or final["budget"]["remaining_wall_seconds"] != 10 - charged_ns / 1e9):
        raise ValueError("budget snapshot differs from actual settled charges")
    first_valid = next(t["monotonic"] for t in trace
        if any(o["validation"]["status"] == "valid"
               for r in t["snapshot"]["recorded_evidence"]["results"]
               for o in r["observation_records"]))
    statuses = {status: sum(o["validation"]["status"] == status for o in observations)
                for status in ("valid", "invalid", "unknown")}
    return {
        "selection": {"task_id": selected["ref"]["task_id"],
            "candidate": left["values"]["candidate"], "cost": left["values"]["cost"],
            "protocol": left["protocol_fingerprint"], "scope": left["comparison_scope"],
            "dataset_sha256": left["values"]["dataset_sha256"],
            "action_signature": _action_signature(selected)},
        "native_actions": len(attempts), "reserved_seconds": charged_ns / 1e9,
        "analysis_actions": sum(o["values"]["operation"] == "analyze" for o in observations),
        "observation_statuses": statuses,
        "cli_wall_seconds": record["wall_seconds"],
        "first_valid_consumed_seconds": first_valid - record["started_monotonic"],
        "confirmed_selection_seconds": final["monotonic"] - record["started_monotonic"],
        "worker_cpu_seconds": sum(o["values"]["worker_cpu_seconds"] for o in observations),
        "worker_wall_seconds": sum(o["values"]["worker_wall_seconds"] for o in observations),
        "native_elapsed_seconds": sum(t["elapsed_wall_seconds"] for t in terminals),
    }


def run_one(root, domain, variant, arm, *, deadline):
    from orze.engine.supervised_process import prepare_supervised, SupervisionUncertain
    import yaml
    module = {"sorting": sorting, "compression": compression}[domain]
    dataset = copy.deepcopy(module.DEFAULT_DATASET if variant == "default"
                            else module.COUNTERFACTUAL_DATASET)
    cfg = {"execution": {"version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 10},
        "results_dir": str(root / "results"), "idea_lake_db": str(root / "lake.db"),
        "ideas_file": str(root / "ideas.md"), "min_disk_gb": 0,
        "action_domain": {"version": 1, "kind": "acceptance_" + domain,
                          "config": {"dataset": dataset}},
        "action_policy": {"version": 1, "kind": "acceptance" if arm == "A" else "dominance_pruning",
                          "idle": "stop", "wait_seconds": .05,
                          "config": {} if arm == "A" else {"validation_only_protocols": [
                              digest({"schema": "acceptance.protocol.v1", "protocol": module.PROTOCOL})]}}}
    root.mkdir()
    (root / "orze.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    bootstrap = ("import sys;sys.path.insert(0," + repr(str(APPLICATION)) + ");"
                 "from examples.research_efficiency.__main__ import main;"
                 "raise SystemExit(main())")
    command = [sys.executable, "-I", "-c", bootstrap, "-c", str(root / "orze.yaml")]
    env = {"PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
           "LANG": "C.UTF-8", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    record = {"root": str(root), "domain": domain, "variant": variant, "arm": arm,
              "config": cfg, "command": command, "environment_keys": sorted(env),
              "exit_code": None, "error": None, "controller_closure": None}
    process = None
    started = time.monotonic()
    with (root / "stdout.log").open("xb") as out, (root / "stderr.log").open("xb") as err:
        try:
            try:
                process = prepare_supervised(command, identity={"scope": str(root),
                    "research_efficiency": arm}, cwd=str(root), env=env, stdout=out, stderr=err)
            except SupervisionUncertain as exc:
                process = exc.process
                raise
            record["controller_binding"] = process.binding
            save(root / "controller-ready.json", record["controller_binding"])
            if deadline - time.monotonic() < 20:
                raise TimeoutError("insufficient global budget for GO and owned cleanup")
            process.start()
            record["exit_code"] = process.wait(timeout=min(30, deadline - time.monotonic() - 10))
            record["controller_closure"] = process.closure_receipt()
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if process is not None:
                try:
                    process.stop(timeout=10)
                    record["controller_closure"] = process.closure_receipt()
                    record["exit_code"] = process.returncode
                except Exception as cleanup:
                    record["cleanup_error"] = {"type": type(cleanup).__name__, "message": str(cleanup)}
    record.update(started_monotonic=started, finished_monotonic=time.monotonic())
    record["wall_seconds"] = record["finished_monotonic"] - started
    record["stdout"] = (root / "stdout.log").read_text()
    record["stderr"] = (root / "stderr.log").read_text()
    record["trace"] = []
    record["pruning"] = []
    record["artifact_bytes"] = {}
    try:
        record["trace"] = [json.loads(line.split("=", 1)[1]) for line in record["stdout"].splitlines()
                           if line.startswith("ACCEPTANCE_DECISION=")]
        record["pruning"] = [json.loads(line.split("=", 1)[1]) for line in record["stdout"].splitlines()
                             if line.startswith("PRUNING_DECISION=")]
        record["database"] = database(root)
        for row in record["database"]["research_artifacts"]:
            artifact = json.loads(row["record_json"]); path = Path(artifact["path"])
            if not path.is_relative_to(root):
                raise ValueError("artifact path escaped private project")
            record["artifact_bytes"][artifact["artifact_id"]] = path.read_text()
        record["metrics"] = qualify(record)
        record["quality_passed"] = True
    except Exception as exc:
        record["quality_passed"] = False
        record["quality_error"] = {"type": type(exc).__name__, "message": str(exc)}
    path = save(root / "run.json", record)
    print("EFFICIENCY_RUN=" + json.dumps({"path": path, "domain": domain, "variant": variant,
        "arm": arm, "quality_passed": record["quality_passed"],
        "metrics": record.get("metrics"), "error": record.get("quality_error")}), flush=True)
    return record, path


def summarize(records):
    pairs = []
    for repetition in range(6):
        for domain in ("sorting", "compression"):
            for variant in ("default", "counterfactual"):
                matching = [r for r in records if (r["repetition"], r["domain"], r["variant"])
                            == (repetition, domain, variant)]
                arms = {r["arm"]: r for r in matching}
                quality = (len(matching) == 2 and set(arms) == {"A", "B"}
                           and all(r["quality_passed"] for r in matching)
                           and arms["A"]["metrics"]["selection"] == arms["B"]["metrics"]["selection"])
                pair = {"repetition": repetition, "domain": domain, "variant": variant,
                        "quality_passed": quality}
                if quality:
                    a, b = arms["A"]["metrics"], arms["B"]["metrics"]
                    pair["metrics"] = {key: {"A": a[key], "B": b[key], "B_minus_A": b[key] - a[key]}
                        for key in ("native_actions", "reserved_seconds", "analysis_actions",
                                    "cli_wall_seconds", "first_valid_consumed_seconds",
                                    "confirmed_selection_seconds", "worker_cpu_seconds",
                                    "worker_wall_seconds", "native_elapsed_seconds")}
                    expected = (5, 4) if variant == "default" else (3, 3)
                    pair["preregistered_action_hypothesis_passed"] = (
                        a["native_actions"], b["native_actions"]) == expected
                pairs.append(pair)
    groups = {}
    for domain in ("sorting", "compression"):
        for variant in ("default", "counterfactual"):
            group = [p for p in pairs if p["domain"] == domain and p["variant"] == variant]
            item = {"pairs": len(group), "quality_passed_pairs": sum(p["quality_passed"] for p in group)}
            # Never present a selected successful subset as the group result.
            if all(p["quality_passed"] for p in group):
                item["median_paired_differences"] = {key: statistics.median(
                    p["metrics"][key]["B_minus_A"] for p in group) for key in group[0]["metrics"]}
                item["median_by_arm"] = {arm: {key: statistics.median(p["metrics"][key][arm] for p in group)
                    for key in group[0]["metrics"]} for arm in ("A", "B")}
            groups[domain + "_" + variant] = item
    outcomes = {}
    statuses = {}
    for record in records:
        for row in record.get("database", {}).get("execution_attempts", []):
            terminal = json.loads(row["terminal_json"]) if row.get("terminal_json") else {}
            key = terminal.get("outcome", "unterminated")
            outcomes[key] = outcomes.get(key, 0) + 1
        for row in record.get("database", {}).get("research_observations", []):
            key = json.loads(row["record_json"])["validation"]["status"]
            statuses[key] = statuses.get(key, 0) + 1
    return {"pairs": pairs, "groups": groups,
            "counts": {"planned_cli": 48, "executed_cli": len(records),
                "not_executed_cli": 48 - len(records),
                "nonzero_or_unknown_exit": sum(r.get("exit_code") != 0 for r in records),
                "quality_failed_cli": sum(not r["quality_passed"] for r in records),
                "native_outcomes": outcomes, "observation_statuses": statuses},
            "all_quality_passed": len(records) == 48 and all(p["quality_passed"] for p in pairs),
            "action_hypothesis_passed": all(p.get("preregistered_action_hypothesis_passed") is True for p in pairs),
            "scope": "Fixed public CPU tasks; less validation coverage; no general scientific/LLM/GPU benefit claim"}


def capture_sources(core_package):
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others",
        "--exclude-standard", "-z", "--", "examples/acceptance", "examples/research_efficiency"],
        cwd=APPLICATION).decode().split("\0")
    result = {p: hashlib.sha256((APPLICATION / p).read_bytes()).hexdigest()
              for p in sorted(set(paths)) if p and (APPLICATION / p).is_file()}
    result.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in core_package.rglob("*.py")})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--core-wheel", type=Path, required=True,
                        help="Exact original local wheel used by both arms")
    parser.add_argument("--smoke", action="store_true",
                        help="One A/B default-sorting development pair; never formal evidence")
    args = parser.parse_args()
    root = args.output_root.absolute()
    root.mkdir()  # Refuse any prior experiment directory; never overwrite evidence.
    import orze
    if (sys.prefix == sys.base_prefix or Path(orze.__file__).is_relative_to(APPLICATION)
            or not Path(orze.__file__).is_relative_to(Path(sys.prefix))):
        raise RuntimeError("experiment requires a non-editable isolated installed Core")
    try:
        metadata.distribution("orze-pro")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("this CPU-only protocol must not load Pro or license state")
    wheel_raw = args.core_wheel.read_bytes()
    with zipfile.ZipFile(args.core_wheel) as wheel:
        members = [name for name in wheel.namelist() if name.startswith("orze/") and not name.endswith("/")]
        actual_python = {"orze/" + str(p.relative_to(Path(orze.__file__).parent))
                         for p in Path(orze.__file__).parent.rglob("*.py")}
        if not members or actual_python != {n for n in members if n.endswith(".py")}:
            raise ValueError("declared wheel does not contain this exact Core Python package")
        for name in members:
            if ".." in Path(name).parts:
                raise ValueError("unsafe wheel member")
            if (Path(orze.__file__).parent.parent / name).read_bytes() != wheel.read(name):
                raise ValueError("installed Core differs from the declared wheel")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=APPLICATION, text=True).strip()
    source_before = capture_sources(Path(orze.__file__).parent)
    if not args.smoke and subprocess.check_output(["git", "status", "--porcelain"], cwd=APPLICATION):
        raise RuntimeError("formal run requires a clean committed application checkout")
    schedule = []
    for repetition in range(6):
        for domain in ("sorting", "compression"):
            for variant in ("default", "counterfactual"):
                for arm in ("A", "B") if repetition % 2 == 0 else ("B", "A"):
                    schedule.append({"repetition": repetition, "domain": domain,
                                     "variant": variant, "arm": arm})
    if args.smoke:
        schedule = schedule[:2]
    manifest = {"version": 1, "development_smoke": args.smoke, "application_commit": commit,
        "core_version": metadata.version("orze"), "core_import": orze.__file__,
        "core_wheel": {"path": str(args.core_wheel.absolute()), "bytes": len(wheel_raw),
            "sha256": hashlib.sha256(wheel_raw).hexdigest(), "installed_members_verified": len(members)},
        "python": sys.executable, "python_version": platform.python_version(),
        "platform": platform.platform(), "schedule": schedule, "source_before": source_before,
        "dependencies": {d.metadata["Name"]: d.version for d in metadata.distributions()},
        "datasets": {name + "_" + variant: digest(getattr(module, constant))
            for name, module in (("sorting", sorting), ("compression", compression))
            for variant, constant in (("default", "DEFAULT_DATASET"), ("counterfactual", "COUNTERFACTUAL_DATASET"))},
        "outer_timeout_seconds": 30, "global_timeout_seconds": 900}
    save(root / "manifest.json", manifest)
    records, pointers = [], []
    deadline = time.monotonic() + 900
    for ordinal, spec in enumerate(schedule):
        # Leave room for preparation, wait and the exact owner's STOP/closure.
        # Unexecuted planned entries remain in the failure denominator.
        if deadline - time.monotonic() < 60:
            break
        folder = root / (str(ordinal).zfill(2) + "-" + spec["domain"] + "-" + spec["variant"] + "-" + spec["arm"])
        record, pointer = run_one(folder, spec["domain"], spec["variant"], spec["arm"], deadline=deadline)
        records.append({**spec, **record})
        pointers.append({**spec, **pointer})
        if record.get("cleanup_error") or record["controller_closure"] is None:
            break  # Unknown owner is not permission to continue launching.
    source_after = capture_sources(Path(orze.__file__).parent)
    summary = summarize(records) if not args.smoke else {
        "development_smoke": True, "quality_passed": all(r["quality_passed"] for r in records),
        "run_count": len(records)}
    summary.update(records=pointers, source_after=source_after,
                   source_unchanged=source_after == source_before)
    save(root / "summary.json", summary)
    print("EFFICIENCY_SUMMARY=" + json.dumps({"path": str(root / "summary.json"),
        "development_smoke": args.smoke, "runs": len(records),
        "source_unchanged": summary["source_unchanged"],
        "all_quality_passed": summary.get("all_quality_passed", summary.get("quality_passed"))}), flush=True)
    return 0 if summary["source_unchanged"] and summary.get(
        "all_quality_passed", summary.get("quality_passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
