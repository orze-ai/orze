"""Read-only adapter for the already published 2026-09-12 CPU comparison.

These exact public records are controls for the measurement code, not a new
task set, replication, model run, deployment or prospective registration.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from examples.research_efficiency.run import qualify
from .protocol import digest, schedule
from .report import METRICS, compare


PINS = {
    "formal-01.tar.gz": "f712ade45710619d8c15e6b9a43a8971e7384514b528c69f561bef58a762ebd1",
    "manifest.json": "b0a165ab2311deeb51ba4fd9c7b7200706683c495c0b3a55178e8d87f968c819",
    "summary.json": "4c586c565501fe45766c71ae827c44d121f4732adc9840660689fb071ac3e72e",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_acceptance(record, task, arm):
    raw = record["raw"]
    if raw["arm"] != arm:
        raise ValueError("historical arm identity mismatch")
    measured = qualify(raw)  # Recompute from attempts, observations, artifacts and trace.
    if task is not None:
        if (task["id"] != raw["domain"] + "_" + raw["variant"]
                or measured["selection"]["dataset_sha256"] != task["inputs"]["data"]):
            raise ValueError("historical task or data identity mismatch")
    metrics = {name: measured.get(name) for name in METRICS}
    # This experiment has no provider usage ledger. Do not manufacture measured
    # zero tokens/dollars/GPU-seconds from its CPU-only declared execution scope.
    outcomes = {}
    for row in raw["database"]["execution_attempts"]:
        outcome = json.loads(row["terminal_json"])["outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    selection = dict(measured["selection"])
    score = selection.pop("cost")
    return {"status": "completed", "metrics": metrics,
            "quality": {"valid": True, "confirmed": True, "score": score,
                        "comparison_key": selection},
            "observations": measured["observation_statuses"], "native_outcomes": outcomes}


def replay(evidence_dir):
    """Re-read hash-pinned archived reports without extracting or executing them."""
    folder = Path(evidence_dir)
    inputs = {}
    for name, expected in PINS.items():
        raw = (folder / name).read_bytes()
        if sha(raw) != expected:
            raise ValueError("historical input differs: " + name)
        if name.endswith(".json"):
            inputs[name] = json.loads(raw)
    old = inputs["manifest.json"]
    previous = inputs["summary.json"]
    if old["source_before"] != previous["source_after"] or not previous["source_unchanged"]:
        raise ValueError("historical source was not frozen")
    application = Path(__file__).resolve().parents[2]
    dependencies = ("examples/research_efficiency/run.py", "examples/acceptance/policy.py",
                    "examples/acceptance/common.py", "examples/research_comparison/protocol.py",
                    "examples/research_comparison/report.py", "examples/research_comparison/legacy.py")
    verifier_files = {name: sha((application / name).read_bytes()) for name in dependencies}
    plan = {
        "schema": 1, "comparison_id": "historical-cpu-20260912-reanalysis",
        "mode": "retrospective_offline", "repetitions": 6, "ordering": "repetition",
        "arms": {arm: {"artifact_sha256": old["core_wheel"]["sha256"],
                       "treatment_sha256": digest({"source": old["source_before"], "arm": arm})}
                 for arm in ("A", "B")},
        "shared": {"model": digest({"provider": "none", "scope": "historical_cpu"}),
                   "tools": digest(old["source_before"]),
                   "environment": digest({k: old[k] for k in (
                       "python_version", "platform", "dependencies")})},
        "verifier_sha256": digest(verifier_files), "tasks": [],
    }
    for domain in ("sorting", "compression"):
        for variant in ("default", "counterfactual"):
            name = domain + "_" + variant
            plan["tasks"].append({
                "id": name, "domain": domain,
                "role": "target" if variant == "default" else "negative_control",
                "seeds": [None] * 6,  # Fixed deterministic data, no invented RNG seed.
                "inputs": {"data": old["datasets"][name],
                           "evaluator": old["source_before"]["examples/acceptance/" + domain + ".py"],
                           "instructions": digest({"protocol": PINS["manifest.json"], "task": name}),
                           "initial_history": digest([]), "initial_memory": digest(None)},
                "quality": {"direction": "minimize", "max_regression": 0,
                            "minimum_valid_observations": 2},
                "limits": {"provider_calls": None, "provider_tokens": None, "provider_cost_usd": None,
                           "reserved_seconds": 10, "gpu_seconds": None, "cli_wall_seconds": 30},
            })
    slots = {(s["task_id"], s["repetition"], s["arm"]): s for s in schedule(plan)}
    records, pointers = [], []
    with tarfile.open(folder / "formal-01.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        by_name = {m.name: m for m in members}
        if len(by_name) != len(members):
            raise ValueError("duplicate archive member")
        for ordinal, pointer in enumerate(previous["records"]):
            spec = {k: pointer[k] for k in ("domain", "variant", "repetition", "arm")}
            if spec != old["schedule"][ordinal]:
                raise ValueError("record order differs from the original registered schedule")
            path = PurePosixPath(pointer["path"])
            name = str(PurePosixPath("formal-01") / path.parent.name / "run.json")
            member = by_name.get(name)
            if member is None or not member.isfile() or member.size != pointer["bytes"]:
                raise ValueError("missing or unexpected historical record")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("historical record unreadable")
            with stream:
                raw = stream.read()
            if sha(raw) != pointer["sha256"]:
                raise ValueError("historical record hash mismatch")
            value = json.loads(raw)
            if any(value[k] != spec[k] for k in ("domain", "variant", "arm")):
                raise ValueError("historical record identity mismatch")
            slot = slots[(spec["domain"] + "_" + spec["variant"], spec["repetition"], spec["arm"])]
            records.append({**slot, "protocol_sha256": digest(plan), "raw": value})
            pointers.append({"run_id": slot["run_id"], "member": name, "sha256": pointer["sha256"]})
    result = compare(plan, records, verify=verify_acceptance)
    matched = True
    for name, original in previous["groups"].items():
        group = result["groups"][name]
        for field in ("median_by_arm", "median_paired_differences"):
            matched = matched and group.get(field) == original.get(field)
    if len(records) != 48 or not result["all_pairs_qualified"] or not matched:
        raise ValueError("reanalysis disagrees with the original 24-pair result")
    result["historical_summary_exact"] = matched
    result["provenance"] = {"historical_inputs": PINS, "verifier_files": verifier_files,
                            "records": pointers, "original_application_commit": old["application_commit"],
                            "new_cpu_executions": 0, "new_provider_calls": 0,
                            "seed_note": "Fixed historical tasks used no RNG seed; null is not an unknown random seed"}
    # Re-read all three immutable input byte streams before publishing a report.
    for name, expected in PINS.items():
        if sha((folder / name).read_bytes()) != expected:
            raise ValueError("historical input changed during read")
    return plan, result
