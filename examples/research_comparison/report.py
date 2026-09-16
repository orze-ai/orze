"""Paired accounting over adapter-derived measurements, never cached verdicts.

Adapters are trusted application code. They must verify raw sources, input and
execution identities, current qualification, independent confirmation, clocks,
all attempts/retries and actual usage. This reducer provides no such authority
on its own, and has no launch, network, credential or process-control interface.
"""
import copy
import math
import statistics

from .protocol import LIMITS, digest, keys, number, schedule


METRICS = ("native_actions", "reserved_seconds", "analysis_actions", "cli_wall_seconds",
           "first_valid_consumed_seconds", "confirmed_selection_seconds", "worker_cpu_seconds",
           "worker_wall_seconds", "native_elapsed_seconds", "provider_calls", "provider_tokens",
           "provider_cost_usd", "gpu_seconds")
COUNTS = frozenset(("native_actions", "analysis_actions", "provider_calls", "provider_tokens"))


def _measured(value):
    keys(value, ("status", "metrics", "quality", "observations", "native_outcomes"), "measurement")
    if value["status"] not in ("completed", "failed", "unknown"):
        raise ValueError("invalid run status")
    keys(value["metrics"], METRICS, "metrics")
    for name, actual in value["metrics"].items():
        if actual is not None and not number(actual, integer=name in COUNTS):
            raise ValueError("invalid metric: " + name)
    quality = value["quality"]
    keys(quality, ("valid", "confirmed", "score", "comparison_key"), "quality result")
    if type(quality["valid"]) is not bool or type(quality["confirmed"]) is not bool:
        raise ValueError("quality verdicts must be booleans")
    score = quality["score"]
    if score is not None and (type(score) not in (int, float)
                              or not -(2**63 - 1) <= score <= 2**63 - 1):
        raise ValueError("quality score must be finite or unknown")
    if not isinstance(quality["comparison_key"], dict) or not quality["comparison_key"]:
        raise ValueError("quality needs a nonempty comparison identity")
    digest(quality["comparison_key"])
    keys(value["observations"], ("valid", "invalid", "unknown"), "observations")
    if (not isinstance(value["native_outcomes"], dict) or not value["native_outcomes"]
            or any(not isinstance(k, str) or not k for k in value["native_outcomes"])):
        raise ValueError("native outcomes must retain their observed categories")
    for counts in (value["observations"], value["native_outcomes"]):
        if not all(number(v, integer=True) for v in counts.values()):
            raise ValueError("invalid observation or outcome count")
    wall = value["metrics"]["cli_wall_seconds"]
    first = value["metrics"]["first_valid_consumed_seconds"]
    confirmed = value["metrics"]["confirmed_selection_seconds"]
    if (wall is not None and any(v is not None and v > wall for v in (first, confirmed))
            or first is not None and confirmed is not None and first > confirmed):
        raise ValueError("measurement clocks are out of order")
    return copy.deepcopy(value)


def _run(slot, raw, task, arm, verify):
    result = dict(slot)
    if raw is None:
        return {**result, "status": "missing", "qualified": False,
                "measurement_error": "missing_run"}
    try:
        measured = _measured(verify(copy.deepcopy(raw), copy.deepcopy(task), arm))
    except Exception as exc:
        # Keep the planned slot. Do not copy unverified caller metrics, exception
        # text (which may include provider data), or a precomputed success flag.
        return {**result, "status": "unknown", "qualified": False,
                "measurement_error": type(exc).__name__}
    result.update(measured)
    checks = {}
    for name, limit in task["limits"].items():
        actual = measured["metrics"][name]
        checks[name] = ("not_specified" if limit is None else "unknown" if actual is None
                        else "passed" if actual <= limit else "exceeded")
    quality = measured["quality"]
    result["budget_checks"] = checks
    result["qualified"] = (
        measured["status"] == "completed" and quality["valid"] and quality["confirmed"]
        and quality["score"] is not None
        and measured["observations"]["valid"] >= task["quality"]["minimum_valid_observations"]
        and all(v in ("passed", "not_specified") for v in checks.values()))
    return result


def compare(plan, records, *, verify):
    """Keep the full planned denominator and independently unknown cost columns.

``verify(raw, task, arm)`` must derive the normalized measurement from original
evidence, or raise. Its implementation and dependencies belong in the frozen
protocol's verifier hash. Merely returning a caller's ``quality_passed`` field
is not a valid workload adapter. Input argument objects are not modified.
"""
    planned = schedule(plan)
    slots = {item["run_id"]: item for item in planned}
    bound = {}
    for raw in records:
        if not isinstance(raw, dict) or raw.get("run_id") not in slots or raw["run_id"] in bound:
            raise ValueError("duplicate or unplanned run")
        slot = slots[raw["run_id"]]
        if (any(type(raw.get(k)) is not type(v) or raw[k] != v for k, v in slot.items())
                or raw.get("protocol_sha256") != digest(plan)):
            raise ValueError("run identity or protocol digest mismatch")
        bound[raw["run_id"]] = raw
    if list(bound) != [slot["run_id"] for slot in planned if slot["run_id"] in bound]:
        raise ValueError("provided run order differs from the declared schedule")
    tasks = {task["id"]: task for task in plan["tasks"]}
    runs = [_run(slot, bound.get(slot["run_id"]), tasks[slot["task_id"]], slot["arm"], verify)
            for slot in planned]
    lookup = {(r["task_id"], r["repetition"], r["arm"]): r for r in runs}
    pairs = []
    for repetition in range(plan["repetitions"]):
        for task in plan["tasks"]:
            a, b = (lookup[(task["id"], repetition, arm)] for arm in ("A", "B"))
            pair = {"task_id": task["id"], "repetition": repetition, "seed": task["seeds"][repetition],
                    "qualified": False}
            if a["qualified"] and b["qualified"]:
                qa, qb = a["quality"], b["quality"]
                regression = qb["score"] - qa["score"]
                if task["quality"]["direction"] == "maximize":
                    regression = -regression
                pair["quality_regression"] = regression
                pair["qualified"] = (digest(qa["comparison_key"]) == digest(qb["comparison_key"])
                                     and regression <= task["quality"]["max_regression"])
            if "observations" in a and "observations" in b:
                pair["observation_differences"] = {k: b["observations"][k] - a["observations"][k]
                                                   for k in ("valid", "invalid", "unknown")}
            if pair["qualified"]:
                pair["metrics"] = {name: {"A": a["metrics"][name], "B": b["metrics"][name],
                                         "B_minus_A": b["metrics"][name] - a["metrics"][name]}
                                   for name in METRICS if a["metrics"][name] is not None
                                   and b["metrics"][name] is not None}
            pairs.append(pair)
    groups = {}
    for task in plan["tasks"]:
        group = [p for p in pairs if p["task_id"] == task["id"]]
        summary = {"domain": task["domain"], "role": task["role"], "planned_pairs": len(group),
                   "qualified_pairs": sum(p["qualified"] for p in group), "cost_totals": {}}
        for arm in ("A", "B"):
            selected = [r for r in runs if r["task_id"] == task["id"] and r["arm"] == arm]
            totals = {}
            for metric in METRICS:
                values = [r.get("metrics", {}).get(metric) for r in selected]
                known = [v for v in values if v is not None]
                totals[metric] = {"known_sum": sum(known), "known_runs": len(known),
                                  "missing_runs": len(values) - len(known), "complete": len(known) == len(values)}
            summary["cost_totals"][arm] = totals
        if all(p["qualified"] for p in group):
            available = [k for k in METRICS if all(k in p["metrics"] for p in group)]
            summary["median_paired_differences"] = {
                k: statistics.median(p["metrics"][k]["B_minus_A"] for p in group) for k in available}
            summary["median_by_arm"] = {arm: {
                k: statistics.median(p["metrics"][k][arm] for p in group) for k in available}
                for arm in ("A", "B")}
        groups[task["id"]] = summary
    return {"schema": 1, "protocol_sha256": digest(plan), "comparison_id": plan["comparison_id"],
            "mode": plan["mode"], "new_research_evidence": False,
            "scope": "Offline reduction of adapter-verified measurements; no new execution or general research claim",
            "counts": {"planned_runs": len(planned), "provided_runs": len(bound),
                       "missing_runs": len(planned) - len(bound)},
            "all_pairs_qualified": all(p["qualified"] for p in pairs),
            "runs": runs, "pairs": pairs, "groups": groups}
