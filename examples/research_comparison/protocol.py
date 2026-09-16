"""Content-addressed comparison plans, independent of product stopping policy.

A digest fixes bytes; it does not establish when a plan was registered, prove
that its inputs were used, or authorize a model invocation. A workload adapter
must verify those facts against its own raw execution and evaluation evidence.
"""
import hashlib
import json
import math
import re


LIMITS = ("provider_calls", "provider_tokens", "provider_cost_usd",
          "reserved_seconds", "gpu_seconds", "cli_wall_seconds")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(label + " has missing or unknown fields")


def number(value, *, integer=False):
    if type(value) is int:
        return 0 <= value <= 2**63 - 1
    return (not integer and type(value) is float and math.isfinite(value)
            and 0 <= value <= 2**63 - 1)


def read_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def hashes(value, expected, label):
    keys(value, expected, label)
    if any(not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None
           for v in value.values()):
        raise ValueError(label + " requires SHA-256 values")


def identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,95}", value) is not None


def validate_protocol(plan):
    keys(plan, ("schema", "comparison_id", "mode", "repetitions", "ordering", "arms", "shared",
                "verifier_sha256", "tasks"), "protocol")
    if type(plan["schema"]) is not int or plan["schema"] != 1:
        raise ValueError("unsupported protocol schema")
    if not identifier(plan["comparison_id"]):
        raise ValueError("invalid comparison identifier")
    if plan["mode"] not in ("prospective", "retrospective_offline"):
        raise ValueError("unknown comparison mode")
    if plan["ordering"] not in ("repetition", "task_and_repetition"):
        raise ValueError("unknown paired order")
    count = plan["repetitions"]
    if not number(count, integer=True) or not 1 <= count <= 1000:
        raise ValueError("repetitions must be an integer in 1..1000")
    keys(plan["arms"], ("A", "B"), "arms")
    for arm in plan["arms"].values():
        hashes(arm, ("artifact_sha256", "treatment_sha256"), "arm")
    hashes(plan["shared"], ("model", "tools", "environment"), "shared inputs")
    hashes({"verifier": plan["verifier_sha256"]}, ("verifier",), "verifier")
    tasks = plan["tasks"]
    if not isinstance(tasks, list) or not 2 <= len(tasks) <= 128:
        raise ValueError("comparison requires target and negative-control tasks")
    names, roles = set(), set()
    for task in tasks:
        keys(task, ("id", "domain", "role", "seeds", "inputs", "quality", "limits"), "task")
        if not identifier(task["id"]) or task["id"] in names or not identifier(task["domain"]):
            raise ValueError("invalid or duplicate task identity")
        names.add(task["id"])
        if task["role"] not in ("target", "negative_control"):
            raise ValueError("unknown task role")
        roles.add(task["role"])
        seeds = task["seeds"]
        deterministic_history = (plan["mode"] == "retrospective_offline"
                                 and isinstance(seeds, list) and seeds == [None] * count)
        if (not isinstance(seeds, list) or len(seeds) != count
                or not deterministic_history and (
                    not all(number(v, integer=True) for v in seeds) or len(set(seeds)) != count)):
            raise ValueError("each repetition needs a distinct nonnegative integer seed")
        hashes(task["inputs"], ("data", "evaluator", "instructions", "initial_history", "initial_memory"),
               "task inputs")
        quality = task["quality"]
        keys(quality, ("direction", "max_regression", "minimum_valid_observations"), "quality")
        if (quality["direction"] not in ("minimize", "maximize")
                or not number(quality["max_regression"])
                or not number(quality["minimum_valid_observations"], integer=True)):
            raise ValueError("invalid quality gate")
        keys(task["limits"], LIMITS, "limits")
        for name, value in task["limits"].items():
            if value is None and plan["mode"] == "retrospective_offline":
                continue  # Historical unmeasured cost remains explicitly unknown.
            if not number(value, integer=name in ("provider_calls", "provider_tokens")):
                raise ValueError("every prospective limit must be explicit: " + name)
    if roles != {"target", "negative_control"}:
        raise ValueError("a negative control and a target are both required")
    return plan


def schedule(plan):
    validate_protocol(plan)
    result = []
    for repetition in range(plan["repetitions"]):
        for index, task in enumerate(plan["tasks"]):
            offset = index if plan["ordering"] == "task_and_repetition" else 0
            order = ("A", "B") if (repetition + offset) % 2 == 0 else ("B", "A")
            for arm in order:
                result.append({"run_id": f"{task['id']}-{repetition:04d}-{arm}",
                               "task_id": task["id"], "repetition": repetition,
                               "seed": task["seeds"][repetition], "arm": arm})
    return result
