"""Branch-local discovery and prefix-only replay, inspired by Dream-RSI.

Policies are trusted Python objects with ``decide(view)``. The same decision
loop drives a caller's existing batch executor or reads saved outcomes. This
module supplies no process runner, model provider, execution permission or
sandbox. Executors must restore the supplied branch context and retain their
usual admission, artifact qualification and resource checks.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def detached(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _number(value, name, minimum=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (minimum is not None and value < minimum)):
        raise ValueError(f"invalid {name}")


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def validate_spec(spec):
    """Fixed task/model/evaluator/context identity plus explicit exploration bounds.

``score`` is higher-is-better; ``score_scale`` is fixed before exploration,
not estimated from a trace's eventual maximum. ``problem_id`` groups related
rollouts for held-out validation. Protocol identity must include the generator,
evaluator, data and fixed shared context supplied by the integration.
"""
    spec = detached(spec)
    if set(spec) != {"problem_id", "protocol_id", "root", "plan", "score_scale"}:
        raise ValueError("invalid exploration specification fields")
    for key in ("problem_id", "protocol_id"):
        if type(spec[key]) is not str or not spec[key].strip():
            raise ValueError(f"invalid {key}")
    root = spec["root"]
    if type(root) is not dict or set(root) != {"score", "artifact", "feedback"}:
        raise ValueError("root requires score, artifact and feedback")
    _number(root["score"], "root score")
    if any(type(root[k]) is not dict for k in ("artifact", "feedback")):
        raise ValueError("root artifact and feedback must be objects")
    _number(spec["score_scale"], "score scale")
    if spec["score_scale"] <= 0:
        raise ValueError("score scale must be positive")
    plan = spec["plan"]
    if type(plan) is not dict or set(plan) != {"branches", "depth", "calls", "workers"}:
        raise ValueError("plan requires branches, depth, calls and workers")
    for key, value in plan.items():
        if type(value) is not int or not 1 <= value <= 100000:
            raise ValueError(f"invalid plan {key}")
    return spec


def _action(branch, step):
    return {"id": f"b{branch}-s{step}", "branch": branch, "step": step,
            "parent": f"b{branch}-s{step - 1}" if step else None}


def _view(spec, nodes):
    plan = spec["plan"]
    legal = []
    for branch in range(plan["branches"]):
        step = sum(n["branch"] == branch for n in nodes)
        if step < plan["depth"]:
            legal.append(_action(branch, step))
    return detached({"root": spec["root"], "score_scale": spec["score_scale"],
                     "observed": nodes, "legal": legal,
                     "remaining_calls": plan["calls"] - len(nodes),
                     "workers": plan["workers"]})


def _decision(policy, view):
    decision = detached(policy.decide(detached(view)))
    if (type(decision) is not dict or set(decision) != {"actions", "reason"}
            or type(decision["actions"]) is not list
            or any(type(x) is not str for x in decision["actions"])
            or type(decision["reason"]) is not str or not decision["reason"].strip()):
        raise ValueError("policy must return actions and a reason")
    ids = decision["actions"]
    legal = {a["id"] for a in view["legal"]}
    if (len(ids) != len(set(ids)) or not set(ids) <= legal
            or len(ids) > min(view["workers"], view["remaining_calls"])):
        raise ValueError("policy batch exceeds the revealed frontier or budget")
    return decision


def _outcome(value):
    value = detached(value)
    if type(value) is not dict or set(value) != {
            "score", "status", "feedback", "artifact", "cost", "seconds"}:
        raise ValueError("invalid attempt outcome fields")
    if value["status"] not in ("ok", "repairable", "blocked"):
        raise ValueError("invalid attempt status")
    if value["status"] == "ok" and value["score"] is not None:
        _number(value["score"], "score")
    elif value["status"] != "ok" and value["score"] is not None:
        raise ValueError("a failed attempt cannot provide a quality score")
    for key in ("feedback", "artifact"):
        if type(value[key]) is not dict:
            raise ValueError(f"{key} must be an object")
    for key in ("cost", "seconds"):
        if value[key] is not None:
            _number(value[key], key, 0)
    return value


def _contexts(spec, nodes, actions):
    # No sibling outcomes, global leader or mutable shared conversation. A
    # parent's complete path includes failed attempts and their artifacts.
    return [{"action": a, "root": spec["root"],
             "history": [n for n in nodes if n["branch"] == a["branch"]]}
            for a in actions]


def _rollout(spec, policy, transition):
    nodes, rounds = [], []
    reason = "budget"
    stop_reason = None
    while len(nodes) < spec["plan"]["calls"]:
        view = _view(spec, nodes)
        if not view["legal"]:
            reason = "grid_complete"
            break
        decision = _decision(policy, view)
        if not decision["actions"]:
            reason = "policy_stop"
            stop_reason = decision["reason"]
            break
        actions = [{a["id"]: a for a in view["legal"]}[i] for i in decision["actions"]]
        # All contexts are captured before any result in the batch is visible.
        contexts = detached(_contexts(spec, nodes, actions))
        outcomes = transition(contexts, len(rounds))
        if outcomes is None:
            rounds.append({"decision": decision, "observations": [], "supported": False})
            reason = "out_of_support"
            break
        if type(outcomes) is not dict or set(outcomes) != set(decision["actions"]):
            raise ValueError("executor must return exactly one outcome per requested action")
        new_nodes = [{**a, **_outcome(outcomes[a["id"]])} for a in actions]
        rounds.append({"decision": decision, "observations": new_nodes, "supported": True})
        nodes.extend(new_nodes)
    best = max([spec["root"]["score"]] + [n["score"] for n in nodes if n["score"] is not None])
    return {"schema": 1, "mode": "online", "spec": spec, "rounds": rounds,
            "termination": reason, "stop_reason": stop_reason,
            "metrics": {"supported": reason != "out_of_support", "best_score": best,
                        "quality_gain": (best - spec["root"]["score"]) / spec["score_scale"],
                        "calls": len(nodes), "rounds": len(rounds),
                        "cost": (sum(n["cost"] for n in nodes)
                                 if all(n["cost"] is not None for n in nodes) else None),
                        # A scheduling estimate, not a measured counterfactual
                        # wall clock. Executor/provider contention may differ.
                        "parallel_seconds_estimate": (sum(max(
                            (n["seconds"] for n in r["observations"]), default=0)
                            for r in rounds) if all(n["seconds"] is not None for n in nodes) else None)}}


def run_online(spec, policy=None, execute_batch=None, output=None):
    """Drive an existing executor and archive one non-resumable discovery world.

The output directory is exclusively created BEFORE any execution. A failed or
interrupted directory cannot be reused. ``execute_batch(contexts)`` returns an
id -> outcome mapping; it owns actual execution, model calls, and authorization.
Exceptions stop this rollout without retrying. Each request is saved before
dispatch, so an uncertain call is never silently counted as free or replayed.
Omitting ``policy`` uses the Dream-RSI portfolio bootstrap. A replay-selected
policy can replace it explicitly on the next rollout. The executor and output
remain required; selecting a default never grants execution authority.
"""
    if not callable(execute_batch) or output is None:
        raise ValueError("execute_batch and a fresh output directory are required")
    if policy is None:
        from orze.research.exploration_policies import Portfolio
        policy = Portfolio()
    spec = validate_spec(spec)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "spec.json", spec)

    def transition(contexts, index):
        _write(output / f"request-{index:04d}.json", contexts)
        result = execute_batch(detached(contexts))
        _write(output / f"response-{index:04d}.json", result)
        return result

    try:
        trace = _rollout(spec, policy, transition)
        _write(output / "trace.json", trace)
        return detached(trace)
    except BaseException as exc:
        _write(output / "failure.json", {"type": type(exc).__name__, "retry_allowed": False})
        raise


def validate_trace(trace):
    """Reconstruct the recorded decisions, including batch visibility and metrics."""
    trace = detached(trace)
    if (type(trace) is not dict or set(trace) != {
            "schema", "mode", "spec", "rounds", "termination", "stop_reason", "metrics"}
            or type(trace["schema"]) is not int or trace["schema"] != 1
            or trace["mode"] != "online"
            or type(trace["rounds"]) is not list
            or trace["termination"] == "out_of_support"):
        raise ValueError("replay requires a complete online trace")
    spec = validate_spec(trace["spec"])
    index = 0

    class Recorded:
        def decide(self, view):
            if index == len(trace["rounds"]):
                return {"actions": [], "reason": trace["stop_reason"]}
            return trace["rounds"][index]["decision"]

    def transition(contexts, _):
        nonlocal index
        row = trace["rounds"][index]
        if set(row) != {"decision", "observations", "supported"} or row["supported"] is not True:
            raise ValueError("invalid recorded batch")
        observations = row["observations"]
        if (type(observations) is not list or len(observations) != len(contexts)
                or [n["id"] for n in observations] != [c["action"]["id"] for c in contexts]):
            raise ValueError("recorded batch identities differ")
        outcomes = {}
        for c, n in zip(contexts, observations):
            if any(n[k] != v for k, v in c["action"].items()):
                raise ValueError("recorded ancestry differs")
            outcomes[n["id"]] = {k: v for k, v in n.items() if k not in c["action"]}
        index += 1
        return outcomes

    rebuilt = _rollout(spec, Recorded(), transition)
    if rebuilt != trace or index != len(trace["rounds"]):
        raise ValueError("trace metrics, decisions or termination do not reconstruct")
    return trace


def replay(trace, policy, *, calls=None):
    """Reveal recorded outcomes only after decisions; never synthesize missing cells.

Legal actions come from the declared grid, NOT the hidden tree's coverage. An
unrecorded continuation makes the rollout unsupported; it cannot earn a policy
selection score. Scores/work before that point remain available for diagnosis.
"""
    trace = validate_trace(trace)
    outcomes = {n["id"]: {k: v for k, v in n.items() if k not in _action(n["branch"], n["step"])}
                for row in trace["rounds"] for n in row["observations"]}

    def transition(contexts, _):
        ids = [c["action"]["id"] for c in contexts]
        if any(i not in outcomes for i in ids):
            return None
        return {i: outcomes[i] for i in ids}

    spec = detached(trace["spec"])
    if calls is not None:
        if type(calls) is not int or not 1 <= calls <= spec["plan"]["calls"]:
            raise ValueError("replay call cap must be within the recorded declaration")
        spec["plan"]["calls"] = calls
    result = _rollout(spec, policy, transition)
    result["mode"] = "replay"
    return result
