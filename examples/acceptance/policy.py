"""Finite evidence-driven example policy, deliberately not a Core default.

The initial three-candidate exploration order is an explicit example prior.
Selection uses recorded valid comparable costs, never domain names or expected
numbers. A replica repeats the selected observation-producing action: repeating
an analysis is not a fresh run of its input algorithm or statistical independence.
"""
import copy
import json
import time

from .common import OUTPUTS, digest

TASKS = {"baseline": "idea-baseline", "challenger": "idea-challenger",
         "unchecked": "idea-unchecked", "analysis": "idea-analysis"}


def request(candidate, operation="measure", sources=()):
    return {"version": 1, "purpose": operation + " candidate " + candidate,
        "inputs": {}, "timeout_seconds": 2, "outputs": copy.deepcopy(OUTPUTS),
        "input_artifact_ids": list(sources),
        "payload": {"candidate": candidate, "operation": operation}}


def _observation(result):
    observations = result["observation_records"]
    if len(observations) != 1 or observations[0]["name"] != "objective":
        raise ValueError("finite acceptance policy requires one explicit objective")
    observation = observations[0]
    if (type(observation["values"].get("cost")) is not int
            or observation["values"]["cost"] < 0
            or observation["validation"]["status"] not in ("valid", "invalid", "unknown")):
        raise ValueError("finite acceptance objective is malformed")
    return observation


def _comparable(left, right):
    return (left["protocol_fingerprint"] == right["protocol_fingerprint"]
        and left["comparison_scope"] == right["comparison_scope"]
        and left["values"]["dataset_sha256"] == right["values"]["dataset_sha256"])


def _action_signature(result):
    # Artifact specification hashes bind the materialized action, including
    # argv, inputs, source bindings and output contract. IDs differ per run.
    return sorted((item["logical_name"], item["spec_fingerprint"])
                  for item in result["artifact_records"])


class CommonPolicy:
    def __init__(self, declaration):
        if declaration.get("config", {}) != {}:
            raise ValueError("the frozen acceptance policy has no domain parameters")
        self.declaration = copy.deepcopy(declaration)

    def _proposal(self, key, declared):
        return {"kind": "Propose", "request_id": "plan-" + key,
            "task_id": TASKS[key], "reason": "obtain the next explicitly required evidence",
            "domain_request": declared}

    def _decide(self, snapshot, budget):
        view = snapshot["recorded_evidence"]
        if view["unavailable"] or view["more_available"]:
            raise ValueError("incomplete evidence is not scientific invalidity or convergence")
        if budget["stopped"]:
            return {"kind": "Stop", "reason": "scope_stopped", "wakeup": None}
        results = view["results"]
        if any(item["outcome"] != "completed" for item in results):
            return {"kind": "Stop", "reason": "execution_did_not_complete", "wakeup": None}
        by_task = {item["ref"]["task_id"]: item for item in results}
        if len(by_task) != len(results):
            raise ValueError("ambiguous result occurrence")
        for result in results:
            _observation(result)
        outcomes = snapshot["recorded_proposals"]["results"]
        if any(item["status"] not in ("inserted", "already_present_exact") for item in outcomes):
            return {"kind": "Stop", "reason": "proposal_not_admitted", "wakeup": None}
        # Every queued task still enters the ordinary Core claim/budget path.
        # This application neither creates task rows nor launches a worker.
        if snapshot["queue"]:
            if len(snapshot["queue"]) != 1:
                raise ValueError("finite sequential acceptance project has unexpected work")
            item = snapshot["queue"][0]
            if item["request"]["timeout_seconds"] > budget["remaining_wall_seconds"]:
                return {"kind": "Stop", "reason": "insufficient_example_budget", "wakeup": None}
            return {"kind": "Execute", "task_id": item["idea_id"]}
        if snapshot["active"] or budget["active_reservations"]:
            return {"kind": "Wait", "reason": "wait_for_actual_terminal",
                    "wakeup": snapshot["now"] + self.declaration["wait_seconds"]}
        baseline = by_task.get(TASKS["baseline"])
        if baseline is None:
            return self._proposal("baseline", request("baseline"))
        base_observation = _observation(baseline)
        if base_observation["validation"]["status"] != "valid":
            return {"kind": "Stop", "reason": "baseline_not_valid", "wakeup": None}
        challenger = by_task.get(TASKS["challenger"])
        if challenger is None:
            return self._proposal("challenger", request("challenger"))
        challenge_observation = _observation(challenger)
        challenge_better = (challenge_observation["validation"]["status"] == "valid"
            and _comparable(base_observation, challenge_observation)
            and challenge_observation["values"]["cost"] < base_observation["values"]["cost"])
        if not challenge_better:
            unchecked = by_task.get(TASKS["unchecked"])
            if unchecked is None:
                return self._proposal("unchecked", request("unchecked"))
            if (_observation(unchecked)["validation"]["status"] == "unknown"
                    and TASKS["analysis"] not in by_task):
                sources = []
                for key in ("baseline", "challenger", "unchecked"):
                    producer = by_task[TASKS[key]]
                    artifacts = producer["artifact_records"]
                    if len(artifacts) != 1 or artifacts[0]["logical_name"] != "result":
                        raise ValueError("analysis needs one actual artifact from each producer")
                    sources.append(artifacts[0]["artifact_id"])
                return self._proposal("analysis", request("unchecked", "analyze", sources))
        candidates = [item for task, item in by_task.items() if task in TASKS.values()
            and _observation(item)["validation"]["status"] == "valid"
            and _comparable(base_observation, _observation(item))]
        winner = min(candidates, key=lambda item: (
            _observation(item)["values"]["cost"], item["ref"]["task_id"]))
        selected = _observation(winner)
        # Correlate equivalent materialized-action occurrences from recorded
        # evidence, not a private in-memory "replication succeeded" flag.
        repeats = [item for task, item in by_task.items() if task not in TASKS.values()
                   and _action_signature(item) == _action_signature(winner)]
        if repeats:
            confirmed = any(_observation(item)["validation"]["status"] == "valid"
                and _comparable(selected, _observation(item))
                and _observation(item)["values"]["cost"] == selected["values"]["cost"]
                for item in repeats)
            return {"kind": "Stop",
                    "reason": "confirmed_selection" if confirmed else "confirmation_failed",
                    "wakeup": None}
        return {"kind": "Replicate", "source_ref": copy.deepcopy(winner["ref"]),
            "request_id": "confirm-" + digest(winner["ref"])[:24],
            "reason": "confirm the lowest-cost valid comparable recorded action"}

    def decide(self, snapshot, budget):
        decision = self._decide(snapshot, budget)
        # A diagnostic trace, not a Lake commit/GO/settlement receipt. Full
        # bounded metadata allows independent replay of evidence-only choices.
        record = {"version": 1, "monotonic": time.monotonic(),
            "snapshot_sha256": digest(snapshot), "snapshot": snapshot,
            "budget": budget, "decision": decision}
        print("ACCEPTANCE_DECISION=" + json.dumps(record, sort_keys=True,
              separators=(",", ":"), ensure_ascii=False, allow_nan=False), flush=True)
        return decision
