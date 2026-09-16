"""Prune strictly dominated validation-only analysis under an explicit protocol.

This application policy never edits evidence or makes an unknown result valid.
The capability is a prospective assertion by this application's trusted owner,
not an automatic inference from metric names or a general Domain guarantee.
"""
import copy
import json
import re

from examples.acceptance.common import digest
from examples.acceptance.policy import (
    CommonPolicy, TASKS, _action_signature, _comparable, _observation,
)


class DominancePruningPolicy(CommonPolicy):
    def __init__(self, declaration):
        config = declaration.get("config", {})
        if type(config) is not dict or set(config) != {"validation_only_protocols"}:
            raise ValueError("explicit validation-only protocol declaration required")
        protocols = config["validation_only_protocols"]
        if (type(protocols) is not list or len(protocols) > 16
                or any(type(p) is not str or re.fullmatch(r"[0-9a-f]{64}", p) is None
                       for p in protocols)
                or len(set(protocols)) != len(protocols)):
            raise ValueError("invalid validation-only protocol identities")
        ordinary = copy.deepcopy(declaration)
        ordinary["config"] = {}
        super().__init__(ordinary)
        self.declaration = copy.deepcopy(declaration)
        self.validation_only_protocols = frozenset(protocols)

    def _decide(self, snapshot, budget):
        ordinary = super()._decide(snapshot, budget)
        if not (ordinary.get("kind") == "Propose"
                and ordinary.get("task_id") == TASKS["analysis"]):
            return ordinary
        results = snapshot["recorded_evidence"]["results"]
        by_task = {r["ref"]["task_id"]: r for r in results}
        unchecked = by_task[TASKS["unchecked"]]
        observed = _observation(unchecked)
        baseline = _observation(by_task[TASKS["baseline"]])
        candidates = [item for task, item in by_task.items()
                      if task in TASKS.values()
                      and _observation(item)["validation"]["status"] == "valid"
                      and _comparable(baseline, _observation(item))]
        winner = min(candidates, key=lambda item: (
            _observation(item)["values"]["cost"], item["ref"]["task_id"]))
        selected = _observation(winner)
        if not (observed["validation"]["status"] == "unknown"
                and observed["protocol_fingerprint"] in self.validation_only_protocols
                and _comparable(selected, observed)
                and observed["values"]["cost"] > selected["values"]["cost"]):
            return ordinary

        # These are actual recorded occurrences. No synthetic "analysis done"
        # result is inserted to force the original policy down another branch.
        repeats = [item for task, item in by_task.items()
                   if task not in TASKS.values()
                   and _action_signature(item) == _action_signature(winner)]
        if repeats:
            confirmed = any(_observation(item)["validation"]["status"] == "valid"
                            and _comparable(selected, _observation(item))
                            and _observation(item)["values"]["cost"] == selected["values"]["cost"]
                            for item in repeats)
            decision = {"kind": "Stop",
                        "reason": "confirmed_selection" if confirmed else "confirmation_failed",
                        "wakeup": None}
        else:
            decision = {"kind": "Replicate", "source_ref": copy.deepcopy(winner["ref"]),
                        "request_id": "confirm-" + digest(winner["ref"])[:24],
                        "reason": "confirm the lowest-cost valid comparable recorded action"}
        print("PRUNING_DECISION=" + json.dumps({
            "version": 1, "reason": "strictly_dominated_validation_only",
            "unknown_ref": unchecked["ref"], "incumbent_ref": winner["ref"],
            "unknown_cost": observed["values"]["cost"],
            "incumbent_cost": selected["values"]["cost"],
            "validation_only_protocol": observed["protocol_fingerprint"],
            "comparison_scope": observed["comparison_scope"],
            "dataset_sha256": observed["values"]["dataset_sha256"],
            "snapshot_sha256": digest(snapshot), "decision": decision,
        }, sort_keys=True, allow_nan=False), flush=True)
        return decision
