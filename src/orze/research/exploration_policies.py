"""Small, replaceable exploration policies; no scientific directions are prescribed.

These are starting policies, not learned scientific knowledge. New policies may
implement the same ``decide(view)`` interface without inheriting either class.
"""
from __future__ import annotations


class ParallelRefine:
    """Open the declared roots, then refine the least explored paths in parallel."""

    def decide(self, view):
        actions = sorted(view["legal"], key=lambda a: (a["step"], a["branch"]))
        size = min(view["workers"], view["remaining_calls"])
        return {"actions": [a["id"] for a in actions[:size]],
                "reason": "Open independent paths and give each equal refinement depth."}


class Portfolio:
    """Compare whole trajectories; preserve weak early ideas and repairable paths.

A successful evaluation without improvement is evidence about the direction;
an implementation failure is not. The policy retains a path's successful anchor
through failures, spends bounded effort on repair, and opens new paths alongside
promising continuations. The patience parameters count evidence, not tokens or
monetary cost. They are fixed for a rollout and must be validated across worlds.
"""

    def __init__(self, min_valid=2, patience=2, repair_patience=2):
        for value in (min_valid, patience, repair_patience):
            if type(value) is not int or value < 1:
                raise ValueError("portfolio patience must be positive integers")
        self.min_valid = min_valid
        self.patience = patience
        self.repair_patience = repair_patience

    def decide(self, view):
        scale = view["score_scale"]
        baseline = view["root"]["score"]
        roots, repairs, normal = [], [], []
        closed = 0
        for action in view["legal"]:
            path = [n for n in view["observed"] if n["branch"] == action["branch"]]
            if not path:
                roots.append(action)
                continue
            valid = [n for n in path if n["status"] == "ok" and n["score"] is not None]
            anchor = max([baseline] + [n["score"] for n in valid])
            stale = 0
            best = baseline
            for n in valid:
                if n["score"] > best:
                    best, stale = n["score"], 0
                else:
                    stale += 1
            latest = path[-1]
            failures = 0
            for n in reversed(path):
                if n["status"] == "ok":
                    break
                failures += 1
            gain = ((valid[-1]["score"] - valid[-2]["score"]) / scale
                    if len(valid) > 1 else 0.0)
            # Prefix-relative ranking, no task-specific score or winning IDs.
            priority = (anchor - baseline) / scale + max(0.0, gain)
            key = (-priority, len(path), action["branch"])
            if latest["status"] == "blocked":
                closed += 1
            elif latest["status"] == "repairable":
                if failures <= self.repair_patience:
                    repairs.append((key, action))
                else:
                    closed += 1
            elif len(valid) < self.min_valid or stale < self.patience:
                normal.append((key, action))
            else:
                closed += 1
        normal.sort(key=lambda pair: pair[0])
        repairs.sort(key=lambda pair: pair[0])
        slots = min(view["workers"], view["remaining_calls"])
        chosen = []
        # One-worker runs rotate competing roles deterministically. With more
        # workers, represent available roles then fill with useful refinements.
        roles = [([a for _, a in normal]), roots, ([a for _, a in repairs][:1])]
        roles = [role for role in roles if role]
        if slots == 1 and roles:
            chosen = roles[len(view["observed"]) % len(roles)][:1]
        else:
            for role in roles:
                if len(chosen) < slots:
                    chosen.append(role[0])
            remaining = [a for _, a in normal] + roots
            for action in remaining:
                if len(chosen) < slots and action not in chosen:
                    chosen.append(action)
        return {"actions": [a["id"] for a in chosen],
                "reason": (f"Revealed trajectories: {len(normal)} promising/underexplored, "
                           f"{len(roots)} unopened, {len(repairs)} repairable, {closed} closed. "
                           "Preserve successful anchors and compare independent continuations.")}
