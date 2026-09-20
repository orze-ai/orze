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


class BreadthThenAdaptive:
    """Give each available path a minimum depth, then delegate allocation.

    ``min_steps`` counts attempts, including failures and unscored analyses.
    With four paths, min_steps=2 and 14 calls, the first eight attempts cover
    each path twice; the adaptive policy allocates the remaining six. A smaller
    budget still opens paths before deepening any of them. Blocked paths stay
    closed. The delegate must decide from the complete revealed prefix rather
    than require callbacks for the initial rounds.

    This is an explicit experimental policy, not evidence of better quality.
    """

    def __init__(self, adaptive, min_steps=2):
        if not callable(getattr(adaptive, "decide", None)):
            raise ValueError("adaptive policy requires decide(view)")
        if type(min_steps) is not int or min_steps < 1:
            raise ValueError("minimum exploration depth must be a positive integer")
        self.adaptive = adaptive
        self.min_steps = min_steps

    def decide(self, view):
        latest = {n["branch"]: n for n in view["observed"]}
        legal = [a for a in view["legal"]
                 if latest.get(a["branch"], {}).get("status") != "blocked"]
        initial = sorted((a for a in legal if a["step"] < self.min_steps),
                         key=lambda a: (a["step"], a["branch"]))
        size = min(view["workers"], view["remaining_calls"])
        if initial:
            return {"actions": [a["id"] for a in initial[:size]],
                    "reason": f"Give each available path {self.min_steps} attempts before adaptive allocation."}
        if not legal or not size:
            return {"actions": [], "reason": "No available paths or remaining attempts."}
        return self.adaptive.decide(dict(view, legal=legal))


class Portfolio:
    """Compare whole trajectories; preserve weak early ideas and repairable paths.

A successful evaluation without improvement is evidence about the direction;
an implementation failure is not. The policy retains a path's successful anchor
through failures, spends bounded effort on repair, and opens new paths alongside
promising continuations. The patience parameters count evidence, not tokens or
monetary cost. They are fixed for a rollout and must be validated across worlds.
"""

    def __init__(self, min_valid=2, patience=2, repair_patience=2, *, stop_on_plateau=False):
        for value in (min_valid, patience, repair_patience):
            if type(value) is not int or value < 1:
                raise ValueError("portfolio patience must be positive integers")
        self.min_valid = min_valid
        self.patience = patience
        self.repair_patience = repair_patience
        if type(stop_on_plateau) is not bool:
            raise ValueError("stop_on_plateau must be boolean")
        self.stop_on_plateau = stop_on_plateau

    def decide(self, view):
        scale = view["score_scale"]
        baseline = view["root"]["score"]
        roots, repairs, normal, dormant = [], [], [], []
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
                dormant.append((key, action))
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
        # A short plateau does not establish that a research path is exhausted.
        # Spend remaining authorized attempts on underexplored dormant paths;
        # blocked paths and exhausted repair episodes stay closed.
        if not self.stop_on_plateau:
            for _, action in sorted(dormant, key=lambda pair: (
                    pair[1]["step"], pair[0])):
                if len(chosen) < slots:
                    chosen.append(action)
        return {"actions": [a["id"] for a in chosen],
                "reason": (f"Revealed trajectories: {len(normal)} promising/underexplored, "
                           f"{len(roots)} unopened, {len(repairs)} repairable, {closed} closed. "
                           "Preserve successful anchors and compare independent continuations.")}
