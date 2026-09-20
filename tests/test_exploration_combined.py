"""The combined policy must actually expose broad evidence before allocating depth."""
import pytest

from orze.research.exploration import replay, run_online
from orze.research.exploration_policies import BreadthThenAdaptive


class Deepen:
    def __init__(self):
        self.views = []

    def decide(self, view):
        self.views.append(view)
        return {"actions": [view["legal"][0]["id"]], "reason": "Deepen the first available path."}


def run(tmp_path, *, calls=14, workers=1, blocked=False):
    spec = {"problem_id": "combined-fixture", "protocol_id": "fixed-executor",
            "score_scale": 1, "root": {"score": 0, "artifact": {}, "feedback": {}},
            "plan": {"branches": 4, "depth": 6, "calls": calls, "workers": workers}}
    delegate = Deepen()
    policy = BreadthThenAdaptive(delegate)
    seen = []
    def execute(contexts):
        result = {}
        for context in contexts:
            a = context["action"]
            assert len(context["history"]) == a["step"]
            assert all(n["branch"] == a["branch"] for n in context["history"])
            seen.append(a["id"])
            status = "blocked" if blocked and a["branch"] == 0 else "ok"
            # An unscored diagnostic remains an attempt and carries evidence.
            score = None if status == "blocked" or a["step"] == 0 else 10 - a["branch"]
            result[a["id"]] = {"score": score, "status": status,
                "artifact": {"kind": "analysis" if a["step"] == 0 else "method"},
                "feedback": {"finding": a["id"]}, "cost": 1, "seconds": 1}
        return result
    trace = run_online(spec, policy, execute, tmp_path / "run")
    return trace, delegate, seen


def test_broad_evidence_is_consumed_before_adaptive_depth(tmp_path):
    trace, delegate, seen = run(tmp_path)
    assert seen[:8] == [f"b{b}-s{s}" for s in range(2) for b in range(4)]
    assert seen[8:12] == [f"b0-s{s}" for s in range(2, 6)]
    assert len(delegate.views[0]["observed"]) == 8
    assert delegate.views[0]["remaining_calls"] == 6
    assert delegate.views[0]["observed"][0]["feedback"] == {"finding": "b0-s0"}
    assert replay(trace, BreadthThenAdaptive(Deepen()))["rounds"] == trace["rounds"]


@pytest.mark.parametrize("calls,workers", [(3, 1), (7, 3), (14, 3)])
def test_small_budget_and_batches_obey_the_same_foundation(tmp_path, calls, workers):
    trace, delegate, seen = run(tmp_path, calls=calls, workers=workers)
    assert len(seen) == calls
    assert set(seen[:min(4, calls)]) == {f"b{b}-s0" for b in range(min(4, calls))}
    if delegate.views:
        assert all(sum(n["branch"] == b for n in delegate.views[0]["observed"]) >= 2
                   for b in range(4))


def test_blocked_path_is_not_forced_to_reach_the_minimum(tmp_path):
    trace, delegate, seen = run(tmp_path, blocked=True)
    assert [x for x in seen if x.startswith("b0-")] == ["b0-s0"]
    assert len(delegate.views[0]["observed"]) == 7
    assert all(a["branch"] != 0 for v in delegate.views for a in v["legal"])


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_minimum_requires_a_positive_integer(value):
    with pytest.raises(ValueError):
        BreadthThenAdaptive(Deepen(), min_steps=value)
