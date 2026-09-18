"""Causal visibility, replay support and actual branch execution contracts."""
import json

import pytest

from orze.research.exploration import detached, replay, run_online, validate_trace
from orze.research.exploration_policies import ParallelRefine, Portfolio


def spec(branches=3, depth=3, calls=9, workers=2):
    return {"problem_id": "fixture", "protocol_id": "fixture-generator-and-evaluator-v1",
            "root": {"score": 0, "artifact": {"source": "initial"}, "feedback": {}},
            "score_scale": 1, "plan": dict(branches=branches, depth=depth, calls=calls, workers=workers)}


def outcome(score, status="ok", **kwargs):
    return dict(score=score, status=status, artifact={}, feedback={}, cost=1, seconds=2, **kwargs)


def executor(contexts):
    return {c["action"]["id"]: outcome(c["action"]["step"] + 1) for c in contexts}


def test_default_reaches_late_breakthrough_after_plateau(tmp_path):
    def late(contexts):
        return {c["action"]["id"]: outcome(9 if c["action"]["step"] == 4 else 0)
                for c in contexts}
    declaration = spec(branches=1, depth=6, calls=6)
    trace = run_online(declaration, execute_batch=late, output=tmp_path / "default")
    assert trace["metrics"]["best_score"] == 9
    assert trace["metrics"]["calls"] == 6
    assert replay(trace, Portfolio())["metrics"] == trace["metrics"]
    early = run_online(declaration, Portfolio(stop_on_plateau=True), late,
                       tmp_path / "explicit-early-stop")
    assert early["termination"] == "policy_stop"
    assert early["metrics"]["best_score"] == 0


def test_default_still_requires_explicit_executor_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="execute_batch"):
        run_online(spec(), output=tmp_path / "missing-executor")
    assert not (tmp_path / "missing-executor").exists()


def test_online_and_replay_have_identical_decisions_and_resource_accounting(tmp_path):
    trace = run_online(spec(), ParallelRefine(), executor, tmp_path / "online")
    assert validate_trace(trace) == trace
    assert replay(trace, ParallelRefine()) == {**trace, "mode": "replay"}
    assert trace["metrics"] == {"supported": True, "best_score": 3, "quality_gain": 3,
                                "calls": 9, "rounds": 5, "cost": 9,
                                "parallel_seconds_estimate": 10}


def test_siblings_and_same_batch_results_never_enter_generation_context(tmp_path):
    seen = []

    def generate(contexts):
        for context in contexts:
            branch = context["action"]["branch"]
            assert all(n["branch"] == branch for n in context["history"])
            assert len(context["history"]) == context["action"]["step"]
        seen.extend(detached(contexts))
        result = executor(contexts)
        contexts[0]["root"]["artifact"]["source"] = "attempted mutation"
        return result

    trace = run_online(spec(), ParallelRefine(), generate, tmp_path / "online")
    assert len(seen) == 9
    assert all(c["root"]["artifact"]["source"] == "initial" for c in seen)
    assert trace["spec"]["root"]["artifact"]["source"] == "initial"


def test_unseen_results_do_not_change_policy_prefix(tmp_path):
    trace = run_online(spec(), ParallelRefine(), executor, tmp_path / "online")
    changed = detached(trace)
    changed["rounds"][-1]["observations"][0]["score"] = 900
    changed["metrics"].update(best_score=900, quality_gain=900)
    views = []

    class Observe:
        def decide(self, view):
            views.append(view)
            return ParallelRefine().decide(view)

    replay(trace, Observe())
    first = detached(views)
    views.clear()
    replay(changed, Observe())
    assert views == first
    assert all("rounds" not in v and "spec" not in v for v in views)


def test_missing_continuation_is_unsupported_and_hidden_from_legal_actions(tmp_path):
    class StopEarly:
        def decide(self, view):
            return {"actions": [] if view["observed"] else ["b0-s0"], "reason": "one probe"}

    trace = run_online(spec(), StopEarly(), executor, tmp_path / "online")
    result = replay(trace, ParallelRefine())
    assert result["termination"] == "out_of_support"
    assert result["metrics"]["supported"] is False
    assert result["rounds"][0]["decision"]["actions"] == ["b0-s0", "b1-s0"]
    # No partial batch revelation and no invented outcome for b1.
    assert result["rounds"][0]["observations"] == []
    with pytest.raises(ValueError, match="complete online trace"):
        validate_trace(result)


@pytest.mark.parametrize("actions", [["b0-s1"], ["b0-s0", "b0-s0"],
                                      ["b0-s0", "b1-s0", "b2-s0"]])
def test_illegal_batch_rejected_before_execution(tmp_path, actions):
    class Bad:
        def decide(self, view):
            return {"actions": actions, "reason": "invalid"}
    reached = []
    with pytest.raises(ValueError, match="frontier or budget"):
        run_online(spec(), Bad(), lambda c: reached.append(c), tmp_path / "online")
    assert reached == []


def test_interrupted_or_completed_online_directory_is_never_rerun(tmp_path):
    def interrupted(contexts):
        assert (tmp_path / "online/request-0000.json").exists()
        raise RuntimeError("unknown callback outcome")
    with pytest.raises(RuntimeError):
        run_online(spec(), ParallelRefine(), interrupted, tmp_path / "online")
    assert not (tmp_path / "online/trace.json").exists()
    assert json.loads((tmp_path / "online/failure.json").read_text())["retry_allowed"] is False
    with pytest.raises(FileExistsError):
        run_online(spec(), ParallelRefine(), executor, tmp_path / "online")


@pytest.mark.parametrize("change", ["parent", "score", "budget", "metrics", "extra"])
def test_corrupt_trace_cannot_enter_pool(tmp_path, change):
    trace = run_online(spec(), ParallelRefine(), executor, tmp_path / "online")
    if change == "parent":
        trace["rounds"][1]["observations"][1]["parent"] = "b2-s0"
    elif change == "score":
        trace["rounds"][0]["observations"][0].update(status="repairable", score=99)
    elif change == "budget":
        trace["spec"]["plan"]["calls"] = 1
    elif change == "metrics":
        trace["metrics"]["calls"] = 1
    else:
        trace["rounds"][0]["observations"][0]["future_score"] = 99
    with pytest.raises(ValueError):
        validate_trace(trace)


def test_quality_can_recover_after_implementation_failure(tmp_path):
    def recover(contexts):
        result = {}
        for c in contexts:
            step = c["action"]["step"]
            if step == 1:
                value = outcome(None, "repairable")
                value["artifact"] = {"source": "broken but retained"}
            else:
                if step == 2:
                    assert c["history"][-1]["artifact"]["source"] == "broken but retained"
                value = outcome(1 if step == 0 else 5)
            result[c["action"]["id"]] = value
        return result

    trace = run_online(spec(branches=1, calls=3), Portfolio(), recover, tmp_path / "online")
    assert trace["metrics"]["best_score"] == 5
    assert trace["metrics"]["cost"] == 3
    assert replay(trace, Portfolio()) == {**trace, "mode": "replay"}


def test_bad_early_score_alone_does_not_prune_new_direction(tmp_path):
    def late_gain(contexts):
        return {c["action"]["id"]: outcome(-1 if c["action"]["step"] == 0 else 5)
                for c in contexts}
    trace = run_online(spec(branches=1, calls=2), Portfolio(), late_gain, tmp_path / "online")
    assert trace["metrics"]["best_score"] == 5


def test_repair_patience_counts_current_episode_not_all_historical_failures(tmp_path):
    def recover_twice(contexts):
        return {c["action"]["id"]: outcome(None, "repairable") if c["action"]["step"] in (0, 2)
                else outcome(c["action"]["step"]) for c in contexts}
    trace = run_online(spec(branches=1, depth=4, calls=4), Portfolio(repair_patience=1),
                       recover_twice, tmp_path / "online")
    assert trace["metrics"]["best_score"] == 3


def test_failed_quality_never_replaces_successful_anchor(tmp_path):
    def fail_after_success(contexts):
        return {c["action"]["id"]: outcome(5) if not c["action"]["step"]
                else outcome(None, "blocked") for c in contexts}
    trace = run_online(spec(branches=1, depth=5, calls=5), Portfolio(), fail_after_success,
                       tmp_path / "online")
    assert trace["termination"] == "policy_stop"
    assert trace["metrics"]["best_score"] == 5
    assert trace["metrics"]["calls"] == 2


def test_analysis_is_success_without_inventing_quality_or_unknown_cost(tmp_path):
    def analyze(contexts):
        result = executor(contexts)
        for value in result.values():
            value.update(score=None, cost=None, seconds=None, feedback={"finding": "competing explanations"})
        return result
    trace = run_online(spec(branches=1), Portfolio(), analyze, tmp_path / "online")
    assert trace["metrics"]["best_score"] == 0
    assert trace["metrics"]["calls"] == 3
    assert trace["metrics"]["cost"] is None
    assert trace["metrics"]["parallel_seconds_estimate"] is None
    assert replay(trace, Portfolio())["metrics"] == trace["metrics"]
