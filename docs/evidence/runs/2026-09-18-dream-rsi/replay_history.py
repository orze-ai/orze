"""Convert PRE-final development paths; do not execute models or evaluators.

These histories have one path each. They support prefix/stopping comparisons,
not new sibling outcomes. Source facts/programs are preserved; unknown cost and
duration remain unknown. No independence between same-dataset paths is claimed.
"""
from pathlib import Path
import hashlib
import json
import sys

from orze.research.exploration import fingerprint, replay, validate_trace
from orze.research.exploration_policies import ParallelRefine, Portfolio


def main(source, output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in ("opus-A", "opus-B", "fable-A", "fable-B"):
        path = source / name / "state-006-b-execute.json"
        raw = path.read_bytes()
        state = json.loads(raw)
        history = state["history"]
        assert len(history) == 9 and state["cycle"] == 6 and not state.get("finished")
        initial, attempts = history[:3], history[3:]
        assert all(h["valid"] for h in history)
        baseline = min(h["facts"]["loss"] for h in initial)
        source_sha = hashlib.sha256(raw).hexdigest()
        spec = {"problem_id": "uci275-same-development-split", "protocol_id": fingerprint({
                    "source": source_sha, "original_protocol": "context-experiment-v1", "trajectory": name}),
                "root": {"score": -baseline,
                         "artifact": {"baselines": [{"action": h["action"], "facts": h["facts"]} for h in initial]},
                         "feedback": {"original_source": str(path.relative_to(source)),
                                      "sha256": source_sha, "scope": "historical single-path stopping only"}},
                "score_scale": baseline,
                "plan": {"branches": 1, "depth": 6, "calls": 6, "workers": 1}}
        rounds = []
        for step, h in enumerate(attempts):
            node = {"id": f"b0-s{step}", "branch": 0, "step": step,
                    "parent": f"b0-s{step-1}" if step else None,
                    "score": -h["facts"]["loss"] if h["action"]["kind"] == "method" else None,
                    "status": "ok", "feedback": h["facts"],
                    "artifact": {"task_id": h["task_id"], "action": h["action"]},
                    "cost": None, "seconds": None}
            rounds.append({"decision": {"actions": [node["id"]], "reason": "original next research round"},
                           "observations": [node], "supported": True})
        best = max([-baseline] + [r["observations"][0]["score"] for r in rounds
                                 if r["observations"][0]["score"] is not None])
        trace = {"schema": 1, "mode": "online", "spec": spec, "rounds": rounds, "termination": "budget",
                 "stop_reason": None, "metrics": {"supported": True, "best_score": best,
                   "quality_gain": (best + baseline) / baseline, "calls": 6, "rounds": 6,
                   "cost": None, "parallel_seconds_estimate": None}}
        validate_trace(trace)
        results = {"fixed": replay(trace, ParallelRefine()), "portfolio": replay(trace, Portfolio())}
        for label, value in {"trace": trace, **results}.items():
            with (output / f"{name}-{label}.json").open("x") as stream:
                json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        rows.append({"name": name, "source_sha256": source_sha,
                     **{k: {"calls": r["metrics"]["calls"], "best_development_rmse": -r["metrics"]["best_score"],
                            "termination": r["termination"]} for k, r in results.items()}})
    summary = {"new_model_calls": 0, "new_scientific_executions": 0, "rows": rows,
               "independent_validation_problems": 0,
               "promotion": "none: related single-path histories; portfolio loses delayed gains",
               "general_research_improvement_proven": False}
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
