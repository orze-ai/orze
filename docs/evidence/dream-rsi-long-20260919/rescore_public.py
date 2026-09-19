"""Rescore the exported confirmation predictions; requires Python 3 and scipy.

This checks prediction arithmetic and aggregation, not execution provenance or
label isolation. Those require the separately archived full study verifier.
No model calls or scientific program executions are performed.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

PROBLEMS = ("extrapolation", "interaction", "dynamics", "robust", "transport")
ARMS = ("control", "dream")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rmse(prediction, truth):
    assert len(prediction) == len(truth) == 600
    assert all(type(x) in (int, float) and math.isfinite(x)
               for x in [*prediction, *truth])
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(prediction, truth)) / len(truth))


def rescore(evidence):
    from scipy.stats import t

    assert evidence["schema"] == 1
    cases = {(r["problem"], r["repetition"]): r for r in evidence["cases"]}
    assert len(cases) == len(evidence["cases"]) == 10
    assert set(cases) == set(itertools.product(PROBLEMS, (0, 1)))
    paired = []
    for name, rep in itertools.product(PROBLEMS, (0, 1)):
        row = cases[name, rep]
        baseline = rmse(row["baseline_prediction"], row["truth"])
        assert baseline > 0
        assert set(row["selected"]) == set(ARMS)
        pair = {"problem": name, "repetition": rep}
        for arm in ARMS:
            selected = row["selected"][arm]
            assert type(selected["valid"]) is bool
            if selected["valid"]:
                gain = 1 - rmse(selected["prediction"], row["truth"]) / baseline
            else:
                assert selected["prediction"] is None
                gain = -1.0  # Frozen study rule, never silently drop a failure.
            pair[arm + "_gain"] = gain
        pair["delta"] = pair["dream_gain"] - pair["control_gain"]
        paired.append(pair)
    groups = []
    for name in PROBLEMS:
        own = [r for r in paired if r["problem"] == name]
        groups.append({"problem": name, **{
            key: statistics.fmean(r[key] for r in own)
            for key in ("control_gain", "dream_gain", "delta")}})
    deltas = [r["delta"] for r in groups]
    mean = statistics.fmean(deltas)
    margin = float(t.ppf(.975, 4)) * statistics.stdev(deltas) / math.sqrt(5)
    permutations = [statistics.fmean(d * sign for d, sign in zip(deltas, signs))
                    for signs in itertools.product((-1, 1), repeat=5)]
    return {
        "paired": paired, "problem_means": groups,
        "control_mean_gain": statistics.fmean(r["control_gain"] for r in groups),
        "dream_mean_gain": statistics.fmean(r["dream_gain"] for r in groups),
        "mean_delta": mean, "problem_t_95_interval": [mean - margin, mean + margin],
        "exact_problem_sign_flip_two_sided_p": sum(abs(x) >= abs(mean) - 1e-15
                                                   for x in permutations) / 32,
        "independent_problem_count": 5, "paired_runs": 10,
    }


def compare(actual, reported):
    if isinstance(actual, dict):
        for key, value in actual.items():
            compare(value, reported[key])
    elif isinstance(actual, list):
        assert len(actual) == len(reported)
        for a, b in zip(actual, reported):
            compare(a, b)
    elif isinstance(actual, (int, float)):
        assert math.isclose(actual, reported, rel_tol=0, abs_tol=1e-12), (actual, reported)
    else:
        assert actual == reported, (actual, reported)


def export(study):
    """Export only after the full private verifier has passed."""
    verification = read(study / "verification.json")
    assert verification["status"] == "passed"
    assert verification["summary_sha256"] == sha(study / "summary.json")
    choices = read(study / "all-choices-frozen.json")
    confirmation = read(study / "confirmation-results.json")
    assert confirmation["frozen_choices_sha256"] == sha(study / "all-choices-frozen.json")
    selected_policy = read(study / "policy-selection.json")
    result = {
        "schema": 1,
        "scope": "Five synthetic problems, two paired data repetitions, one scientist model; confirmation predictions only.",
        "provenance": {name: sha(study / name) for name in (
            "plan.json", "all-choices-frozen.json", "confirmation-results.json",
            "summary.json", "verification.json", "policy-selection.json")},
        "dream_policy_identity": selected_policy["identity"],
        "dream_policy_source_sha256": selected_policy["manifest"]["source_sha256"],
        "cases": [],
    }
    for name, rep in itertools.product(PROBLEMS, (0, 1)):
        prefix = f"{name}-{rep}"
        data = read(study / "inputs" / f"{prefix}.json")
        baseline = study / "confirmation" / f"{prefix}-baseline"
        base_source = choices["choices"][f"{prefix}-control"]["baseline"]["source"]
        row = {"problem": name, "repetition": rep, "truth": data["confirmation"]["y"],
               "baseline_prediction": read(baseline / "prediction.json")["prediction"],
               "baseline_source_sha256": hashlib.sha256(base_source.encode()).hexdigest(),
               "selected": {}}
        for arm in ARMS:
            label = f"{prefix}-{arm}"
            choice = choices["choices"][label]
            assert choice["trace_sha256"] == sha(study / "episodes" / label / "online/trace.json")
            records = [r for r in confirmation["rows"]
                       if (r["problem"], r["repetition"], r["arm"]) == (name, rep, arm)]
            assert len(records) == 1
            valid = records[0]["valid"]
            folder = baseline if choice["selected"]["source"] == base_source else study / "confirmation" / label
            row["selected"][arm] = {
                "valid": valid,
                "prediction": read(folder / "prediction.json")["prediction"] if valid else None,
                "origin": choice["selected"]["origin"],
                "source_sha256": hashlib.sha256(choice["selected"]["source"].encode()).hexdigest(),
                "trace_sha256": choice["trace_sha256"],
                "program_stdout_sha256": sha(folder / "stdout.log"),
                "closure_sha256": sha(folder / "closure.json"),
            }
        result["cases"].append(row)
    recomputed = rescore(result)
    compare(recomputed, read(study / "summary.json"))
    result["reported"] = recomputed
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="?", type=Path)
    parser.add_argument("--export-study", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.export_study:
        if not args.output or args.evidence:
            parser.error("--export-study requires --output and no evidence argument")
        evidence = export(args.export_study)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(evidence, f, allow_nan=False, separators=(",", ":"))
    else:
        if not args.evidence or args.output:
            parser.error("provide an evidence JSON file, or use --export-study with --output")
        evidence = read(args.evidence)
    result = rescore(evidence)
    compare(result, evidence["reported"])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
