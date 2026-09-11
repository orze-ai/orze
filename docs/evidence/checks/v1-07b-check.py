#!/usr/bin/env python3
"""Read-only original-holdout evidence consistency, not execution attestation."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

if sys.flags.optimize:
    raise SystemExit("Run without -O: evidence assertions must be enabled.")
ROOT = Path(__file__).resolve().parents[3]
PRO = ROOT.parent / "orze-pro"
CANONICAL = ROOT / "docs/evidence/2026-09-11-v1-07b-holdout.json"
NAMES = {"workflow", "boundary", "duplicate_id", "missing_prerequisite", "overload", "malformed"}
A_NAMES = {d + "_" + v for d in ("sorting", "compression") for v in ("default", "counterfactual")}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def same(left, right):
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert encode(left) == encode(right)


def read(path):
    return json.loads((ROOT / path).read_bytes())


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def manifest(record):
    assert record["exit_code"] == 0
    lines = record["output"].splitlines()
    assert all(re.fullmatch(r"[0-9a-f]{64}  .+", row) for row in lines)
    parsed = {row[66:]: row[:64] for row in lines}
    assert len(parsed) == len(lines)
    return parsed


def footer(output, expected):
    pattern = re.escape(expected) + r"(?:, [1-9][0-9]* warnings?)? in [0-9]+(?:\.[0-9]+)?s(?: \([^\n]*\))?"
    assert any(re.fullmatch(pattern, row.strip("= ")) for row in output.splitlines()), output[-3000:]


def suite(record, expected):
    chunks = record["chunks"]
    assert chunks and chunks[-1]["exit_code"] == 0
    assert all(row.get("exit_code") is None for row in chunks[:-1])
    output = "".join(row["output"] for row in chunks)
    assert "Warning: truncated output" not in output
    assert not re.search(r"\b[1-9][0-9]* failed\b|KeyboardInterrupt|Interrupted:", output)
    footer(output, expected)
    return output


def required_support():
    required = {
        "docs/holdout-acceptance.md",
        "docs/plans/2026-09-11-v1-07b-holdout.zh-CN.md",
        "docs/evidence/challenges/v1-07b-original.md",
        "docs/evidence/checks/v1-07b-check.py",
        "docs/evidence/checks/v1-07b-domain-check.py",
        "docs/evidence/checks/v1-07b-runs-check.py",
        "docs/evidence/checks/v1-07a-runs-check.py",
        "docs/evidence/2026-09-11-v1-07b-domain-author.json",
        "docs/evidence/2026-09-11-v1-07b-domain-independent-review.json",
        "docs/evidence/2026-09-11-v1-07b-product-independent-review.json",
        "docs/evidence/2026-09-11-v1-07b-target-history.json",
        "docs/evidence/snapshots/v1-07b-before-unique-snapshot-labels-testing.py",
        "docs/evidence/snapshots/v1-07b-before-unique-snapshot-labels-product-tests.py",
    }
    required.update("docs/evidence/runs/v1-07b-" + epoch + "-" + name + ".json"
                    for epoch in ("candidate1", "target", "full") for name in NAMES)
    required.update("docs/evidence/runs/v1-07b-known-full-" + name + ".json" for name in A_NAMES)
    return required


def report_exports(output, marker, reports, names, prefix):
    assert output.count(marker + "=") == 1
    actual = json.loads(output.split(marker + "=")[1].split("\n")[0])
    assert len(actual) == len(reports) == len(names)
    assert {row["name"] for row in actual} == {row["name"] for row in reports} == names
    assert sorted(actual, key=lambda row: row["name"]) == sorted(
        [{k: row[k] for k in ("name", "path", "sha256", "bytes")} for row in reports],
        key=lambda row: row["name"])
    for row in reports:
        assert row["archive"] == "docs/evidence/runs/" + prefix + "-" + row["name"] + ".json"
        raw = (ROOT / row["archive"]).read_bytes()
        assert sha(raw) == row["sha256"] and len(raw) == row["bytes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree-only", action="store_true")
    args = parser.parse_args()
    evidence = json.loads(CANONICAL.read_bytes())
    assert evidence["schema"] == 1 and evidence["slice"] == "V1-07B"
    assert evidence["core_product_frozen"] == "8f61ae5f68c0846d5149aceb4df55c69263c9df2"
    support = evidence["support"]
    assert len(support) == len({row["path"] for row in support})
    assert {row["path"] for row in support} == required_support()
    for row in support:
        raw = (ROOT / row["path"]).read_bytes()
        assert sha(raw) == row["sha256"], row["path"]
        if not args.worktree_only:
            assert raw == git(ROOT, "show", evidence["fixed_code_commits"]["core"] + ":" + row["path"])
    freeze = evidence["full_epoch"]
    total = 0
    for name, repo, count in (("core", ROOT, 755), ("pro", PRO, 223)):
        before = manifest(freeze["before"][name])
        after = manifest(freeze["after"][name])
        assert before == after and len(before) == count
        live = git(repo, "ls-files", "-c", "-o", "--exclude-standard", "--",
                   "src", "tests", "examples", "pyproject.toml").decode().splitlines()
        assert set(live) == set(before)
        for path, digest in before.items():
            assert sha((repo / path).read_bytes()) == digest, (name, path)
        if not args.worktree_only:
            commit = evidence["fixed_code_commits"][name]
            assert re.fullmatch(r"[0-9a-f]{40}", commit)
            fixed = git(repo, "ls-tree", "-r", "--name-only", commit, "--",
                        "src", "tests", "examples", "pyproject.toml").decode().splitlines()
            assert set(fixed) == set(before)
            for path, digest in before.items():
                assert sha(git(repo, "show", commit + ":" + path)) == digest
        total += count
    extra = manifest(freeze["before"]["domain_checker"])
    assert extra == manifest(freeze["after"]["domain_checker"])
    assert extra == {"docs/evidence/checks/v1-07b-domain-check.py":
                     "7907844dc7cb2362efc75f56415a900b9e1ed729f1cecb8aa8c4311d3c842c3e"}
    for path, digest in extra.items():
        assert sha((ROOT / path).read_bytes()) == digest
        if not args.worktree_only:
            assert sha(git(ROOT, "show", evidence["fixed_code_commits"]["core"] + ":" + path)) == digest
    old = "ce88e12dfbc2fe6091e9c49d7996df7ab13a8e19"
    old_paths = git(ROOT, "ls-tree", "-r", "--name-only", old, "--",
                    "src", "tests", "examples", "pyproject.toml").decode().splitlines()
    assert len(old_paths) == 746
    for path in old_paths:
        assert (ROOT / path).read_bytes() == git(ROOT, "show", old + ":" + path)
    product_paths = git(ROOT, "ls-tree", "-r", "--name-only", evidence["core_product_frozen"],
                        "--", "src", "pyproject.toml").decode().splitlines()
    assert len(product_paths) == 278
    assert {p for p in manifest(freeze["before"]["core"]) if p.startswith("src/") or p == "pyproject.toml"} == set(product_paths)
    for path in product_paths:
        assert (ROOT / path).read_bytes() == git(ROOT, "show", evidence["core_product_frozen"] + ":" + path)
    additions = set(manifest(freeze["before"]["core"])) - set(old_paths)
    expected_tests = {"tests/test_holdout_domain.py": (45, 38),
        "tests/test_holdout_domain_review.py": (27, 19),
        "tests/test_holdout_product.py": (9, 41),
        "tests/test_holdout_product_review.py": (8, 73)}
    assert additions == set(expected_tests) | {"examples/holdout/" + name for name in
        ("__init__.py", "__main__.py", "instance.json", "scheduling.py", "testing.py")}
    for path, (cases, asserts) in expected_tests.items():
        assert evidence["new_tests"][path] == {"cases": cases, "literal_asserts": asserts}
        assert sum(isinstance(n, ast.Assert) for n in ast.walk(ast.parse((ROOT / path).read_bytes()))) == asserts
    assert set(evidence["new_tests"]) == set(expected_tests)
    collection = evidence["collection"]
    assert collection["result"]["exit_code"] == 0
    collected = collection["result"]["output"]
    assert "Warning: truncated output" not in collected
    footer(collected, "89 tests collected")
    nodes = [row for row in collected.splitlines() if row.startswith("tests/")]
    assert len(nodes) == len(set(nodes)) == 89
    assert Counter(row.split("::")[0] for row in nodes) == {p: v[0] for p, v in expected_tests.items()}
    challenge = (ROOT / "docs/evidence/challenges/v1-07b-original.md").read_bytes()
    assert sha(challenge) == "fd1a4795c25829e5891680980bdaaef311985e92c4878f8a1552caa260023245"
    source_json = re.search(rb"```json\s*\n(.*?)\n```", challenge, re.S)
    assert source_json and json.loads(source_json[1]) == read("examples/holdout/instance.json")
    history = read("docs/evidence/2026-09-11-v1-07b-target-history.json")
    first = history["candidate1"]
    assert first["final"]["exit_code"] == 1
    footer(first["final"]["output"], "3 failed, 85 passed")
    assert manifest(first["before"]) == manifest(first["after"])
    regression = history["regression"]
    assert regression["red"]["exit_code"] == 1 and regression["green"]["exit_code"] == 0
    footer(regression["red"]["output"], "1 failed")
    footer(regression["green"]["output"], "1 passed")
    target = history["final_target"]
    assert target["final"]["exit_code"] == 0
    footer(target["final"]["output"], "89 passed")
    assert manifest(target["before"]) == manifest(target["after"]) == manifest(freeze["before"]["core"])
    snapshots = ("testing.py", "product-tests.py")
    expected_shas = ("18a387b49dd45760bdd53ac4d7ece71b81d3ef6bd59ed447f7c54c2c167d4c27",
                     "6f6b102a6287f5a65ffdf555a0ba10899aa2ba01f079814e979630b42e8fd63f")
    prior_manifest = manifest(first["before"])
    final_manifest = manifest(target["before"])
    changed_paths = {p for p in prior_manifest if prior_manifest[p] != final_manifest.get(p)}
    assert set(prior_manifest) == set(final_manifest)
    assert changed_paths == {"examples/holdout/testing.py", "tests/test_holdout_product.py"}
    assert prior_manifest["examples/holdout/testing.py"] == expected_shas[0]
    assert prior_manifest["tests/test_holdout_product.py"] == expected_shas[1]
    for name, digest in zip(snapshots, expected_shas):
        assert sha((ROOT / ("docs/evidence/snapshots/v1-07b-before-unique-snapshot-labels-" + name)).read_bytes()) == digest
    old_test = ast.parse((ROOT / "docs/evidence/snapshots/v1-07b-before-unique-snapshot-labels-product-tests.py").read_bytes())
    current = {n.name: n for n in ast.parse((ROOT / "tests/test_holdout_product.py").read_bytes()).body
               if isinstance(n, ast.FunctionDef)}
    for node in old_test.body:
        if isinstance(node, ast.FunctionDef):
            assert ast.dump(node, include_attributes=False) == ast.dump(current[node.name], include_attributes=False)
    output = suite(freeze["suites"]["core"], "4195 passed, 7 skipped")
    suite(freeze["suites"]["pro"], "974 passed")
    suite(freeze["suites"]["paired"], "60 passed")
    report_exports(first["final"]["output"], "HOLDOUT_REPORTS", first["reports"], NAMES, "v1-07b-candidate1")
    for epoch, reports, stdout in (
        ("target", target["reports"], target["final"]["output"]),
        ("full", freeze["holdout_reports"], output)):
        report_exports(stdout, "HOLDOUT_REPORTS", reports, NAMES, "v1-07b-" + epoch)
        result = json.loads(subprocess.check_output([sys.executable,
            str(ROOT / "docs/evidence/checks/v1-07b-runs-check.py"),
            "--prefix", "v1-07b-" + epoch], cwd=ROOT))
        assert result["status"] == "passed"
        assert result["native_actions"] == 17 and result["reserved_wall_seconds"] == 34
        assert result["fresh_cli_invocations"] == 22 and result["additional_workers_started"] == 0
        measurements = [{"name": row["name"], **row["timings"],
            "native_actions": row["native_actions"], "reserved_wall_seconds": row["reserved_wall_seconds"],
            "fresh_cli_invocations": row["fresh_cli_invocations"]} for row in result["reports"]]
        same(evidence["measurements"][epoch], measurements)
        captured_runs = [read(row["archive"]) for row in reports]
        terminals = [json.loads(row["terminal_json"]) for run in captured_runs
                     for row in run["database"]["execution_attempts"]]
        reservations = [row for run in captured_runs for row in run["database"]["cpu_action_reservations"]]
        same(evidence["acceptance"]["per_complete_holdout_epoch"], {
            "private_projects": len(captured_runs), "native_actions": len(terminals),
            "reserved_wall_seconds": result["reserved_wall_seconds"],
            "fresh_cli_invocations": result["fresh_cli_invocations"],
            "completed": sum(t["outcome"] == "completed" for t in terminals),
            "failed": sum(t["outcome"] == "failed" for t in terminals),
            "settled": sum(r["state"] == "SETTLED" for r in reservations),
            "workflow_observations": sum(len(r["observations"]) for r in captured_runs if "condition" not in r),
            "boundary_observations": sum(len(r["observations"]) for r in captured_runs if "condition" in r)})
        workflow = next(row for row in result["reports"] if row["name"] == "workflow")
        verdicts = {row["task_id"]: row["verdict"] for row in workflow["verdicts"]}
        boundary = next(row for row in result["reports"] if row["name"] == "boundary")
        same(evidence["acceptance"]["domain_values"], {
            "baseline_valid_value": verdicts["idea-baseline-v1"]["scheduled_value"],
            "challenger_v1_valid_value": verdicts["idea-challenger-v1-recovered"]["scheduled_value"],
            "challenger_v2": verdicts["idea-challenger-v2"],
            "boundary_valid_value": boundary["verdicts"][0]["verdict"]["scheduled_value"]})
    report_exports(output, "ACCEPTANCE_REPORTS", freeze["known_reports"], A_NAMES, "v1-07b-known-full")
    known = json.loads(subprocess.check_output([sys.executable,
        str(ROOT / "docs/evidence/checks/v1-07a-runs-check.py"),
        "--prefix", "v1-07b-known-full"], cwd=ROOT))
    assert known["status"] == "passed" and known["native_actions"] == 16
    assert known["reserved_wall_seconds"] == 32 and known["additional_workers_started"] == 0
    assert evidence["completion"] == {
        "original_holdout_declarative_workflow": True,
        "known_domain_acceptance_preserved": True,
        "new_policy_implementation": False,
        "autonomous_holdout_proposal_policy_proven": False,
        "controller_crash_recovery_proven": False,
        "v1_all": False, "research_efficiency_gain_proven": False}
    print(json.dumps({"record_consistency": "passed", "source_files": total,
        "extra_frozen_test_imports": len(extra), "unchanged_baseline_files": len(old_paths),
        "new_cases": 89, "new_asserts": 171, "whole_core_passed": 4195,
        "whole_pro_passed": 974, "paired_passed": 60,
        "fixed_commit_checked": not args.worktree_only}, sort_keys=True))


if __name__ == "__main__":
    main()
