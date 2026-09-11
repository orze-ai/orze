#!/usr/bin/env python3
"""Read-only evidence consistency check, not cryptographic execution attestation."""
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
CANONICAL = ROOT / "docs/evidence/2026-09-11-v1-07a-cross-domain.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(relative):
    return json.loads((ROOT / relative).read_bytes())


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def manifest(output):
    lines = output.splitlines()
    assert all(re.fullmatch(r"[0-9a-f]{64}  .+", line) for line in lines)
    result = {line[66:]: line[:64] for line in lines}
    assert len(lines) == len(result)
    return result


def footer(output, expected):
    pattern = re.escape(expected) + r" in [0-9]+(?:\.[0-9]+)?s(?: \([^\n]*\))?"
    assert any(re.fullmatch(pattern, line.strip("= ")) for line in output.splitlines()), output[-3000:]


def suite(record, expected):
    chunks = record["chunks"]
    assert chunks and chunks[-1]["exit_code"] == 0
    assert all(item.get("exit_code") is None for item in chunks[:-1])
    output = "".join(item["output"] for item in chunks)
    assert "Warning: truncated output" not in output
    assert not re.search(r"\b[1-9][0-9]* failed\b|Interrupted:|KeyboardInterrupt", output)
    footer(output, expected)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree-only", action="store_true",
                        help="before code commit only; never a fixed-commit verification")
    args = parser.parse_args()
    evidence = json.loads(CANONICAL.read_bytes())
    validation_ref = evidence["root_validation"]
    assert sha((ROOT / validation_ref["path"]).read_bytes()) == validation_ref["sha256"]
    validation = read(validation_ref["path"])
    expected_support = evidence["support"]
    assert len({item["path"] for item in expected_support}) == len(expected_support)
    names = {domain + "_" + variant for domain in ("sorting", "compression")
             for variant in ("default", "counterfactual")}
    required = {"docs/cross-domain-acceptance.md",
        "docs/plans/2026-09-11-v1-07-acceptance.zh-CN.md",
        "docs/evidence/checks/v1-07a-check.py", "docs/evidence/checks/v1-07a-runs-check.py",
        "docs/evidence/snapshots/v1-07a-before-report-export-testing.py",
        "docs/evidence/snapshots/v1-07-before-unchecked-oracle-compression.py",
        "docs/evidence/snapshots/v1-07-before-unchecked-oracle-compression-tests.py",
        "docs/evidence/2026-09-11-v1-07a-target-history.json"}
    required.update("docs/evidence/2026-09-11-v1-07-" + label + ".json" for label in (
        "source-inventory-review", "compression-author", "sorting-author",
        "domain-independent-review", "policy-independent-review"))
    required.update("docs/evidence/runs/v1-07a-" + epoch + "-" + name + ".json"
                    for epoch in ("target", "full") for name in names)
    assert {item["path"] for item in expected_support} == required
    for item in expected_support:
        assert sha((ROOT / item["path"]).read_bytes()) == item["sha256"], item["path"]
    before, after = validation["source_before"], validation["source_after"]
    total = 0
    for label, repo, count in (("core", ROOT, 746), ("pro", PRO, 223)):
        assert before[label]["exit_code"] == after[label]["exit_code"] == 0
        start, end = manifest(before[label]["output"]), manifest(after[label]["output"])
        assert start == end and len(start) == count
        live = git(repo, "ls-files", "-c", "-o", "--exclude-standard", "--",
                   "src", "tests", "examples", "pyproject.toml").decode().splitlines()
        assert set(live) == set(start)
        for path, digest in start.items():
            assert sha((repo / path).read_bytes()) == digest, (label, path)
        if not args.worktree_only:
            commit = evidence["fixed_code_commits"][label]
            assert re.fullmatch(r"[0-9a-f]{40}", commit)
            fixed = git(repo, "ls-tree", "-r", "--name-only", commit, "--",
                        "src", "tests", "examples", "pyproject.toml").decode().splitlines()
            assert set(fixed) == set(start)
            for path, digest in start.items():
                assert sha(git(repo, "show", commit + ":" + path)) == digest
        total += count
    assert evidence["inventory_path"] == "docs/evidence/2026-09-11-v1-07-source-inventory-review.json"
    assert evidence["target_history_path"] == "docs/evidence/2026-09-11-v1-07a-target-history.json"
    inventory_report = read(evidence["inventory_path"])
    inventory = inventory_report["static_audit"]["parsed_actual_stdout"]
    assert inventory["baseline"] == evidence["core_implementation_frozen"]
    assert inventory["counts"] == {
        "example_python_files": 7, "new_literal_asserts": 147, "new_test_files": 4,
        "old_python_tests": 457, "old_test_files": 457, "source_and_build_files": 278}
    for group in ("sources", "old_tests"):
        paths = ("src", "pyproject.toml") if group == "sources" else ("tests",)
        baseline_paths = git(ROOT, "ls-tree", "-r", "--name-only", inventory["baseline"],
                             "--", *paths).decode().splitlines()
        assert set(inventory[group]) == set(baseline_paths)
        assert len(baseline_paths) == (278 if group == "sources" else 457)
        for path, item in inventory[group].items():
            actual = (ROOT / path).read_bytes()
            assert sha(actual) == item["sha256"]
            assert actual == git(ROOT, "show", inventory["baseline"] + ":" + path)
    assert "tests/test_acceptance_matrix.py" not in inventory["new_tests"]
    counts = {"tests/test_acceptance_sorting.py": 22, "tests/test_acceptance_compression.py": 31,
              "tests/test_acceptance_policy_review.py": 8, "tests/test_acceptance_product.py": 9}
    assert set(inventory["new_tests"]) == set(counts)
    assert set(inventory["examples"]) == {"examples/acceptance/" + name + ".py" for name in (
        "__init__", "__main__", "common", "compression", "policy", "sorting", "testing")}
    for path, item in inventory["examples"].items():
        assert sha((ROOT / path).read_bytes()) == item["sha256"]
    collection = inventory_report["collection"]
    assert collection["exit_code"] == 0
    footer(collection["output"], "70 tests collected")
    nodes = [line for line in collection["output"].splitlines() if line.startswith("tests/")]
    assert len(nodes) == len(set(nodes)) == 70
    assert Counter(node.split("::")[0] for node in nodes) == counts
    asserts = 0
    for path, item in inventory["new_tests"].items():
        raw = (ROOT / path).read_bytes()
        count = sum(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(raw)))
        assert sha(raw) == item["sha256"] and count == item["literal_assert_count"]
        asserts += count
    assert asserts == 147
    old_tests = ast.parse((ROOT / "docs/evidence/snapshots/v1-07-before-unchecked-oracle-compression-tests.py").read_bytes())
    final_tests = ast.parse((ROOT / "tests/test_acceptance_compression.py").read_bytes())
    new_nodes = {node.name: node for node in final_tests.body if isinstance(node, ast.FunctionDef)}
    for node in old_tests.body:
        if isinstance(node, ast.FunctionDef):
            assert ast.dump(node, include_attributes=False) == ast.dump(new_nodes[node.name], include_attributes=False)
    history = read(evidence["target_history_path"])
    assert history["first_candidate"]["final"]["exit_code"] == 0
    footer(history["first_candidate"]["final"]["output"], "17 passed")
    assert history["target"]["final"]["exit_code"] == 0
    footer(history["target"]["final"]["output"], "70 passed")
    assert history["target"]["native_actions"] == 16
    compression = read("docs/evidence/2026-09-11-v1-07-compression-author.json")
    finding = compression["candidate_finding"]
    for key in ("first_red", "whole_candidate_replay"):
        assert finding[key]["exit_code"] == 1
        assert "unchecked measurement must defer scientific validation" in finding[key]["output"]
        footer(finding[key]["output"], "1 failed")
    assert compression["final_bounded"]["exit_code"] == 0
    footer(compression["final_bounded"]["output"], "31 passed")
    output = suite(validation["suites"]["core"], "4106 passed, 7 skipped")
    suite(validation["suites"]["pro"], "974 passed")
    suite(validation["suites"]["paired"], "60 passed")
    for epoch, reports, captured_output in (
            ("target", history["target"]["reports"], history["target"]["final"]["output"]),
            ("full", validation["full_reports"], output)):
        assert captured_output.count("ACCEPTANCE_REPORTS=") == 1
        captured = json.loads(captured_output.split("ACCEPTANCE_REPORTS=")[1].split("\n")[0])
        assert len(captured) == len(reports) == 4
        assert {row["name"] for row in captured} == {row["name"] for row in reports} == names
        assert sorted(captured, key=lambda row: row["name"]) == sorted(
            [{key: row[key] for key in ("name", "path", "bytes", "sha256")} for row in reports],
            key=lambda row: row["name"])
        for item in reports:
            assert item["archived_path"] == "docs/evidence/runs/v1-07a-" + epoch + "-" + item["name"] + ".json"
            raw = (ROOT / item["archived_path"]).read_bytes()
            assert len(raw) == item["bytes"] and sha(raw) == item["sha256"]
        verified = json.loads(subprocess.check_output([sys.executable,
            str(ROOT / "docs/evidence/checks/v1-07a-runs-check.py"),
            "--prefix", "v1-07a-" + epoch], cwd=ROOT))
        assert verified["status"] == "passed" and verified["native_actions"] == 16
        assert verified["reserved_wall_seconds"] == 32 and verified["additional_workers_started"] == 0
        assert len(verified["reports"]) == 4
        assert {Path(row["path"]).name: row["sha256"] for row in verified["reports"]} == {
            Path(row["archived_path"]).name: row["sha256"] for row in reports}
    assert evidence["completion"] == {
        "v1_07a_known_domains": True, "v1_07_all": False, "v1_all": False,
        "core_interfaces_frozen_for_holdout": True, "holdout_opened": False,
        "research_efficiency_gain_proven": False}
    print(json.dumps({"source_files": total, "unchanged_old_tests": 457,
        "new_cases": 70, "new_asserts": asserts, "whole_core_passed": 4106,
        "whole_pro_passed": 974, "paired_passed": 60, "record_consistency": "passed",
        "fixed_commit_checked": not args.worktree_only}, sort_keys=True))


if __name__ == "__main__":
    main()
