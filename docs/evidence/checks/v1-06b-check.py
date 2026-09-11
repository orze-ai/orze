"""Read-only V1-06B evidence verifier. Run from the Core repository root.

Checks recorded bytes, exact commits, test preservation and command syntax.
Does not execute replay commands, tests, providers or live experiments.
--worktree-only is pre-commit validation, not proof of remote publication.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--worktree-only", action="store_true")
args = parser.parse_args()
root = Path.cwd()
repos = {"core": root, "pro": root.parent / "orze-pro"}
evidence = json.loads((root / "docs/evidence/2026-09-11-v1-06b-research-interfaces.json").read_bytes())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(repo, *words):
    return subprocess.check_output(["git", "-C", str(repos[repo]), *words])


def referenced(ref):
    raw = (root / ref["path"]).read_bytes()
    assert digest(raw) == ref["sha256"], ref["path"]
    return json.loads(raw)


def manifest_digest(entries):
    return digest("".join(f"{entries[path]}  {path}\n" for path in sorted(entries)).encode())


assert evidence["schema"] == 1 and evidence["slice"] == "V1-06B"
for key in ("whole_v1_complete", "whole_v1_06_complete", "interface_frozen", "holdout_opened",
            "cpu_same_request_replica_implemented"):
    assert evidence[key] is False, key
assert evidence["accepted_epoch"] == "first_frozen_full_validation"
inventory = referenced(evidence["source_inventory"])
validation = referenced(evidence["root_validation"])
provenance = referenced(evidence["snapshot_provenance"])
assert validation["final_epoch"] == evidence["final_epoch"]
assert inventory["source_bytes_equal_before_after"] is True
assert evidence["final_epoch"]["source_bytes_equal_before_after"] is True
assert validation["test_classification"]["old_tracked_tests_modified"] == 0
assert validation["safety"]["holdout_opened"] is False

count = 0
for repo, entries in inventory["files"].items():
    actual = sorted(set(p.decode() for p in git(repo, "ls-files", "-c", "-o",
        "--exclude-standard", "-z", "--", "src", "tests", "pyproject.toml").split(b"\0") if p))
    assert actual == sorted(entries), (repo, "inventory paths")
    if not args.worktree_only:
        git(repo, "merge-base", "--is-ancestor", evidence["code_commits"][repo], "HEAD")
    for path, expected in entries.items():
        assert digest((repos[repo] / path).read_bytes()) == expected, (repo, path, "worktree")
        if not args.worktree_only:
            assert digest(git(repo, "show", evidence["code_commits"][repo] + ":" + path)) == expected, (repo, path, "commit")
        count += 1
    assert len(entries) == evidence["final_epoch"]["file_counts"][repo]
    for point in ("before", "after"):
        capture = inventory["manifest_captures"][point][repo]
        assert capture["exit_code"] == 0 and capture["file_count"] == len(entries)
        assert capture["stdout_sha256"] == manifest_digest(entries)
assert count == 935

commands, heredocs, inline = set(), set(), set()


def inspect_commands(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if (key in {"cmd", "command"} or key.endswith("_command")) and isinstance(child, str):
                commands.add(child)
            inspect_commands(child)
    elif isinstance(value, list):
        for child in value:
            inspect_commands(child)


for record in (evidence, validation, inventory, provenance):
    inspect_commands(record)
for ref in evidence["supporting_files"]:
    raw = (root / ref["path"]).read_bytes()
    assert digest(raw) == ref["sha256"], ref["path"]
    if ref["path"].endswith(".json"):
        inspect_commands(json.loads(raw))
    elif ref["path"].endswith(".py"):
        ast.parse(raw, filename=ref["path"])

snapshots = evidence["snapshots"]
assert snapshots == provenance["snapshots"] and len(snapshots) == 21
assert {p.as_posix() for p in Path("docs/evidence/snapshots").glob("v1-06b-*.py")} == {s["path"] for s in snapshots}
baseline_count = 0
for item in snapshots:
    raw = (root / item["path"]).read_bytes()
    assert digest(raw) == item["sha256"] and len(raw) == item["bytes"], item["path"]
    ast.parse(raw, filename=item["path"])
    assert digest((root / item["source"]).read_bytes()) == item["current_source_sha256"]
    if item["origin"] == "baseline":
        assert item["baseline_commit"] == evidence["baseline_commits"]["core"]
        assert raw == git("core", "show", item["baseline_commit"] + ":" + item["source"])
        baseline_count += 1
    else:
        assert item["baseline_byte_exact"] is None
        assert item["not_a_released_or_prior_commit_behavior_red"] is True
assert baseline_count == 7

fixture_asserts = 0
for audit in provenance["fixture_corrections"].values():
    old = ast.parse((root / audit["snapshot"]).read_bytes())
    new = ast.parse((root / audit["current"]).read_bytes())
    # Additional new assertions are allowed; no original Assert may disappear
    # or change, even when a fixture's setup/purpose/raises precondition changes.
    before = [ast.dump(n) for n in ast.walk(old) if isinstance(n, ast.Assert)]
    after = [ast.dump(n) for n in ast.walk(new) if isinstance(n, ast.Assert)]
    remaining = iter(after)
    assert all(any(item == candidate for candidate in remaining) for item in before)
    assert len(before) == audit["ast_audit"]["old_assertions"]
    fixture_asserts += len(before)
assert fixture_asserts == 66

tests = evidence["test_inventory"]["files"]
static_refs = [ref for ref in evidence["supporting_files"]
               if ref["path"].endswith("v1-06b-final-static-review.json")]
assert len(static_refs) == 1
static_review = referenced(static_refs[0])
assert json.loads(static_review["collection"]["raw_output"]) == static_review["collection"]["result"]
assert tests == static_review["collection"]["result"]["files"]
assert len(tests) == 11
assert sum(t["cases"] for t in tests) == 104
assert sum(t["literal_asserts"] for t in tests) == 203
for entry in tests:
    raw = (root / entry["path"]).read_bytes()
    assert digest(raw) == entry["sha256"]
    assert sum(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(raw))) == entry["literal_asserts"]
expected_new = {entry["path"] for entry in tests}
end = [] if args.worktree_only else [evidence["code_commits"]["core"]]
changes = git("core", "diff", "--name-status", evidence["baseline_commits"]["core"], *end,
              "--", "src", "tests", "pyproject.toml").decode().splitlines()
test_changes = [line for line in changes if line.split("\t")[-1].startswith("tests/")]
assert all(line.startswith("A\t") for line in test_changes), "old tests modified"
added = {line.split("\t")[-1] for line in test_changes}
if args.worktree_only:
    added |= set(git("core", "ls-files", "--others", "--exclude-standard", "--", "tests").decode().splitlines())
assert added == expected_new
all_changes = list(changes)
if args.worktree_only:
    all_changes += ["A\t" + path for path in git("core", "ls-files", "--others",
        "--exclude-standard", "--", "src", "tests", "pyproject.toml").decode().splitlines()]
assert sorted(all_changes) == sorted(evidence["changed_code_paths"])
pro_end = [] if args.worktree_only else [evidence["code_commits"]["pro"]]
assert git("pro", "diff", "--name-only", evidence["baseline_commits"]["pro"], *pro_end,
           "--", "src", "tests", "pyproject.toml") == b""
assert not git("pro", "ls-files", "--others", "--exclude-standard", "--", "src", "tests", "pyproject.toml")

for key, expected in {"core": (3835, 7), "pro": (974, 0), "paired": (60, 0), "target": (104, 0)}.items():
    run = evidence["final_epoch"]["results"][key]
    assert run["exit_code"] == 0 and (run["passed"], run["skipped"]) == expected
    assert "".join(chunk["output"] for chunk in run["tool_chunks"]) == run["complete_output"]
    assert run["tool_chunks"][-1]["exit_code"] == run["exit_code"]
    pattern = str(expected[0]) + " passed" + (f", {expected[1]} skipped" if expected[1] else "")
    assert re.fullmatch(pattern + r" in [0-9.]+s(?: \([^\n]+\))?", run["complete_output"].strip().splitlines()[-1])
    elapsed = re.search(r" in ([0-9.]+)s", run["complete_output"].strip().splitlines()[-1])
    assert float(elapsed[1]) == run["seconds"]

for command in sorted(commands):
    subprocess.run(["bash", "-n"], input=command.encode(), check=True)
    for match in re.finditer(r"<<'(?P<delimiter>PY[A-Z0-9_]*)'\n(?P<body>.*?)\n(?P=delimiter)(?:\n|$)", command, re.S):
        ast.parse(match["body"])
        heredocs.add(match["body"])
    if "<<" not in command:
        words = shlex.split(command)
        for index, word in enumerate(words[:-2]):
            if Path(word).name in {"python", "python3"} and words[index + 1] == "-c":
                ast.parse(words[index + 2])
                inline.add(words[index + 2])
print(json.dumps({"result": "verified", "mode": "worktree_only" if args.worktree_only else "worktree_and_commits",
                  "source_files": count, "supporting_files": len(evidence["supporting_files"]),
                  "snapshots": 21, "baseline_snapshots": baseline_count,
                  "new_test_files": 11, "new_cases": 104, "literal_asserts": 203,
                  "new_fixture_original_asserts_preserved": fixture_asserts,
                  "shell_commands": len(commands), "python_heredocs": len(heredocs), "python_inline": len(inline),
                  "cpu_same_request_replica_implemented": False, "whole_v1_06_complete": False,
                  "whole_v1_complete": False}, sort_keys=True))
