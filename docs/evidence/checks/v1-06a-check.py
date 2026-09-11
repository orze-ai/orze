"""Read-only V1-06A evidence verifier; run from the Core repository root.

Verifies recorded bytes, ancestry, assertions and command syntax. Does not run
replays, tests, providers or live experiments. --worktree-only is explicitly a
pre-commit check and is not evidence of committed or remotely published bytes.
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
canonical_path = root / "docs/evidence/2026-09-11-v1-06a-cpu-actions.json"
evidence = json.loads(canonical_path.read_bytes())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def manifest_digest(entries):
    # Exact sha256sum stdout format used by the captured sorted file command.
    return digest("".join(f"{entries[path]}  {path}\n" for path in sorted(entries)).encode())


def git(repo, *words):
    return subprocess.check_output(["git", "-C", str(repos[repo]), *words])


def referenced(ref):
    raw = (root / ref["path"]).read_bytes()
    assert digest(raw) == ref["sha256"], ref["path"]
    return json.loads(raw)


assert evidence["schema"] == 1 and evidence["slice"] == "V1-06A"
assert evidence["whole_v1_complete"] is False
assert evidence["whole_v1_06_complete"] is False
assert evidence["interface_frozen"] is False
assert evidence["holdout_opened"] is False
assert evidence["accepted_epoch"] == "second_frozen_full_validation"
inventory = referenced(evidence["source_inventory"])
validation = referenced(evidence["root_validation"])
provenance = referenced(evidence["snapshot_provenance"])
assert validation["final_epoch"] == evidence["final_epoch"]
file_count = 0
for repo, entries in inventory["files"].items():
    actual_paths = sorted(set(p.decode() for p in git(
        repo, "ls-files", "-c", "-o", "--exclude-standard", "-z", "--",
        "src", "tests", "pyproject.toml").split(b"\0") if p))
    assert actual_paths == sorted(entries), (repo, "source inventory changed")
    if not args.worktree_only:
        git(repo, "merge-base", "--is-ancestor", evidence["code_commits"][repo], "HEAD")
    for path, expected in entries.items():
        assert digest((repos[repo] / path).read_bytes()) == expected, (repo, path, "worktree")
        if not args.worktree_only:
            assert digest(git(repo, "show", evidence["code_commits"][repo] + ":" + path)) == expected, (repo, path, "commit")
        file_count += 1
    assert len(entries) == evidence["final_epoch"]["file_counts"][repo]
    for point in ("before", "after"):
        capture = inventory["manifest_captures"][point][repo]
        assert capture["exit_code"] == 0
        assert capture["file_count"] == len(entries)
        assert capture["stdout_sha256"] == manifest_digest(entries)
assert file_count == 919
assert inventory["source_bytes_equal_before_after"] is True
assert evidence["final_epoch"]["source_bytes_equal_before_after"] is True

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


for record in (validation, provenance, inventory):
    inspect_commands(record)


for ref in evidence["supporting_files"]:
    path = root / ref["path"]
    raw = path.read_bytes()
    assert digest(raw) == ref["sha256"], ref["path"]
    if path.suffix == ".json":
        inspect_commands(json.loads(raw))
    elif path.suffix == ".py":
        ast.parse(raw, filename=str(path))

# Candidate hashes belong to their own historical epoch, not final source.
snapshots = evidence["snapshots"]
assert snapshots == provenance["snapshots"]
assert len(snapshots) == 36
assert {p.as_posix() for p in Path("docs/evidence/snapshots").glob("v1-06a-*.py")} == {s["path"] for s in snapshots}
baseline_count = 0
for snapshot in snapshots:
    raw = (root / snapshot["path"]).read_bytes()
    assert digest(raw) == snapshot["sha256"], snapshot["path"]
    ast.parse(raw, filename=snapshot["path"])
    if snapshot.get("commit"):
        assert snapshot["commit"] == evidence["baseline_commits"]["core"]
        assert raw == git("core", "show", snapshot["commit"] + ":" + snapshot["source"])
        baseline_count += 1
    else:
        assert snapshot["origin"] in {
            "actual_uncommitted_implementation_draft_candidate",
            "complete_new_test_fixture_version",
        }
assert baseline_count == 14
fixture_asserts = fixture_preserved = 0
for audit in provenance["fixture_audits"]:
    before = (root / audit["snapshot"]).read_bytes()
    after = (root / audit["source"]).read_bytes()
    assert digest(before) == audit["snapshot_sha256"]
    assert digest(after) == audit["current_sha256"]
    original = [ast.dump(n) for n in ast.walk(ast.parse(before)) if isinstance(n, ast.Assert)]
    current = [ast.dump(n) for n in ast.walk(ast.parse(after)) if isinstance(n, ast.Assert)]
    assert len(original) == audit["before_assert_count"]
    assert len(current) == audit["after_assert_count"]
    same = sum(a == b for a, b in zip(original, current))
    assert same == audit["positional_exact_asserts"]
    assert (original == current) is audit["all_original_assert_ast_exact"]
    fixture_asserts += len(original)
    fixture_preserved += same
assert (fixture_asserts, fixture_preserved) == (65, 64)

inspect_commands(evidence)
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

tests = evidence["test_inventory"]["files"]
assert len(tests) == 13
assert sum(t["cases"] for t in tests) == 92
assert sum(t["literal_asserts"] for t in tests) == 218
for entry in tests:
    raw = (root / entry["path"]).read_bytes()
    assert digest(raw) == entry["sha256"]
    assert sum(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(raw))) == entry["literal_asserts"]
expected_new = {entry["path"] for entry in tests}
if args.worktree_only:
    changed = git("core", "diff", "--name-status", evidence["baseline_commits"]["core"], "--", "tests").decode().splitlines()
    assert all(line.startswith("A\t") for line in changed), "old tests changed before commit"
    staged = {line.split("\t")[1] for line in changed}
    untracked = set(git("core", "ls-files", "--others", "--exclude-standard", "--", "tests").decode().splitlines())
    assert staged | untracked == expected_new and not staged & untracked
else:
    changed = git("core", "diff", "--name-status", evidence["baseline_commits"]["core"], evidence["code_commits"]["core"], "--", "tests").decode().splitlines()
    assert {line.split("\t")[1] for line in changed} == expected_new
    assert all(line.startswith("A\t") for line in changed), "old tests changed"
    assert git("pro", "diff", "--name-only", evidence["baseline_commits"]["pro"], evidence["code_commits"]["pro"], "--", "src", "tests", "pyproject.toml") == b""


def verify_footer(run, passed, skipped, failed=0):
    assert run["exit_code"] == (1 if failed else 0)
    footer = run["complete_output"].strip().splitlines()[-1]
    pattern = (rf"{failed} failed, " if failed else "") + rf"{passed} passed"
    if skipped:
        pattern += rf", {skipped} skipped"
    assert re.fullmatch(pattern + r" in [0-9.]+s(?: \([^\n]+\))?", footer), footer


for key, counts in {"core": (3731, 7), "pro": (974, 0), "paired": (60, 0), "target": (92, 0)}.items():
    run = evidence["final_epoch"]["results"][key]
    assert (run["passed"], run["skipped"]) == counts
    verify_footer(run, *counts)
discarded = validation["discarded_first_full_epoch"]
assert discarded["accepted_as_final"] is False
verify_footer(discarded["results"]["core"], 3727, 7, 3)
assert len(discarded["failures"]) == 3
for nodeid in discarded["failures"]:
    # --tb=short -rs prints path/line and the case title separately, not a
    # FAILED nodeid summary. Require both actual trace components.
    path, case = nodeid.split("::", 1)
    assert path in discarded["results"]["core"]["complete_output"]
    assert case in discarded["results"]["core"]["complete_output"]
delta = discarded["delta_to_second_epoch"]
assert delta["pro"] == [] and {item["path"] for item in delta["core"]} == {
    "src/orze/engine/cpu_phase.py", "src/orze/engine/launcher.py",
    "src/orze/engine/phases.py", "tests/test_cpu_action_shared_idle.py"}
for item in delta["core"]:
    assert item["second_epoch_sha256"] == inventory["files"]["core"][item["path"]]
    if item["before_sha256"] is not None:
        assert any(s["source"] == item["path"] and s["sha256"] == item["before_sha256"] for s in snapshots)
for repo in ("core", "pro"):
    preceding = dict(inventory["files"][repo])
    for item in delta[repo]:
        if item["before_sha256"] is None:
            del preceding[item["path"]]
        else:
            preceding[item["path"]] = item["before_sha256"]
    capture = discarded["before_manifest_captures"][repo]
    assert capture["exit_code"] == 0
    assert len(preceding) == capture["file_count"] == discarded["before_file_counts"][repo]
    assert manifest_digest(preceding) == capture["stdout_sha256"]
assert validation["test_classification"]["old_tracked_tests_modified"] == 0
assert validation["safety"]["holdout_opened"] is False
print(json.dumps({"result": "verified", "mode": "worktree_only" if args.worktree_only else "worktree_and_commits",
                  "source_files": file_count, "supporting_files": len(evidence["supporting_files"]),
                  "snapshots": len(snapshots), "baseline_snapshots": baseline_count,
                  "new_test_files": len(tests), "new_cases": 92, "literal_asserts": 218,
                  "fixture_asserts": fixture_asserts, "fixture_asserts_preserved": fixture_preserved,
                  "shell_commands": len(commands), "python_heredocs": len(heredocs),
                  "python_inline": len(inline), "whole_v1_06_complete": False,
                  "whole_v1_complete": False}, sort_keys=True))
