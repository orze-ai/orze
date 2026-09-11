"""Read-only V1-06C evidence verifier. Run from the Core repository root.

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
evidence = json.loads((root / "docs/evidence/2026-09-11-v1-06c-cpu-replication.json").read_bytes())


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


assert evidence["schema"] == 1 and evidence["slice"] == "V1-06C"
for key in ("whole_v1_complete", "whole_v1_06_complete", "interface_frozen", "holdout_opened"):
    assert evidence[key] is False, key
assert evidence["cpu_same_request_replica_implemented"] is True
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
assert count == 943

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
assert snapshots == provenance["snapshots"] and len(snapshots) == 17
assert {p.as_posix() for p in Path("docs/evidence/snapshots").glob("v1-06c-*.py")} == {s["path"] for s in snapshots}
baseline_count = 0
baseline_equal = 0
for item in snapshots:
    raw = (root / item["path"]).read_bytes()
    assert digest(raw) == item["sha256"] and len(raw) == item["bytes"], item["path"]
    ast.parse(raw, filename=item["path"])
    assert digest((root / item["source"]).read_bytes()) == item["current_source_sha256"]
    if item["origin"] == "baseline":
        assert item["baseline_commit"] == evidence["baseline_commits"]["core"]
        assert raw == git("core", "show", item["baseline_commit"] + ":" + item["source"])
        baseline_count += 1
    if item["baseline_byte_exact"] is True:
        assert raw == git("core", "show", evidence["baseline_commits"]["core"] + ":" + item["source"])
        baseline_equal += 1
assert baseline_count == 7
assert baseline_equal == 8

# Each new-fixture correction is explicit; old tracked tests stay byte-exact.
# C includes two Assert AST changes that repair setup/accessor premises, so do
# not reuse B's claim that every new-fixture assertion remains text-identical.
fixture_asserts = 0
fixture_asserts_changed = 0
for audit in provenance["fixture_corrections"].values():
    old = ast.parse((root / audit["snapshot"]).read_bytes())
    new = ast.parse((root / audit["current"]).read_bytes())
    before_functions = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
    after_functions = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
    assert set(audit["exact_per_function_audit"]) == set(before_functions)
    for name, before_function in before_functions.items():
        after_function = after_functions[name]
        stats = audit["exact_per_function_audit"][name]
        before = [n for n in ast.walk(before_function) if isinstance(n, ast.Assert)]
        after = [n for n in ast.walk(after_function) if isinstance(n, ast.Assert)]
        assert len(before) == stats["before_count"]
        assert len(after) == stats["after_count"]
        assert len(before) == len(after), "changed original fixture assertion count"
        changes = [{"index_in_function": index, "before": ast.unparse(a),
                    "after": ast.unparse(b), "before_ast": ast.dump(a), "after_ast": ast.dump(b)}
                   for index, (a, b) in enumerate(zip(before, after)) if ast.dump(a) != ast.dump(b)]
        assert changes == stats["changed"]
        assert len(before) - len(changes) == stats["unchanged"]
        assert (ast.dump(before_function) == ast.dump(after_function)) is stats["whole_function_ast_exact"]
        fixture_asserts += len(before)
        fixture_asserts_changed += len(changes)
assert (fixture_asserts, fixture_asserts_changed) == (76, 2)
initial = provenance["reconstructed_initial_runtime_fixture"]
assert digest((root / initial["path"]).read_bytes()) == initial["sha256"]
original = ast.parse((root / initial["path"]).read_bytes())
current = ast.parse((root / "tests/test_cpu_replication.py").read_bytes())
current_functions = {n.name: n for n in current.body if isinstance(n, ast.FunctionDef)}
for node in original.body:
    if isinstance(node, ast.FunctionDef):
        assert ast.dump(node) == ast.dump(current_functions[node.name])

tests = evidence["test_inventory"]["files"]
static_refs = [ref for ref in evidence["supporting_files"]
               if ref["path"].endswith("v1-06c-interfaces-independent-review.json")]
assert len(static_refs) == 1
static_review = referenced(static_refs[0])
collection_lines = static_review["collection"]["raw_output"].splitlines()
collection_json = [line.removeprefix("COLLECTION ") for line in collection_lines
                   if line.startswith("COLLECTION ")]
assert len(collection_json) == 1
collected = json.loads(collection_json[0])
assert [{"path": path, **collected[path]} for path in sorted(collected)] == tests
assert tests == static_review["collection"]["result"]["files"]
nodeids = [line for line in collection_lines if line.startswith("tests/") and "::" in line]
assert len(nodeids) == len(set(nodeids)) == 75
for entry in tests:
    assert sum(node.split("::")[0] == entry["path"] for node in nodeids) == entry["cases"]
assert len(tests) == 7
assert sum(t["cases"] for t in tests) == 75
assert sum(t["literal_asserts"] for t in tests) == 166
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

for key, expected in {"core": (3910, 7), "pro": (974, 0), "paired": (60, 0), "target": (75, 0)}.items():
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
                  "snapshots": 17, "primary_baseline_snapshots": baseline_count,
                  "baseline_equal_snapshots": baseline_equal,
                  "new_test_files": 7, "new_cases": 75, "literal_asserts": 166,
                  "new_fixture_original_asserts": fixture_asserts,
                  "new_fixture_assert_changes": fixture_asserts_changed,
                  "shell_commands": len(commands), "python_heredocs": len(heredocs), "python_inline": len(inline),
                  "cpu_same_request_replica_implemented": True, "whole_v1_06_complete": False,
                  "whole_v1_complete": False}, sort_keys=True))
