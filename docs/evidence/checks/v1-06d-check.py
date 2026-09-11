"""Read-only V1-06D evidence verifier. Run from the Core repository root.

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
evidence = json.loads((root / "docs/evidence/2026-09-11-v1-06d-cpu-policy-proposals.json").read_bytes())


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


assert evidence["schema"] == 1 and evidence["slice"] == "V1-06D"
for key in ("whole_v1_complete", "whole_v1_06_complete", "interface_frozen", "holdout_opened"):
    assert evidence[key] is False, key
assert evidence["policy_proposals_implemented"] is True
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
assert count == 953

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
assert snapshots == provenance["snapshots"] and len(snapshots) == 14
assert {p.as_posix() for p in Path("docs/evidence/snapshots").glob("v1-06d-*.py")} == {s["path"] for s in snapshots}
baseline_count = 0
baseline_equal = 0
for item in snapshots:
    raw = (root / item["path"]).read_bytes()
    assert digest(raw) == item["sha256"] and len(raw) == item["bytes"], item["path"]
    ast.parse(raw, filename=item["path"])
    assert digest((root / item["source"]).read_bytes()) == item["current_source_sha256"]
    if item["origin"] == "baseline":
        assert raw == git("core", "show", evidence["baseline_commits"]["core"] + ":" + item["source"])
        baseline_count += 1
    if item["baseline_byte_exact"] is True:
        assert raw == git("core", "show", evidence["baseline_commits"]["core"] + ":" + item["source"])
        baseline_equal += 1
assert baseline_count == 4
assert baseline_equal == 4

# Parameterization expands new cases without weakening their original assertions.
# Reconstructed original fixtures are explicitly not pre-edit disk captures.
def assertions(raw):
    tree = ast.parse(raw)
    return {node.name: [ast.dump(item) for item in ast.walk(node) if isinstance(item, ast.Assert)]
            for node in tree.body if isinstance(node, ast.FunctionDef)}

snaproot = root / "docs/evidence/snapshots"
old_tx = (snaproot / "v1-06d-first-proposal-transaction-review-tests.py").read_bytes()
new_tx = (root / "tests/test_cpu_proposals_transaction_review.py").read_bytes()
assert assertions(old_tx) == assertions(new_tx)
assert sum(map(len, assertions(old_tx).values())) == 22
normal_path = root / "tests/test_proposal_admission_transactions.py"
normal = normal_path.read_text()
original_normal = normal.replace('@pytest.mark.parametrize("table", ["ideas", "IdEaS"])\n', '').replace(
    'def test_temporary_catalog_cannot_redirect_normal_admission(lake, table):',
    'def test_temporary_catalog_cannot_redirect_normal_admission(lake):').replace(
    'lake.conn.execute(f"CREATE TEMP TABLE {table} AS SELECT * FROM main.ideas")',
    'lake.conn.execute("CREATE TEMP TABLE ideas AS SELECT * FROM main.ideas")')
assert digest(original_normal.encode()) == "c3fd4374fd68f3ec420b4edda190db2c90ca12db4dd226effdd539a1bb640d01"
assert assertions(original_normal) == assertions(normal)
assert sum(map(len, assertions(original_normal).values())) == 48
old_product = (snaproot / "v1-06d-root-product-first-tests.py").read_bytes()
assert digest(old_product) == "cc08d33b424820f9a2ec89e74e7aad9a22e548537ebe4cf43132c788570b3129"
current_functions = {node.name: node for node in ast.parse(
    (root / "tests/test_cpu_proposal_product.py").read_bytes()).body if isinstance(node, ast.FunctionDef)}
for node in ast.parse(old_product).body:
    if isinstance(node, ast.FunctionDef):
        assert ast.dump(node) == ast.dump(current_functions[node.name])
assert sum(map(len, assertions(old_product).values())) == 23

tests = evidence["test_inventory"]["files"]
assert tests == provenance["collection"]["files"]
collection = validation["collection"]
assert collection["complete"]["exit_code"] == 0
collection_lines = collection["complete"]["output"].splitlines()
assert "Warning: truncated" not in collection["complete"]["output"]
assert re.fullmatch(r"96 tests collected in [0-9.]+s", collection_lines[-1] or collection_lines[-2])
nodeids = [line for line in collection_lines if line.startswith("tests/") and "::" in line]
assert len(nodeids) == len(set(nodeids)) == 96
assert len(tests) == 8 and sum(t["cases"] for t in tests) == 96
assert sum(t["literal_asserts"] for t in tests) == 206
for entry in tests:
    raw = (root / entry["path"]).read_bytes()
    assert digest(raw) == entry["sha256"]
    assert sum(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(raw))) == entry["literal_asserts"]
    assert sum(node.split("::")[0] == entry["path"] for node in nodeids) == entry["cases"]
old_tests = [p for p in git("core", "ls-tree", "-r", "--name-only", evidence["baseline_commits"]["core"],
             "--", "tests").decode().splitlines() if p.endswith(".py")]
assert len(old_tests) == 444
for path in old_tests:
    assert (root / path).read_bytes() == git("core", "show", evidence["baseline_commits"]["core"] + ":" + path)

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

for key, expected in {"core": (4006, 7), "pro": (974, 0), "paired": (60, 0), "target": (96, 0)}.items():
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
                  "snapshots": len(snapshots), "primary_baseline_snapshots": baseline_count,
                  "baseline_equal_snapshots": baseline_equal, "new_test_files": len(tests),
                  "new_cases": 96, "literal_asserts": 206, "old_tests_byte_exact": len(old_tests),
                  "preserved_new_fixture_asserts": 93, "new_fixture_assert_changes": 0,
                  "shell_commands": len(commands), "python_heredocs": len(heredocs), "python_inline": len(inline),
                  "policy_proposals_implemented": True, "whole_v1_06_complete": False,
                  "whole_v1_complete": False}, sort_keys=True))
