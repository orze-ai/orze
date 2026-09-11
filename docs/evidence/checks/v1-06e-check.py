"""Read-only V1-06E verifier. Run from the frozen Core worktree.
Checks exact bytes, fixed commits, preserved tests and recorded command syntax.
Does not execute tests/replays or contact providers. Worktree-only is precommit.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

if not __debug__:
    raise SystemExit("Evidence verification requires assertions; do not use optimized Python.")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--worktree-only", action="store_true")
args = parser.parse_args()
root = Path.cwd()
repos = {"core": root, "pro": root.parent / "orze-pro"}
evidence = json.loads((root / "docs/evidence/2026-09-11-v1-06e-cpu-retirement.json").read_bytes())

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def git(repo, *words):
    return subprocess.check_output(["git", "-C", str(repos[repo]), *words])

def referenced(ref):
    raw = (root / ref["path"]).read_bytes()
    assert sha(raw) == ref["sha256"], ref["path"]
    return json.loads(raw)

assert evidence["schema"] == 1 and evidence["slice"] == "V1-06E"
assert evidence["baseline_commits"] == {
    "core": "188af473184463fd2b5835f57cdcc3b5ed9af569",
    "pro": "e5c3d75c00c5e157747bd39dcb59e8c3151e504d",
}
for key in ("whole_v1_complete", "whole_v1_06_complete", "interface_frozen", "holdout_opened"):
    assert evidence[key] is False
assert evidence["confirmed_retirement_implemented"] is True
inventory = referenced(evidence["source_inventory"])
validation = referenced(evidence["root_validation"])
provenance = referenced(evidence["snapshot_provenance"])
assert inventory["source_bytes_equal_before_after"] is True
assert validation["final_epoch"] == evidence["final_epoch"]
assert evidence["final_epoch"]["source_bytes_equal_before_after"] is True
assert validation["test_classification"]["baseline_new_requirement_failures"] == 11
assert validation["test_classification"]["candidate_behavior_failures"] == 2
assert validation["test_classification"]["other_controls_and_mechanisms"] == 17
assert validation["test_classification"]["old_tracked_tests_modified"] == 0
assert validation["test_classification"]["new_fixture_representation_assert_changes"] == 1

count = 0
assert set(inventory["files"]) == {"core", "pro"}
assert {key: len(value) for key, value in inventory["files"].items()} == {"core": 735, "pro": 223}
for repo, entries in inventory["files"].items():
    actual = sorted(set(p.decode() for p in git(repo, "ls-files", "-c", "-o",
        "--exclude-standard", "-z", "--", "src", "tests", "pyproject.toml").split(b"\0") if p))
    assert actual == sorted(entries), (repo, "inventory paths")
    if not args.worktree_only:
        git(repo, "merge-base", "--is-ancestor", evidence["code_commits"][repo], "HEAD")
    for path, expected in entries.items():
        assert sha((repos[repo] / path).read_bytes()) == expected, (repo, path)
        if not args.worktree_only:
            assert sha(git(repo, "show", evidence["code_commits"][repo] + ":" + path)) == expected
        count += 1
    assert len(entries) == evidence["final_epoch"]["file_counts"][repo]
    digest = sha("".join(f"{entries[path]}  {path}\n" for path in sorted(entries)).encode())
    for point in ("before", "after"):
        capture = inventory["manifest_captures"][point][repo]
        assert capture["exit_code"] == 0 and capture["file_count"] == len(entries)
        assert capture["stdout_sha256"] == digest
assert count == 958

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

for item in (evidence, validation, inventory, provenance):
    inspect_commands(item)
required_support = {
    "docs/evidence/2026-09-11-v1-06e-" + tail + ".json" for tail in (
        "interface-lifetime-author", "native-cpu-retirement-author",
        "retirement-independent-review", "lifetime-final-review",
        "snapshot-provenance", "cpu-retention-product")
} | {"docs/cpu-retirement.md", "docs/evidence/checks/v1-06e-check.py"}
assert len(evidence["supporting_files"]) == len(required_support)
assert {ref["path"] for ref in evidence["supporting_files"]} == required_support
for ref in evidence["supporting_files"]:
    raw = (root / ref["path"]).read_bytes()
    assert sha(raw) == ref["sha256"], ref["path"]
    if ref["path"].endswith(".json"):
        inspect_commands(json.loads(raw))
    elif ref["path"].endswith(".py"):
        ast.parse(raw, filename=ref["path"])
support_by_path = {ref["path"]: ref for ref in evidence["supporting_files"]}
for ref in provenance["available_support_reports_at_capture"]:
    assert ref["path"] in support_by_path
    assert ref["sha256"] == support_by_path[ref["path"]]["sha256"]
    assert len((root / ref["path"]).read_bytes()) == ref["bytes"]

snapshots = evidence["snapshots"]
assert snapshots == provenance["snapshots"]
assert len(snapshots) == 14
assert Counter(s["origin"] for s in snapshots) == {
    "baseline": 6, "candidate": 2, "first_fixture": 4, "replay_transcript": 2}
assert {p.as_posix() for p in Path("docs/evidence/snapshots").glob("v1-06e-*")} == {s["path"] for s in snapshots}
baseline_count = 0
for item in snapshots:
    raw = (root / item["path"]).read_bytes()
    assert sha(raw) == item["sha256"] and len(raw) == item["bytes"], item["path"]
    if item["path"].endswith(".py"):
        ast.parse(raw, filename=item["path"])
    if item.get("source"):
        assert sha((root / item["source"]).read_bytes()) == item["current_source_sha256"]
    if item["origin"] == "baseline":
        assert raw == git("core", "show", evidence["baseline_commits"]["core"] + ":" + item["source"])
        baseline_count += 1
assert baseline_count == 6

tests = evidence["test_inventory"]["files"]
assert len(tests) == 5 and sum(t["cases"] for t in tests) == 30
assert sum(t["literal_asserts"] for t in tests) == 86
collection = validation["collection"]["complete"]
assert collection["exit_code"] == 0
nodeids = [line for line in collection["output"].splitlines() if line.startswith("tests/") and "::" in line]
assert len(nodeids) == len(set(nodeids)) == 30
assert re.fullmatch(r"30 tests collected in [0-9.]+s", collection["output"].strip().splitlines()[-1])
for entry in tests:
    raw = (root / entry["path"]).read_bytes()
    assert sha(raw) == entry["sha256"]
    assert sum(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(raw))) == entry["literal_asserts"]
    assert sum(n.split("::")[0] == entry["path"] for n in nodeids) == entry["cases"]

def definitions(raw):
    return {n.name: ast.dump(n) for n in ast.parse(raw).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
for before, after in [
    ("v1-06e-first-cpu-interface-lifetime-tests.py", "tests/test_cpu_interface_lifetime.py"),
    ("v1-06e-first-cpu-retirement-product-tests.py", "tests/test_cpu_retirement_product.py"),
    ("v1-06e-retirement-review-first-three-tests.py", "tests/test_cpu_retirement_interfaces_review.py"),
]:
    old = definitions((root / "docs/evidence/snapshots" / before).read_bytes())
    new = definitions((root / after).read_bytes())
    assert all(new[key] == value for key, value in old.items())
prior = (root / "docs/evidence/snapshots/v1-06e-before-signal-wrapper-retirement-review-tests.py").read_text()
current = (root / "tests/test_cpu_retirement_interfaces_review.py").read_text()
assert prior.count("assert signal.getsignal(signal.SIGTERM).__self__ is engine") == 1
assert prior.replace("assert signal.getsignal(signal.SIGTERM).__self__ is engine",
                     "assert callable(signal.getsignal(signal.SIGTERM))") == current
def assertions(raw):
    return [ast.dump(n) for n in ast.walk(ast.parse(raw)) if isinstance(n, ast.Assert)]
append_only_count = sum(len(assertions((root / "docs/evidence/snapshots" / path).read_bytes())) for path in (
    "v1-06e-first-cpu-interface-lifetime-tests.py", "v1-06e-first-cpu-retirement-product-tests.py"))
prior_asserts, current_asserts = assertions(prior), assertions(current)
assert len(prior_asserts) == len(current_asserts) == 27
audited_count = append_only_count + len(prior_asserts)
changed_count = sum(before != after for before, after in zip(prior_asserts, current_asserts))
preserved_count = audited_count - changed_count
assert (audited_count, preserved_count, changed_count) == (57, 56, 1)
old_tests = [p for p in git("core", "ls-tree", "-r", "--name-only", evidence["baseline_commits"]["core"],
             "--", "tests").decode().splitlines() if p.endswith(".py")]
assert len(old_tests) == 452
for path in old_tests:
    assert (root / path).read_bytes() == git("core", "show", evidence["baseline_commits"]["core"] + ":" + path)
end = [] if args.worktree_only else [evidence["code_commits"]["core"]]
changes = git("core", "diff", "--name-status", evidence["baseline_commits"]["core"], *end,
              "--", "src", "tests", "pyproject.toml").decode().splitlines()
if args.worktree_only:
    changes += ["A\t" + path for path in git("core", "ls-files", "--others", "--exclude-standard",
        "--", "src", "tests", "pyproject.toml").decode().splitlines()]
assert sorted(changes) == sorted(evidence["changed_code_paths"])
assert {line.split("\t")[-1] for line in changes if line.startswith("A\ttests/")} == {t["path"] for t in tests}
pro_end = [] if args.worktree_only else [evidence["code_commits"]["pro"]]
assert git("pro", "diff", "--name-only", evidence["baseline_commits"]["pro"], *pro_end,
           "--", "src", "tests", "pyproject.toml") == b""
assert not git("pro", "ls-files", "--others", "--exclude-standard", "--", "src", "tests", "pyproject.toml")
for key, expected in {"core": (4036, 7), "pro": (974, 0), "paired": (60, 0), "target": (30, 0)}.items():
    run = evidence["final_epoch"]["results"][key]
    assert run["exit_code"] == 0 and (run["passed"], run["skipped"]) == expected
    assert "".join(chunk["output"] for chunk in run["tool_chunks"]) == run["complete_output"]
    assert run["tool_chunks"][-1]["exit_code"] == run["exit_code"]
    assert all("exit_code" not in chunk for chunk in run["tool_chunks"][:-1])
    assert not re.search(r"(?mi)^\d+ failed(?:, [^\n]*)? in [0-9.]+s", run["complete_output"])
    assert not re.search(r"(?i)warning:[^\n]*truncat|output truncated", run["complete_output"])
    pattern = str(expected[0]) + " passed" + (f", {expected[1]} skipped" if expected[1] else "")
    footer = run["complete_output"].strip().splitlines()[-1]
    assert re.fullmatch(pattern + r" in [0-9.]+s(?: \([^\n]+\))?", footer)
    assert float(re.search(r" in ([0-9.]+)s", footer)[1]) == run["seconds"]
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
    "snapshots": len(snapshots), "baseline_snapshots": baseline_count,
    "new_cases": 30, "literal_asserts": 86, "old_tests_byte_exact": len(old_tests),
    "unique_audited_new_fixture_asserts": audited_count, "preserved": preserved_count,
    "explicit_representation_change": changed_count,
    "shell_commands": len(commands), "python_heredocs": len(heredocs), "python_inline": len(inline),
    "record_consistency_not_cryptographic_execution_attestation": True,
    "whole_v1_complete": False, "whole_v1_06_complete": False}, sort_keys=True))
