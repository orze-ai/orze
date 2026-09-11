"""Read-only C3a evidence verifier. Run from the Core repository root.

Checks recorded bytes/commits/command syntax, not live evaluation or power loss.
The expected failing replay is preserved separately in the author report.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


root = Path.cwd()
repos = {"core": root, "pro": root.parent / "orze-pro"}
evidence = json.loads((root / "docs/evidence/2026-09-11-v1-05c3a-benchmark-reservation.json").read_bytes())
inventory = json.loads((root / evidence["source_inventory"]["path"]).read_bytes())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repos[repo]), *args])


assert evidence["schema"] == 1 and evidence["slice"] == "V1-05C3a"
assert evidence["whole_v1_complete"] is False
assert evidence["test_classification"] == {
    "preceding_commit_behavior_reds": 2, "controls": 2,
    "new_storage_mechanisms": 7, "draft_reds": 0, "total_new_cases": 11,
}
assert digest((root / evidence["source_inventory"]["path"]).read_bytes()) == evidence["source_inventory"]["sha256"]
file_count = 0
for repo, entries in inventory["files"].items():
    actual_paths = sorted(set(p.decode() for p in git(
        repo, "ls-files", "-c", "-o", "--exclude-standard", "-z", "--",
        "src", "tests", "pyproject.toml").split(b"\0") if p))
    assert actual_paths == sorted(entries), (repo, "source inventory changed")
    for path, expected in entries.items():
        assert digest((repos[repo] / path).read_bytes()) == expected, (repo, path, "worktree")
        assert digest(git(repo, "show", evidence["code_commits"][repo] + ":" + path)) == expected, (repo, path, "commit")
        file_count += 1
    assert len(entries) == evidence["final_epoch"]["file_counts"][repo]

commands = set()
heredocs = set()
inline = set()


def inspect_commands(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if (key in {"cmd", "command"} or key.endswith("_command")) and isinstance(child, str):
                commands.add(child)
            inspect_commands(child)
    elif isinstance(value, list):
        for child in value:
            inspect_commands(child)


for ref in evidence["supporting_files"]:
    path = root / ref["path"]
    raw = path.read_bytes()
    assert digest(raw) == ref["sha256"], ref["path"]
    if path.suffix == ".json":
        inspect_commands(json.loads(raw))
    if path.suffix == ".py":
        ast.parse(raw, filename=str(path))
for snapshot in evidence["snapshots"]:
    raw = (root / snapshot["path"]).read_bytes()
    assert digest(raw) == snapshot["sha256"]
    assert raw == git("core", "show", snapshot["commit"] + ":" + snapshot["source"])
    ast.parse(raw, filename=snapshot["path"])

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

expected_new = {"tests/test_benchmark_reservation_fencing.py", "tests/test_benchmark_reservation_storage.py"}
changed = git("core", "diff", "--name-status", evidence["baseline_commits"]["core"], evidence["code_commits"]["core"], "--", "tests").decode().splitlines()
assert {line.split("\t")[1] for line in changed} == expected_new
assert all(line.startswith("A\t") for line in changed), "old tests changed"
assert git("pro", "diff", "--name-only", evidence["baseline_commits"]["pro"], evidence["code_commits"]["pro"], "--", "src", "tests", "pyproject.toml") == b""
for key, expected in {"core": (3626, 7), "pro": (974, 0), "paired": (60, 0), "target": (11, 0)}.items():
    run = evidence["final_epoch"]["results"][key]
    assert run["exit_code"] == 0
    assert (run["passed"], run["skipped"]) == expected
    assert f"{expected[0]} passed" in run["complete_output"]
    if expected[1]:
        assert f"{expected[1]} skipped" in run["complete_output"]
assert evidence["final_epoch"]["source_bytes_equal_before_after"] is True
print(json.dumps({"result": "verified", "source_files_worktree_and_commits": file_count,
                  "supporting_files": len(evidence["supporting_files"]),
                  "snapshots": len(evidence["snapshots"]),
                  "shell_commands": len(commands), "python_heredocs": len(heredocs),
                  "python_inline": len(inline), "whole_v1_complete": False}, sort_keys=True))
