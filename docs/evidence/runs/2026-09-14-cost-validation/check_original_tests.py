"""Read-only byte audit; one expressly approved Core race-injection migration.

Usage: python check_original_tests.py REPO BASELINE [--core-fixture-migration]
The exception is an exact four-replacement transformation, not a skipped file.
All other bytes, including every original assertion and competing write, must
match the named Git baseline. Pro has no exception.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


TARGET = "tests/test_idea_ingress_contract.py"
REPLACEMENTS = (
    (b"original_get_ids = instance.lake.get_all_ids\n",
     b"original_get_ids = instance.lake.find_existing_ids\n"),
    (b"def stale_ids_then_competing_admission():\n",
     b"def stale_ids_then_competing_admission(idea_ids):\n"),
    (b"old_ids = original_get_ids()\n",
     b"old_ids = original_get_ids(idea_ids)\n"),
    (b'monkeypatch.setattr(instance.lake, "get_all_ids", stale_ids_then_competing_admission)',
     b'monkeypatch.setattr(instance.lake, "find_existing_ids", stale_ids_then_competing_admission)'),
)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def assertions(raw):
    return [ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(raw))
            if isinstance(node, ast.Assert)]


def main():
    if not __debug__:
        raise SystemExit("byte audit requires assertions enabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("baseline")
    parser.add_argument("--core-fixture-migration", action="store_true")
    args = parser.parse_args()
    repo = args.repository.resolve()
    commit = git(repo, "rev-parse", args.baseline + "^{commit}").decode().strip()
    names = git(repo, "ls-tree", "-rz", "--name-only", commit, "--", "tests").split(b"\0")
    results = {}
    for name in filter(None, names):
        relative = name.decode()
        original = git(repo, "show", commit + ":" + relative)
        current = repo / relative
        raw = current.read_bytes() if current.is_file() else None
        item = {"original_sha256": sha(original), "current_sha256": sha(raw) if raw is not None else None,
                "byte_exact": original == raw, "approved_migration": False}
        if args.core_fixture_migration and relative == TARGET:
            expected = original
            for before, after in REPLACEMENTS:
                assert expected.count(before) == 1, before
                expected = expected.replace(before, after, 1)
            item.update(expected_migrated_sha256=sha(expected),
                        approved_migration=raw == expected,
                        original_assertion_count=len(assertions(original)),
                        all_original_assertion_ast_exact=raw is not None and assertions(raw) == assertions(original),
                        replacements=[{"before": a.decode(), "after": b.decode()} for a, b in REPLACEMENTS])
        item["preserved"] = item["byte_exact"] or item["approved_migration"]
        results[relative] = item
    assert results
    if args.core_fixture_migration:
        assert TARGET in results
    report = {"baseline": commit, "repository": str(repo), "files": results,
              "original_test_files": len(results),
              "byte_exact_files": sum(r["byte_exact"] for r in results.values()),
              "approved_migrated_files": sum(r["approved_migration"] for r in results.values()),
              "all_exact": all(r["byte_exact"] for r in results.values()),
              "all_preserved_with_explicit_exception": all(r["preserved"] for r in results.values()),
              "audit_script_sha256": sha(Path(__file__).read_bytes()),
              "limits": ["File preservation audit only, not executed test evidence.",
                         "The sole optional exception is the exact race-injection seam migration; no assertion may change."]}
    print(json.dumps(report, sort_keys=True, indent=2))
    raise SystemExit(0 if report["all_preserved_with_explicit_exception"] else 1)


if __name__ == "__main__":
    main()
