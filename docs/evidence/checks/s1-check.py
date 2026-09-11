"""Verify S1's limited consolidation, fixed source epoch and retained test logs."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

if not __debug__:
    raise SystemExit("Evidence verification requires assertions; do not use -O.")

ROOT = Path(__file__).resolve().parents[3]
PRO = ROOT.parent / "orze-pro"
PATH = "docs/evidence/2026-09-11-s1-report-rules.json"
BASE = {"core": "dbba0a69d8f76994f15930d2f3b59a27186b3e4b",
        "pro": "59cafe463360fe1256a5b2eb2831efa96006814b"}
CHANGED = {"core": {"src/orze/engine/rebuild_state.py", "src/orze/reporting/evidence.py"},
           "pro": {"src/orze_pro/agents/research_context.py"}}
NEW_TEST = "tests/test_report_rule_consolidation.py"
TEST_SHA = {"core": "c6f9491cf02d2a4b1c1f3f01745bacad45128efc710fae8e81445afc53464df1",
            "pro": "1a04a4629dd4228a458a4af6e153989026ad8c7fef4afe01e6a022741f7aaf68"}

def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def manifest(result):
    assert result["exit_code"] == 0
    rows = result["output"].splitlines()
    parsed = {row[66:]: row[:64] for row in rows}
    assert len(parsed) == len(rows)
    assert all(re.fullmatch("[0-9a-f]{64}", value) for value in parsed.values())
    return parsed

def dump(node):
    return ast.dump(node, include_attributes=False)

def function(tree, name):
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]

def footer(chunks, passed, failed=0, skipped=0):
    assert chunks[-1]["exit_code"] == (1 if failed else 0)
    output = "".join(chunk["output"] for chunk in chunks)
    assert "truncated output" not in output.lower()
    pattern = r"(\d+ [a-z]+(?:, \d+ [a-z]+)*) in \d+(?:\.\d+)?s(?: \([^\n]+\))?"
    lines = [line.strip("= ") for line in output.rstrip().splitlines()]
    summaries = [re.fullmatch(pattern, line) for line in lines]
    assert sum(item is not None for item in summaries) == 1
    final = summaries[-1]
    assert final is not None
    pairs = [item.split(" ") for item in final.group(1).split(", ")]
    counts = {label: int(count) for count, label in pairs}
    assert len(counts) == len(pairs)
    assert set(counts) <= {"passed", "failed", "skipped", "warning", "warnings"}
    assert all(counts.get(label, 0) == count for label, count in (
        ("passed", passed), ("failed", failed), ("skipped", skipped)))
    return output

def junit_cases(raw, passed, skipped):
    suites = ET.fromstring(raw).findall("testsuite")
    assert len(suites) == 1
    suite = suites[0]
    assert int(suite.attrib["tests"]) == passed + skipped
    assert int(suite.attrib["failures"]) == int(suite.attrib["errors"]) == 0
    assert int(suite.attrib["skipped"]) == skipped
    cases = suite.findall("testcase")
    assert len(cases) == passed + skipped
    identities = [(case.attrib["classname"], case.attrib["name"]) for case in cases]
    assert len(set(identities)) == len(identities)
    assert all(case.find("failure") is None and case.find("error") is None for case in cases)
    assert sum(len(case.findall("skipped")) for case in cases) == skipped
    assert all(len(case.findall("skipped")) <= 1 for case in cases)
    return cases

def structural_checks():
    old_rebuild = ast.parse(git(ROOT, "show", BASE["core"] + ":src/orze/engine/rebuild_state.py"))
    current_rebuild = ast.parse((ROOT / "src/orze/engine/rebuild_state.py").read_bytes())
    old_raw = git(ROOT, "show", BASE["core"] + ":src/orze/reporting/evidence.py").decode()
    new_raw = (ROOT / "src/orze/reporting/evidence.py").read_text()
    old_evidence, new_evidence = ast.parse(old_raw), ast.parse(new_raw)
    original = function(old_rebuild, "_eligible_metric")
    moved = copy.deepcopy(function(new_evidence, "legacy_archive_metric_value"))
    assert isinstance(moved.body[0], ast.Expr) and isinstance(moved.body[0].value, ast.Constant)
    assert isinstance(moved.body[0].value.value, str)
    moved.body.pop(0)
    moved.name = original.name
    assert dump(moved) == dump(original)
    assert ast.get_source_segment(old_raw, function(old_evidence, "dataset_metric_keys")) == (
        ast.get_source_segment(new_raw, function(new_evidence, "dataset_metric_keys")))

    # The complete modules differ only in the declared relocation/imports.
    normalized = copy.deepcopy(new_evidence)
    normalized.body.remove(function(normalized, "legacy_archive_metric_value"))
    old_typing = [node for node in old_evidence.body
                  if isinstance(node, ast.ImportFrom) and node.module == "typing"]
    new_typing = [node for node in normalized.body
                  if isinstance(node, ast.ImportFrom) and node.module == "typing"]
    assert len(old_typing) == len(new_typing) == 1
    expected_typing = copy.deepcopy(old_typing[0])
    expected_typing.names.append(ast.alias(name="Optional", asname=None))
    assert dump(new_typing[0]) == dump(expected_typing)
    new_typing[0].names.pop()
    assert dump(normalized) == dump(old_evidence)
    old_rebuild.body = [node for node in old_rebuild.body if not (
        isinstance(node, ast.FunctionDef) and node.name in {"_eligible_metric", "_report_dataset_keys"}
        or isinstance(node, ast.Import) and [name.name for name in node.names] == ["math"])]
    aliases = [node for node in current_rebuild.body if (
        isinstance(node, ast.ImportFrom) and node.module == "orze.reporting.evidence")]
    assert len(aliases) == 1
    assert [(name.name, name.asname) for name in aliases[0].names] == [
        ("dataset_metric_keys", "_report_dataset_keys"),
        ("legacy_archive_metric_value", "_eligible_metric")]
    current_rebuild.body.remove(aliases[0])
    assert dump(old_rebuild) == dump(current_rebuild)

    path = "src/orze_pro/agents/research_context.py"
    old_pro = ast.parse(git(PRO, "show", BASE["pro"] + ":" + path))
    new_pro = ast.parse((PRO / path).read_bytes())
    def dependencies(tree):
        return [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and node.module == "orze.engine.rebuild_state"]
    assert len(dependencies(old_pro)) == 2 and not dependencies(new_pro)
    class ReverseRoutes(ast.NodeTransformer):
        names = {"legacy_archive_metric_value": "_eligible_metric",
                 "dataset_metric_keys": "_report_dataset_keys"}
        def visit_ImportFrom(self, node):
            if node.module == "orze.reporting.evidence" and any(name.name in self.names for name in node.names):
                node.module = "orze.engine.rebuild_state"
                for name in node.names:
                    name.name = self.names.get(name.name, name.name)
            return node
        def visit_Name(self, node):
            node.id = self.names.get(node.id, node.id)
            return node
    assert dump(ReverseRoutes().visit(new_pro)) == dump(old_pro)
    return {"dataset_selector_implementations_before": 2, "dataset_selector_implementations_after": 1,
            "pro_private_engine_imports_before": 2, "pro_private_engine_imports_after": 0,
            "executable_archive_AST_preserved": True, "other_product_AST_preserved": True}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree-only", action="store_true")
    args = parser.parse_args()
    document = json.loads((ROOT / PATH).read_bytes())
    assert document["schema"] == 1 and document["slice"] == "S1"
    assert document["whole_v1_complete"] is False and document["research_benefit_proven"] is False
    assert sha(Path(__file__).read_bytes()) == document["checker_sha256"]
    baseline_path = ROOT / "docs/evidence/2026-09-11-s1-report-rules-baseline.json"
    assert sha(baseline_path.read_bytes()) == document["baseline_evidence_sha256"]
    baseline = json.loads(baseline_path.read_bytes())
    assert baseline["code_baseline"] == BASE
    footer(baseline["existing_neighborhood"]["core"], 70)
    footer(baseline["existing_neighborhood"]["pro"], 60)
    footer(baseline["new_tests"]["core"]["chunks"], 22, 2)
    footer(baseline["new_tests"]["pro"]["chunks"], 2, 2)
    counts, old_tests = {}, {}
    for name, repo in (("core", ROOT), ("pro", PRO)):
        before = manifest(document["final_epoch"]["before"][name])
        after = manifest(document["final_epoch"]["after"][name])
        assert before == after and len(before) == {"core": 762, "pro": 224}[name]
        paths = git(repo, "ls-files", "-c", "-o", "--exclude-standard", "-z",
                    "--", "src", "tests", "examples", "pyproject.toml").split(b"\0")
        current = {path.decode(): sha((repo / path.decode()).read_bytes()) for path in set(paths) if path}
        assert current == before
        previous = manifest(baseline["manifest"][name])
        assert previous.keys() == before.keys()
        assert {path for path in before if before[path] != previous[path]} == CHANGED[name]
        assert before[NEW_TEST] == TEST_SHA[name]
        fixed_paths = git(repo, "ls-tree", "-r", "--name-only", BASE[name], "--",
                          "src", "tests", "examples", "pyproject.toml").decode().splitlines()
        assert set(fixed_paths) | {NEW_TEST} == set(before)
        for path in fixed_paths:
            old = sha(git(repo, "show", BASE[name] + ":" + path))
            assert (old == before[path]) == (path not in CHANGED[name])
        old_tests[name] = sum(path.startswith("tests/") and path.endswith(".py") for path in fixed_paths)
        assert old_tests[name] == {"core": 471, "pro": 133}[name]
        commit = document["code_commits"][name]
        if not args.worktree_only:
            assert re.fullmatch("[0-9a-f]{40}", commit)
            committed_paths = git(repo, "ls-tree", "-r", "--name-only", commit, "--",
                                  "src", "tests", "examples", "pyproject.toml").decode().splitlines()
            assert set(committed_paths) == set(before)
            for path, digest in before.items():
                assert sha(git(repo, "show", commit + ":" + path)) == digest
        counts[name] = len(before)
        footer(document["target"][name]["chunks"], {"core": 94, "pro": 64}[name])
    extra_path = "docs/evidence/checks/v1-07b-domain-check.py"
    assert set(manifest(document["final_epoch"]["before"]["extra"])) == {extra_path}
    assert manifest(document["final_epoch"]["before"]["extra"]) == (
        manifest(document["final_epoch"]["after"]["extra"]))
    assert sha((ROOT / extra_path).read_bytes()) == (
        manifest(document["final_epoch"]["before"]["extra"])[extra_path])
    assert sha(git(ROOT, "show", BASE["core"] + ":" + extra_path)) == sha((ROOT / extra_path).read_bytes())
    if not args.worktree_only:
        assert sha(git(ROOT, "show", document["code_commits"]["core"] + ":" + extra_path)) == (
            sha((ROOT / extra_path).read_bytes()))
    for name, passed, skipped in (("core", 4260, 7), ("pro", 978, 0), ("paired", 60, 0)):
        footer(document["final_epoch"]["chunks"][name], passed, skipped=skipped)
        command = document["final_epoch"]["commands"][name]
        assert "PYTHONDONTWRITEBYTECODE=1" in command and "--junitxml=" in command
        assert "--basetemp=" in command and "-k " not in command
        report = document["junit"][name]
        raw = (ROOT / report["archive"]).read_bytes()
        assert sha(raw) == report["sha256"] and len(raw) == report["bytes"]
        cases = junit_cases(raw, passed, skipped)
        if name != "paired":
            new_cases = [case for case in cases
                         if case.attrib["classname"].split(".")[-1] == "test_report_rule_consolidation"]
            assert len(new_cases) == {"core": 24, "pro": 4}[name]
            assert all(case.find("skipped") is None for case in new_cases)
    if not args.worktree_only:
        relative = str(Path(__file__).resolve().relative_to(ROOT))
        assert sha(git(ROOT, "show", document["code_commits"]["core"] + ":" + relative)) == document["checker_sha256"]
    result = structural_checks()
    result.update(result="verified", fixed_commits_checked=not args.worktree_only,
                  source_files=counts, old_test_files_unchanged=old_tests,
                  extra_imported_checkers=1, new_cases=28,
                  record_consistency_not_independent_execution_attestation=True)
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__":
    main()
