"""S2 record verification; reuse S1's frozen log/JUnit validators."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess

if not __debug__:
    raise SystemExit("Evidence verification requires assertions; do not use -O.")
ROOT = Path(__file__).resolve().parents[3]
PRO = ROOT.parent / "orze-pro"
BASE = {"core": "694754b0927a251ace72a7600e7338142caa4d49",
        "pro": "1384c12d40069c98a430d0e25647de102a7e38de"}
CHANGED = {"src/orze/engine/claim_authority.py", "src/orze/engine/native_cpu_action.py",
           "src/orze/core/cpu_action_budget.py"}
TESTS = {"tests/test_claim_reader_consolidation.py": "b049d9aeec26567d9ca1a2d7747f4057d73a190794a39f311ae5c858cb5642e0",
         "tests/test_cpu_claim_reader_consolidation.py": "eff390610b65fc0ca09b751d72e7e1823e9b7591ff25dfd2af1bb94465379b1b"}
HELPER = "docs/evidence/checks/s1-check.py"
HELPER_SHA = "139621de658f450f9cdbd1471f1f95bbc29153b473839b2c5e17d589acf00211"
REQUIRED_SUPPORT = {HELPER, "docs/claim-reader-consolidation.md",
    "docs/plans/2026-09-11-s2-claim-reader-consolidation.zh-CN.md",
    "docs/evidence/2026-09-11-s2-claim-reader-baseline.json",
    "docs/evidence/2026-09-11-s2-test-revision.json",
    "docs/evidence/2026-09-11-s2-disappearance-review.json",
    *{"docs/evidence/runs/s2-" + name + "-junit.xml" for name in ("core", "pro", "paired")}}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])

assert sha((ROOT / HELPER).read_bytes()) == HELPER_SHA
H = runpy.run_path(str(ROOT / HELPER), run_name="s2_frozen_record_helpers")
footer, junit_cases, manifest, dump = (H[k] for k in ("footer", "junit_cases", "manifest", "dump"))

def structural_checks():
    for path, old_name in (("src/orze/engine/native_cpu_action.py", "_read"),
                           ("src/orze/core/cpu_action_budget.py", "read_claim")):
        old = ast.parse(git(ROOT, "show", BASE["core"] + ":" + path))
        new = ast.parse((ROOT / path).read_bytes())
        imports, calls = [], []
        class ReverseRoute(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                if node.level == 0 and node.module == "orze.engine" and [(n.name, n.asname) for n in node.names] == [("claim_authority", None)]:
                    imports.append(node)
                    return ast.ImportFrom(module="orze.engine.training_attempts",
                        names=[ast.alias(name="_read", asname=None if old_name == "_read" else old_name)], level=0)
                return node
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and (
                    node.func.value.id == "claim_authority" and node.func.attr == "read_claim_snapshot"):
                    assert len(node.args) == 1
                    assert dump(ast.Tuple(elts=[kw.value for kw in node.keywords], ctx=ast.Load())) == (
                        dump(ast.Tuple(elts=[ast.Constant(8192), ast.Constant(True)], ctx=ast.Load())))
                    assert [kw.arg for kw in node.keywords] == ["limit", "required"]
                    calls.append(node)
                    return ast.Call(func=ast.Name(id=old_name, ctx=ast.Load()),
                                    args=node.args + [ast.Constant(8192)], keywords=[])
                return self.generic_visit(node)
        normalized = ReverseRoute().visit(new)
        assert len(imports) == len(calls) == 1 and dump(normalized) == dump(old)
    path = "src/orze/engine/claim_authority.py"
    old = ast.parse(git(ROOT, "show", BASE["core"] + ":" + path))
    new = ast.parse((ROOT / path).read_bytes())
    for name in ("read_claim", "read_claim_snapshot"):
        H["function"](new, name)  # reject shadow definitions
    old.body.remove(H["function"](old, "read_claim"))
    new.body.remove(H["function"](new, "read_claim"))
    new.body.remove(H["function"](new, "read_claim_snapshot"))
    imports = [n for n in new.body if isinstance(n, ast.Import) and
               [(a.name, a.asname) for a in n.names] in [[("errno", None)], [("hashlib", None)]]]
    assert [[(a.name, a.asname) for a in node.names] for node in imports] == [
        [("errno", None)], [("hashlib", None)]]
    for node in imports:
        new.body.remove(node)
    assert dump(old) == dump(new)
    return {"cpu_private_training_reader_imports_before": 2, "after": 0,
            "other_cpu_AST_exact": True, "other_claim_authority_AST_exact": True}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree-only", action="store_true")
    args = parser.parse_args()
    d = json.loads((ROOT / "docs/evidence/2026-09-11-s2-claim-reader.json").read_bytes())
    assert d["schema"] == 1 and d["slice"] == "S2" and d["research_benefit_proven"] is False
    assert sha(Path(__file__).read_bytes()) == d["checker_sha256"]
    assert set(d["support_sha256"]) == REQUIRED_SUPPORT
    for path, digest in d["support_sha256"].items():
        assert sha((ROOT / path).read_bytes()) == digest
        if not args.worktree_only:
            assert sha(git(ROOT, "show", d["code_commits"]["core"] + ":" + path)) == digest
    baseline = json.loads((ROOT / "docs/evidence/2026-09-11-s2-claim-reader-baseline.json").read_bytes())
    assert baseline["code_baseline"] == BASE
    footer(baseline["existing_neighborhood"]["chunks"], 78)
    footer(baseline["new_tests"]["unit"]["chunks"], 9, failed=1)
    footer(baseline["new_tests"]["cpu"]["chunks"], 2, failed=2)
    revision = json.loads((ROOT / "docs/evidence/2026-09-11-s2-test-revision.json").read_bytes())
    test_path = "tests/test_claim_reader_consolidation.py"
    v1 = git(ROOT, "show", "ae0f8aeacf31b47588832aa08cf1849487084f88:" + test_path)
    assert sha(v1) == "0bbf92f7b3ee1f718703a88b65785af7d7339984c96db51158fb06e0cb5799fb"
    assert revision["original_test_sha256"] == sha(v1)
    assert revision["revised_test_sha256"] == TESTS[test_path]
    assert baseline["test_sha256"][test_path] == sha(v1)
    old_test = ast.parse(v1)
    new_test = ast.parse((ROOT / test_path).read_bytes())
    def snapshot_class(tree):
        found = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TestNewSnapshot"]
        assert len(found) == 1
        return found[0]
    old_class, new_class = snapshot_class(old_test), snapshot_class(new_test)
    fault = H["function"](old_class, "test_after_read_identity_changes_never_return_a_snapshot")
    assignment = ast.parse("rejection = OSError if change == 'disappear' else AttemptEffectBusy").body[0]
    assignments = [n for n in fault.body if dump(n) == dump(assignment)]
    assert len(assignments) == 1
    fault.body.remove(assignments[0])
    raises = [n for n in ast.walk(fault) if isinstance(n, ast.Call) and
              dump(n.func) == dump(ast.parse("pytest.raises", mode="eval").body)]
    assert len(raises) == 1 and len(raises[0].args) == 1
    assert dump(raises[0].args[0]) == dump(ast.Name(id="rejection", ctx=ast.Load()))
    raises[0].args[0] = ast.Name(id="AttemptEffectBusy", ctx=ast.Load())
    new_class.body.remove(H["function"](new_class, "test_disappearance_after_safe_file_before_open_is_os_error"))
    assert dump(old_test) == dump(new_test)
    footer(revision["candidate"]["chunks"], 35, failed=1)
    footer(revision["revised_new_tests"]["chunks"], 37)
    footer(d["target"]["chunks"], 115)
    counts, old_tests = {}, {}
    for name, repo in (("core", ROOT), ("pro", PRO)):
        before, after = [manifest(d["final_epoch"][epoch][name]) for epoch in ("before", "after")]
        assert before == after and len(before) == {"core": 764, "pro": 224}[name]
        paths = git(repo, "ls-files", "-c", "-o", "--exclude-standard", "-z", "--",
                    "src", "tests", "examples", "pyproject.toml").split(b"\0")
        current = {p.decode(): sha((repo / p.decode()).read_bytes()) for p in set(paths) if p}
        assert current == before
        old_paths = git(repo, "ls-tree", "-r", "--name-only", BASE[name], "--",
                        "src", "tests", "examples", "pyproject.toml").decode().splitlines()
        assert set(old_paths) | (set(TESTS) if name == "core" else set()) == set(before)
        for path in old_paths:
            same = sha(git(repo, "show", BASE[name] + ":" + path)) == before[path]
            assert same == (name != "core" or path not in CHANGED)
        old_tests[name] = sum(p.startswith("tests/") and p.endswith(".py") for p in old_paths)
        assert old_tests[name] == {"core": 472, "pro": 134}[name]
        if name == "core":
            assert all(before[p] == h for p, h in TESTS.items())
        if not args.worktree_only:
            commit = d["code_commits"][name]
            assert re.fullmatch("[0-9a-f]{40}", commit)
            committed = git(repo, "ls-tree", "-r", "--name-only", commit, "--",
                            "src", "tests", "examples", "pyproject.toml").decode().splitlines()
            assert set(committed) == set(before)
            assert all(sha(git(repo, "show", commit + ":" + p)) == h for p, h in before.items())
        counts[name] = len(before)
    extra = "docs/evidence/checks/v1-07b-domain-check.py"
    extra_sha = sha(git(ROOT, "show", BASE["core"] + ":" + extra))
    assert manifest(d["final_epoch"]["before"]["extra"]) == manifest(d["final_epoch"]["after"]["extra"]) == {extra: extra_sha}
    assert sha((ROOT / extra).read_bytes()) == extra_sha
    if not args.worktree_only:
        assert sha(git(ROOT, "show", d["code_commits"]["core"] + ":" + extra)) == extra_sha
        assert sha(git(ROOT, "show", d["code_commits"]["core"] + ":" +
                       str(Path(__file__).resolve().relative_to(ROOT)))) == d["checker_sha256"]
    assert sha(git(ROOT, "show", BASE["core"] + ":" + HELPER)) == HELPER_SHA
    for name, passed, skipped in (("core", 4297, 7), ("pro", 978, 0), ("paired", 60, 0)):
        footer(d["final_epoch"]["chunks"][name], passed, skipped=skipped)
        command = d["final_epoch"]["commands"][name]
        assert "--basetemp=" in command and "--junitxml=" in command and "-k " not in command
        report = d["junit"][name]
        raw = (ROOT / report["archive"]).read_bytes()
        assert sha(raw) == report["sha256"] and len(raw) == report["bytes"]
        cases = junit_cases(raw, passed, skipped)
        if name == "core":
            for module, count in (("test_claim_reader_consolidation", 31), ("test_cpu_claim_reader_consolidation", 6)):
                selected = [c for c in cases if module in c.attrib["classname"].split(".")]
                assert len(selected) == count and all(c.find("skipped") is None for c in selected)
    result = structural_checks()
    print(json.dumps(dict(result, result="verified", fixed_commits_checked=not args.worktree_only,
          source_files=counts, old_test_files_unchanged=old_tests, new_cases=37,
          research_benefit_proven=False, record_consistency_not_execution_attestation=True), sort_keys=True))

if __name__ == "__main__":
    main()
