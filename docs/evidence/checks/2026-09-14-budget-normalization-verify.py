"""Independent-process byte/AST/result readback; imports no product code.

This is mechanical verification, not a claim of review by a second author.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import tarfile
import xml.etree.ElementTree as ET
import zipfile

REPO = Path(__file__).resolve().parents[3]
PRO = REPO.parent / "pro"
OUT = REPO / "docs/evidence/runs/2026-09-14-budget-normalization"
BASE = "558f8c69db03eb7a5087d8200f54d4ce0fe46dcd"
PRO_BASE = "ac937e4"
sha = lambda raw: hashlib.sha256(raw).hexdigest()
dump = lambda node: ast.dump(node, include_attributes=False)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def data(path):
    return json.loads(path.read_bytes())


def unchanged_tests(repo, commit):
    names = git(repo, "ls-tree", "-r", "--name-only", commit, "tests").decode().splitlines()
    for name in names:
        assert (repo / name).read_bytes() == git(repo, "show", commit + ":" + name), name
    return len(names)


def ast_contract():
    relative = "src/orze/core/cpu_action_budget.py"
    baseline = git(REPO, "show", BASE + ":" + relative)
    assert baseline == (OUT / "baseline/cpu_action_budget.py").read_bytes()
    old, new = ast.parse(baseline), ast.parse((REPO / relative).read_bytes())
    old_funcs = {node.name: node for node in old.body if isinstance(node, ast.FunctionDef)}
    new_funcs = {node.name: node for node in new.body if isinstance(node, ast.FunctionDef)}
    assert set(new_funcs) == set(old_funcs) | {"_validate_declaration", "_validate_scope", "_validate_permit"}
    for name in old_funcs:
        if name in {"_declaration", "_scope", "_permit", "_totals"}:
            continue
        assert dump(old_funcs[name]) == dump(new_funcs[name]), name
    for name in ("declaration", "scope", "permit"):
        before = old_funcs["_" + name]
        body = before.body[:-1]
        assert dump(before.body[-1]) == dump(ast.parse("return _decode(_json(value))").body[0])
        if name == "scope":
            for node in body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "_declaration":
                    node.value.func.id = "_validate_declaration"
        if name == "permit":
            updated = []
            for node in body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "_scope":
                    node.value = node.value.args[0]
                    updated.extend([node, ast.parse("_validate_scope(scope)").body[0]])
                else:
                    updated.append(node)
            body = updated
        assert [dump(node) for node in body] == [dump(node) for node in new_funcs["_validate_" + name].body]
        expected = ast.parse("_validate_" + name + "(value)\nreturn _decode(_json(value))").body
        assert [dump(node) for node in new_funcs["_" + name].body] == [dump(node) for node in expected]
    totals = old_funcs["_totals"]
    loop, = [node for node in totals.body if isinstance(node, ast.For)]
    assert dump(loop.body[0]) == dump(ast.parse("permit = _permit(_decode(stored[4]))").body[0])
    loop.body[:1] = ast.parse("permit = _decode(stored[4])\n_validate_permit(permit)").body
    assert dump(totals) == dump(new_funcs["_totals"])
    assert [dump(n) for n in old.body if not isinstance(n, ast.FunctionDef)] == [
        dump(n) for n in new.body if not isinstance(n, ast.FunctionDef)]
    return {"baseline_sha256": sha(baseline), "candidate_sha256": sha((REPO / relative).read_bytes()),
            "other_functions_ast_unchanged": len(old_funcs) - 4,
            "all_existing_validation_branches_retained": True}


def frozen(path, *, path_failures=()):
    record = data(path / "run.json")
    assert record["exit_code"] == (1 if path_failures else 0)
    assert record["frozen"] and record["before"] == record["after"]
    for name, digest in record["files"].items():
        raw = (path / name).read_bytes()
        assert sha(raw) == digest["sha256"] and len(raw) == digest["bytes"]
    for label, files in record["after"].items():
        root = Path(record["repositories"][label])
        for relative, digest in files.items():
            assert sha((root / relative).read_bytes()) == digest
    suite, = ET.parse(path / "junit.xml").getroot().findall("testsuite")
    counts = {k: int(suite.attrib[k]) for k in ("tests", "failures", "errors", "skipped")}
    assert counts["errors"] == 0 and counts["failures"] == len(path_failures)
    failures = {t.attrib["name"]: t.find("failure") for t in suite.findall("testcase") if t.find("failure") is not None}
    assert set(failures) == set(path_failures)
    for failure in failures.values():
        assert "/core/src/orze/cli.py" in failure.text and 'endswith' in failure.text
    passed = counts["tests"] - counts["skipped"] - counts["failures"]
    assert re.search(r"\b" + str(passed) + r" passed\b", (path / "stdout.log").read_text())
    return {**counts, "passed": passed, "run_sha256": sha((path / "run.json").read_bytes())}


def main(commit):
    report = {"ast": ast_contract(), "unchanged_existing_tests": {
        "core": unchanged_tests(REPO, BASE), "pro": unchanged_tests(PRO, PRO_BASE)}}
    report["suites"] = {"core_full": frozen(OUT / "core-full-corrected", path_failures=(
        "test_benchmark_verifies_exact_blocked_receipts_at_acceptance_scale",
        "test_benchmark_fails_closed_on_one_false_receipt")),
        "core_path_rerun": frozen(OUT / "core-path-rerun"),
        "pro": frozen(PRO / "docs/evidence/runs/2026-09-14-budget-normalization/pro-full"),
        "optional": frozen(PRO / "docs/evidence/runs/2026-09-14-budget-normalization/core-pro-optional")}
    full_input = data(OUT / "core-full-corrected/run.json")["after"]["primary"]
    assert full_input == data(OUT / "core-path-rerun/run.json")["before"]["primary"]
    full_cases = {(t.attrib["classname"], t.attrib["name"]): t for t in
                  ET.parse(OUT / "core-full-corrected/junit.xml").getroot().iter("testcase")}
    rerun_cases = {(t.attrib["classname"], t.attrib["name"]): t for t in
                   ET.parse(OUT / "core-path-rerun/junit.xml").getroot().iter("testcase")}
    failed_ids = {key for key, t in full_cases.items() if t.find("failure") is not None}
    assert failed_ids <= set(rerun_cases) <= set(full_cases)
    full_cases.update(rerun_cases)
    assert not any(t.find("failure") is not None or t.find("error") is not None for t in full_cases.values())
    report["core_combined_coverage"] = {"passed": sum(t.find("skipped") is None for t in full_cases.values()),
        "skipped": sum(t.find("skipped") is not None for t in full_cases.values()),
        "same_frozen_source_and_tests": True, "single_green_full_run": False}
    archives = data(OUT / "archive-index.json")
    for key, record in archives.items():
        if "archive" not in record:
            for relative, expected in record["files"].items():
                raw = (OUT / relative).read_bytes()
                assert len(raw) == expected["bytes"] and sha(raw) == expected["sha256"]
            continue
        path = OUT / record["archive"]
        assert sha(path.read_bytes()) == record["sha256"]
        with tarfile.open(path, "r:gz") as tar:
            assert {m.name for m in tar.getmembers()} == set(record["members"])
            for member in tar.getmembers():
                raw = tar.extractfile(member).read()
                assert {"bytes": len(raw), "sha256": sha(raw)} == record["members"][member.name]
    installed = data(OUT / "installed-product.json")
    assert installed["exit_code"] == 0 and installed["source_tests_prepost_exact"]
    build = data(OUT / "package/build-install.json")
    wheel = OUT / "package" / Path(build["wheel"]).name
    assert sha(wheel.read_bytes()) == build["wheel_sha256"] == installed["wheel_sha256"]
    with zipfile.ZipFile(wheel) as archive:
        for name, expected in build["package_files"].items():
            raw = archive.read(name)
            assert sha(raw) == expected
            assert raw == (REPO / "src" / name).read_bytes()
            if commit:
                assert raw == git(REPO, "show", commit + ":src/" + name)
        for name, record in installed["loaded_orze_modules"].items():
            assert sha(archive.read(record["wheel_member"])) == record["sha256"]
            assert sha(Path(record["file"]).read_bytes()) == record["sha256"]
    cases = [event for event in installed["events"] if event["when"] == "call"]
    assert len(cases) == 58 and all(event["outcome"] == "passed" for event in cases)
    report["installed"] = {"passed": len(cases), "loaded_modules": len(installed["loaded_orze_modules"]),
                           "wheel_sha256": build["wheel_sha256"]}
    workers = []
    product_root = Path(archives["installed-product"]["root"]) / "pytest"
    for path in product_root.rglob("lake.db"):
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='execution_attempts'").fetchone():
                continue
            for row in conn.execute("SELECT * FROM execution_attempts"):
                terminal = json.loads(row["terminal_json"]) if row["terminal_json"] else {}
                tree = terminal.get("process_tree")
                if tree is None:
                    continue  # The explicit metadata prefixes have no worker.
                assert row["state"] == "TERMINAL" and terminal["outcome"] == "completed"
                assert tree["event"] == "TREE_CLOSED" and tree["wait_proof"] == "ECHILD_WALL"
                ref = tree["binding"]["identity"]["attempt_ref"]
                assert all(row[key] == value for key, value in ref.items())
                reservation, = [r for r in conn.execute("SELECT * FROM cpu_action_reservations")
                                 if r["ref_json"] and json.loads(r["ref_json"]) == ref]
                assert reservation["state"] == "SETTLED"
                assert reservation["terminal_sha256"] == sha(row["terminal_json"].encode())
                workers.append({"database": str(path.relative_to(product_root)), "ref": ref,
                                "terminal_sha256": reservation["terminal_sha256"]})
    assert len(workers) == 6
    report["installed"]["native_workers_with_closed_tree_and_settled_budget"] = workers
    bench = data(OUT / "benchmark-v2.json")
    for row in bench["rows"]:
        assert row["ledger_sha256_before"] == row["ledger_sha256_after"]
        size = row["rows_in_scope"]
        for operation in ("totals", "snapshot", "policy_continuation"):
            arms = row[operation]["arms"]
            assert arms["old"]["instrumented"]["value"] == arms["new"]["instrumented"]["value"]
            assert len(arms["old"]["seconds"]) == len(arms["new"]["seconds"]) == 5
        assert row["totals"]["arms"]["old"]["instrumented"]["calls"]["_decode"] == size * 5
        assert row["totals"]["arms"]["new"]["instrumented"]["calls"]["_decode"] == size * 2
        assert all(arm["instrumented"]["calls"]["_totals"] == 2 for arm in row["policy_continuation"]["arms"].values())
    report.update(passed=True, commit=commit, limits=[
        "Mechanical verification in a separate process, not a second human/agent review.",
        "Pro licensing uses test substitutes; no production deployment or scientific improvement claimed.",
        "Core first full run interrupted for fixture sibling layout; no completed summary exists for that attempt."])
    target = OUT / ("verification-committed.json" if commit else "verification.json")
    assert not target.exists()
    target.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit")
    main(parser.parse_args().commit)
