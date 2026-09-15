"""Separate-process mechanical audit of ordered reads and unchanged admission rules."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
PRO = ROOT.parent / "pro"
OUT = ROOT / "docs/evidence/runs/2026-09-15-status-merge-queries"
POUT = PRO / "docs/evidence/runs/2026-09-15-status-merge-queries"
BASE = "f9d69e24c3a31569cb8235d617366b5d5a8894f5"
PBASE = "78ba34a54ca05a3dc4081b52cc7859d70013b0fa"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


prior = module("prior_audit", ROOT / "docs/evidence/checks/2026-09-15-identity-repair-verify.py")
prior.OUT, prior.POUT, prior.BASE, prior.PBASE = OUT, POUT, BASE, PBASE
common = prior.common
sha, data, git = common.sha, common.data, common.git


def function(tree, name):
    return next(n for n in tree.body if getattr(n, "name", None) == name)


def source_checks():
    baseline = data(OUT / "baseline/source.json")
    assert baseline["head"] == BASE and baseline["pro_head"] == PBASE
    for name, digest in baseline["files"].items():
        raw = (OUT / "baseline" / Path(name).name).read_bytes()
        assert raw == git(ROOT, "show", BASE + ":" + name) and sha(raw) == digest
    old = ast.parse((OUT / "baseline/config_identity_repair.py").read_bytes())
    new = ast.parse((ROOT / "src/orze/core/config_identity_repair.py").read_bytes())
    new.body = [n for n in new.body if not (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_ORDERED_MISSING" for t in n.targets))]
    old_source = next(n for n in ast.walk(function(old, "_copy_snapshot"))
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr == "execute" and isinstance(n.args[0], ast.BinOp))
    new_source = next(n for n in ast.walk(function(new, "_copy_snapshot"))
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr == "execute" and isinstance(n.args[0], ast.Name)
                      and n.args[0].id == "_ORDERED_MISSING")
    new_source.args[0] = copy.deepcopy(old_source.args[0])
    assert ast.dump(old) == ast.dump(new), "only the source SELECT changed in repair"
    old = ast.parse((OUT / "baseline/proposal_admission.py").read_bytes())
    new = ast.parse((ROOT / "src/orze/core/proposal_admission.py").read_bytes())
    original = function(old, "_dedup_owner")
    changed = function(new, "_dedup_owner")
    old_projection = next(n.value.value for n in original.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "candidate" for t in n.targets))
    new_projection = next(n.value.value for n in function(new, "_candidate_rows").body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "candidate" for t in n.targets))
    assert new_projection == old_projection.replace("status COLLATE NOCASE IN (?, ?, ?, ?)", "status COLLATE NOCASE = ?")
    assert [ast.dump(n) for n in original.body if isinstance(n, ast.If)] == [ast.dump(n) for n in changed.body if isinstance(n, ast.If)]
    old_loop = next(i for i, n in enumerate(original.body) if isinstance(n, ast.For))
    new_loop = next(i for i, n in enumerate(changed.body) if isinstance(n, ast.For))
    assert [ast.dump(n) for n in original.body[old_loop:]] == [ast.dump(n) for n in changed.body[new_loop:]]
    for tree in (old, new):
        tree.body = [n for n in tree.body if getattr(n, "name", None) not in {"_dedup_owner", "_candidate_rows"}]
    assert ast.dump(old) == ast.dump(new), "normal and caller-owned admission bodies remain identical"
    path = "tests/test_config_identity_staging.py"
    old = ast.parse((OUT / "baseline" / Path(path).name).read_bytes())
    new = ast.parse((ROOT / path).read_bytes())
    old_execute = function(function(old, "SourceConnection"), "execute")
    new_execute = function(function(new, "SourceConnection"), "execute")
    old_if = next(n for n in old_execute.body if isinstance(n, ast.If))
    new_if = next(n for n in new_execute.body if isinstance(n, ast.If))
    assert ast.dump(new_if.test) != ast.dump(old_if.test)
    new_if.test = copy.deepcopy(old_if.test)
    assert ast.dump(old) == ast.dump(new), "only observer recognition widened; every assertion retained"
    return {"unchanged_core_sources": common.unchanged(ROOT, BASE, "src", {
                "src/orze/core/config_identity_repair.py", "src/orze/core/proposal_admission.py"}),
            "unchanged_core_tests": common.unchanged(ROOT, BASE, "tests", {path}),
            "adapted_test_observer": path,
            "unchanged_pro_sources": common.unchanged(PRO, PBASE, "src"),
            "unchanged_pro_tests": common.unchanged(PRO, PBASE, "tests")}


def archives_check():
    archives = data(OUT / "archives.json")
    checked = {}
    for name, record in archives.items():
        assert sha((OUT / record["archive"]).read_bytes()) == record["sha256"]
        with tarfile.open(OUT / record["archive"], "r:gz") as tar:
            assert {m.name for m in tar.getmembers()} == set(record["members"])
            for member in tar.getmembers():
                raw = tar.extractfile(member).read()
                assert sha(raw) == record["members"][member.name]
                assert raw == (Path(record["root"]) / member.name).read_bytes()
        checked[name] = {"sha256": record["sha256"], "files": len(record["members"])}
    return archives, checked


def source_benchmarks(archives):
    checked = {}
    for name in ("source_v1", "source_v2", "source_memory", "source_allocator"):
        root = Path(archives[name]["root"])
        report = data(root / "report.json")
        mode = {"source_memory": "memory", "source_allocator": "allocator"}.get(name, "time")
        repetitions = 3 if mode == "time" else 1
        assert report["passed"] and report["mode"] == mode and report["repetitions"] == repetitions
        assert report["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-status-merge-source-benchmark.py").read_bytes())
        assert report["allocator_helper_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-status-merge-allocator.py").read_bytes())
        for path, digest in report["source_files"].items():
            assert sha((ROOT / path).read_bytes()) == digest
        for path, digest in report["baseline_files"].items():
            assert sha((OUT / "baseline" / path).read_bytes()) == digest
        assert [(r["records"], r["payload_characters"]) for r in report["rows"]] == [(100, 1024), (1000, 1024), (5000, 1024), (500, 8192)]
        for row in report["rows"]:
            project = root / f'{row["records"]}-{row["payload_characters"]}'
            assert prior.db_digest(project / "input.db") == row["input_database_sha256"]
            assert row["orders"] == [["old", "new"], ["new", "old"], ["old", "new"]][:repetitions]
            identities = set()
            for arm in row["arms"].values():
                assert len(arm) == repetitions
                for event in arm:
                    assert event["repaired"] == row["records"]
                    assert prior.db_digest(Path(event["database"])) == event["database_sha256"]
                    identities.add(event["database_sha256"])
                    assert (event["current_peak_python_bytes"] is not None) == (mode == "memory")
                    assert (event["sqlite_allocation_after"] is not None) == (mode == "allocator")
                    if mode == "allocator":
                        assert event["sqlite_allocation_after"]["peak"] >= event["sqlite_allocation_before"]["current"] > 0
            assert len(identities) == 1
            for event in row.get("separate_writer_audit", {}).values():
                assert event["events"] == ["BEGIN IMMEDIATE", "COMMIT"] and event["hold_seconds"] > 0
                assert prior.db_digest(Path(event["database"])) in identities
                stage, = event["stage"]
                assert stage["staged_rows"] == row["records"]
                assert (stage["journal_mode"], stage["synchronous"], stage["locking_mode"], stage["cache_size"], stage["directory_mode"]) == ("delete", 2, "normal", -512, 0o700)
                assert stage["bytes_before_main_apply"] > 0 and not Path(stage["directory"]).exists()
        checked[name] = sha((root / "report.json").read_bytes())
    return checked


def admission_benchmarks(archives):
    checked = {}
    expected = [(100, "exact"), (1000, "exact"), (5000, "exact"), (5000, "semantic"),
                (1000, "crowded"), (5000, "crowded"), (20000, "crowded"),
                (1000, "missing"), (5000, "missing"), (20000, "missing")]
    for name in ("admission_v1", "admission_v2"):
        root = Path(archives[name]["root"])
        report = data(root / "report.json")
        assert report["passed"] and report["baseline"] == BASE and report["repetitions"] == 3
        assert report["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-status-merge-admission-benchmark.py").read_bytes())
        assert report["allocator_helper_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-status-merge-allocator.py").read_bytes())
        for path, digest in report["sources"].items():
            assert sha(Path(path).read_bytes()) == digest
        assert [(r["history"], r["mode"]) for r in report["rows"]] == expected
        for row in report["rows"]:
            history, mode = row["history"], row["mode"]
            project = root / f"{history}-{mode}"
            assert prior.db_digest(project / "lake.db") == row["database_sha256_before"] == row["database_sha256_after"]
            assert sha((project / "ideas.md").read_bytes()) == row["source_sha256"]
            assert row["entry"] == ("direct_admission" if mode == "missing" else "complete_ingress")
            assert row["orders"] == [["old", "new"], ["new", "old"], ["old", "new"]]
            status = "rejected" if mode in {"crowded", "missing"} and history > 1024 else "config_duplicate"
            for arm in row["arms"].values():
                assert len(arm["seconds"]) == len(arm["io_deltas"]) == 3
                assert arm["traced_peak_bytes"] >= arm["traced_current_bytes"] >= 0
            for counter in row["counters"].values():
                assert counter["admission_status"] == status
                assert counter["begin_immediate"] == counter["rollbacks"] == 128
                assert counter["yaml_loads"] == (0 if mode == "missing" else 384 if mode == "semantic" else 256)
                assert len(counter["candidate_queries"]) == len(counter["candidate_plans"]) == (1 if mode == "crowded" and status == "rejected" else 2)
            assert all("USE TEMP B-TREE" not in step[-1] for plan in row["counters"]["new"]["candidate_plans"] for step in plan)
            for native in row["sqlite_memory"].values():
                assert native["after"]["peak"] >= native["before"] > 0
            if mode == "missing":
                assert len(row["value"]["outcomes"]) == 128
                assert all(v["status"] == status for v in row["value"]["outcomes"])
            else:
                assert row["value"]["inserted"] == []
        checked[name] = sha((root / "report.json").read_bytes())
    return checked


def differential_and_prototype(archives):
    root = Path(archives["differential"]["root"])
    report = data(root / "report.json")
    assert report["passed"] and report["baseline"] == BASE and len(report["cases"]) == 64
    assert report["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-dedup-candidate-query-differential.py").read_bytes())
    for path, digest in report["sources"].items():
        assert sha(Path(path).read_bytes()) == digest
    for index, case in enumerate(report["cases"]):
        project = root / f"case-{index:03d}"
        before = prior.db_digest(project / "baseline.db")
        assert before == case["database_sha256_before"]
        for entry, values in case["outcomes"].items():
            assert values["old"] == values["new"]
            for arm in ("old", "new"):
                expected = values[arm]["database_sha256_after"] if entry == "normal" else values[arm]["rollback_sha256"]
                assert prior.db_digest(project / f"{entry}-{arm}.db") == expected
    proto = Path(archives["prototype"]["root"])
    assert data(proto / "report.json")["passed"] and data(proto / "repair-report.json")["passed"]
    assert data(proto / "tests.json")["exit_code"] == 0
    assert data(proto / "current-result.json")["exit_code"] == 1
    assert data(proto / "candidate-result.json")["exit_code"] == 0
    for name in ("tests.json", "current-result.json", "candidate-result.json"):
        record = data(proto / name)
        for path, digest in record["files"].items():
            assert sha((proto / path).read_bytes()) == digest
        for path, digest in record.get("frozen_inputs", {}).items():
            assert sha(git(ROOT, "show", BASE + ":" + path)) == digest
    return {"differential_sha256": sha((root / "report.json").read_bytes()),
            "prototype_sha256": sha((proto / "repair-report.json").read_bytes())}


def product_check(archives):
    root = Path(archives["product"]["root"])
    product = data(root / "report.json")
    assert product["passed"] and product["missing_before"] == 1500 and product["missing_after"] == 0
    assert product["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-identity-repair-product.py").read_bytes())
    for name, digest in product["source_files"].items():
        assert sha((ROOT / name).read_bytes()) == digest
    helper = module("product_helper", ROOT / "docs/evidence/checks/2026-09-15-ingress-product.py")
    assert product["before"] == product["after"] == helper.rows(root / "lake.db")
    assert product["unprepared"]["status"] == "rejected" and product["unprepared"]["reason"] == "proposal_dedup_capacity"
    assert product["ingress"]["inserted"] == ["idea-tail"]
    assert len(product["invocations"]) == 3 and all(v["exit_code"] == 0 for v in product["invocations"])
    assert sha((root / "ideas.md").read_bytes()) == product["source_sha256_after_execution"]
    attempt, = product["after"]["execution_attempts"]
    reservation, = product["after"]["cpu_action_reservations"]
    artifact, = product["after"]["research_artifacts"]
    terminal = json.loads(attempt["terminal_json"])
    assert attempt["task_id"] == "idea-tail" and attempt["state"] == "TERMINAL" and terminal["outcome"] == "completed"
    tree = terminal["process_tree"]
    assert tree["event"] == "TREE_CLOSED" and tree["wait_proof"] == "ECHILD_WALL"
    ref = tree["binding"]["identity"]["attempt_ref"]
    assert all(attempt[k] == v for k, v in ref.items()) and json.loads(reservation["ref_json"]) == ref
    assert reservation["state"] == "SETTLED" and reservation["terminal_sha256"] == sha(attempt["terminal_json"].encode())
    record = json.loads(artifact["record_json"])
    assert data(Path(record["path"])) == 10 and sha(Path(record["path"]).read_bytes()) == record["content_sha256"]
    return {"ref": ref, "terminal_sha256": reservation["terminal_sha256"]}


def main(preliminary=False):
    result = {"baseline_core": BASE, "baseline_pro": PBASE,
              "audit_script_sha256": sha(Path(__file__).read_bytes()), **source_checks(), "suites": {}, "runs": {}}
    for name, path, passed in (("core", OUT / "core-full", 4956), ("pro", POUT / "pro-full", 1152),
                               ("paired", POUT / "paired", 31), ("targeted", OUT / "targeted-v1", 156)):
        if preliminary and not (path / "run.json").exists():
            result["suites"][name] = {"pending": True, "before_sha256": sha((path / "before.json").read_bytes())}
        else:
            value = common.frozen(path)
            assert value["passed"] == passed
            result["suites"][name] = value
    for name, code in (("baseline-tests", 0), ("baseline-regression", 1), ("differential", 0),
                       ("product", 0), ("source-v1", 0), ("source-v2", 0), ("source-memory", 0),
                       ("source-allocator", 0), ("admission-v1", 0), ("admission-v2", 0)):
        result["runs"][name] = prior.historical_run(OUT / name, code)
    assert result["runs"]["baseline-regression"]["junit"]["failures"] == 3
    raw = (OUT / "baseline-regression/stdout.log").read_text()
    assert all(value in raw for value in ("40600", "50400", "54700", "3 failed, 18 passed"))
    archives, checked = archives_check()
    result["archives"] = checked
    result["benchmarks"] = {**source_benchmarks(archives), **admission_benchmarks(archives)}
    result.update(differential_and_prototype(archives))
    result["workers"] = [product_check(archives)]
    result.update(passed=True, complete=not preliminary,
                  limits=["Separate-process mechanical audit, not another reviewer.",
                          "SQLite plans and opcode counts observed on this local runtime; optional indexes are not an authority requirement.",
                          "One current CPU worker and synthetic history; no real model, scientific speedup or production acceptance.",
                          "Full legacy derivation and one-config/YAML size remain; allocator and Python counters are not RSS."])
    target = OUT / ("preflight.json" if preliminary else "verification.json")
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "complete": not preliminary, "suites": result["suites"],
                      "archive_files": sum(r["files"] for r in checked.values()),
                      "verification_sha256": sha(target.read_bytes())}))


if __name__ == "__main__":
    main(preliminary="--preflight" in sys.argv)
