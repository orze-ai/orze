"""Audit frozen partial-file continuation evidence in a separate process."""
import ast
import base64
import copy
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
PRO = ROOT.parent / "pro"
OUT = ROOT / "docs/evidence/runs/2026-09-15-sidecar-partial-prefix"
POUT = PRO / "docs/evidence/runs/2026-09-15-sidecar-partial-prefix"
BASE = "e5289587a0c4aa445cfc420d77b7e174a9c24c03"
PBASE = "f7ba224a01aa13d8d0739873def24f5272b3ba5e"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


scan = module("scan_audit", ROOT / "docs/evidence/checks/2026-09-15-sidecar-scan-verify.py")
scan.OUT, scan.POUT, scan.BASE, scan.PBASE = OUT, POUT, BASE, PBASE
scan.previous.OUT = OUT
prior = scan.prior
prior.OUT, prior.POUT, prior.BASE, prior.PBASE = OUT, POUT, BASE, PBASE
common = scan.common
sha, data, git = common.sha, common.data, common.git
function = scan.function


def source_checks():
    baseline = data(OUT / "baseline/source.json")
    assert baseline["head"] == BASE and baseline["pro_head"] == PBASE
    for name, digest in baseline["files"].items():
        raw = (OUT / "baseline" / Path(name).name).read_bytes()
        assert raw == git(ROOT, "show", BASE + ":" + name) and sha(raw) == digest
    old = ast.parse((OUT / "baseline/sidecar_prefix.py").read_bytes())
    new = ast.parse((ROOT / "src/orze/engine/sidecar_prefix.py").read_bytes())
    new.body[0] = copy.deepcopy(old.body[0])
    a, b = function(old, "SidecarPrefix"), function(new, "SidecarPrefix")
    init = function(b, "__init__")
    init.body = [n for n in init.body if not (isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Attribute) and t.attr in {"partial", "generation"} for t in n.targets))]
    condition = next(n.test for n in ast.walk(function(b, "verify")) if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp))
    condition.values = [v for v in condition.values if not any(isinstance(n, ast.Attribute) and n.attr == "partial" for n in ast.walk(v))]
    def file_loop(cls):
        return next(n for n in ast.walk(function(cls, "stream")) if isinstance(n, ast.For)
                    and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Attribute) and n.iter.func.attr == "_names")
    old_loop, new_loop = file_loop(a), file_loop(b)
    def amount_index(nodes):
        return next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "amount" for t in n.targets))
    new_loop.body[3:amount_index(new_loop.body)] = copy.deepcopy(old_loop.body[3:amount_index(old_loop.body)])
    assert ast.dump(old) == ast.dump(new), "only partial state, its verification and per-file consumption/publication changed"
    from orze.engine import sidecar_prefix
    assert Path(sidecar_prefix.__file__).resolve() == ROOT / "src/orze/engine/sidecar_prefix.py"
    return {"unchanged_core_sources": common.unchanged(ROOT, BASE, "src", {"src/orze/engine/sidecar_prefix.py"}),
            "unchanged_core_tests": common.unchanged(ROOT, BASE, "tests"),
            "unchanged_pro_sources": common.unchanged(PRO, PBASE, "src"),
            "unchanged_pro_tests": common.unchanged(PRO, PBASE, "tests")}


def prototypes(archives):
    root = Path(archives["prototype"]["root"])
    for path, digest in data(root / "report.json")["reports"].items():
        assert sha((root / path).read_bytes()) == digest
    original = data(root / "baseline/source.json")
    assert original["uncommitted_scan_windows"] is True
    for path, digest in original["files"].items():
        assert sha(git(ROOT, "show", BASE + ":" + path)) == digest
        assert sha((root / "baseline" / Path(path).name).read_bytes()) == digest
    for run, script, failed, passed in (("tests-old", "run_tests.py", 1, 0), ("tests-v1", "run_tests.py", 1, 22),
                                        ("tests-v2", "run_tests.py", 0, 23), ("boundaries-v2", "run_boundaries.py", 0, 17)):
        report = data(root / run / "run.json")
        assert report["exit_code"] == (1 if failed else 0) and report["frozen"] and report["before"] == report["after"]
        assert report["script_sha256"] == sha((root / script).read_bytes())
        for path, digest in report["files"].items():
            assert sha((root / path).read_bytes()) == digest
        for path, digest in report["before"].items():
            assert sha(git(ROOT, "show", BASE + ":" + path)) == digest
        suites = ET.parse(root / run / "junit.xml").getroot().findall("testsuite")
        assert sum(int(s.attrib.get("failures", 0)) for s in suites) == failed
        assert sum(int(s.attrib.get("tests", 0)) for s in suites) == failed + passed
        assert sum(int(s.attrib.get("errors", 0)) for s in suites) == 0
    failure = (root / "tests-v1/stdout.log").read_text()
    assert "test_read_failure_does_not_grant_precedence_to_cached_ids" in failure and "['idea-tail']" in failure
    candidate = ast.parse((root / "candidate-v2.py").read_bytes())
    current = ast.parse((ROOT / "src/orze/engine/sidecar_prefix.py").read_bytes())
    current.body[0] = copy.deepcopy(candidate.body[0])
    assert ast.dump(candidate) == ast.dump(current)
    probe = data(root / "probe-report.json")
    assert probe["passed"] and probe["frozen"] and probe["script_sha256"] == sha((root / "probe.py").read_bytes())
    assert probe["candidate_sha256"] == sha((root / "candidate-v2.py").read_bytes())
    assert probe["baseline_sha256"] == sha((root / "baseline/sidecar_prefix.py").read_bytes())
    for path, digest in probe["source_test_files"].items():
        assert sha(git(ROOT, "show", BASE + ":" + path)) == digest
    for row in probe["rows"]:
        for path, digest in row["sources"].items():
            assert sha(Path(path).read_bytes()) == digest
        project = Path(probe["root"]) / f'{row["layout"]}-{row["records"]}'
        assert prior.db_digest(project / "lake.db") == row["database_sha256"]
        assert row["arms"]["old"]["value"] == row["arms"]["new"]["value"]
    fixture_index = data(Path(archives["prototype_probe"]["root"]) / "report.json")
    assert fixture_index["experiment_report_sha256"] == sha((root / "probe-report.json").read_bytes())
    return {"retained": True, "ordinary_latency_evidence": False, "initial_read_failure_preserved": True}


def benchmarks(archives):
    checked = {}
    for name in ("many_v1", "one_v1", "many_v2", "one_v2", "many_memory", "one_memory"):
        root = Path(archives[name]["root"])
        report = data(root / "report.json")
        layout = name.split("_")[0]
        memory = name.endswith("memory")
        assert report["passed"] and report["baseline"] == BASE and report["layout"] == layout
        assert report["mode"] == ("memory" if memory else "time")
        assert report["repetitions"] == (1 if memory else 3)
        assert report["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-sidecar-partial-benchmark.py").read_bytes())
        assert report["helper_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-ingress-benchmark.py").read_bytes())
        for path, digest in report["source_files"].items():
            assert sha((ROOT / path).read_bytes()) == digest
        for path, digest in report["baseline_files"].items():
            assert sha((OUT / "baseline" / path).read_bytes()) == digest
        assert [r["records"] for r in report["rows"]] == ([100, 1000, 5000] if layout == "many" else [1000, 5000])
        for row in report["rows"]:
            size = row["records"]
            assert prior.db_digest(root / str(size) / "lake.db") == row["database_sha256"]
            for path, digest in row["source_files"].items():
                assert sha(Path(path).read_bytes()) == digest
            for scope in ("first_batch", "full_cycle"):
                result = row[scope]
                assert result["value"]["records"] == (size if scope == "full_cycle" else min(128, size))
                assert result["orders"] == [["old", "new"], ["new", "old"], ["old", "new"]][:report["repetitions"]]
                for arm in result["arms"].values():
                    assert len(arm["seconds"]) == len(arm["io_deltas"]) == report["repetitions"]
                    assert ("traced_peak_bytes" in arm) == memory
                    if memory:
                        assert arm["traced_peak_bytes"] >= arm["traced_current_bytes"] >= 0
            before, after = row["separate_io_audit"]["old"], row["separate_io_audit"]["new"]
            for key in ("sidecar_payload_reads", "sidecar_payload_bytes", "heading_matches", "directory_enumerations", "directory_entries", "begins", "rollbacks"):
                assert before[key] == after[key]
            assert before["begins"] == before["rollbacks"] == size
            assert before["directory_enumerations"] == ((size + 511) // 512 if layout == "many" else 1)
            assert after["directory_enumerations"] == ((size + 511) // 512 if layout == "many" else 1)
            assert after["max_name_window"] <= 512
            assert before["heading_matches"] == after["heading_matches"]
            if layout == "many":
                assert before["file_identity_checks"] == after["file_identity_checks"]
                assert before["yaml_loads"] == after["yaml_loads"]
                assert after["max_partial_ids"] == 0
            else:
                assert before["file_identity_checks"] < after["file_identity_checks"]
                assert before["yaml_loads"] > after["yaml_loads"]
                assert 0 < after["max_partial_ids"] <= 32768
            assert after["max_partial_id_bytes"] <= 1024 * 1024
        checked[name] = sha((root / "report.json").read_bytes())
    return checked


def differential(archives):
    root = Path(archives["differential"]["root"])
    report = data(root / "report.json")
    assert report["passed"] and report["baseline"] == BASE and len(report["cases"]) == 64
    assert report["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-sidecar-partial-differential.py").read_bytes())
    assert report["baseline_sha256"] == sha((OUT / "baseline/sidecar_prefix.py").read_bytes())
    assert report["helper_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-ingress-benchmark.py").read_bytes())
    for path, digest in report["sources"].items():
        assert sha((ROOT / path).read_bytes()) == digest
    mutations = ["none", "rewrite_restore_mtime", "replace", "namespace", "delete", "invalid_utf8", "empty_reader", "truncate"]
    for index, record in enumerate(report["cases"]):
        assert record["case"] == index
        assert record["layout"] == ("one", "many", "mixed", "small_groups")[(index // 8) % 4]
        assert record["mutation"] == mutations[index % 8]
        project = root / f"case-{index:03d}"
        plan = data(project / "input.json")
        assert sha((project / "input.json").read_bytes()) == record["input_sha256"]
        assert all(plan[k] == record[k] for k in ("layout", "mutation", "limits"))
        assert all(len(base64.b64decode(value, validate=True)) < 4 * 1024 * 1024 for value in plan["initial_files_base64"].values())
        assert len(record["outputs"][0]["ids"]) == 128 and record["outputs"][-1]["offset"] == 0
        assert all(len(page["ids"]) <= 128 for page in record["outputs"])
        for path, digest in record["sources"].items():
            assert sha((project / path).read_bytes()) == digest
        assert prior.db_digest(project / "lake.db") == record["database_sha256"]
    return sha((root / "report.json").read_bytes())


def main(preliminary=False):
    result = {"baseline_core": BASE, "baseline_pro": PBASE,
              "audit_script_sha256": sha(Path(__file__).read_bytes()), **source_checks(), "suites": {}, "runs": {}}
    for name, path, passed in (("core", OUT / "core-full", 5005), ("pro", POUT / "pro-full", 1152),
                               ("paired", POUT / "paired", 31), ("targeted", OUT / "targeted-v1", 126)):
        if preliminary and not (path / "run.json").exists():
            result["suites"][name] = {"pending": True, "before_sha256": sha((path / "before.json").read_bytes())}
        else:
            value = common.frozen(path)
            assert value["passed"] == passed
            result["suites"][name] = value
    run_names = ("baseline-tests", "baseline-regression", "product", "differential", "many-v1", "one-v1", "many-v2", "one-v2", "many-memory", "one-memory")
    for name in run_names:
        result["runs"][name] = prior.historical_run(OUT / name, 1 if name == "baseline-regression" else 0)
    assert result["runs"]["baseline-tests"]["junit"]["tests"] == 104
    assert result["runs"]["baseline-regression"]["junit"]["failures"] == 1
    assert result["runs"]["baseline-regression"]["junit"]["tests"] == 2
    assert "assert [0] == [128]" in (OUT / "baseline-regression/stdout.log").read_text()
    previous_end = max(data(OUT / name / "run.json")["finished_unix"] for name in ("targeted-v1", "product", "differential"))
    for name in ("many-v1", "one-v1", "many-v2", "one-v2", "many-memory", "one-memory"):
        run = data(OUT / name / "run.json")
        assert run["started_unix"] >= previous_end
        previous_end = run["finished_unix"]
    for path in (OUT / "core-full", POUT / "pro-full", POUT / "paired"):
        if (path / "run.json").exists():
            assert data(path / "run.json")["started_unix"] >= previous_end
    archives, checked = scan.previous.archives_check()
    result["archives"] = checked
    result["benchmarks"] = benchmarks(archives)
    result["differential_sha256"] = differential(archives)
    result["prototypes"] = prototypes(archives)
    result["workers"] = scan.product_check(archives)
    result.update(passed=True, complete=not preliminary,
                  limits=["Separate-process mechanical audit, not another reviewer.",
                          "Two real CPU workers; no real model, paid account, GPU or existing-service change.",
                          "Selected files are fully read again; heading scans, full-cycle ID sets and capacity fallback remain.",
                          "Partial hints share complete-prefix capacity and add identity checks; no cached admission or execution authority.",
                          "Python tracing is not RSS, SQLite allocator accounting or a complete process budget.",
                          "Behavioral differential compares both arms within the frozen script and retains a shared output summary."])
    target = OUT / ("preflight.json" if preliminary else "verification.json")
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "complete": not preliminary, "suites": result["suites"],
                      "archive_files": sum(r["files"] for r in checked.values()), "verification_sha256": sha(target.read_bytes())}))


if __name__ == "__main__":
    main(preliminary="--preflight" in sys.argv)
