"""Mechanical audit of scan windows, unchanged qualification, and real CPU runs."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
PRO = ROOT.parent / "pro"
OUT = ROOT / "docs/evidence/runs/2026-09-15-sidecar-scan-windows"
POUT = PRO / "docs/evidence/runs/2026-09-15-sidecar-scan-windows"
BASE = "a952321b33e6973d6c580f5593139bebdce19e30"
PBASE = "8e556ddc88f1e78a625585683b84de3c04ed6022"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


previous = module("status_audit", ROOT / "docs/evidence/checks/2026-09-15-status-merge-verify.py")
previous.OUT = OUT
prior = previous.prior
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
    old = ast.parse((OUT / "baseline/ideas.py").read_bytes())
    new = ast.parse((ROOT / "src/orze/core/ideas.py").read_bytes())
    new.body.remove(function(new, "_sidecar_sections"))
    a, b = function(old, "_iter_sidecar_text"), function(new, "_iter_sidecar_text")
    a.body = [n for n in a.body if not (isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "sm_list" for t in n.targets))]
    old_for = next(n for n in a.body if isinstance(n, ast.For))
    new_for = next(n for n in b.body if isinstance(n, ast.For))
    old_for.target, old_for.iter = copy.deepcopy(new_for.target), copy.deepcopy(new_for.iter)
    old_for.body = [n for n in old_for.body if not (isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "se" for t in n.targets))]
    assert ast.dump(old) == ast.dump(new), "only section enumeration changed; parsing/precedence and legacy overlay retained"
    old = ast.parse((OUT / "baseline/sidecar_prefix.py").read_bytes())
    new = ast.parse((ROOT / "src/orze/engine/sidecar_prefix.py").read_bytes())
    new.body[0] = copy.deepcopy(old.body[0])
    cls = function(new, "SidecarPrefix")
    cls.body.remove(function(cls, "_names"))
    init = function(cls, "__init__")
    init.body = [n for n in init.body if not (isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Attribute) and t.attr.startswith("name_window") for t in n.targets))]
    call, = [n for n in ast.walk(function(cls, "stream")) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "_names"]
    call.func = ast.Name(id="_names", ctx=ast.Load())
    assert ast.dump(old) == ast.dump(new), "only name retrieval/state added; all identity and cleanup rules retained"
    from orze.core import ideas
    from orze.engine import sidecar_prefix
    assert Path(ideas.__file__).resolve() == ROOT / "src/orze/core/ideas.py"
    assert Path(sidecar_prefix.__file__).resolve() == ROOT / "src/orze/engine/sidecar_prefix.py"
    return {"unchanged_core_sources": common.unchanged(ROOT, BASE, "src", {
                "src/orze/core/ideas.py", "src/orze/engine/sidecar_prefix.py"}),
            "unchanged_core_tests": common.unchanged(ROOT, BASE, "tests"),
            "unchanged_pro_sources": common.unchanged(PRO, PBASE, "src"),
            "unchanged_pro_tests": common.unchanged(PRO, PBASE, "tests")}


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
        assert report["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-sidecar-scan-benchmark.py").read_bytes())
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
            for key in ("sidecar_payload_reads", "sidecar_payload_bytes", "file_identity_checks", "yaml_loads", "begins", "rollbacks"):
                assert before[key] == after[key]
            assert before["begins"] == before["rollbacks"] == size
            assert before["directory_enumerations"] == (size + 127) // 128
            assert after["directory_enumerations"] == ((size + 511) // 512 if layout == "many" else 1)
            assert after["max_name_window"] <= 512
            assert before["heading_matches"] >= after["heading_matches"]
            if layout == "many":
                assert before["heading_matches"] == after["heading_matches"]
        checked[name] = sha((root / "report.json").read_bytes())
    for name in ("invalid_v1", "invalid_v2"):
        root = Path(archives[name]["root"])
        report = data(root / "report.json")
        assert report["passed"] and report["baseline"] == BASE and report["repetitions"] == 3
        assert report["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-sidecar-scan-invalid-benchmark.py").read_bytes())
        for path, digest in report["sources"].items():
            assert sha(Path(path).read_bytes()) == digest
        assert [r["unparsed_headings"] for r in report["rows"]] == [500, 5000, 50000]
        for row in report["rows"]:
            count = row["unparsed_headings"]
            project = root / str(count)
            raw = (project / "ideas.d/all.md").read_bytes()
            assert len(raw) == row["source_bytes"] <= 4 * 1024 * 1024 and sha(raw) == row["source_sha256"]
            assert prior.db_digest(project / "lake.db") == row["database_sha256"]
            assert row["value"]["inserted"] == []
            assert row["orders"] == [["old", "new"], ["new", "old"], ["old", "new"]]
            for arm in row["arms"].values():
                assert len(arm["seconds"]) == len(arm["io_deltas"]) == 3
                assert arm["traced_peak_bytes"] >= arm["traced_current_bytes"] >= 0
            assert all(v == {"begins": 1, "rollbacks": 1, "heading_matches": count + 1, "yaml_loads": 2}
                       for v in row["counters"].values())
        checked[name] = sha((root / "report.json").read_bytes())
    return checked


def differential(archives):
    root = Path(archives["differential"]["root"])
    report = data(root / "report.json")
    assert report["passed"] and len(report["differential"]) == 128
    assert report["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-sidecar-scan-differential.py").read_bytes())
    assert report["original_sha256"] == sha((ROOT / "src/orze/core/ideas.py").read_bytes())
    assert report["baseline_sha256"] == sha((OUT / "baseline/ideas.py").read_bytes())
    for index, record in enumerate(report["differential"]):
        assert sha((root / f"input-{index // 2:03d}.md").read_bytes()) == record["input_sha256"]
        assert record["seed"] == 915 + index // 2 and record["limit"] == (None if index % 2 == 0 else 3)
        assert record["result"]["error"] is None
    return sha((root / "report.json").read_bytes())


def prototypes(archives):
    root = Path(archives["prototype"]["root"])
    for name, digest in data(root / "report.json")["reports"].items():
        assert sha((root / name).read_bytes()) == digest
    failure = data(OUT / "first-archive/failure.json")
    assert failure["observed_exit_code"] == 1 and failure["observed_failure"] == "AssertionError: prototype"
    archive_source = "2026-09-15-sidecar-scan-archive.py"
    assert (OUT / "first-archive" / archive_source).read_bytes() == (ROOT / "docs/evidence/checks" / archive_source).read_bytes()
    assert failure["archive_script_sha256"] == sha((ROOT / "docs/evidence/checks" / archive_source).read_bytes())
    for name, code in (("tests.json", 0), ("name-old.json", 1), ("name-new.json", 0)):
        report = data(root / name)
        assert report["exit_code"] == code
        for path, digest in report["files"].items():
            assert sha((root / path).read_bytes()) == digest
        for path, digest in report["frozen_inputs"].items():
            assert sha(git(ROOT, "show", BASE + ":" + path)) == digest
    for name in ("run_existing.py", "run_name_tests.py", "probe.py", "parser_probe.py"):
        assert (root / name).is_file()
    probe = data(root / "probe-report.json")
    assert probe["passed"] and probe["script_sha256"] == sha((root / "probe.py").read_bytes())
    assert probe["candidate_sha256"] == sha((root / "candidate.py").read_bytes())
    for path, digest in probe["source_test_files"].items():
        assert sha(git(ROOT, "show", BASE + ":" + path)) == digest
    fixture_index = data(Path(archives["prototype_probe"]["root"]) / "report.json")
    assert fixture_index["experiment_report_sha256"] == sha((root / "probe-report.json").read_bytes())
    for row in probe["rows"]:
        for path, digest in row["source_files"].items():
            assert sha(Path(path).read_bytes()) == digest
        project = Path(probe["root"]) / f'{row["records"]}-{row["layout"]}'
        assert prior.db_digest(project / "lake.db") == row["database_sha256"]
    parser = data(root / "parser-report.json")
    assert parser["passed"] and parser["script_sha256"] == sha((root / "parser_probe.py").read_bytes())
    assert len(parser["differential"]) == 128
    assert parser["candidate_sha256"] == sha((root / "parser_candidate.py").read_bytes())
    assert parser["original_sha256"] == sha((OUT / "baseline/ideas.py").read_bytes())
    return {"retained": True, "ordinary_latency_evidence": False}


def product_check(archives):
    root = Path(archives["product"]["root"])
    product = data(root / "report.json")
    assert product["passed"] and len(product["variants"]) == 2
    assert product["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-sidecar-prefix-product.py").read_bytes())
    for name, digest in product["source_files"].items():
        assert sha((ROOT / name).read_bytes()) == digest
    helper = module("product_helper", ROOT / "docs/evidence/checks/2026-09-15-ingress-product.py")
    workers = []
    for variant in product["variants"]:
        project = Path(variant["project"])
        assert len(variant["invocations"]) == 3 and all(r["exit_code"] == 0 for r in variant["invocations"])
        assert [r["inserted"] for r in variant["ingress"]] == [[], [], ["idea-tail"]]
        assert [len(r["ids"]) for r in variant["ingress"]] == [128, 128, 45]
        assert variant["ingress"][-1]["offset"] == 0
        assert variant["before"] == variant["after"] == helper.rows(project / "lake.db")
        for name, digest in variant["source_files"].items():
            assert sha((project / name).read_bytes()) == digest
        attempt, = variant["after"]["execution_attempts"]
        reservation, = variant["after"]["cpu_action_reservations"]
        artifact, = variant["after"]["research_artifacts"]
        terminal = json.loads(attempt["terminal_json"])
        assert attempt["task_id"] == "idea-tail" and attempt["state"] == "TERMINAL" and terminal["outcome"] == "completed"
        tree = terminal["process_tree"]
        assert tree["event"] == "TREE_CLOSED" and tree["wait_proof"] == "ECHILD_WALL"
        ref = tree["binding"]["identity"]["attempt_ref"]
        assert all(attempt[k] == v for k, v in ref.items()) and json.loads(reservation["ref_json"]) == ref
        assert reservation["state"] == "SETTLED" and reservation["terminal_sha256"] == sha(attempt["terminal_json"].encode())
        record = json.loads(artifact["record_json"])
        assert data(Path(record["path"])) == [1, 3, 5] and sha(Path(record["path"]).read_bytes()) == record["content_sha256"]
        workers.append({"ref": ref, "terminal_sha256": reservation["terminal_sha256"]})
    return workers


def main(preliminary=False):
    result = {"baseline_core": BASE, "baseline_pro": PBASE,
              "audit_script_sha256": sha(Path(__file__).read_bytes()), **source_checks(), "suites": {}, "runs": {}}
    for name, path, passed in (("core", OUT / "core-full", 4983), ("pro", POUT / "pro-full", 1152),
                               ("paired", POUT / "paired", 31), ("targeted", OUT / "targeted-v1", 104)):
        if preliminary and not (path / "run.json").exists():
            result["suites"][name] = {"pending": True, "before_sha256": sha((path / "before.json").read_bytes())}
        else:
            value = common.frozen(path)
            assert value["passed"] == passed
            result["suites"][name] = value
    for name in ("baseline-tests", "baseline-regression", "product", "differential", "many-v1", "one-v1",
                 "many-v2", "one-v2", "many-memory", "one-memory", "invalid-v1", "invalid-v2"):
        result["runs"][name] = prior.historical_run(OUT / name, 1 if name == "baseline-regression" else 0)
    assert result["runs"]["baseline-regression"]["junit"]["failures"] == 2
    raw = (OUT / "baseline-regression/stdout.log").read_text()
    assert "assert 3 == 1" in raw and "assert 10000 <= 129" in raw
    archives, checked = previous.archives_check()
    result["archives"] = checked
    result["benchmarks"] = benchmarks(archives)
    result["differential_sha256"] = differential(archives)
    result["prototypes"] = prototypes(archives)
    result["workers"] = product_check(archives)
    result.update(passed=True, complete=not preliminary,
                  limits=["Separate-process mechanical audit, not another reviewer.",
                          "Small local timing differences and complete negative controls; no scientific or production claim.",
                          "Names and heading lookahead do not remove source reading, prefix identity revalidation, or full-cycle ID sets.",
                          "Two real CPU workers; no real-model, paid account, GPU, or service change.",
                          "Python tracing measures traced allocations, not RSS or a complete process budget."])
    target = OUT / ("preflight.json" if preliminary else "verification.json")
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "complete": not preliminary, "suites": result["suites"],
                      "archive_files": sum(r["files"] for r in checked.values()),
                      "verification_sha256": sha(target.read_bytes())}))


if __name__ == "__main__":
    main(preliminary="--preflight" in sys.argv)
