"""Separate-process mechanical audit; not another reviewer or overall acceptance."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
PRO = ROOT.parent / "pro"
OUT = ROOT / "docs/evidence/runs/2026-09-15-ingress-preparation"
POUT = PRO / "docs/evidence/runs/2026-09-15-ingress-preparation"
BASE = "1b8ac7b5e7a398ceae95f58bbd47e8b3ae0ab895"
PBASE = "8ca58e9351f209209db9a9a4c3b7ceb9d276d433"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


previous = module("status_audit", ROOT / "docs/evidence/checks/2026-09-15-status-merge-verify.py")
previous.OUT, previous.POUT, previous.BASE, previous.PBASE = OUT, POUT, BASE, PBASE
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
    old = ast.parse((OUT / "baseline/idea_lake.py").read_bytes())
    new = ast.parse((ROOT / "src/orze/idea_lake.py").read_bytes())
    cls = function(new, "IdeaLake")
    prepare = function(cls, "prepare_admitted_config_hashes")
    lookup = function(function(old, "IdeaLake"), "find_admitted_config_hashes")
    assert ast.dump(prepare.body[1]) == ast.dump(lookup.body[1]), "same complete input normalization before SQL"
    assert len(prepare.body) == 3 and isinstance(prepare.body[-1], ast.If)
    assert ast.dump(prepare.body[-1].body[0]) == ast.dump(lookup.body[3]), "same legacy repair call"
    cls.body.remove(prepare)
    assert ast.dump(new) == ast.dump(old), "public lookup, all other Lake code identical"
    old = ast.parse((OUT / "baseline/idea_ingress.py").read_bytes())
    new = ast.parse((ROOT / "src/orze/engine/idea_ingress.py").read_bytes())
    old_try = next(n for n in ast.walk(old) if isinstance(n, ast.Try)
                   and len(n.body) == 1 and isinstance(n.body[0], ast.Expr)
                   and isinstance(n.body[0].value, ast.Call)
                   and isinstance(n.body[0].value.func, ast.Attribute)
                   and n.body[0].value.func.attr == "find_admitted_config_hashes")
    new_try = next(n for n in ast.walk(new) if isinstance(n, ast.Try)
                   and len(n.body) == 3 and isinstance(n.body[0], ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "prepare" for t in n.body[0].targets))
    assert ast.dump(new_try.body[-1].value.args[0]) == ast.dump(old_try.body[0].value.args[0])
    new_try.body = copy.deepcopy(old_try.body)
    assert ast.dump(new) == ast.dump(old), "only preparation dispatch changed; lock, parse, writer and ACK unchanged"
    adapted = ["tests/test_idea_ingress_cost.py", "tests/test_ingress_bounded_sources.py"]
    for name in adapted:
        old = ast.parse((OUT / "baseline" / Path(name).name).read_bytes())
        new = ast.parse((ROOT / name).read_bytes())
        changes = []
        for node in ast.walk(new):
            if isinstance(node, ast.Attribute) and node.attr == "prepare_admitted_config_hashes":
                node.attr = "find_admitted_config_hashes"
                changes.append("attribute")
            if isinstance(node, ast.Constant) and node.value == "prepare_admitted_config_hashes":
                node.value = "find_admitted_config_hashes"
                changes.append("injection")
        assert sorted(changes) == ["attribute", "injection"]
        assert ast.dump(new) == ast.dump(old), "all source-lock and concurrent-writer business assertions retained"
    name = "test_ingress_identity_preparation.py"
    old = ast.parse((OUT / "first-tests" / name).read_bytes())
    new = ast.parse((ROOT / "tests" / name).read_bytes())
    fixed = function(new, "test_missing_history_is_prepared_before_legal_normal_admission")
    insert = next(n for n in ast.walk(fixed) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == "insert")
    kw, = [k for k in insert.keywords if k.arg == "status"]
    assert kw.value.value == "queued"
    insert.keywords.remove(kw)
    assert ast.dump(old) == ast.dump(new), "first fixture only corrected its mandatory queued status"
    from orze import idea_lake
    from orze.engine import idea_ingress
    assert Path(idea_lake.__file__).resolve() == ROOT / "src/orze/idea_lake.py"
    assert Path(idea_ingress.__file__).resolve() == ROOT / "src/orze/engine/idea_ingress.py"
    return {"unchanged_core_sources": common.unchanged(ROOT, BASE, "src", {
                "src/orze/idea_lake.py", "src/orze/engine/idea_ingress.py"}),
            "unchanged_core_tests": common.unchanged(ROOT, BASE, "tests", set(adapted)),
            "adapted_test_observers": adapted,
            "unchanged_pro_sources": common.unchanged(PRO, PBASE, "src"),
            "unchanged_pro_tests": common.unchanged(PRO, PBASE, "tests")}


def benchmarks(archives):
    checked = {}
    expected = [(100, "exact"), (1000, "exact"), (5000, "exact"), (5000, "semantic"),
                (1000, "crowded"), (5000, "crowded"), (20000, "crowded")]
    for name in ("benchmark_v1", "benchmark_v2"):
        root = Path(archives[name]["root"])
        report = data(root / "report.json")
        assert report["passed"] and report["baseline"] == BASE and report["repetitions"] == 3
        assert report["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-ingress-preparation-benchmark.py").read_bytes())
        assert report["allocator_helper_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-status-merge-allocator.py").read_bytes())
        for path, digest in report["sources"].items():
            assert sha(Path(path).read_bytes()) == digest
        assert [(r["history"], r["mode"]) for r in report["rows"]] == expected
        for row in report["rows"]:
            history, mode = row["history"], row["mode"]
            project = root / f"{history}-{mode}"
            assert prior.db_digest(project / "lake.db") == row["database_sha256_before"] == row["database_sha256_after"]
            assert sha((project / "ideas.md").read_bytes()) == row["source_sha256"]
            assert row["entry"] == "complete_ingress" and row["value"]["inserted"] == []
            assert row["orders"] == [["old", "new"], ["new", "old"], ["old", "new"]]
            for arm in row["arms"].values():
                assert len(arm["seconds"]) == len(arm["io_deltas"]) == 3
                assert arm["traced_peak_bytes"] >= arm["traced_current_bytes"] >= 0
            status = "rejected" if mode == "crowded" and history > 1024 else "config_duplicate"
            for counter in row["counters"].values():
                assert counter["admission_status"] == status
                assert counter["begin_immediate"] == counter["rollbacks"] == 128
                assert counter["yaml_loads"] == (384 if mode == "semantic" else 256)
            assert len(row["counters"]["old"]["owner_mapping_queries"]) == 1
            assert row["counters"]["new"]["owner_mapping_queries"] == []
            assert row["counters"]["old"]["candidate_queries"] == row["counters"]["new"]["candidate_queries"]
            for native in row["sqlite_memory"].values():
                assert native["after"]["peak"] >= native["before"] > 0
        checked[name] = sha((root / "report.json").read_bytes())
    return checked


def differential_and_prototype(archives):
    root = Path(archives["differential"]["root"])
    report = data(root / "report.json")
    assert report["passed"] and report["baseline"] == BASE and len(report["cases"]) == 64
    assert report["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-ingress-preparation-differential.py").read_bytes())
    for path, digest in report["sources"].items():
        assert sha(Path(path).read_bytes()) == digest
    counts = {}
    for index, case in enumerate(report["cases"]):
        project = root / f"case-{index:03d}"
        assert prior.db_digest(project / "baseline.db") == case["database_sha256_before"]
        assert case["outcomes"]["old"] == case["outcomes"]["new"]
        for arm, result in case["outcomes"].items():
            assert prior.db_digest(project / arm / "lake.db") == result["database_sha256_after"]
            assert sha((project / arm / "ideas.md").read_bytes()) == result["source_sha256_after"]
            assert len(result["admissions"]) == 2
        for value in case["outcomes"]["new"]["admissions"]:
            key = value["status"] + ":" + value["reason"]
            counts[key] = counts.get(key, 0) + 1
    assert counts == report["result_counts"] == {"config_duplicate:proposal_config_duplicate": 28,
        "inserted:proposal_admitted": 29, "rejected:proposal_dedup_config_unavailable": 71}
    proto = Path(archives["prototype"]["root"])
    report = data(proto / "report.json")
    assert report["script_sha256"] == sha((proto / "investigate.py").read_bytes())
    for name, digest in report["source_files"].items():
        assert sha(git(ROOT, "show", BASE + ":" + name)) == digest
    assert (proto / "README.zh-CN.md").exists()
    return {"differential_sha256": sha((root / "report.json").read_bytes()),
            "prototype": {"archive_retained": True, "used_for_acceptance": False,
                          "reason": "No recorded runtime import paths/environment; heap outlier unresolved."}}


def main(preliminary=False):
    result = {"baseline_core": BASE, "baseline_pro": PBASE,
              "audit_script_sha256": sha(Path(__file__).read_bytes()), **source_checks(), "suites": {}, "runs": {}}
    for name, path, passed in (("core", OUT / "core-full", 4971), ("pro", POUT / "pro-full", 1152),
                               ("paired", POUT / "paired", 31), ("targeted", OUT / "targeted-v2", 171)):
        if preliminary and not (path / "run.json").exists():
            result["suites"][name] = {"pending": True, "before_sha256": sha((path / "before.json").read_bytes())}
        else:
            value = common.frozen(path)
            assert value["passed"] == passed
            result["suites"][name] = value
    for name, code in (("baseline-tests", 0), ("baseline-regression", 1), ("targeted-v1", 1),
                       ("differential-v1", 0), ("product", 0), ("benchmark-v1", 0), ("benchmark-v2", 0)):
        result["runs"][name] = prior.historical_run(OUT / name, code)
    assert result["runs"]["baseline-regression"]["junit"]["failures"] == 2
    assert "ingress discarded this full matching-history result" in (OUT / "baseline-regression/stdout.log").read_text()
    assert result["runs"]["targeted-v1"]["junit"]["failures"] == 1
    assert "proposal_requires_queued" in (OUT / "targeted-v1/stdout.log").read_text()
    archives, checked = previous.archives_check()
    result["archives"] = checked
    result["benchmarks"] = benchmarks(archives)
    result.update(differential_and_prototype(archives))
    result["workers"] = [previous.product_check(archives)]
    result.update(passed=True, complete=not preliminary,
                  limits=["Separate-process mechanical audit, not another reviewer.",
                          "Two local hot-history complete ingress rounds; sparse controls retain no stable speedup.",
                          "One CPU worker and imported history, no real-model or production/scientific acceptance.",
                          "Legacy preparation, individual config/YAML and per-proposal transactions remain.",
                          "SQLite allocator and Python tracing are separate observations, not RSS."])
    target = OUT / ("preflight.json" if preliminary else "verification.json")
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "complete": not preliminary, "suites": result["suites"],
                      "archive_files": sum(r["files"] for r in checked.values()),
                      "verification_sha256": sha(target.read_bytes())}))


if __name__ == "__main__":
    main(preliminary="--preflight" in sys.argv)
