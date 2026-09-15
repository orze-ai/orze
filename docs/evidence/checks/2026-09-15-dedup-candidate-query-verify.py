"""Separate mechanical audit of final code, abandoned variants and real runs."""
import ast
import importlib.util
import json
from pathlib import Path
import sqlite3
import tarfile
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[3]
PRO = REPO.parent / "pro"
OUT = REPO / "docs/evidence/runs/2026-09-15-dedup-candidate-query"
BASE = "c17ce2e8573bd14d0153cb7e2f030914e57a36fd"
PRO_BASE = "6c12fc898ec0b71b93f61240fdc10a1ea6f5899e"
spec = importlib.util.spec_from_file_location("audit_common", REPO / "docs/evidence/checks/2026-09-15-policy-audit-verify.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
sha, data, git = common.sha, common.data, common.git


def logical(path):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        return sha("\n".join(conn.iterdump()).encode())


def original_run(path):
    run = data(path / "run.json")
    assert run["frozen"] and run["before"] == run["after"] and run["exit_code"] != 0
    for name, record in run["files"].items():
        raw = (path / name).read_bytes()
        assert sha(raw) == record["sha256"] and len(raw) == record["bytes"]
    for key, files in run["before"].items():
        root = Path(run["repositories"][key])
        assert all(sha((root / name).read_bytes()) == digest for name, digest in files.items())
    suite, = ET.parse(path / "junit.xml").getroot().findall("testsuite")
    counts = {k: int(suite.attrib[k]) for k in ("tests", "failures", "errors", "skipped")}
    assert counts["failures"] >= 3
    assert "AF_UNIX path too long" in (path / "stdout.log").read_text()
    return {"exit_code": run["exit_code"], "counts": counts,
            "run_sha256": sha((path / "run.json").read_bytes()), "interruption_requested": True}


def main():
    changed = "src/orze/core/proposal_admission.py"
    old = git(REPO, "show", BASE + ":" + changed)
    assert old == (OUT / "baseline/proposal_admission.py").read_bytes()
    trees = [ast.parse(old), ast.parse((REPO / changed).read_bytes())]
    def stable(tree):
        return [ast.dump(n) for n in tree.body if getattr(n, "name", None) != "_dedup_owner"
                and not (isinstance(n, ast.ImportFrom) and n.module == "heapq")]
    assert stable(trees[0]) == stable(trees[1])
    owners = [next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "_dedup_owner") for t in trees]
    loops = [next(n for n in fn.body if isinstance(n, ast.For)) for fn in owners]
    assert [ast.dump(n) for n in loops[0].body] == [ast.dump(n) for n in loops[1].body]
    result = {"baseline": BASE, "pro_baseline": PRO_BASE,
              "changed_module_sha256": sha((REPO / changed).read_bytes()),
              "unchanged_core_sources": common.unchanged(REPO, BASE, "src", (changed,)),
              "unchanged_core_tests": common.unchanged(REPO, BASE, "tests"),
              "unchanged_pro_sources": common.unchanged(PRO, PRO_BASE, "src"),
              "unchanged_pro_tests": common.unchanged(PRO, PRO_BASE, "tests"),
              "suites": {"core": common.frozen(OUT / "core-full-v2"),
                         "targeted": common.frozen(OUT / "targeted"),
                         "pro": common.frozen(PRO / "docs/evidence/runs/2026-09-15-dedup-candidate-query/pro-full"),
                         "optional": common.frozen(PRO / "docs/evidence/runs/2026-09-15-dedup-candidate-query/core-pro-optional")},
              "first_core_run": original_run(OUT / "core-full")}
    assert "29 passed" in (OUT / "baseline-tests.log").read_text()
    assert "136 passed" in (OUT / "targeted-v1.log").read_text()
    assert "29 passed" in (OUT / "branch-limit-targeted.log").read_text()
    assert "3 failed, 1 passed" in (OUT / "failure-reproduction.log").read_text()
    assert "4 passed" in (OUT / "failure-reproduction-v2.log").read_text()
    assert data(OUT / "interruption-note.json")["signal"] == "SIGINT"
    for path, digest in data(OUT / "harness-inputs.json").items():
        assert sha((REPO / path).read_bytes()) == digest
    archives = data(OUT / "archives.json")
    for record in archives.values():
        assert sha((OUT / record["archive"]).read_bytes()) == record["sha256"]
        with tarfile.open(OUT / record["archive"], "r:gz") as tar:
            assert {m.name for m in tar.getmembers()} == set(record["members"])
            for member in tar.getmembers():
                assert sha(tar.extractfile(member).read()) == record["members"][member.name]
                assert sha((Path(record["root"]) / member.name).read_bytes()) == record["members"][member.name]
    def sources(record, variant):
        for path, digest in record["sources"].items():
            expected = OUT / variant / "proposal_admission.py" if path == str(REPO / changed) and variant else Path(path)
            assert sha(expected.read_bytes()) == digest
    for version, variant in (("v1", "before-branch-limits"), ("v2", "nested-branch-limits"), ("v3", None), ("v4", None)):
        benchmark = data(OUT / f"benchmark-{version}.json")
        root = Path(archives["benchmark_" + version]["root"])
        assert benchmark == data(root / "report.json")
        script = OUT / "benchmark-script-v1.py" if variant else REPO / "docs/evidence/checks/2026-09-15-dedup-candidate-query-benchmark.py"
        assert sha(script.read_bytes()) == benchmark["script_sha256"]
        sources(benchmark, variant)
        assert [(r["history"], r["mode"]) for r in benchmark["rows"]] == [
            (1, "exact"), (100, "exact"), (1000, "exact"), (5000, "exact"),
            (1, "semantic"), (5000, "semantic"), (1000, "crowded"), (5000, "crowded")]
        for row in benchmark["rows"]:
            project = root / f'{row["history"]}-{row["mode"]}'
            assert logical(project / "lake.db") == row["database_sha256_before"] == row["database_sha256_after"]
            assert sha((project / "ideas.md").read_bytes()) == row["source_sha256"]
            assert row["value"]["inserted"] == [] and len(row["orders"]) == 7
            assert all(len(a["seconds"]) == 7 for a in row["arms"].values())
            for arm, counts in row["counters"].items():
                assert counts["begin_immediate"] == counts["rollbacks"] == 128
                assert counts["yaml_loads"] == (384 if row["mode"] == "semantic" else 256)
                expected_status = "rejected" if row["mode"] == "crowded" and row["history"] > 1024 else "config_duplicate"
                assert counts["admission_status"] == expected_status
                if not variant:
                    assert len(counts["candidate_queries"]) == (1 if arm == "old" or expected_status == "rejected" else 2)
    for version, variant in (("v1", "before-branch-limits"), ("v2", None)):
        differential = data(OUT / f"differential-{version}.json")
        root = Path(archives["differential_" + version]["root"])
        assert differential == data(root / "report.json") and differential["passed"]
        assert differential["script_sha256"] == sha((REPO / "docs/evidence/checks/2026-09-15-dedup-candidate-query-differential.py").read_bytes())
        sources(differential, variant)
        assert len(differential["cases"]) == 64
        for index, case in enumerate(differential["cases"]):
            project = root / f"case-{index:03d}"
            assert logical(project / "baseline.db") == case["database_sha256_before"]
            for entry, outcomes in case["outcomes"].items():
                assert outcomes["old"] == outcomes["new"]
                for arm, outcome in outcomes.items():
                    expected = outcome["database_sha256_after"] if entry == "normal" else outcome["rollback_sha256"]
                    assert logical(project / f"{entry}-{arm}.db") == expected
    root = Path(archives["product"]["root"])
    product = data(OUT / "product.json")
    assert product == data(root / "report.json") and product["passed"]
    assert len(product["invocations"]) == 2 and all(i["exit_code"] == 0 for i in product["invocations"])
    assert product["invocations"][0]["database"] == product["invocations"][1]["database"]
    assert sha((root / "ideas.d/actions.md").read_bytes()) == product["source_sha256"]
    assert (root / "ideas.md").read_text() == "# Ideas\n"
    for path, digest in product["source_files"].items():
        assert sha((REPO / path).read_bytes()) == digest
    with sqlite3.connect((root / "lake.db").as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for name, expected in product["invocations"][1]["database"].items():
            assert [dict(r) for r in conn.execute("SELECT * FROM " + name)] == expected
        row, = conn.execute("SELECT * FROM execution_attempts").fetchall()
        reservation, = conn.execute("SELECT * FROM cpu_action_reservations").fetchall()
        artifact, = conn.execute("SELECT * FROM research_artifacts").fetchall()
        terminal = json.loads(row["terminal_json"])
        tree = terminal["process_tree"]
        assert row["state"] == "TERMINAL" and terminal["outcome"] == "completed"
        assert tree["event"] == "TREE_CLOSED" and tree["wait_proof"] == "ECHILD_WALL"
        ref = tree["binding"]["identity"]["attempt_ref"]
        assert json.loads(reservation["ref_json"]) == ref and all(row[k] == v for k, v in ref.items())
        assert reservation["state"] == "SETTLED" and reservation["terminal_sha256"] == sha(row["terminal_json"].encode())
        assert data(Path(json.loads(artifact["record_json"])["path"])) == [1, 3, 5]
    result.update(passed=True, actual_worker={"ref": ref, "terminal_sha256": reservation["terminal_sha256"]},
                  archived_regular_files=sum(len(r["members"]) for r in archives.values()),
                  limits=["Separate mechanical process, not another reviewer.",
                          "Each admission retains its writer. Sparse healthy identities avoid unrelated rows; missing/crowded candidates still cost work.",
                          "Synthetic histories and offline tests; actual CPU execution is distinct from research benefit or production acceptance."])
    assert not (OUT / "verification.json").exists()
    (OUT / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
