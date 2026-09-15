"""Separate-process mechanical audit of repair semantics, cost and CPU settlement.

This is not another reviewer. Private Pro inventories stay in the private repo.
No Python heap figure below is an RSS or complete SQLite-memory bound.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
PRO = ROOT.parent / "pro"
OUT = ROOT / "docs/evidence/runs/2026-09-15-identity-repair-staging"
POUT = PRO / "docs/evidence/runs/2026-09-15-identity-repair-staging"
BASE = "d9cd439c79538a91030d9022ff2347d363df0c9d"
PBASE = "4ef1903ae2ec31e34083ef568bff39abef17bf82"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


common = module("audit_common", ROOT / "docs/evidence/checks/2026-09-15-policy-audit-verify.py")
sha, data, git = common.sha, common.data, common.git


def matching_source(name, digest):
    return any(p.is_file() and sha(p.read_bytes()) == digest for p in
               (ROOT / name, OUT / "baseline" / Path(name).name, OUT / "first-tests" / Path(name).name))


def historical_run(path, code):
    record = data(path / "run.json")
    assert record["frozen"] and record["before"] == record["after"] and record["exit_code"] == code
    assert sha(Path(record["recorder"]["path"]).read_bytes()) == record["recorder"]["sha256"]
    for name, entry in record["files"].items():
        raw = (path / name).read_bytes()
        assert len(raw) == entry["bytes"] and sha(raw) == entry["sha256"]
    for key, files in record["before"].items():
        assert Path(record["repositories"][key]) == ROOT
        for name, digest in files.items():
            assert matching_source(name, digest), name
    result = {"exit_code": code, "run_sha256": sha((path / "run.json").read_bytes())}
    if (path / "junit.xml").exists():
        suite, = ET.parse(path / "junit.xml").getroot().findall("testsuite")
        result["junit"] = {k: int(suite.attrib[k]) for k in ("tests", "errors", "failures", "skipped")}
    return result


def source_checks():
    path = "src/orze/idea_lake.py"
    old = git(ROOT, "show", BASE + ":" + path)
    assert old == (OUT / "baseline/idea_lake.py").read_bytes()
    def stable(raw):
        tree = ast.parse(raw)
        cls, = [n for n in tree.body if getattr(n, "name", None) == "IdeaLake"]
        cls.body = [n for n in cls.body if getattr(n, "name", None) != "_repair_admitted_config_hashes"]
        find, = [n for n in cls.body if getattr(n, "name", None) == "find_admitted_config_hashes"]
        assert isinstance(find.body[0], ast.Expr) and isinstance(find.body[0].value.value, str)
        find.body.pop(0)
        return ast.dump(tree)
    assert stable(old) == stable((ROOT / path).read_bytes())
    baseline = data(OUT / "baseline/source.json")
    assert baseline["head"] == BASE and baseline["pro_head"] == PBASE
    for name, digest in baseline["files"].items():
        assert sha(git(ROOT, "show", BASE + ":" + name)) == digest
        assert sha((OUT / "baseline" / Path(name).name).read_bytes()) == digest
    return {"unchanged_core_sources": common.unchanged(ROOT, BASE, "src", {path}),
            "unchanged_core_tests": common.unchanged(ROOT, BASE, "tests"),
            "unchanged_pro_sources": common.unchanged(PRO, PBASE, "src"),
            "unchanged_pro_tests": common.unchanged(PRO, PBASE, "tests")}


def db_digest(path):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        return sha("\n".join(conn.iterdump()).encode())


def main(preliminary=False):
    result = {"baseline_core": BASE, "baseline_pro": PBASE,
              "audit_script_sha256": sha(Path(__file__).read_bytes()), **source_checks(),
              "suites": {}, "runs": {}, "archives": {}, "benchmarks": {}}
    for name, path, passed in (("core", OUT / "core-full", 4935), ("pro", POUT / "pro-full", 1152),
                               ("paired", POUT / "paired", 31), ("targeted", OUT / "targeted-v2", 72)):
        if preliminary and not (path / "run.json").exists():
            result["suites"][name] = {"pending": True, "before_sha256": sha((path / "before.json").read_bytes())}
        else:
            value = common.frozen(path)
            assert value["passed"] == passed
            result["suites"][name] = value
    for name, code in (("baseline-tests", 0), ("compatibility", 0), ("targeted-v1", 1),
                       ("baseline-regression", 1), ("product", 0), ("benchmark-v1", 0),
                       ("benchmark-v2", 0), ("memory", 0)):
        result["runs"][name] = historical_run(OUT / name, code)
    assert result["runs"]["baseline-regression"]["junit"]["failures"] == 1
    assert "unbounded_legacy_source_read" in (OUT / "baseline-regression/stdout.log").read_text()
    assert result["runs"]["targeted-v1"]["junit"]["failures"] == 1
    assert "AttributeError" in (OUT / "targeted-v1/stdout.log").read_text()
    archives = data(OUT / "archives.json")
    for name, record in archives.items():
        assert sha((OUT / record["archive"]).read_bytes()) == record["sha256"]
        with tarfile.open(OUT / record["archive"], "r:gz") as tar:
            assert {m.name for m in tar.getmembers()} == set(record["members"])
            for member in tar.getmembers():
                raw = tar.extractfile(member).read()
                assert sha(raw) == record["members"][member.name]
                assert raw == (Path(record["root"]) / member.name).read_bytes()
        result["archives"][name] = {"sha256": record["sha256"], "files": len(record["members"])}
    for name in ("benchmark_v1", "benchmark_v2", "memory"):
        root = Path(archives[name]["root"])
        report = data(root / "report.json")
        mode = "memory" if name == "memory" else "time"
        repetitions = 1 if mode == "memory" else 3
        assert report["passed"] and report["mode"] == mode and report["repetitions"] == repetitions
        assert report["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-identity-repair-benchmark.py").read_bytes())
        for path, digest in report["source_files"].items():
            assert sha((ROOT / path).read_bytes()) == digest
        for path, digest in report["baseline_files"].items():
            assert sha((OUT / "baseline" / path).read_bytes()) == digest
        assert [(r["records"], r["payload_characters"]) for r in report["rows"]] == [(100, 1024), (1000, 1024), (5000, 1024), (500, 8192)]
        for row in report["rows"]:
            project = root / f'{row["records"]}-{row["payload_characters"]}'
            assert db_digest(project / "input.db") == row["input_database_sha256"]
            assert row["orders"] == [["old", "new"], ["new", "old"], ["old", "new"]][:repetitions]
            identities = set()
            for arm in row["arms"].values():
                assert len(arm) == repetitions
                for event in arm:
                    assert event["repaired"] == row["records"]
                    assert db_digest(Path(event["database"])) == event["database_sha256"]
                    identities.add(event["database_sha256"])
                    assert (event["current_peak_python_bytes"] is not None) == (mode == "memory")
            assert len(identities) == 1
            for arm, event in row.get("separate_writer_audit", {}).items():
                assert event["events"] == ["BEGIN IMMEDIATE", "COMMIT"] and event["hold_seconds"] > 0
                assert db_digest(Path(event["database"])) in identities
                assert len(event["stage"]) == (1 if arm == "new" else 0)
                for stage in event["stage"]:
                    assert stage["staged_rows"] == row["records"]
                    assert (stage["journal_mode"], stage["synchronous"], stage["locking_mode"], stage["cache_size"], stage["directory_mode"]) == ("delete", 2, "normal", -512, 0o700)
                    assert stage["bytes_before_main_apply"] > 0 and not Path(stage["directory"]).exists()
        result["benchmarks"][name] = sha((root / "report.json").read_bytes())
    root = Path(archives["product"]["root"])
    product = data(root / "report.json")
    assert product["passed"] and product["missing_before"] == 1500 and product["missing_after"] == 0
    assert product["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-identity-repair-product.py").read_bytes())
    for name, digest in product["source_files"].items():
        assert sha((ROOT / name).read_bytes()) == digest
    product_helper = module("product_helper", ROOT / "docs/evidence/checks/2026-09-15-ingress-product.py")
    assert product["before"] == product["after"] == product_helper.rows(root / "lake.db")
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
    with sqlite3.connect((root / "lake.db").as_uri() + "?mode=ro", uri=True) as conn:
        result["source_query_plan"] = [list(row) for row in conn.execute(
            "EXPLAIN QUERY PLAN SELECT idea_id, config FROM ideas WHERE status COLLATE NOCASE IN (?, ?, ?, ?) "
            "AND (config_hash IS NULL OR config_source_sha256 IS NULL) ORDER BY rowid", ("queued", "pending", "running", "completed"))]
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    result.update(passed=True, complete=not preliminary, workers=[{"ref": ref, "terminal_sha256": reservation["terminal_sha256"]}],
                  limits=["Separate-process mechanical audit, not another reviewer.",
                          "Python row collections bounded, not RSS, SQLite sorter memory, one YAML config, total I/O or disk.",
                          "One real CPU action and synthetic history; no real model, scientific speedup or production acceptance."])
    target = OUT / ("preflight.json" if preliminary else "verification.json")
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "complete": not preliminary, "suites": result["suites"],
                      "archive_files": sum(r["files"] for r in result["archives"].values()),
                      "verification_sha256": sha(target.read_bytes())}))


if __name__ == "__main__":
    main(preliminary="--preflight" in sys.argv)
