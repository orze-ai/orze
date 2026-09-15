"""Separate-process mechanical audit of source hints, costs and native execution."""
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
OUT = ROOT / "docs/evidence/runs/2026-09-15-sidecar-prefix"
POUT = PRO / "docs/evidence/runs/2026-09-15-sidecar-prefix"
BASE = "24a7eea460fc528ac573db661afdd405c2b39b64"
PBASE = "c280136eee31b824dd42a7131ba2606f670bd814"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


common = module("audit_common", ROOT / "docs/evidence/checks/2026-09-15-policy-audit-verify.py")
sha, data, git = common.sha, common.data, common.git


def matching_source(name, digest):
    paths = [ROOT / name, OUT / "first-source" / name,
             OUT / "baseline" / Path(name).name,
             OUT / "hardening" / Path(name).name]
    return any(p.is_file() and sha(p.read_bytes()) == digest for p in paths)


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
    changed = {"src/orze/core/ideas.py", "src/orze/engine/idea_ingress.py"}
    for path in changed:
        old = git(ROOT, "show", BASE + ":" + path)
        assert old == (OUT / "baseline" / Path(path).name).read_bytes()
        ignored = ({"_iter_sidecar_ideas", "_iter_sidecar_text"} if path.endswith("/ideas.py")
                   else {"_batch", "ingest_ideas_source"})
        def stable(raw):
            return {n.name: ast.dump(n) for n in ast.parse(raw).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name not in ignored}
        assert stable(old) == stable((ROOT / path).read_bytes())
    old = ast.parse((OUT / "baseline/ideas.py").read_bytes())
    new = ast.parse((ROOT / "src/orze/core/ideas.py").read_bytes())
    original, = [n for n in old.body if getattr(n, "name", None) == "_iter_sidecar_ideas"]
    extracted, = [n for n in new.body if getattr(n, "name", None) == "_iter_sidecar_text"]
    # Moving the parser into a helper preserves its original statements.
    assert ast.dump(original.body[-2]) == ast.dump(extracted.body[1])
    assert [ast.dump(n) for n in original.body[-1].body[-2:]] == [ast.dump(n) for n in extracted.body[2:]]
    return {"unchanged_core_sources": common.unchanged(ROOT, BASE, "src", changed),
            "unchanged_core_tests": common.unchanged(ROOT, BASE, "tests"),
            "unchanged_pro_sources": common.unchanged(PRO, PBASE, "src"),
            "unchanged_pro_tests": common.unchanged(PRO, PBASE, "tests")}


def main(preliminary=False):
    result = {"baseline_core": BASE, "baseline_pro": PBASE, "audit_script_sha256": sha(Path(__file__).read_bytes()),
              **source_checks(), "suites": {}, "runs": {}, "archives": {}}
    for name, path, passed in (("core", OUT / "core-full", 4923), ("pro", POUT / "pro-full", 1152),
                               ("paired", POUT / "paired", 31), ("targeted", OUT / "targeted-v2", 77)):
        if preliminary and not (path / "run.json").exists():
            result["suites"][name] = {"pending": True, "before_sha256": sha((path / "before.json").read_bytes())}
        else:
            value = common.frozen(path)
            assert value["passed"] == passed
            result["suites"][name] = value
    for name, code in (("baseline-tests", 0), ("compatibility", 0), ("targeted", 0),
                       ("baseline-regression", 1), ("hardening", 1), ("product", 0),
                       ("benchmark-v1", 0), ("benchmark-v2", 0), ("benchmark-v3", 0),
                       ("one-file-v1", 0), ("one-file-v2", 0), ("one-file-v3", 0), ("memory", 0)):
        result["runs"][name] = historical_run(OUT / name, code)
    assert result["runs"]["baseline-regression"]["junit"]["failures"] == 1
    assert "257 <= 129" in (OUT / "baseline-regression/stdout.log").read_text()
    assert result["runs"]["hardening"]["junit"]["failures"] == 2
    assert "2 failed, 16 deselected" in (OUT / "hardening/stdout.log").read_text()
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
    result["benchmarks"] = {}
    for prefix, layout, sizes in (("benchmark", "many", [100, 1000, 5000]), ("one_file", "one", [1000])):
        for version in ("v1", "v2", "v3"):
            root = Path(archives[prefix + "_" + version]["root"])
            bench = data(root / "report.json")
            assert bench["passed"] and bench["repetitions"] == 3 and bench["layout"] == layout
            assert bench["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-sidecar-prefix-benchmark.py").read_bytes())
            for name, digest in bench["source_files"].items():
                assert matching_source(name, digest)
            for name, digest in bench["baseline_files"].items():
                assert sha((OUT / "baseline" / name).read_bytes()) == digest
            assert [r["records"] for r in bench["rows"]] == sizes
            for row in bench["rows"]:
                count = row["records"]
                for path, digest in row["source_files"].items():
                    assert sha(Path(path).read_bytes()) == digest
                with sqlite3.connect((root / str(count) / "lake.db").as_uri() + "?mode=ro", uri=True) as conn:
                    assert sha("\n".join(conn.iterdump()).encode()) == row["database_sha256"]
                assert row["first_batch"]["value"]["records"] == min(count, 128)
                assert row["full_cycle"]["value"]["records"] == count
                assert row["full_cycle"]["value"]["batches"] == (count + 127) // 128
                for field in ("first_batch", "full_cycle"):
                    assert row[field]["orders"] == [["old", "new"], ["new", "old"], ["old", "new"]]
                    assert all(len(a["seconds"]) == len(a["io_deltas"]) == 3 for a in row[field]["arms"].values())
                reads = row["separate_io_audit"]
                if layout == "many":
                    assert reads["new"]["sidecar_payload_reads"] == count + (count - 1) // 128
                    assert reads["old"]["sidecar_payload_reads"] >= reads["new"]["sidecar_payload_reads"]
                else:
                    assert reads["old"] == reads["new"] and reads["old"]["sidecar_payload_reads"] == 8
            result["benchmarks"][prefix + "_" + version] = sha((root / "report.json").read_bytes())
    memory_root = Path(archives["memory"]["root"])
    memory = data(memory_root / "report.json")
    assert memory["passed"] and memory["repetitions"] == 1
    assert memory["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-sidecar-prefix-memory.py").read_bytes())
    for name, digest in memory["source_files"].items():
        assert sha((ROOT / name).read_bytes()) == digest
    for row in memory["rows"]:
        assert row["arms"]["old"]["records"] == row["arms"]["new"]["records"] == row["records"]
        assert row["arms"]["new"]["retained_prefix_files"] == row["records"] <= 8192
        assert row["arms"]["new"]["retained_prefix_ids"] == row["records"] <= 32768
        assert row["arms"]["new"]["retained_prefix_id_bytes"] <= 1024 * 1024
    result["memory_sha256"] = sha((memory_root / "report.json").read_bytes())
    root = Path(archives["product"]["root"])
    product = data(root / "report.json")
    assert product["passed"] and len(product["variants"]) == 2
    assert product["script_sha256"] == sha((ROOT / "docs/evidence/checks/2026-09-15-sidecar-prefix-product.py").read_bytes())
    for name, digest in product["source_files"].items():
        assert sha((ROOT / name).read_bytes()) == digest
    product_helper = module("product_helper", ROOT / "docs/evidence/checks/2026-09-15-ingress-product.py")
    result["workers"] = []
    for variant in product["variants"]:
        project = Path(variant["project"])
        assert len(variant["invocations"]) == 3 and all(r["exit_code"] == 0 for r in variant["invocations"])
        assert [r["inserted"] for r in variant["ingress"]] == [[], [], ["idea-tail"]]
        assert [len(r["ids"]) for r in variant["ingress"]] == [128, 128, 45]
        assert variant["ingress"][-1]["offset"] == 0
        assert variant["before"] == variant["after"] == product_helper.rows(project / "lake.db")
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
        result["workers"].append({"ref": ref, "terminal_sha256": reservation["terminal_sha256"]})
    result.update(passed=True, complete=not preliminary, limits=["Separate-process mechanical audit, not a second reviewer.",
        "Synthetic historical metadata and actual CPU tail actions; no scientific speedup or production acceptance.",
        "Cached prefix identities are bounded; current contents still read and admitted normally.",
        "Single-file prefix parsing, whole-directory enumeration, full traversal ID sets and legacy hash work remain."])
    target = OUT / ("preflight.json" if preliminary else "verification.json")
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "complete": not preliminary, "suites": result["suites"],
                      "archive_files": sum(r["files"] for r in result["archives"].values()),
                      "verification_sha256": sha(target.read_bytes())}))


if __name__ == "__main__":
    main(preliminary="--preflight" in sys.argv)
