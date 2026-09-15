"""Separate-process mechanical verification; not another author/reviewer.

Verify frozen regression inputs, unchanged admission/GO budget implementation,
captured old source, alternating measurements, and actual worker settlements.
"""
import ast
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import tarfile
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[3]
PRO = REPO.parent / "pro"
OUT = REPO / "docs/evidence/runs/2026-09-15-policy-audit"
BASE = "cdfbe064849af73b89300d0778be514044c31087"
PRO_BASE = "bac1fdb"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def data(path):
    return json.loads(path.read_text())


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def unchanged(repo, base, folder, exceptions=()):
    files = git(repo, "ls-tree", "-rz", "--name-only", base, folder).decode().split("\0")
    checked = []
    for name in files:
        if name and name not in exceptions:
            assert (repo / name).read_bytes() == git(repo, "show", base + ":" + name), name
            checked.append(name)
    return len(checked)


def frozen(path):
    d = data(path / "run.json")
    assert d["frozen"] and d["exit_code"] == 0 and d["before"] == d["after"]
    for name, record in d["files"].items():
        raw = (path / name).read_bytes()
        assert sha(raw) == record["sha256"] and len(raw) == record["bytes"]
    for key, files in d["before"].items():
        root = Path(d["repositories"][key])
        for relative, digest in files.items():
            assert sha((root / relative).read_bytes()) == digest
    suite, = ET.parse(path / "junit.xml").getroot().findall("testsuite")
    counts = {k: int(suite.attrib[k]) for k in ("tests", "errors", "failures", "skipped")}
    assert counts["errors"] == counts["failures"] == 0
    counts["passed"] = counts["tests"] - counts["skipped"]
    assert str(counts["passed"]) + " passed" in (path / "stdout.log").read_text()
    return {**counts, "run_sha256": sha((path / "run.json").read_bytes())}


def main():
    path = "src/orze/engine/cpu_phase.py"
    original = git(REPO, "show", BASE + ":" + path)
    assert original == (OUT / "baseline/cpu_phase.py").read_bytes()
    old_tree, new_tree = ast.parse(original), ast.parse((REPO / path).read_text())
    def stable(tree):
        return [ast.dump(node) for node in tree.body
                if getattr(node, "name", None) not in {"require_admission", "_require_invocation", "iteration"}]
    assert stable(old_tree) == stable(new_tree)
    result = {"baseline": BASE, "changed_module_sha256": sha((REPO / path).read_bytes()),
              "unchanged_core_source_files": unchanged(REPO, BASE, "src", (path,)),
              "unchanged_core_test_files": unchanged(REPO, BASE, "tests"),
              "unchanged_pro_source_files": unchanged(PRO, PRO_BASE, "src"),
              "unchanged_pro_test_files": unchanged(PRO, PRO_BASE, "tests"),
              "suites": {"core": frozen(OUT / "core-full"),
                         "pro": frozen(PRO / "docs/evidence/runs/2026-09-15-policy-audit/pro-full"),
                         "optional": frozen(PRO / "docs/evidence/runs/2026-09-15-policy-audit/core-pro-optional")}}
    target, = ET.parse(OUT / "targeted.xml").getroot().findall("testsuite")
    assert int(target.attrib["tests"]) == 261
    assert int(target.attrib["errors"]) == int(target.attrib["failures"]) == 0
    result["targeted_passed"] = 261
    assert "8 failed, 3 passed" in (OUT / "baseline.log").read_text()
    result["benchmarks"] = {}
    for name in ("benchmark-v1.json", "benchmark-v2.json"):
        bench = data(OUT / name)
        for path, digest in bench["sources"].items():
            assert sha(Path(path).read_bytes()) == digest
        assert sha((REPO / "docs/evidence/checks/2026-09-15-policy-audit-benchmark.py").read_bytes()) == bench["script_sha256"]
        assert [r["settled_metadata_rows"] for r in bench["rows"]] == [100, 1000, 5000]
        for row in bench["rows"]:
            assert row["ledger_sha256_before"] == row["ledger_sha256_after"]
            assert row["arms"]["old"]["complete_audits"] == 2
            assert row["arms"]["new"]["complete_audits"] == 1
            for arm in row["arms"].values():
                assert len(arm["seconds"]) == bench["repetitions"] == 7
        result["benchmarks"][name] = sha((OUT / name).read_bytes())
    for version, record in data(OUT / "benchmark-archives.json").items():
        assert sha((OUT / record["archive"]).read_bytes()) == record["sha256"]
        with tarfile.open(OUT / record["archive"], "r:gz") as tar:
            assert {member.name for member in tar.getmembers()} == set(record["members"])
            for member in tar.getmembers():
                assert sha(tar.extractfile(member).read()) == record["members"][member.name]
        assert (Path(record["root"]) / "report.json").read_bytes() == (OUT / ("benchmark-" + version + ".json")).read_bytes()
    archive = data(OUT / "product-archive.json")
    root = Path(archive["root"])
    assert sha((OUT / archive["archive"]).read_bytes()) == archive["sha256"]
    with tarfile.open(OUT / archive["archive"], "r:gz") as tar:
        assert {member.name for member in tar.getmembers()} == set(archive["members"])
        for member in tar.getmembers():
            assert sha(tar.extractfile(member).read()) == archive["members"][member.name]
            assert sha((root / member.name).read_bytes()) == archive["members"][member.name]
    report = data(root / "paging-product-report.json")
    assert report["actual_native_workers"] == 2 and report["outcomes"] == [0]
    assert [v["decision"]["kind"] for v in report["trace"]] == ["ReadEvidence"] * 4 + ["Execute"]
    for event in report["trace"]:
        rows = event["database_before_callback"]["cpu_action_reservations"]
        charged = sum(int(json.loads(row["permit_json"])["reserved_nanoseconds"]) for row in rows)
        assert charged == 2_000_000_000
        assert event["budget"]["reserved_wall_seconds"] == charged / 1_000_000_000
        assert event["budget"]["remaining_wall_seconds"] == 10
        assert event["budget"]["free_slots"] == 1 and not event["budget"]["stopped"]
        assert event["budget"]["active_reservations"] == 0
    workers = []
    with sqlite3.connect((root / "lake.db").as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT * FROM execution_attempts"):
            terminal = json.loads(row["terminal_json"])
            if "process_tree" not in terminal:
                continue
            tree = terminal["process_tree"]
            assert row["state"] == "TERMINAL" and terminal["outcome"] == "completed"
            assert tree["event"] == "TREE_CLOSED" and tree["wait_proof"] == "ECHILD_WALL"
            ref = tree["binding"]["identity"]["attempt_ref"]
            assert all(row[key] == value for key, value in ref.items())
            reservation, = [r for r in conn.execute("SELECT * FROM cpu_action_reservations")
                             if r["ref_json"] and json.loads(r["ref_json"]) == ref]
            assert reservation["state"] == "SETTLED"
            assert reservation["terminal_sha256"] == sha(row["terminal_json"].encode())
            workers.append({"ref": ref, "terminal_sha256": reservation["terminal_sha256"]})
    assert len(workers) == 2
    result.update(passed=True, actual_native_workers=workers,
                  limits=["Separate mechanical verifier, not a second reviewer.",
                          "Pro suite uses a test license substitute; no provider or production acceptance.",
                          "Local metadata cost improvement, not end-to-end scientific speedup."])
    target = OUT / "verification.json"
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
