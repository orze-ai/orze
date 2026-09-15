"""Separate-process mechanical verification of ingress evidence and retained failures."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tarfile
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[3]
PRO = REPO.parent / "pro"
OUT = REPO / "docs/evidence/runs/2026-09-15-ingress-bounds"
BASE = "ad7ed16632897ab3d4ad62fb717fec53d6795f34"
PRO_BASE = "520971ecf57370a8c3d8d4017da74dca99b88f55"
spec = importlib.util.spec_from_file_location("audit_common", REPO / "docs/evidence/checks/2026-09-15-policy-audit-verify.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
sha, data, git = common.sha, common.data, common.git


def migrated_test():
    path = "tests/test_idea_ingress_cost.py"
    original = git(REPO, "show", BASE + ":" + path).decode()
    assert original.encode() == (OUT / "baseline/test_idea_ingress_cost.py").read_bytes()
    assert original.count('assert loads == ["actual_cache_loader"]') == 3
    expected = original.replace('assert loads == ["actual_cache_loader"]', 'assert loads == []')
    for old, new in (
        ("real_load = instance._load_config_hashes", "real_lookup = instance.lake.find_admitted_config_hashes"),
        ("def load_and_move_lock():", "def lookup_and_move_lock(identities):"),
        ("result = real_load()", "result = real_lookup(identities)"),
        ('monkeypatch.setattr(instance, "_load_config_hashes", load_and_move_lock)',
         'monkeypatch.setattr(instance.lake, "find_admitted_config_hashes", lookup_and_move_lock)'),
    ):
        assert expected.count(old) == 1
        expected = expected.replace(old, new)
    assert (REPO / path).read_text() == expected
    return sha(expected.encode())


def main():
    changed = ("src/orze/engine/idea_ingress.py", "src/orze/core/ideas.py")
    for path in changed:
        old = git(REPO, "show", BASE + ":" + path)
        assert old == (OUT / "baseline" / Path(path).name).read_bytes()
        before, after = ast.parse(old), ast.parse((REPO / path).read_text())
        ignored = ({"_overlay_sidecar_ideas", "_iter_sidecar_ideas"} if path.endswith("/ideas.py") else
                   {"ingest_ideas_source", "_read_sidecar", "_batch"})
        stable = lambda tree: {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name not in ignored}
        assert stable(before) == stable(after)
    result = {"baseline": BASE, "source_hashes": {p: sha((REPO / p).read_bytes()) for p in changed},
              "unchanged_core_sources": common.unchanged(REPO, BASE, "src", changed),
              "unchanged_core_tests": common.unchanged(REPO, BASE, "tests", ("tests/test_idea_ingress_cost.py",)),
              "migrated_cost_test_sha256": migrated_test(),
              "unchanged_pro_sources": common.unchanged(PRO, PRO_BASE, "src"),
              "unchanged_pro_tests": common.unchanged(PRO, PRO_BASE, "tests"),
              "suites": {"core": common.frozen(OUT / "core-full"),
                         "pro": common.frozen(PRO / "docs/evidence/runs/2026-09-15-ingress-bounds/pro-full"),
                         "optional": common.frozen(PRO / "docs/evidence/runs/2026-09-15-ingress-bounds/core-pro-optional")}}
    suite, = ET.parse(OUT / "targeted.xml").getroot().findall("testsuite")
    assert int(suite.attrib["tests"]) == 132 and int(suite.attrib["failures"]) == int(suite.attrib["errors"]) == 0
    assert "8 failed, 1 passed" in (OUT / "baseline.log").read_text()
    assert "5 failed, 62 passed" in (OUT / "first-targeted.log").read_text()
    result["targeted_passed"] = 132
    result["benchmarks"] = {}
    for version in ("v1", "v2"):
        bench = data(OUT / f"benchmark-{version}.json")
        for path, digest in bench["sources"].items():
            assert sha(Path(path).read_bytes()) == digest
        assert bench["script_sha256"] == sha((REPO / "docs/evidence/checks/2026-09-15-ingress-benchmark.py").read_bytes())
        assert [r["sidecar_records"] for r in bench["rows"]] == [100, 1000, 5000]
        for row in bench["rows"]:
            size = row["sidecar_records"]
            assert row["database_sha256_before"] == row["database_sha256_after"]
            assert row["first_batch"]["value"]["records"] == min(size, 128)
            assert row["full_cycle"]["value"]["records"] == size
            assert row["full_cycle"]["value"]["batches"] == (size + 127) // 128
            for mode in ("first_batch", "full_cycle"):
                for arm in row[mode]["arms"].values():
                    assert len(arm["seconds"]) == bench["repetitions"] == 5
        result["benchmarks"][version] = sha((OUT / f"benchmark-{version}.json").read_bytes())
    archives = data(OUT / "archives.json")
    for record in archives.values():
        assert sha((OUT / record["archive"]).read_bytes()) == record["sha256"]
        with tarfile.open(OUT / record["archive"], "r:gz") as tar:
            assert {m.name for m in tar.getmembers()} == set(record["members"])
            for member in tar.getmembers():
                assert sha(tar.extractfile(member).read()) == record["members"][member.name]
                assert sha((Path(record["root"]) / member.name).read_bytes()) == record["members"][member.name]
    for version in ("v1", "v2"):
        assert data(Path(archives["benchmark_" + version]["root"]) / "report.json") == data(OUT / f"benchmark-{version}.json")
    product_root = Path(archives["product"]["root"])
    duplicate_root = Path(archives["duplicate_control"]["root"])
    control = data(OUT / "duplicate-control.json")
    assert control == data(duplicate_root / "report.json")
    assert control["database_sha256_before"] == control["database_sha256_after"]
    assert control["value"]["inserted"] == []
    assert len(control["orders"]) == 7
    assert all(len(arm["seconds"]) == 7 for arm in control["arms"].values())
    assert control["source_sha256"] == sha((duplicate_root / "ideas.md").read_bytes())
    with sqlite3.connect((duplicate_root / "lake.db").as_uri() + "?mode=ro", uri=True) as conn:
        assert sha("\n".join(conn.iterdump()).encode()) == control["database_sha256_after"]
        assert conn.execute("SELECT idea_id, status FROM ideas").fetchall() == [("idea-winner", "completed")]
    result["duplicate_control_sha256"] = sha((OUT / "duplicate-control.json").read_bytes())
    product = data(product_root / "report.json")
    assert product["passed"] and len(product["invocations"]) == 2
    assert all(i["exit_code"] == 0 for i in product["invocations"])
    assert product["invocations"][0]["database"] == product["invocations"][1]["database"]
    assert sha((product_root / "ideas.d/actions.md").read_bytes()) == product["source_sha256"]
    assert (product_root / "ideas.md").read_text() == "# Ideas\n"
    for path, digest in product["source_files"].items():
        assert sha((REPO / path).read_bytes()) == digest
    with sqlite3.connect((product_root / "lake.db").as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for name, expected in product["invocations"][1]["database"].items():
            assert [dict(r) for r in conn.execute("SELECT * FROM " + name)] == expected
        artifact, = conn.execute("SELECT * FROM research_artifacts").fetchall()
        record = json.loads(artifact["record_json"])
        assert data(Path(record["path"])) == [1, 3, 5]
        row, = conn.execute("SELECT * FROM execution_attempts").fetchall()
        reservation, = conn.execute("SELECT * FROM cpu_action_reservations").fetchall()
        terminal = json.loads(row["terminal_json"])
        assert row["state"] == "TERMINAL" and terminal["outcome"] == "completed"
        tree = terminal["process_tree"]
        assert tree["event"] == "TREE_CLOSED" and tree["wait_proof"] == "ECHILD_WALL"
        ref = tree["binding"]["identity"]["attempt_ref"]
        assert json.loads(reservation["ref_json"]) == ref and all(row[k] == v for k, v in ref.items())
        assert reservation["state"] == "SETTLED" and reservation["terminal_sha256"] == sha(row["terminal_json"].encode())
    result.update(passed=True, actual_worker={"ref": ref, "terminal_sha256": reservation["terminal_sha256"]},
                  limits=["Separate mechanical process, not a second reviewer.",
                          "Pro test license substitute; no production provider or deployment acceptance.",
                          "First-batch latency and full traversal cost are distinct; prefix work remains."])
    target = OUT / "verification.json"
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
