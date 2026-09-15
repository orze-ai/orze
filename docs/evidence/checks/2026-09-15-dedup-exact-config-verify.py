"""Separate mechanical process: exact edit, old tests, frozen runs and CPU proof."""
import ast
import importlib.util
import json
from pathlib import Path
import sqlite3
import tarfile

REPO = Path(__file__).resolve().parents[3]
PRO = REPO.parent / "pro"
OUT = REPO / "docs/evidence/runs/2026-09-15-dedup-exact-config"
BASE = "080edd8199e5772353f1c7e03924ff7dc1e13b13"
PRO_BASE = "d25754a33884e1d640930d84f2c0674f777d7c68"
spec = importlib.util.spec_from_file_location("audit_common", REPO / "docs/evidence/checks/2026-09-15-policy-audit-verify.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
sha, data, git = common.sha, common.data, common.git


def main():
    changed = "src/orze/core/proposal_admission.py"
    old = git(REPO, "show", BASE + ":" + changed)
    new = (REPO / changed).read_bytes()
    assert old == (OUT / "baseline/proposal_admission.py").read_bytes()
    assert new == (OUT / "optimized-proposal-admission.py").read_bytes()
    assert (OUT / "baseline/idea_lake.py").read_bytes() == git(REPO, "show", BASE + ":src/orze/idea_lake.py")
    before, after = ast.parse(old), ast.parse(new)
    owner = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == "_dedup_owner")
    loop = next(n for n in owner.body if isinstance(n, ast.For))
    removed = loop.body.pop(1)
    expected = ast.parse('if row["config"] == prepared["config"]:\n    return row["idea_id"]\n').body[0]
    assert ast.dump(removed) == ast.dump(expected)
    assert ast.dump(before) == ast.dump(after), "only exact current text branch may change"
    result = {"baseline": BASE, "pro_baseline": PRO_BASE, "changed_module_sha256": sha(new),
              "unchanged_core_sources": common.unchanged(REPO, BASE, "src", (changed,)),
              "unchanged_core_tests": common.unchanged(REPO, BASE, "tests"),
              "unchanged_pro_sources": common.unchanged(PRO, PRO_BASE, "src"),
              "unchanged_pro_tests": common.unchanged(PRO, PRO_BASE, "tests"),
              "suites": {"core": common.frozen(OUT / "core-full"),
                         "targeted": common.frozen(OUT / "targeted"),
                         "pro": common.frozen(PRO / "docs/evidence/runs/2026-09-15-dedup-exact-config/pro-full"),
                         "optional": common.frozen(PRO / "docs/evidence/runs/2026-09-15-dedup-exact-config/core-pro-optional")}}
    assert "4 failed, 6 passed" in (OUT / "baseline-tests.log").read_text()
    assert "4 failed, 103 passed" in (OUT / "targeted-v1.log").read_text()
    assert "10 passed" in (OUT / "baseline-tests-v2.log").read_text()
    assert "NOT NULL constraint failed: ideas.raw_markdown" in (OUT / "baseline-tests.log").read_text()
    assert (OUT / "test-script-v1.py").is_file()
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
    for version in ("v1", "v2"):
        benchmark = data(OUT / f"benchmark-{version}.json")
        assert benchmark == data(Path(archives["benchmark_" + version]["root"]) / "report.json")
        assert benchmark["script_sha256"] == sha((REPO / "docs/evidence/checks/2026-09-15-dedup-exact-config-benchmark.py").read_bytes())
        for path, digest in benchmark["sources"].items():
            assert sha(Path(path).read_bytes()) == digest
        assert [(r["history"], r["mode"]) for r in benchmark["rows"]] == [
            (1, "exact"), (100, "exact"), (1000, "exact"), (5000, "exact"), (1, "semantic"), (5000, "semantic")]
        for row in benchmark["rows"]:
            assert row["database_sha256_before"] == row["database_sha256_after"]
            assert row["value"]["inserted"] == [] and len(row["orders"]) == 7
            assert all(len(a["seconds"]) == 7 for a in row["arms"].values())
            assert all(c["begin_immediate"] == c["rollbacks"] == 128 for c in row["counters"].values())
            assert row["counters"]["old"]["yaml_loads"] == 384
            assert row["counters"]["new"]["yaml_loads"] == (256 if row["mode"] == "exact" else 384)
            project = Path(archives["benchmark_" + version]["root"]) / f'{row["history"]}-{row["mode"]}'
            assert sha((project / "ideas.md").read_bytes()) == row["source_sha256"]
            with sqlite3.connect((project / "lake.db").as_uri() + "?mode=ro", uri=True) as conn:
                assert sha("\n".join(conn.iterdump()).encode()) == row["database_sha256_after"]
                assert conn.execute("SELECT COUNT(*) FROM ideas").fetchone()[0] == row["history"]
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
        for name, expected_rows in product["invocations"][1]["database"].items():
            assert [dict(r) for r in conn.execute("SELECT * FROM " + name)] == expected_rows
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
                          "All current transactions and SQL candidate scans remain; exact-text parsing only.",
                          "Synthetic history, real CPU worker, no scientific speedup or production provider acceptance."])
    assert not (OUT / "verification.json").exists()
    (OUT / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
