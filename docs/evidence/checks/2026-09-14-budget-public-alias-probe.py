"""Bounded public-snapshot differential; metadata only, never execution proof.

Run with Core src on PYTHONPATH and one fresh, empty scratch directory argument.
Current initialize creates the ordinary schema. No product function, CHECK,
index, clock, process, or ownership evidence is replaced by this probe.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

if not __debug__:
    raise SystemExit("public alias probe requires assertions enabled")

from orze.core import cpu_action_budget as current
from orze.idea_lake import IdeaLake

REPO = Path(__file__).resolve().parents[3]
FIXED = "df446ab35e745daf02cc7c4a19cadffd57518103"
BASE_COMMIT = "e606ba6688b4e1972e26a08617821a427668dd24"
BASE = REPO / "docs/evidence/runs/2026-09-14-cost-equivalence/baseline/src/orze/core/cpu_action_budget.py"
BASE_SHA = "71bc89977cb582aa86a218003398431bb975ce11410f1df0d43b385ae4667169"
FILES = ["src/orze/core/cpu_action_budget.py", "src/orze/core/execution_attempts.py",
         "src/orze/idea_lake.py", "src/orze/engine/idea_ingress.py",
         "tests/test_cpu_budget_scan.py", "tests/test_cpu_budget_scan_review.py",
         "tests/test_idea_ingress_cost.py", "tests/test_idea_ingress_contract.py"]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def fingerprints():
    return {name: sha((REPO / name).read_bytes()) for name in FILES}


def outcome(call):
    try:
        return {"accepted": True, "value": call()}
    except Exception as exc:
        return {"accepted": False, "exception": type(exc).__name__, "reason": str(exc)}


def row(scope, label, seconds):
    task = "metadata-row-" + label
    identity = sha(task.encode())[:48]
    permit = {"schema": 1, "budget_scope": scope, "reservation_id": identity,
              "task_id": task, "slot": 0, "wall_limit_seconds": seconds,
              "reserved_nanoseconds": str(seconds * 1_000_000_000)}
    ref = {"task_id": task, "phase": "action", "attempt_id": "metadata-attempt-" + label,
           "generation": 1}
    return [identity, scope["results_dir"], task, 0, canonical(permit), canonical(ref),
            "SETTLED", sha(("metadata-terminal-marker-" + label).encode())], permit


def rows(conn):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM main.cpu_action_reservations ORDER BY reservation_id")]


def main(scratch):
    scratch = scratch.resolve()
    assert scratch.is_dir() and not any(scratch.iterdir()), "fresh empty scratch required"
    before = fingerprints()
    for name in FILES:
        assert before[name] == sha(subprocess.check_output(
            ["git", "show", FIXED + ":" + name], cwd=REPO)), name
    baseline = BASE.read_bytes()
    assert sha(baseline) == BASE_SHA
    assert baseline == subprocess.check_output(
        ["git", "show", BASE_COMMIT + ":src/orze/core/cpu_action_budget.py"], cwd=REPO)
    assert Path(current.__file__).resolve() == REPO / "src/orze/core/cpu_action_budget.py"
    spec = importlib.util.spec_from_file_location("orze.core._public_alias_baseline", BASE)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    results = scratch / "results"
    results.mkdir()
    lake = IdeaLake(scratch / "lake.db")
    report = {"schema": 1, "classification": "public snapshot metadata-only SQL-ID alias differential",
              "baseline_commit": BASE_COMMIT, "baseline_sha256": BASE_SHA,
              "fixed_current_commit": FIXED, "source_test_before": before,
              "probe_sha256": sha(Path(__file__).read_bytes()),
              "limits": ["Two SETTLED rows are deliberately synthetic metadata, not actual workers, attempts, settlement or TREE proof.",
                         "This tests actual public snapshot including schema/route checks; it does not demonstrate a public writer can generate the corrupted alias.",
                         "No PRAGMA ignore_check_constraints write, index removal, product-function replacement, provider, GPU or native worker.",
                         "The complete old budget module uses the otherwise current shared dependencies; only that whole module differs."]}
    try:
        scope = current.initialize(lake, results, {
            "version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 10})
        a, permit_a = row(scope, "a", 2)
        b, permit_b = row(scope, "b", 3)
        lake.conn.executemany("INSERT INTO main.cpu_action_reservations VALUES (?,?,?,?,?,?,?,?)", [a, b])
        lake.conn.commit()
        assert lake.conn.execute("PRAGMA ignore_check_constraints").fetchone()[0] == 0
        # Real enforcement control, rolled back inside this scratch database.
        rejected = None
        lake.conn.execute("SAVEPOINT check_enforcement")
        try:
            lake.conn.execute("UPDATE main.cpu_action_reservations SET state='INVALID' WHERE reservation_id=?",
                              (permit_a["reservation_id"],))
        except sqlite3.IntegrityError as exc:
            rejected = {"exception": type(exc).__name__, "reason": str(exc)}
        finally:
            lake.conn.execute("ROLLBACK TO check_enforcement")
            lake.conn.execute("RELEASE check_enforcement")
        assert rejected is not None and not lake.conn.in_transaction
        report.update(scope=scope, check_enforcement=rejected,
                      ignore_check_constraints=lake.conn.execute("PRAGMA ignore_check_constraints").fetchone()[0],
                      sqlite_version=sqlite3.sqlite_version,
                      schema_rows=[dict(r) for r in lake.conn.execute(
                          "SELECT type,name,tbl_name,sql FROM main.sqlite_master "
                          "WHERE name LIKE 'cpu_action_%' ORDER BY type,name")],
                      normal_rows=rows(lake.conn))
        normal_dump = "\n".join(lake.conn.iterdump())
        report["normal"] = {"old": outcome(lambda: old.snapshot(lake, scope)),
                            "new": outcome(lambda: current.snapshot(lake, scope))}
        report["normal_reads_unchanged"] = normal_dump == "\n".join(lake.conn.iterdump())
        # Only A.permit_json aliases existing B. A's own SQL identity, task,
        # unique full Ref, terminal marker and legal SETTLED state stay intact.
        lake.conn.execute("UPDATE main.cpu_action_reservations SET permit_json=? WHERE reservation_id=?",
                          (canonical(permit_b), permit_a["reservation_id"]))
        lake.conn.commit()
        aliased = rows(lake.conn)
        expected = {r["reservation_id"]: dict(r) for r in report["normal_rows"]}
        expected[permit_a["reservation_id"]]["permit_json"] = canonical(permit_b)
        assert aliased == [expected[k] for k in sorted(expected)]
        before_reads = "\n".join(lake.conn.iterdump())
        report["alias_rows"] = aliased
        report["alias"] = {"old": outcome(lambda: old.snapshot(lake, scope)),
                           "new": outcome(lambda: current.snapshot(lake, scope))}
        report["alias_reads_unchanged"] = before_reads == "\n".join(lake.conn.iterdump())
        report["integrity_check"] = [r[0] for r in lake.conn.execute("PRAGMA integrity_check")]
        report["foreign_key_check"] = [list(r) for r in lake.conn.execute("PRAGMA foreign_key_check")]
        report["schema_unchanged"] = report["schema_rows"] == [dict(r) for r in lake.conn.execute(
            "SELECT type,name,tbl_name,sql FROM main.sqlite_master WHERE name LIKE 'cpu_action_%' ORDER BY type,name")]
    finally:
        lake.close()
    report["source_test_after"] = fingerprints()
    report["source_tests_exact"] = before == report["source_test_after"]
    report["baseline_still_exact"] = sha(BASE.read_bytes()) == BASE_SHA
    report["observed_old_accepts_current_refuses"] = (
        report["alias"]["old"]["accepted"] and not report["alias"]["new"]["accepted"])
    target = scratch / "report.json"
    raw = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    target.write_text(raw)
    print(raw, end="", flush=True)
    print("PUBLIC_ALIAS_REPORT=" + canonical({"path": str(target), "sha256": sha(target.read_bytes())}), flush=True)
    assert report["source_tests_exact"] and report["baseline_still_exact"]
    assert report["schema_unchanged"] and report["normal_reads_unchanged"] and report["alias_reads_unchanged"]
    assert report["integrity_check"] == ["ok"] and report["foreign_key_check"] == []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scratch", type=Path)
    main(parser.parse_args().scratch)
