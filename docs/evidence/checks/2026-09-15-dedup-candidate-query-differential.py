"""Frozen old/new admission over synthetic mixed metadata and real SQLite.

Uses both public insert(if_absent=True) and the caller-owned entry. Fixed clocks
make complete database projections comparable; no synthetic row is execution
evidence. Transaction ownership and rollback are checked as actual operations.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

from orze.core import proposal_admission as current
from orze.core.integrity import hash_config
from orze.idea_lake import IdeaLake

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs/evidence/runs/2026-09-15-dedup-candidate-query"
BASE = "c17ce2e8573bd14d0153cb7e2f030914e57a36fd"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot(lake):
    return sha("\n".join(lake.conn.iterdump()).encode())


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    baseline = OUT / "baseline/proposal_admission.py"
    spec = importlib.util.spec_from_file_location("old_admission", baseline)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    normal_entries = {"old": old.admit_proposal, "new": current.admit_proposal}
    caller_entries = {"old": old.admit_proposal_in_tx, "new": current.admit_proposal_in_tx}
    paths = [baseline, REPO / "src/orze/core/proposal_admission.py", REPO / "src/orze/idea_lake.py"]
    sources = {str(p): sha(p.read_bytes()) for p in paths}
    report = {"baseline": BASE, "script_sha256": sha(Path(__file__).read_bytes()), "sources": sources,
              "cases": [], "limits": ["Seeded synthetic metadata; no execution or scientific evidence.",
                                         "Fixed clock only to compare database projections across independent files.",
                                         "Diagnostic differential oracle; unit tests separately assert required outcomes."]}
    match = hash_config({"seed": 13})
    configs = ["seed: 13\n", "{seed: 13}\n", "seed: 99\n", "seed: [\n", b"seed: 13\n", "[]\n",
               "seed: 13\n#" + "x" * 65536]
    fingerprints = [match, None, "wrong", b"wrong", match.upper()]
    counts = {}
    for index in range(64):
        project = root / f"case-{index:03d}"
        project.mkdir()
        prototype = IdeaLake(str(project / "baseline.db"))
        rng = random.Random(index + 915)
        try:
            for i in range(24):
                prototype.conn.execute(
                    "INSERT INTO ideas(idea_id,title,config,raw_markdown,status,kind,config_hash,config_source_sha256) "
                    "VALUES (?,'Imported',?,'',?,?,?,?)",
                    (f"idea-history-{i:03d}", rng.choice(configs),
                     rng.choice(["completed", "CoMpLeTeD", "queued", "PENDING", "running", "failed", "archived", None]),
                     rng.choice(["train", "native_cpu_action", "analysis"]), rng.choice(fingerprints),
                     rng.choice([None, "source", b"source"])),
                )
            for bit, name in enumerate(("idx_status_config_hash_nocase", "idx_missing_config_identity")):
                if index & (1 << bit):
                    prototype.conn.execute(f"DROP INDEX {name}")
            prototype.conn.commit()
            before = snapshot(prototype)
            case = {"seed": index + 915, "database_sha256_before": before, "outcomes": {}}
            for entry in ("normal", "caller"):
                values = {}
                for arm in ("old", "new"):
                    lake = IdeaLake(str(project / f"{entry}-{arm}.db"))
                    try:
                        prototype.conn.backup(lake.conn)
                        assert snapshot(lake) == before
                        lake._transition_time = lambda conn: "2026-09-15T00:00:00Z"
                        kwargs = {"kind": "native_cpu_action" if index % 2 else "train"}
                        if entry == "normal":
                            with patch.object(current, "admit_proposal", normal_entries[arm]):
                                result = lake.insert("idea-candidate", "Candidate", "seed: 13\n", "raw",
                                                     status="queued", if_absent=True, **kwargs)
                            assert not lake.conn.in_transaction
                            after = snapshot(lake)
                            if result["status"] != "inserted":
                                assert after == before
                            values[arm] = {"result": result, "database_sha256_after": after}
                        else:
                            prepared = lake.prepare_proposal("idea-candidate", "Candidate", "seed: 13\n", "raw", **kwargs)
                            lake.conn.execute("BEGIN IMMEDIATE")
                            lake.conn.execute("UPDATE id_sequence SET next_id=12345")
                            try:
                                result = caller_entries[arm](lake, prepared)
                            except (old.ProposalAdmissionError, current.ProposalAdmissionError) as exc:
                                result = {"exception_reason": str(exc)}
                            assert lake.conn.in_transaction
                            assert lake.conn.execute("SELECT next_id FROM id_sequence").fetchone()[0] == 12345
                            provisional = snapshot(lake)
                            lake.conn.rollback()
                            assert snapshot(lake) == before
                            values[arm] = {"result": result, "provisional_sha256": provisional,
                                           "rollback_sha256": before}
                    finally:
                        if lake.conn.in_transaction:
                            lake.conn.rollback()
                        lake.close()
                assert values["old"] == values["new"], (index, entry, values)
                case["outcomes"][entry] = values
                value = values["new"]["result"]
                key = entry + ":" + value.get("status", value.get("exception_reason", "unknown"))
                counts[key] = counts.get(key, 0) + 1
            assert snapshot(prototype) == before
            report["cases"].append(case)
        finally:
            prototype.close()
    assert sources == {str(p): sha(p.read_bytes()) for p in paths}
    report.update(passed=True, result_counts=counts)
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "cases": len(report["cases"]), "old_new_pairs": 128,
                      "actual_admission_calls": 256, "result_counts": counts}))


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
