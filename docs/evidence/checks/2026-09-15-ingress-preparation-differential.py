"""Compare complete old/current ingress against identical imported histories.

Independent real SQLite/source files; fixed transition timestamps only make
database projections comparable. No historical row claims actual execution.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

from orze.core import proposal_admission
from orze.core.integrity import hash_config
from orze.engine import idea_ingress
from orze.engine.orchestrator import Orze
from orze.idea_lake import IdeaLake

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs/evidence/runs/2026-09-15-ingress-preparation"
BASE = "1b8ac7b5e7a398ceae95f58bbd47e8b3ae0ab895"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot(lake):
    return sha("\n".join(lake.conn.iterdump()).encode())


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    baseline = OUT / "baseline/idea_ingress.py"
    spec = importlib.util.spec_from_file_location("orze.engine._preparation_old", baseline)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    entries = {"old": old.ingest_ideas_source, "new": idea_ingress.ingest_ideas_source}
    paths = [baseline, REPO / "src/orze/engine/idea_ingress.py", REPO / "src/orze/idea_lake.py",
             REPO / "src/orze/core/proposal_admission.py", REPO / "src/orze/core/config_identity_repair.py"]
    sources = {str(p): sha(p.read_bytes()) for p in paths}
    report = {"baseline": BASE, "script_sha256": sha(Path(__file__).read_bytes()), "sources": sources,
              "cases": [], "limits": ["Seeded imported metadata and actual source/SQLite admission, no workers.",
                "Fixed transition timestamp solely for complete database comparison.",
                "Differential equivalence is complemented by independently asserted unit-test outcomes.",
                "Source admission and ACK compared for primary and additive sidecar inputs."]}
    match = hash_config({"seed": 13})
    configs = ["seed: 13\n", "{seed: 13}\n", "seed: 99\n", "seed: [\n", b"seed: 13\n", "[]\n",
               "seed: 13\n#" + "x" * 65536]
    fingerprints = [match, None, "wrong", b"wrong", match.upper()]
    counts = {}
    for index in range(64):
        project = root / f"case-{index:03d}"
        project.mkdir()
        prototype = IdeaLake(project / "baseline.db")
        rng = random.Random(index + 916)
        try:
            for i in range(24):
                prototype.conn.execute(
                    "INSERT INTO ideas(idea_id,title,config,raw_markdown,status,kind,config_hash,config_source_sha256) "
                    "VALUES (?,'Imported',?,'',?,?,?,?)",
                    (f"idea-history-{i:03d}", rng.choice(configs),
                     rng.choice(["completed", "CoMpLeTeD", "queued", "PENDING", "running", "failed", "archived", None]),
                     rng.choice(["train", "native_cpu_action", "analysis"]), rng.choice(fingerprints),
                     rng.choice([None, "source", b"source"])))
            for bit, name in enumerate(("idx_status_config_hash_nocase", "idx_missing_config_identity")):
                if index & (1 << bit):
                    prototype.conn.execute(f"DROP INDEX {name}")
            prototype.conn.commit()
            before = snapshot(prototype)
            case = {"seed": index + 916, "database_sha256_before": before,
                    "source_kind": "sidecar" if index & 4 else "primary", "outcomes": {}}
            kind = "native_cpu_action" if index % 2 else "train"
            authored = (f"## idea-candidate: Candidate\n- **Kind**: {kind}\n```yaml\nseed: 13\n```\n\n"
                        "## idea-fresh: Fresh\n```yaml\nseed: -17\n```\n\n")
            for arm, entry in entries.items():
                arm_root = project / arm
                arm_root.mkdir()
                results = arm_root / "results"
                results.mkdir()
                source = arm_root / "ideas.md"
                source.write_text("# Ideas\n" if index & 4 else authored)
                if index & 4:
                    (arm_root / "ideas.d").mkdir()
                    (arm_root / "ideas.d/input.md").write_text(authored)
                lake = IdeaLake(arm_root / "lake.db")
                try:
                    prototype.conn.backup(lake.conn)
                    assert snapshot(lake) == before
                    lake._transition_time = lambda conn: "2026-09-15T00:00:00Z"
                    engine = Orze.__new__(Orze)
                    cfg = {"ideas_file": str(source), "results_dir": str(results),
                           "idea_lake_db": str(arm_root / "lake.db"), "_orze_dir": str(arm_root / ".orze")}
                    engine.cfg, engine.results_dir, engine.active_roles, engine.lake = cfg, results, {}, lake
                    admissions = []
                    original = proposal_admission.admit_proposal

                    def observed(*args, **kwargs):
                        value = original(*args, **kwargs)
                        admissions.append(value)
                        return value

                    with patch.object(proposal_admission, "admit_proposal", observed):
                        raw, inserted = entry(engine, cfg)
                    assert not lake.conn.in_transaction and len(admissions) == 2
                    if index & 4:
                        assert (arm_root / "ideas.d/input.md").read_text() == authored
                    value = {"raw_sha256": sha(json.dumps(raw, sort_keys=True).encode()),
                             "inserted": inserted, "admissions": admissions,
                             "database_sha256_after": snapshot(lake),
                             "source_sha256_after": sha(source.read_bytes())}
                    case["outcomes"][arm] = value
                finally:
                    lake.close()
            assert case["outcomes"]["old"] == case["outcomes"]["new"], (index, case)
            for outcome in case["outcomes"]["new"]["admissions"]:
                key = outcome["status"] + ":" + outcome["reason"]
                counts[key] = counts.get(key, 0) + 1
            assert snapshot(prototype) == before
            report["cases"].append(case)
        finally:
            prototype.close()
    assert sources == {str(p): sha(p.read_bytes()) for p in paths}
    report.update(passed=True, result_counts=counts)
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "complete_ingress_pairs": 64,
                      "actual_admission_calls": 256, "result_counts_per_arm": counts}))


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
