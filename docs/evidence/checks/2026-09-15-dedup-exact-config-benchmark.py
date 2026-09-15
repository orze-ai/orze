"""Alternate complete ingress over identical unchanged databases and sources.

Only the normal admission implementation changes between arms. Timing excludes
setup, independent parse/transaction counts, and independent memory tracing.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs/evidence/runs/2026-09-15-dedup-exact-config"
spec = importlib.util.spec_from_file_location("ingress_measure", REPO / "docs/evidence/checks/2026-09-15-ingress-benchmark.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
from orze.core import proposal_admission as current
from orze.core.integrity import hash_config


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    old = b.load("old_dedup", OUT / "baseline/proposal_admission.py")
    arms = {"old": old.admit_proposal, "new": current.admit_proposal}
    inputs = [REPO / "src/orze/core/proposal_admission.py", REPO / "src/orze/idea_lake.py",
              REPO / "src/orze/engine/idea_ingress.py", OUT / "baseline/proposal_admission.py",
              REPO / "docs/evidence/checks/2026-09-15-ingress-benchmark.py"]
    fingerprints = {str(p): b.sha(p.read_bytes()) for p in inputs}
    report = {"baseline": "080edd8199e5772353f1c7e03924ff7dc1e13b13", "sources": fingerprints,
              "script_sha256": b.sha(Path(__file__).read_bytes()), "repetitions": 7, "rows": [],
              "limits": ["Synthetic completed history; no worker evidence or scientific quality comparison.",
                         "128 retained cross-ID duplicates per invocation. All current transactions still execute.",
                         "First-valid semantic owner preserved even when YAML text differs; that arm is a no-benefit control.",
                         "Private local /tmp, non-exclusive host; Python traced allocations are not RSS.",
                         "Warning output equally suppressed; setup, counters and tracing outside normal timing."]}
    for history, mode in [(1, "exact"), (100, "exact"), (1000, "exact"), (5000, "exact"),
                          (1, "semantic"), (5000, "semantic")]:
        project = root / f"{history}-{mode}"
        project.mkdir()
        results = project / "results"
        results.mkdir()
        source = project / "ideas.md"
        text = "".join(f"## idea-copy-{i:03d}: Copy\n```yaml\nseed: 13\n```\n" for i in range(128))
        source.write_text(text)
        cfg = {"ideas_file": str(source), "results_dir": str(results),
               "idea_lake_db": str(project / "lake.db"), "_orze_dir": str(project / ".orze")}
        engine = b.Orze.__new__(b.Orze)
        engine.cfg, engine.results_dir, engine.active_roles = cfg, results, {}
        engine.lake = b.IdeaLake(cfg["idea_lake_db"])
        try:
            owner = "seed: 13\n" if mode == "exact" else "{seed: 13} # equivalent\n"
            for i in range(history):
                value = owner if i == 0 else f"seed: {1000 + i}\n"
                engine.lake.conn.execute(
                    "INSERT INTO ideas(idea_id,title,config,raw_markdown,status,config_hash,config_source_sha256) "
                    "VALUES (?,'History',?,'','completed',?,?)",
                    (f"idea-history-{i:05d}", value, hash_config({"seed": 13 if i == 0 else 1000 + i}),
                     hashlib.sha256(value.encode()).hexdigest()),
                )
            engine.lake.conn.commit()
            before = b.sha("\n".join(engine.lake.conn.iterdump()).encode())

            def call(entry):
                engine._idea_ingress_cursor = None
                with patch.object(current, "admit_proposal", entry):
                    raw, inserted = b.idea_ingress.ingest_ideas_source(engine, cfg)
                assert not inserted and len(raw) == 128 and source.read_text() == text
                return {"raw_sha256": b.sha(json.dumps(raw, sort_keys=True).encode()), "inserted": inserted}

            with patch.object(b.idea_ingress.logger, "warning", lambda *a, **k: None):
                measured = b.measure({name: lambda entry=entry: call(entry) for name, entry in arms.items()},
                                     7, memory=True)
                # Independent instrumentation checks what was removed without
                # attributing profiler overhead to the normal timing samples.
                counters = {}
                for name, entry in arms.items():
                    calls, statements = [], []
                    load = current.yaml.safe_load
                    def counted(value, *args, **kwargs):
                        calls.append(1)
                        return load(value, *args, **kwargs)
                    engine.lake.conn.set_trace_callback(statements.append)
                    try:
                        with patch.object(current.yaml, "safe_load", counted):
                            assert call(entry) == measured["value"]
                    finally:
                        engine.lake.conn.set_trace_callback(None)
                    counters[name] = {"yaml_loads": len(calls),
                                      "begin_immediate": sum(s == "BEGIN IMMEDIATE" for s in statements),
                                      "rollbacks": sum(s == "ROLLBACK" for s in statements)}
                assert counters["old"]["yaml_loads"] - counters["new"]["yaml_loads"] == (128 if mode == "exact" else 0)
                assert all(c["begin_immediate"] == c["rollbacks"] == 128 for c in counters.values())
            after = b.sha("\n".join(engine.lake.conn.iterdump()).encode())
            assert before == after
            report["rows"].append({"history": history, "mode": mode, **measured, "counters": counters,
                                   "database_sha256_before": before, "database_sha256_after": after,
                                   "source_sha256": b.sha(text.encode())})
            (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"history": history, "mode": mode, "ms": {
                name: arm["median_seconds"] * 1000 for name, arm in measured["arms"].items()},
                "counters": counters}), flush=True)
        finally:
            engine.lake.close()
    assert fingerprints == {str(p): b.sha(p.read_bytes()) for p in inputs}


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
