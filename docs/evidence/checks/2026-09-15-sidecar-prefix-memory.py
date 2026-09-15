"""Separate full-traversal allocation measurement, including retained prefix IDs.

One traced execution per arm/size, not a latency distribution. All historical
rows are synthetic metadata; real ingress/admission still run unchanged.
"""
import gc
import importlib.util
import json
from pathlib import Path
import sys
import tracemalloc
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "docs/evidence/checks/2026-09-15-sidecar-prefix-benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def main(root):
    from orze.engine import idea_ingress
    root.mkdir(parents=True, exist_ok=False)
    baseline = ROOT / "docs/evidence/runs/2026-09-15-sidecar-prefix/baseline"
    original = bench.load("memory_original", baseline / "idea_ingress.py")
    original_ideas = bench.load("memory_original_ideas", baseline / "ideas.py")
    original._iter_sidecar_ideas = original_ideas._iter_sidecar_ideas
    sources = bench.fingerprint()
    report = {"script_sha256": bench.sha(Path(__file__).read_bytes()),
              "benchmark_script_sha256": bench.sha(Path(bench.__file__).read_bytes()),
              "source_files": sources, "rows": [], "repetitions": 1,
              "limits": ["Traced Python allocations, not process RSS or uninstrumented latency.",
                         "Full real ingress over synthetic metadata, including controller-retained hints."]}
    for size in (100, 1000, 5000):
        project = root / str(size)
        side = project / "ideas.d"
        side.mkdir(parents=True)
        results = project / "results"
        results.mkdir()
        source = project / "ideas.md"
        source.write_text("# Ideas\n")
        cfg = {"ideas_file": str(source), "results_dir": str(results),
               "idea_lake_db": str(project / "lake.db"), "_orze_dir": str(project / ".orze")}
        engine = bench.helper.Orze.__new__(bench.helper.Orze)
        engine.cfg, engine.results_dir, engine.active_roles = cfg, results, {}
        engine.lake = bench.helper.IdeaLake(cfg["idea_lake_db"])
        try:
            for i in range(size):
                key = f"idea-side-{i:05d}"
                engine.lake.conn.execute("INSERT INTO ideas(idea_id,title,config,raw_markdown,status) VALUES (?,?,?,'','completed')",
                    (key, "Stored metadata", f"seed: {i}\n"))
                (side / f"{i:05d}.md").write_text(f"## {key}: Conflicting authored title\n```yaml\nseed: {i}\n```\n" + "note " * 40)
            engine.lake.conn.commit()
            before = bench.sha("\n".join(engine.lake.conn.iterdump()).encode())
            row = {"records": size, "database_sha256": before, "arms": {}}
            with patch.object(idea_ingress.logger, "warning", lambda *a, **k: None):
                for name, module in (("old", original), ("new", idea_ingress)):
                    engine._idea_ingress_cursor = None
                    engine._idea_sidecar_prefix = None
                    gc.collect()
                    tracemalloc.start()
                    count = batches = 0
                    while True:
                        raw, inserted = module.ingest_ideas_source(engine, cfg)
                        assert not inserted
                        count += len(raw)
                        batches += 1
                        if engine._idea_ingress_cursor[2] == 0:
                            break
                        assert batches <= (size + 127) // 128 + 1
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    assert count == size
                    prefix = engine._idea_sidecar_prefix
                    row["arms"][name] = {"records": count, "batches": batches, "python_current_bytes": current,
                        "python_peak_bytes": peak, "retained_prefix_files": len(prefix.files) if prefix else 0,
                        "retained_prefix_ids": prefix.id_count if prefix else 0,
                        "retained_prefix_id_bytes": prefix.id_bytes if prefix else 0}
            assert before == bench.sha("\n".join(engine.lake.conn.iterdump()).encode())
            report["rows"].append(row)
            (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps(row), flush=True)
        finally:
            engine.lake.close()
    assert sources == bench.fingerprint()
    report["passed"] = True
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
