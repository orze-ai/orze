"""Alternate real ingress over unchanged, conflicting sidecars and JSON cache.

Synthetic historical rows, no worker or scientific evidence. Every batch uses
real source locking, fresh source reads and immutable SQLite admission.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import statistics
import sys
import time
import tracemalloc
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
from orze.core import integrity
from orze.engine import idea_ingress
from orze.idea_lake import IdeaLake
with patch.dict(sys.modules, {"orze_pro": None}):
    from orze.engine.orchestrator import Orze


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def usage():
    io = {k: int(v) for k, v in (line.split(":") for line in Path("/proc/self/io").read_text().splitlines())}
    r = resource.getrusage(resource.RUSAGE_SELF)
    return {**io, "minor_faults": r.ru_minflt, "major_faults": r.ru_majflt}


def measure(calls, repetitions, *, memory=False):
    expected = None
    arms = {name: {"seconds": [], "io_deltas": []} for name in calls}
    for call in calls.values():
        value = call()
        if expected is None:
            expected = value
        assert value == expected
    orders = []
    for i in range(repetitions):
        order = list(calls) if i % 2 == 0 else list(reversed(calls))
        orders.append(order)
        for name in order:
            before = usage()
            start = time.perf_counter()
            value = calls[name]()
            elapsed = time.perf_counter() - start
            after = usage()
            assert value == expected
            arms[name]["seconds"].append(elapsed)
            arms[name]["io_deltas"].append({k: v - before[k] for k, v in after.items()})
    for name, call in calls.items():
        arm = arms[name]
        arm["median_seconds"] = statistics.median(arm["seconds"])
        if memory:
            tracemalloc.start()
            assert call() == expected
            arm["traced_current_bytes"], arm["traced_peak_bytes"] = tracemalloc.get_traced_memory()
            tracemalloc.stop()
    return {"arms": arms, "orders": orders, "value": expected}


def run(root, repetitions):
    root.mkdir(parents=True, exist_ok=False)
    base = REPO / "docs/evidence/runs/2026-09-15-ingress-bounds/baseline"
    old_ideas = load("orze.core._ingress_old_ideas", base / "ideas.py")
    old = load("orze.engine._ingress_old", base / "idea_ingress.py")
    old._overlay_sidecar_ideas = old_ideas._overlay_sidecar_ideas
    modules = {"old": old, "new": idea_ingress}
    source_paths = [Path(m.__file__) for m in modules.values()] + [base / "ideas.py", REPO / "src/orze/core/ideas.py"]
    sources = {str(p): sha(p.read_bytes()) for p in source_paths}
    report = {"baseline": "ad7ed16632897ab3d4ad62fb717fec53d6795f34", "sources": sources,
              "script_sha256": sha(Path(__file__).read_bytes()), "repetitions": repetitions, "rows": [],
              "limits": ["Synthetic historical metadata and conflicting authored sidecars; no workers.",
                         "First batch and complete inspection cycle measured separately on the same unchanged database.",
                         "Fresh private /tmp projects; host not exclusive; Python traced peak is not RSS.",
                         "Memory tracing only for first batch; setup and I/O counter reads outside timing.",
                         "Invalidation and genuine new CPU actions are covered by separate product tests."]}
    for size in (100, 1000, 5000):
        project = root / str(size)
        project.mkdir()
        results = project / "results"
        results.mkdir()
        source = project / "ideas.md"
        source.write_text("# Ideas\n")
        side = project / "ideas.d"
        side.mkdir()
        cfg = {"ideas_file": str(source), "results_dir": str(results),
               "idea_lake_db": str(project / "lake.db"), "_orze_dir": str(project / ".orze")}
        engine = Orze.__new__(Orze)
        engine.cfg, engine.results_dir, engine.active_roles = cfg, results, {}
        engine.lake = IdeaLake(cfg["idea_lake_db"])
        try:
            cache = {}
            for i in range(size):
                idea_id = f"idea-side-{i:05d}"
                fingerprint = integrity.hash_config({"seed": i})
                cache[fingerprint] = idea_id
                engine.lake.conn.execute(
                    "INSERT INTO ideas(idea_id,title,config,raw_markdown,status) VALUES (?,?,?,'','completed')",
                    (idea_id, "Stored metadata", f"seed: {i}\n"))
                (side / f"{i:05d}.md").write_text(
                    f"## {idea_id}: Conflicting authored title\n```yaml\nseed: {i}\n```\n" + "note " * 400)
            engine.lake.conn.commit()
            cache_file = project / ".orze/state/config_hashes.json"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_text(json.dumps(cache, sort_keys=True))
            before = sha("\n".join(engine.lake.conn.iterdump()).encode())

            def call(module, cycle):
                engine._idea_ingress_cursor = None
                count, batches, digest = 0, 0, hashlib.sha256()
                while True:
                    raw, inserted = module.ingest_ideas_source(engine, cfg)
                    assert not inserted
                    count += len(raw)
                    batches += 1
                    digest.update(json.dumps(raw, sort_keys=True).encode())
                    if not cycle or engine._idea_ingress_cursor[2] == 0:
                        break
                    assert batches <= (size + 127) // 128 + 1
                return {"records": count, "batches": batches, "payload_sha256": digest.hexdigest()}

            # Logging conflict warnings is real product behavior; omit only
            # terminal output in this cost harness for both versions equally.
            with patch.object(idea_ingress.logger, "warning", lambda *a, **k: None):
                first = measure({k: lambda m=m: call(m, False) for k, m in modules.items()}, repetitions, memory=True)
                print(json.dumps({"size": size, "first_batch_ms": {k: a["median_seconds"] * 1000 for k, a in first["arms"].items()}}), flush=True)
                cycle = measure({k: lambda m=m: call(m, True) for k, m in modules.items()}, repetitions)
            after = sha("\n".join(engine.lake.conn.iterdump()).encode())
            assert before == after
            row = {"sidecar_records": size, "cache_bytes": cache_file.stat().st_size,
                   "first_batch": first, "full_cycle": cycle,
                   "database_sha256_before": before, "database_sha256_after": after}
            report["rows"].append(row)
            (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"size": size, "full_cycle_ms": {k: a["median_seconds"] * 1000 for k, a in cycle["arms"].items()}}), flush=True)
        finally:
            engine.lake.close()
    assert sources == {str(p): sha(p.read_bytes()) for p in source_paths}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    run(args.output, args.repetitions)
