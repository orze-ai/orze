"""Alternating old/new metadata-reader and real Policy continuation costs.

Synthetic ledger rows are not real attempts, settlements or research gains.
The production Orze iteration, pagers and BoundPolicy run without a worker.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import resource
import statistics
import sys
import time
import tracemalloc
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests"))
from test_cpu_budget_normalization import load_baseline
from test_cpu_evidence_paging_product import _metadata_prefix
import orze.core
from orze.core import cpu_action_budget as current
from orze.core import research_interfaces as api
from orze.engine import cpu_phase
# This diagnostic is Core-only even if the global harness has a Pro stub.
# Block optional extension imports before importing the Orze class; the CPU
# iteration, policy, pagers and budget functions remain the real code.
with patch.dict(sys.modules, {"orze_pro": None}):
    from orze.engine.orchestrator import Orze


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class PagingPolicy:
    def __init__(self, declaration):
        self.last = None

    def decide(self, snapshot, budget):
        records = snapshot["recorded_evidence"]
        self.last = {"budget": budget, "page": snapshot["evidence_page"]["page"],
                     "results": records["results"], "unavailable": records["unavailable"]}
        return {"kind": "ReadEvidence", "cursor": snapshot["evidence_page"]["next_cursor"]}


@contextmanager
def selected(module):
    with patch.dict(sys.modules, {"orze.core.cpu_action_budget": module}), patch.object(
            orze.core, "cpu_action_budget", module):
        yield


def fingerprint(conn):
    return sha("\n".join(conn.iterdump()).encode())


def usage():
    io = {key: int(value) for key, value in (
        line.split(":") for line in Path("/proc/self/io").read_text().splitlines())}
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return {**io, "minor_faults": ru.ru_minflt, "major_faults": ru.ru_majflt,
            "in_blocks": ru.ru_inblock, "out_blocks": ru.ru_oublock}


def observe(module, call):
    counts = Counter()
    originals = {name: getattr(module, name) for name in ("_decode", "_json", "_totals")}
    def count(name):
        def run(*args, **kwargs):
            counts[name] += 1
            return originals[name](*args, **kwargs)
        return run
    try:
        for name in originals:
            setattr(module, name, count(name))
        result = call()
    finally:
        for name, original in originals.items():
            setattr(module, name, original)
    return {"calls": dict(counts), "value": result}


def measure(prepare, modules, repetitions):
    expected = None
    arms = {name: {"seconds": [], "io_deltas": []} for name in modules}
    for name, module in modules.items():
        with selected(module):
            value = prepare(module)()
            if expected is None:
                expected = value
            assert value == expected
    orders = []
    for iteration in range(repetitions):
        names = list(modules) if iteration % 2 == 0 else list(reversed(modules))
        orders.append(names)
        for name in names:
            with selected(modules[name]):
                call = prepare(modules[name])
                before = usage()
                start = time.perf_counter()
                value = call()
                elapsed = time.perf_counter() - start
                after = usage()
                assert value == expected
                arms[name]["seconds"].append(elapsed)
                arms[name]["io_deltas"].append({k: after[k] - v for k, v in before.items()})
    for name, module in modules.items():
        with selected(module):
            arm = arms[name]
            arm["median_seconds"] = statistics.median(arm["seconds"])
            call = prepare(module)
            arm["instrumented"] = observe(module, call)
            call = prepare(module)
            tracemalloc.start()
            assert call() == expected
            arm["traced_current_bytes"], arm["traced_peak_bytes"] = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            call = prepare(module)
            profiler = cProfile.Profile()
            profiler.runcall(call)
            stats = pstats.Stats(profiler)
            arm["profile"] = [{"file": key[0], "line": key[1], "function": key[2],
                               "primitive_calls": value[0], "calls": value[1],
                               "self_seconds": value[2], "cumulative_seconds": value[3]}
                              for key, value in stats.stats.items()
                              if key[0] == module.__file__ and key[2] in {
                                  "_totals", "_permit", "_scope", "_declaration", "_decode", "_json",
                                  "_validate_permit", "_validate_scope", "_validate_declaration"}]
    return {"arms": arms, "orders": orders}


def main(root, repetitions):
    root.mkdir(parents=True, exist_ok=False)
    baseline = load_baseline()
    modules = {"old": baseline, "new": current}
    sources = {str(Path(m.__file__)): sha(Path(m.__file__).read_bytes()) for m in modules.values()}
    api.register_policy("normalization_cost", "normalization.cost.v1", PagingPolicy)
    report = {"baseline_commit": "558f8c69db03eb7a5087d8200f54d4ce0fe46dcd",
              "sources": sources, "script_sha256": sha(Path(__file__).read_bytes()),
              "python": sys.version, "repetitions": repetitions, "rows": [],
              "limits": ["Synthetic SETTLED metadata, not native executions or actual settlement receipts.",
                         "Policy calls run production Orze iteration and pagers with 33 metadata prefix attempts.",
                         "Private warm /tmp storage, no provider, GPU or production service.",
                         "Instrumentation, profiling, memory tracing and continuation setup are outside timed intervals.",
                         "I/O counters include reading /proc/self/io; memory is Python traced peak, not process peak RSS.",
                         "Full audits and execution authorization remain uncached; this is not scientific speedup."]}
    for size in (100, 1000, 5000):
        project = root / str(size)
        project.mkdir()
        (project / "ideas.md").write_text("# Ideas\n")
        cfg = {"execution": {"version": 2, "resource": "cpu", "slots": 1, "wall_budget_seconds": None},
               "results_dir": str(project / "results"), "ideas_file": str(project / "ideas.md"),
               "idea_lake_db": str(project / "lake.db"), "min_disk_gb": 0,
               "action_policy": {"version": 2, "kind": "normalization_cost", "idle": "wait",
                                 "wait_seconds": .01, "evidence_page_size": 8}}
        engine = Orze([], cfg)
        try:
            cpu_phase.start(engine)
            _metadata_prefix(engine.lake, engine.results_dir)
            other = project / "other"
            other.mkdir()
            scopes = (engine._cpu_scope, current.initialize(engine.lake, other, cfg["execution"]))
            rows = []
            for prefix, scope in enumerate(scopes):
                for index in range(size):
                    task = f"metadata-{prefix}-{index:06d}"
                    identity = sha(task.encode())[:48]
                    permit = {"schema": 1, "budget_scope": scope, "reservation_id": identity,
                              "task_id": task, "slot": 0, "wall_limit_seconds": 2,
                              "reserved_nanoseconds": "2000000000"}
                    ref = {"task_id": task, "phase": "action", "attempt_id": identity, "generation": 1}
                    rows.append((identity, scope["results_dir"], task, 0, baseline._json(permit),
                                 baseline._json(ref), "SETTLED", "0" * 64))
            engine.lake.conn.executemany("INSERT INTO cpu_action_reservations VALUES (?,?,?,?,?,?,?,?)", rows)
            engine.lake.conn.commit()
            before = fingerprint(engine.lake.conn)

            def continuation(module):
                cpu_phase._release_evidence_scan(engine)
                assert cpu_phase.iteration(engine) is True
                def call():
                    assert cpu_phase.iteration(engine) is True
                    # BoundPolicy owns the concrete callback; its captured
                    # input is stable despite fresh cursor/scan identities.
                    return {"continued": True, "queue": engine._cpu_evidence_queue,
                            "page": engine._cpu_evidence_view["evidence_page"]["page"],
                            "evidence": engine._cpu_evidence_view["recorded_evidence"]}
                return call

            row = {"rows_in_scope": size, "rows_in_other_scope": size,
                   "totals": measure(lambda m: lambda: m._totals(engine.lake.conn, engine._cpu_scope), modules, repetitions),
                   "snapshot": measure(lambda m: lambda: m.snapshot(engine.lake, engine._cpu_scope), modules, repetitions),
                   "policy_continuation": measure(continuation, modules, repetitions)}
            row.update(ledger_sha256_before=before, ledger_sha256_after=fingerprint(engine.lake.conn))
            assert row["ledger_sha256_before"] == row["ledger_sha256_after"]
            report["rows"].append(row)
            (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"rows": size, "median_ms": {op: {name: arm["median_seconds"] * 1000
                  for name, arm in row[op]["arms"].items()} for op in ("totals", "snapshot", "policy_continuation")}}), flush=True)
        finally:
            cpu_phase.close(engine)
    assert sources == {path: sha(Path(path).read_bytes()) for path in sources}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    main(args.output, args.repetitions)
