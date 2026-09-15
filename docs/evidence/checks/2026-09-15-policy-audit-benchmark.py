"""Alternate fixed old/new Policy continuations on the same synthetic ledger.

Production iteration, BoundPolicy, pagers and complete budget reader; metadata
history only, no claim of real settlements, workers or research speedup.
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
sys.path.insert(0, str(REPO / "tests"))
from test_cpu_evidence_paging_product import _metadata_prefix
from test_cpu_budget_scan import add_row
from orze.core import cpu_action_budget as budget, research_interfaces as api
from orze.engine import cpu_phase
with patch.dict(sys.modules, {"orze_pro": None}):
    from orze.engine.orchestrator import Orze


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Policy:
    def __init__(self, declaration):
        self.last = None

    def decide(self, snapshot, allowance):
        self.last = {"budget": allowance, "page": snapshot["evidence_page"]["page"],
                     "evidence": snapshot["recorded_evidence"]}
        return {"kind": "ReadEvidence", "cursor": snapshot["evidence_page"]["next_cursor"]}


def usage():
    io = {k: int(v) for k, v in
          (line.split(":") for line in Path("/proc/self/io").read_text().splitlines())}
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return {**io, "minor_faults": ru.ru_minflt, "major_faults": ru.ru_majflt,
            "in_blocks": ru.ru_inblock, "out_blocks": ru.ru_oublock}


def run(root, repetitions):
    root.mkdir(parents=True, exist_ok=False)
    old_path = REPO / "docs/evidence/runs/2026-09-15-policy-audit/baseline/cpu_phase.py"
    spec = importlib.util.spec_from_file_location("orze.engine._audit_baseline", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    modules = {"old": old, "new": cpu_phase}
    sources = {str(Path(m.__file__)): sha(Path(m.__file__).read_bytes()) for m in modules.values()}
    sources[str(Path(budget.__file__))] = sha(Path(budget.__file__).read_bytes())
    report = {"baseline_commit": "cdfbe064849af73b89300d0778be514044c31087",
              "sources": sources, "script_sha256": sha(Path(__file__).read_bytes()),
              "python": sys.version, "rows": [], "repetitions": repetitions,
              "limits": ["Synthetic settled ledger and 33 metadata-only prefix attempts.",
                         "Actual production Policy continuation; no worker or scientific result.",
                         "Private warm /tmp filesystem; host not exclusive.",
                         "Setup/counts/memory tracing outside elapsed measurements.",
                         "Python traced peak is not RSS; I/O includes /proc/self/io reads."]}
    api.register_policy("audit_cost", "audit.cost.v1", Policy)
    for size in (100, 1000, 5000):
        project = root / str(size)
        project.mkdir()
        (project / "ideas.md").write_text("# Ideas\n")
        cfg = {"execution": {"version": 2, "resource": "cpu", "slots": 1, "wall_budget_seconds": None},
               "results_dir": str(project / "results"), "ideas_file": str(project / "ideas.md"),
               "idea_lake_db": str(project / "lake.db"), "min_disk_gb": 0,
               "action_policy": {"version": 2, "kind": "audit_cost", "idle": "wait",
                                 "wait_seconds": .01, "evidence_page_size": 8}}
        engine = Orze([], cfg)
        try:
            cpu_phase.start(engine)
            _metadata_prefix(engine.lake, engine.results_dir)
            for i in range(size):
                add_row(engine.lake.conn, engine._cpu_scope, i + 1)
            before = sha("\n".join(engine.lake.conn.iterdump()).encode())

            def prepare(module):
                cpu_phase._release_evidence_scan(engine)
                assert module.iteration(engine)

                def call():
                    assert module.iteration(engine)
                    return {"page": engine._cpu_evidence_view["evidence_page"]["page"],
                            "evidence": engine._cpu_evidence_view["recorded_evidence"],
                            "queue": engine._cpu_evidence_queue}
                return call

            expected = None
            arms = {name: {"seconds": [], "io_deltas": []} for name in modules}
            for name, module in modules.items():
                value = prepare(module)()
                if expected is None:
                    expected = value
                assert value == expected
            orders = []
            for i in range(repetitions):
                order = list(modules) if i % 2 == 0 else list(reversed(modules))
                orders.append(order)
                for name in order:
                    call = prepare(modules[name])
                    before_io = usage()
                    start = time.perf_counter()
                    value = call()
                    elapsed = time.perf_counter() - start
                    after_io = usage()
                    assert value == expected
                    arms[name]["seconds"].append(elapsed)
                    arms[name]["io_deltas"].append({k: v - before_io[k] for k, v in after_io.items()})
            for name, module in modules.items():
                arm = arms[name]
                call = prepare(module)
                with patch.object(budget, "_totals", wraps=budget._totals) as totals:
                    assert call() == expected
                    arm["complete_audits"] = totals.call_count
                call = prepare(module)
                tracemalloc.start()
                assert call() == expected
                arm["traced_current_bytes"], arm["traced_peak_bytes"] = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                arm["median_seconds"] = statistics.median(arm["seconds"])
            after = sha("\n".join(engine.lake.conn.iterdump()).encode())
            assert before == after
            report["rows"].append({"settled_metadata_rows": size, "arms": arms, "orders": orders,
                                   "ledger_sha256_before": before, "ledger_sha256_after": after,
                                   "result_sha256": sha(json.dumps(expected, sort_keys=True).encode())})
            (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"rows": size, "median_ms": {
                name: arm["median_seconds"] * 1000 for name, arm in arms.items()}}), flush=True)
        finally:
            cpu_phase.close(engine)
    assert sources == {path: sha(Path(path).read_bytes()) for path in sources}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    run(args.output, args.repetitions)
