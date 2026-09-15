"""Alternate old/new status-query repair; normal time and allocation separately.

Fixture database creation, copying, schema open, final equality and observer runs
are outside the normal timed interval. Temporary storage is observed before the
main write, not sampled for a peak. Process /proc I/O counters are not physical
shared-filesystem traffic or fsync counts. All databases use DELETE/FULL.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import time
import tracemalloc
from unittest.mock import patch

from orze.core import config_identity_repair as staging
from orze.idea_lake import IdeaLake
from orze import idea_lake as lake_api

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "docs/evidence/runs/2026-09-15-status-merge-queries/baseline"
spec = importlib.util.spec_from_file_location("old_repair_stage", BASE / "config_identity_repair.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
FUNCTIONS = {"old": lambda lake: old.repair_admitted_identities(
                lake, retry=lake_api._retry_on_busy, hasher=lake_api.hash_config, logger=lake_api.logger),
             "new": IdeaLake._repair_admitted_config_hashes}
allocator_path = ROOT / "docs/evidence/checks/2026-09-15-status-merge-allocator.py"
spec = importlib.util.spec_from_file_location("sqlite_allocator", allocator_path)
allocator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(allocator)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint():
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in (ROOT / "src").rglob("*.py")}


def io():
    return {k: int(v) for k, v in (line.split(":") for line in Path("/proc/self/io").read_text().splitlines())}


def database_identity(lake):
    return sha("\n".join(lake.conn.iterdump()).encode())


class WriterObserver:
    def __init__(self, connection):
        self.connection = connection
        self.events = []
        self.elapsed = None

    def __getattr__(self, key):
        return getattr(self.connection, key)

    def execute(self, statement, *args):
        value = self.connection.execute(statement, *args)
        if statement == "BEGIN IMMEDIATE":
            self.started = time.perf_counter()
            self.events.append("BEGIN IMMEDIATE")
        return value

    def commit(self):
        self.connection.commit()
        self.elapsed = time.perf_counter() - self.started
        self.events.append("COMMIT")


def observe_stage(connection, stage):
    path = Path(stage.execute("PRAGMA database_list").fetchone()[2])
    result = {"journal_mode": stage.execute("PRAGMA journal_mode").fetchone()[0],
              "synchronous": stage.execute("PRAGMA synchronous").fetchone()[0],
              "locking_mode": stage.execute("PRAGMA locking_mode").fetchone()[0],
              "cache_size": stage.execute("PRAGMA cache_size").fetchone()[0],
              "directory_mode": path.parent.stat().st_mode & 0o777,
              "bytes_before_main_apply": path.stat().st_size,
              "allocated_bytes_before_main_apply": path.stat().st_blocks * 512,
              "staged_rows": stage.execute("SELECT COUNT(*) FROM inputs").fetchone()[0],
              "directory": str(path.parent)}
    assert (result["journal_mode"], result["synchronous"], result["locking_mode"]) == ("delete", 2, "normal")
    assert result["cache_size"] == -512 and result["directory_mode"] == 0o700
    return result


def main(root, mode):
    root.mkdir(parents=True, exist_ok=False)
    sources = fingerprint()
    report = {"mode": mode, "source_files": sources, "script_sha256": sha(Path(__file__).read_bytes()),
              "allocator_helper_sha256": sha(allocator_path.read_bytes()),
              "baseline_files": {p.name: sha(p.read_bytes()) for p in BASE.glob("*.py")},
              "rows": [], "limits": ["Imported metadata only, no scientific or worker throughput claim.",
                 "Python tracemalloc and SQLite allocator highwater measured in separate modes, excluded from latency results.",
                 "SQLite MEMORY_USED excludes separately configured auxiliary page-cache memory; neither counter is RSS.",
                 "One config and YAML expansion, total history I/O and temporary disk remain unbounded.",
                 "Stage files normally removed; their size before main apply is not a peak disk measurement.",
                 "Writer hold observations are separate instrumented runs, including Python and temporary reads.",
                 "Already-repaired calls still execute a legacy-missing lookup; no new persistent cache."]}
    repetitions = 3 if mode == "time" else 1
    report["repetitions"] = repetitions
    for count, payload in ((100, 1024), (1000, 1024), (5000, 1024), (500, 8192)):
        project = root / f"{count}-{payload}"
        project.mkdir()
        input_path = project / "input.db"
        with_lake = IdeaLake(input_path)
        try:
            with_lake.conn.executemany(
                "INSERT INTO ideas(idea_id,title,config,raw_markdown,status) VALUES (?,?,?,'','completed')",
                ((f"legacy-{n:05d}", "Imported metadata", f"seed: {n}\npayload: " + "x" * payload + "\n")
                 for n in range(count)),
            )
            with_lake.conn.commit()
            input_digest = database_identity(with_lake)
        finally:
            with_lake.close()
        row = {"records": count, "payload_characters": payload, "input_database_sha256": input_digest,
               "orders": [], "arms": {name: [] for name in FUNCTIONS}}
        expected = None
        for repeat in range(repetitions):
            order = ["old", "new"] if repeat % 2 == 0 else ["new", "old"]
            row["orders"].append(order)
            for name in order:
                path = project / f"{repeat}-{name}.db"
                shutil.copy2(input_path, path)
                lake = IdeaLake(path)
                try:
                    assert database_identity(lake) == input_digest
                    gc.collect()
                    counters = io()
                    native_before = allocator.snapshot(reset=True) if mode == "allocator" else None
                    if mode == "memory":
                        tracemalloc.start()
                    started = time.perf_counter()
                    repaired = FUNCTIONS[name](lake)
                    elapsed = time.perf_counter() - started
                    native_after = allocator.snapshot() if mode == "allocator" else None
                    memory = tracemalloc.get_traced_memory() if mode == "memory" else None
                    if mode == "memory":
                        tracemalloc.stop()
                    counters_after = io()
                    assert repaired == count
                    digest = database_identity(lake)
                    expected = digest if expected is None else expected
                    assert digest == expected
                    assert lake.conn.execute("SELECT COUNT(*) FROM ideas WHERE config_hash IS NULL OR config_source_sha256 IS NULL").fetchone()[0] == 0
                    hot_start = time.perf_counter()
                    assert FUNCTIONS[name](lake) == 0
                    hot_elapsed = time.perf_counter() - hot_start
                    assert database_identity(lake) == digest
                    row["arms"][name].append({"repaired": repaired, "seconds": elapsed,
                        "hot_seconds": hot_elapsed, "current_peak_python_bytes": memory,
                        "sqlite_allocation_before": native_before, "sqlite_allocation_after": native_after,
                        "io_delta": {k: counters_after[k] - counters[k] for k in counters},
                        "database_sha256": digest, "database": str(path)})
                finally:
                    lake.close()
        if mode == "time":
            row["separate_writer_audit"] = {}
            for name in FUNCTIONS:
                path = project / f"writer-{name}.db"
                shutil.copy2(input_path, path)
                lake = IdeaLake(path)
                observer = WriterObserver(lake.conn)
                lake.conn = observer
                stage_info = []
                stage_module = old if name == "old" else staging
                original_apply = stage_module._apply

                def apply(connection, stage):
                    stage_info.append(observe_stage(connection, stage))
                    return original_apply(connection, stage)

                try:
                    with patch.object(stage_module, "_apply", apply):
                        assert FUNCTIONS[name](lake) == count
                    assert observer.events == ["BEGIN IMMEDIATE", "COMMIT"]
                    assert database_identity(lake) == expected
                    assert all(not Path(info["directory"]).exists() for info in stage_info)
                    row["separate_writer_audit"][name] = {"events": observer.events,
                        "hold_seconds": observer.elapsed, "stage": stage_info, "database": str(path)}
                finally:
                    lake.close()
        report["rows"].append(row)
        (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"mode": mode, "records": count, "payload": payload,
            "seconds": {k: statistics.median(v["seconds"] for v in a) for k, a in row["arms"].items()},
            "memory": {k: a[0]["current_peak_python_bytes"] for k, a in row["arms"].items()},
            "sqlite": {k: a[0]["sqlite_allocation_after"] for k, a in row["arms"].items()}}), flush=True)
    assert sources == fingerprint()
    report["passed"] = True
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--mode", choices=("time", "memory", "allocator"), default="time")
    args = parser.parse_args()
    main(args.root.resolve(), args.mode)
