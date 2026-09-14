"""Paired same-machine metadata-only history cost diagnostic.

Adapted from the archived 2026-09-14 history-cost diagnostic. Complete old
modules are loaded from Git-byte-exact snapshots, not rewritten algorithms.
No native worker, TREE, actual settlement, provider, GPU or Pro is involved.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import orze
from orze.idea_lake import IdeaLake
from orze.core import cpu_action_budget as current
from orze.core.integrity import hash_config, load_hashes
from orze.engine import idea_ingress as new_ingress

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "docs/evidence/runs/2026-09-14-cost-equivalence/baseline/src/orze"
FILES = ["src/orze/core/cpu_action_budget.py", "src/orze/engine/idea_ingress.py",
         "src/orze/idea_lake.py", "src/orze/core/integrity.py",
         "src/orze/core/proposal_admission.py", "src/orze/core/ideas.py",
         "src/orze/engine/cpu_phase.py", "src/orze/engine/native_cpu_action.py"]
sha = lambda raw: hashlib.sha256(raw).hexdigest()
hashes = lambda: {p: sha((REPO / p).read_bytes()) for p in FILES}
assert Path(orze.__file__).resolve().is_relative_to(REPO / "src")
assert importlib.util.find_spec("orze_pro") is None


def old_module(name, path, expected):
    assert sha(path.read_bytes()) == expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old = old_module("orze.core._cost_scale_old_budget", BASE / "core/cpu_action_budget.py",
                 "71bc89977cb582aa86a218003398431bb975ce11410f1df0d43b385ae4667169")
old_ingress = old_module("orze.engine._cost_scale_old_ingress", BASE / "engine/idea_ingress.py",
                         "9361e034b8cce04583bdd6fce6a5d9662b1ba38fc369141916c63d2f7a6932ce")
old_lake = old_module("orze._cost_scale_old_lake", BASE / "idea_lake.py",
                      "cdaa8243827eb0370ef1db0d3d2c76677791ecca7a49093bc246120a4051a928")


def paired(old_call, new_call, repetitions=5):
    expected = old_call()
    assert new_call() == expected
    values = {"old": [], "new": []}
    for i in range(repetitions):
        for name, call in (("old", old_call), ("new", new_call)) if i % 2 == 0 else (
                ("new", new_call), ("old", old_call)):
            started = time.perf_counter()
            actual = call()
            values[name].append(time.perf_counter() - started)
            assert actual == expected
    return {name: {"seconds": times, "median_seconds": statistics.median(times)}
            for name, times in values.items()}


def trace_call(conn, call, path, module=None):
    statements, counts = [], [0]
    original = None if module is None else module._decode
    if original is not None:
        def counted(raw):
            counts[0] += 1
            return original(raw)
        module._decode = counted
    conn.set_trace_callback(statements.append)
    try:
        value = call()
    finally:
        conn.set_trace_callback(None)
        if original is not None:
            module._decode = original
    raw = ("\n".join(statements) + ("\n" if statements else "")).encode()
    path.write_bytes(raw)
    return {"value": value, "sql_count": len(statements),
            "select_count": sum(s.lstrip().upper().startswith("SELECT") for s in statements),
            "decode_calls": counts[0] if original is not None else None,
            "sql_trace": str(path), "sql_trace_sha256": sha(raw),
            "statement_prefixes": dict(Counter(s.partition(" WHERE ")[0] for s in statements)),
            "query_plan": [list(r) for r in conn.execute("EXPLAIN QUERY PLAN " + statements[0])]
                          if statements and module is not None else None}


def main(root):
    root.mkdir(parents=True, exist_ok=True)
    assert not any(root.iterdir()), "fresh private output directory required"
    before = hashes()
    measured = []
    for n in (100, 1000, 5000):
        project = root / str(n)
        project.mkdir()
        results, other_results = project / "results", project / "other"
        results.mkdir()
        other_results.mkdir()
        lake = IdeaLake(project / "lake.db")
        peer = old_lake.IdeaLake(project / "lake.db")
        try:
            declaration = {"version": 2, "resource": "cpu", "slots": 1,
                           "wall_budget_seconds": None}
            scope = current.initialize(lake, results, declaration)
            other = current.initialize(lake, other_results, declaration)
            rows = []
            for target, prefix in ((scope, "main"), (other, "other")):
                for i in range(n):
                    task = "metadata-" + prefix + "-" + str(i).zfill(6)
                    identity = sha(task.encode())[:48]
                    permit = {"schema": 1, "budget_scope": target, "reservation_id": identity,
                              "task_id": task, "slot": 0, "wall_limit_seconds": 2,
                              "reserved_nanoseconds": "2000000000"}
                    ref = {"task_id": task, "phase": "action", "attempt_id": identity, "generation": 1}
                    rows.append((identity, target["results_dir"], task, 0, old._json(permit),
                                 old._json(ref), "SETTLED", "0" * 64))
            lake.conn.executemany("INSERT INTO cpu_action_reservations VALUES (?,?,?,?,?,?,?,?)", rows)
            cache, ideas = {}, []
            for i in range(n):
                task = "idea-metadata-" + str(i).zfill(6)
                raw = json.dumps({"seed": i})
                fingerprint = hash_config({"seed": i})
                cache[fingerprint] = task
                ideas.append((task, "metadata-only", raw, fingerprint, sha(raw.encode()),
                              "", "completed", "native_cpu_action"))
            lake.conn.executemany("INSERT INTO ideas(idea_id,title,config,config_hash,"
                "config_source_sha256,raw_markdown,status,kind) VALUES (?,?,?,?,?,?,?,?)", ideas)
            lake.conn.commit()
            cfg = {"ideas_file": str(project / "ideas.md"), "_orze_dir": str(project / ".orze"),
                   "_env_ORZE_RESULTS_DIR": str(results), "results_dir": str(results)}
            Path(cfg["ideas_file"]).write_text("# Ideas\n")
            cache_path = project / ".orze/state/config_hashes.json"
            cache_path.parent.mkdir(parents=True)
            cache_path.write_text(json.dumps(cache, sort_keys=True))
            def engine(actual_lake):
                return SimpleNamespace(results_dir=results, lake=actual_lake, active_roles={},
                    _config_override_hash=hash_config,
                    _load_config_hashes=lambda: load_hashes(results, cfg))
            old_engine, new_engine = engine(peer), engine(lake)
            ledger = lambda: sha(json.dumps([list(r) for r in lake.conn.execute(
                "SELECT * FROM cpu_action_reservations ORDER BY reservation_id")],
                sort_keys=True, separators=(",", ":")).encode())
            ledger_before = ledger()
            row = {"rows_in_scope": n, "rows_in_other_scope": n, "idea_metadata_rows": n,
                   "config_cache_bytes": cache_path.stat().st_size,
                   "totals": paired(lambda: old._totals(lake.conn, scope),
                                    lambda: current._totals(lake.conn, scope)),
                   "snapshot": paired(lambda: old.snapshot(lake, scope),
                                      lambda: current.snapshot(lake, scope)),
                   "empty_ingress": paired(lambda: old_ingress.ingest_ideas_source(old_engine, cfg),
                                           lambda: new_ingress.ingest_ideas_source(new_engine, cfg))}
            row["old_totals_trace"] = trace_call(lake.conn, lambda: old._totals(lake.conn, scope),
                                                  project / "old-totals.sql", old)
            row["new_totals_trace"] = trace_call(lake.conn, lambda: current._totals(lake.conn, scope),
                                                  project / "new-totals.sql", current)
            row["old_empty_ingress_trace"] = trace_call(peer.conn,
                lambda: old_ingress.ingest_ideas_source(old_engine, cfg), project / "old-empty-ingress.sql")
            row["new_empty_ingress_trace"] = trace_call(lake.conn,
                lambda: new_ingress.ingest_ideas_source(new_engine, cfg), project / "new-empty-ingress.sql")
            assert row["old_totals_trace"]["value"] == row["new_totals_trace"]["value"] == (n * 2_000_000_000, {})
            assert row["old_totals_trace"]["sql_count"] == n + 1
            assert row["new_totals_trace"]["sql_count"] == 1
            assert row["new_empty_ingress_trace"]["sql_count"] == 0
            assert ledger_before == ledger()
            row["ledger_sha256_before"] = ledger_before
            row["ledger_sha256_after"] = ledger()
            measured.append(row)
        finally:
            peer.close()
            lake.close()
    report = {"schema": 1, "baseline_commit": "e606ba6688b4e1972e26a08617821a427668dd24",
              "source_root": str(REPO), "fixture_root": str(root),
              "before": before, "after": hashes(), "rows": measured,
              "measurement": "One warmup per arm then five paired repetitions, alternating arm order; traces/parser spies outside timing.",
              "limits": ["Synthetic SETTLED-shaped rows are metadata fixtures, not actual attempts, TREE/effects or settlements.",
                         "One actual supplied IdeaLake with an equal-size other scope; no new index, cache, SUM, refund or skipped bad row.",
                         "Warm private /tmp same-host measurements, not deployed filesystem throughput or research/scientific speedup.",
                         "Budget remains O(N) full row auditing even if SELECT count is constant.",
                         "Empty ingress uses real SourceLock and complete fresh source/sidecar/batch processing, but no worker/provider.",
                         "Old/new full modules and old/new IdeaLake classes share exactly the same scratch DB and source bytes.",
                         "No Pro import, GPU, provider, license, production history, source edit or old-test edit."]}
    assert before == report["after"]
    (root / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)

