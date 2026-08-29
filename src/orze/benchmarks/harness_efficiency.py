"""Reproducible CPU benchmark for Orze's research control plane.

This benchmark never launches a trainer, evaluator, model, or accelerator
probe. It exercises the production IdeaLake, queue selection, filesystem
claim, and audited lifecycle APIs against a disposable database on the chosen
filesystem. A receipt is VERIFIED only at or above the declared acceptance
scale and when every latency and correctness target passes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import platform
import sqlite3
import statistics
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from orze.core.fs import atomic_write
from orze.engine.phases import OrzePhaseMixin
from orze.engine.scheduler import claim, get_unclaimed
from orze.idea_lake import IdeaLake


SCHEMA_VERSION = 2
MIN_ACCEPTANCE_IDEAS = 10_000
# A nearest-rank p95 needs at least 20 observations; with 15, p95 is the
# single maximum and one unrelated filesystem scheduling pause determines the
# verdict. Twenty still fails on two slow observations while exposing raw
# samples for audit.
MIN_ACCEPTANCE_ITERATIONS = 20
MIN_BULK_INSERT_ROWS_PER_SECOND = 1_000.0
DEFAULT_TARGETS_MS = {
    "connection_reopen_p95_ms": 75.0,
    "admitted_identity_lookup_p95_ms": 50.0,
    "queue_query_p95_ms": 60.0,
    "warm_queue_sync_p95_ms": 120.0,
    "steady_control_tick_p95_ms": 200.0,
    "lifecycle_round_trip_p95_ms": 500.0,
}


class _QueueParser(OrzePhaseMixin):
    """Use the exact queue parsing cache without constructing an orchestrator."""


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one sample")
    rank = max(1, int(len(ordered) * probability + 0.999999))
    return ordered[min(rank, len(ordered)) - 1]


def _latency_summary(values: Iterable[float]) -> dict:
    samples = list(values)
    return {
        "samples": len(samples),
        "samples_ms": [round(value, 3) for value in samples],
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(_percentile(samples, 0.95), 3),
        "max_ms": round(max(samples), 3),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timed(call: Callable[[], object]) -> tuple[object, float]:
    started = time.perf_counter()
    value = call()
    return value, (time.perf_counter() - started) * 1000.0


def evaluate_targets(metrics: dict, targets: Optional[dict] = None) -> dict:
    """Evaluate latency targets and non-latency correctness invariants."""
    limits = dict(DEFAULT_TARGETS_MS if targets is None else targets)
    actuals = {
        "connection_reopen_p95_ms": metrics[
            "connection_reopen"
        ]["p95_ms"],
        "admitted_identity_lookup_p95_ms": metrics[
            "admitted_identity_lookup"
        ]["p95_ms"],
        "queue_query_p95_ms": metrics["queue_query"]["p95_ms"],
        "warm_queue_sync_p95_ms": metrics[
            "warm_queue_sync"
        ]["p95_ms"],
        "steady_control_tick_p95_ms": metrics[
            "steady_control_tick"
        ]["p95_ms"],
        "lifecycle_round_trip_p95_ms": metrics[
            "lifecycle_round_trip"
        ]["p95_ms"],
    }
    checks = {
        name: {
            "actual_ms": actuals[name],
            "maximum_ms": float(limit),
            "passed": actuals[name] <= float(limit),
        }
        for name, limit in limits.items()
    }
    throughput = {
        "actual_rows_per_second": metrics["bulk_insert_rows_per_second"],
        "minimum_rows_per_second": MIN_BULK_INSERT_ROWS_PER_SECOND,
        "passed": metrics["bulk_insert_rows_per_second"]
        >= MIN_BULK_INSERT_ROWS_PER_SECOND,
    }
    invariants = {
        "identity_coverage_exact": metrics["identity_coverage_count"]
        == metrics["idea_count"],
        "identity_probe_count_exact": metrics["identity_probe_matches"]
        == metrics["identity_probe_count"],
        "queue_count_exact": metrics["queue_count"]
        == min(metrics["idea_count"], metrics["queue_limit"]),
        "queue_order_stable": metrics["queue_order_stable"],
        "lifecycle_states_exact": metrics["lifecycle_states_exact"],
        "atomic_claim_single_winner": metrics[
            "atomic_claim_successes"
        ] == 1,
        "schema_bootstrap_cache_hits_exact": metrics[
            "schema_bootstrap_cache_hits"
        ] == metrics["iterations"],
        "gpu_compute_requested": metrics["gpu_compute_requested"],
    }
    # gpu_compute_requested is the sole negative invariant.
    invariant_passes = {
        key: (not value if key == "gpu_compute_requested" else bool(value))
        for key, value in invariants.items()
    }
    return {
        "latency": checks,
        "throughput": throughput,
        "invariants": {
            key: {"observed": invariants[key], "passed": passed}
            for key, passed in invariant_passes.items()
        },
        "passed": (
            all(item["passed"] for item in checks.values())
            and throughput["passed"]
            and all(invariant_passes.values())
        ),
    }


def _build_rows(count: int) -> list:
    priorities = ("critical", "high", "medium", "low")
    return [
        {
            "idea_id": f"idea-{index:06d}",
            "title": f"Synthetic control-plane idea {index}",
            "priority": priorities[index % len(priorities)],
            "config_yaml": (
                "strategy: null\n"
                "train_mode: lora\n"
                f"seed: {index}\n"
            ),
            "raw_markdown": "",
            "status": "queued",
        }
        for index in range(count)
    ]


def run_benchmark(
    work_dir: Path,
    *,
    idea_count: int = MIN_ACCEPTANCE_IDEAS,
    queue_limit: int = 2_000,
    iterations: int = MIN_ACCEPTANCE_ITERATIONS,
    claim_workers: int = 8,
    targets: Optional[dict] = None,
) -> dict:
    if idea_count < 1:
        raise ValueError("idea_count must be positive")
    if queue_limit < 1:
        raise ValueError("queue_limit must be positive")
    if iterations < 3:
        raise ValueError("iterations must be at least 3")
    if claim_workers < 2:
        raise ValueError("claim_workers must be at least 2")

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    benchmark_started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="orze-harness-efficiency-", dir=work_dir
    ) as temporary:
        root = Path(temporary)
        results_dir = root / "results"
        results_dir.mkdir()
        db_path = root / "idea_lake.db"
        lake = IdeaLake(db_path)

        rows = _build_rows(idea_count)
        _, insert_ms = _timed(lambda: lake.bulk_insert(rows))
        lake.close()

        connection_reopen_samples = []
        schema_bootstrap_cache_hits = 0
        for _ in range(iterations):
            opened, elapsed = _timed(lambda: IdeaLake(db_path))
            connection_reopen_samples.append(elapsed)
            schema_bootstrap_cache_hits += int(
                opened.schema_bootstrap_cache_hit
            )
            opened.close()

        lake = IdeaLake(db_path)
        parser = _QueueParser()
        probe_rows = lake.conn.execute(
            "SELECT config_hash FROM ideas ORDER BY rowid LIMIT 64"
        ).fetchall()
        identity_probes = [row[0] for row in probe_rows]
        identity_coverage_count = lake.conn.execute(
            "SELECT COUNT(*) FROM ideas WHERE config_hash IS NOT NULL "
            "AND config_source_sha256 IS NOT NULL"
        ).fetchone()[0]
        identity_samples = []
        queue_samples = []
        sync_samples = []
        tick_samples = []
        order_digests = []
        identity_probe_matches = 0
        queue_count = 0

        # The first tick fills the exact-source parse cache. It is recorded as
        # cold-start information but excluded from steady-state p95.
        cold_tick_ms = None
        for tick in range(iterations + 1):
            tick_started = time.perf_counter()
            identities, identity_ms = _timed(
                lambda: lake.find_admitted_config_hashes(identity_probes)
            )
            queue_rows, queue_ms = _timed(
                lambda: lake.get_queue(limit=queue_limit)
            )

            sync_started = time.perf_counter()
            queue_ids = []
            ideas = {}
            for row in queue_rows:
                idea_id = row["idea_id"]
                queue_ids.append(idea_id)
                parsed = parser._parse_lake_queue_config(
                    idea_id, row["config"] or ""
                )
                ideas[idea_id] = {
                    "title": row["title"],
                    "priority": row["priority"],
                    "config": parsed,
                    "raw": "",
                }
            parser._prune_lake_queue_config_cache(queue_ids)
            unclaimed = get_unclaimed(ideas, results_dir)
            sync_ms = (time.perf_counter() - sync_started) * 1000.0
            tick_ms = (time.perf_counter() - tick_started) * 1000.0
            order_digests.append(hashlib.sha256(
                "\n".join(unclaimed).encode("utf-8")
            ).hexdigest())
            identity_probe_matches = len(identities)
            queue_count = len(queue_rows)
            if tick == 0:
                cold_tick_ms = round(tick_ms, 3)
            else:
                identity_samples.append(identity_ms)
                queue_samples.append(queue_ms)
                sync_samples.append(sync_ms)
                tick_samples.append(tick_ms)

        lifecycle_samples = []
        lifecycle_states_exact = True
        lifecycle_ids = [row["idea_id"] for row in queue_rows[:iterations]]
        for idea_id in lifecycle_ids:
            started = time.perf_counter()
            ok = claim(idea_id, results_dir, 4, lake=lake)
            ok = ok and lake.record_state_transition(
                idea_id, "CLAIMED", "IN_PROGRESS", reason="efficiency_benchmark"
            )
            ok = ok and lake.record_stage_transition(
                idea_id, "training", "IN_PROGRESS", "COMPLETE",
                "efficiency_benchmark",
            )
            ok = ok and lake.record_stage_transition(
                idea_id, "evaluation", "PENDING", "IN_PROGRESS",
                "efficiency_benchmark",
            )
            ok = ok and lake.record_state_transition(
                idea_id, "IN_PROGRESS", "COMPLETE",
                reason="efficiency_benchmark",
            )
            lifecycle_samples.append(
                (time.perf_counter() - started) * 1000.0
            )
            lifecycle_states_exact = lifecycle_states_exact and bool(ok) and (
                lake.get_fsm_state(idea_id) == "COMPLETE"
                and lake.get_stage_state(idea_id, "training") == "COMPLETE"
                and lake.get_stage_state(idea_id, "evaluation") == "COMPLETE"
            )

        race_id = "idea-atomic-claim-race"
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=claim_workers
        ) as executor:
            futures = [
                executor.submit(claim, race_id, results_dir, 4, None)
                for _ in range(claim_workers)
            ]
            atomic_claim_successes = sum(bool(item.result()) for item in futures)

        journal_mode = lake.conn.execute("PRAGMA journal_mode").fetchone()[0]
        lake.close()

    module_path = Path(__file__).resolve()
    source_root = module_path.parents[1]
    source_files = {
        "benchmark": module_path,
        "idea_lake": source_root / "idea_lake.py",
        "phases": source_root / "engine" / "phases.py",
        "scheduler": source_root / "engine" / "scheduler.py",
    }
    metrics = {
        "idea_count": idea_count,
        "queue_limit": queue_limit,
        "iterations": iterations,
        "bulk_insert_rows_per_second": round(
            idea_count / (insert_ms / 1000.0), 3
        ),
        "connection_reopen": _latency_summary(connection_reopen_samples),
        "schema_bootstrap_cache_hits": schema_bootstrap_cache_hits,
        "admitted_identity_lookup": _latency_summary(identity_samples),
        "queue_query": _latency_summary(queue_samples),
        "warm_queue_sync": _latency_summary(sync_samples),
        "steady_control_tick": _latency_summary(tick_samples),
        "cold_control_tick_ms": cold_tick_ms,
        "lifecycle_round_trip": _latency_summary(lifecycle_samples),
        "identity_coverage_count": identity_coverage_count,
        "identity_probe_count": len(identity_probes),
        "identity_probe_matches": identity_probe_matches,
        "queue_count": queue_count,
        "queue_order_stable": len(set(order_digests)) == 1,
        "lifecycle_states_exact": lifecycle_states_exact,
        "atomic_claim_workers": claim_workers,
        "atomic_claim_successes": atomic_claim_successes,
        "gpu_compute_requested": False,
    }
    target_evaluation = evaluate_targets(metrics, targets)
    acceptance_scale = (
        idea_count >= MIN_ACCEPTANCE_IDEAS
        and iterations >= MIN_ACCEPTANCE_ITERATIONS
    )
    passed = target_evaluation["passed"] and acceptance_scale
    try:
        orze_version = metadata.version("orze")
    except metadata.PackageNotFoundError:
        orze_version = "source-checkout"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "VERIFIED" if passed else (
            "DIAGNOSTIC" if target_evaluation["passed"] else "FAILED"
        ),
        "benchmark": "orze_cpu_control_plane",
        "verified_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "duration_seconds": round(time.perf_counter() - benchmark_started, 3),
        "scope": {
            "minimum_acceptance_ideas": MIN_ACCEPTANCE_IDEAS,
            "minimum_acceptance_iterations": MIN_ACCEPTANCE_ITERATIONS,
            "acceptance_scale_met": acceptance_scale,
            "filesystem_work_dir": str(work_dir),
            "accelerator_access": "none",
            "model_or_evaluation_executed": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            "orze": orze_version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "sqlite_journal_mode": journal_mode,
        },
        "source_sha256": {
            name: _sha256(path) for name, path in source_files.items()
        },
        "metrics": metrics,
        "targets": target_evaluation,
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir", type=Path, default=Path.cwd(),
        help="filesystem on which the disposable database is benchmarked",
    )
    parser.add_argument("--ideas", type=int, default=MIN_ACCEPTANCE_IDEAS)
    parser.add_argument("--queue-limit", type=int, default=2_000)
    parser.add_argument(
        "--iterations", type=int, default=MIN_ACCEPTANCE_ITERATIONS
    )
    parser.add_argument("--claim-workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = run_benchmark(
        args.work_dir,
        idea_count=args.ideas,
        queue_limit=args.queue_limit,
        iterations=args.iterations,
        claim_workers=args.claim_workers,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(args.output, rendered)
    print(rendered, end="")
    return 0 if receipt["status"] in {"VERIFIED", "DIAGNOSTIC"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
