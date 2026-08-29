"""Acceptance benchmark for blocked launch-policy feedback latency.

The benchmark repeatedly starts the real CLI process and requires a stable,
machine-readable blocked result.  It does not run the comprehensive preflight
and does not grant accelerator visibility to the subprocess.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Callable, Optional, Sequence

from orze.core.fs import atomic_write


SCHEMA_VERSION = 1
MIN_ACCEPTANCE_RUNS = 20
DEFAULT_MAX_P95_MS = 250.0
_EXPECTED_RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "launch_allowed_by_policy",
    "blockers",
    "configured_physical_scope",
    "checks",
    "accelerator_access",
    "accelerator_compute_access",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * probability))
    return ordered[min(rank, len(ordered)) - 1]


def _source_manifest() -> dict:
    package = Path(__file__).resolve().parents[1]
    paths = {
        "benchmarks/launch_policy_latency.py": Path(__file__).resolve(),
        "cli.py": package / "cli.py",
        "cli_setup.py": package / "cli_setup.py",
    }
    files = {name: _sha256(path) for name, path in paths.items()}
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "files": files,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def run_benchmark(
    config_path: Path,
    *,
    expected_scope: Sequence[int],
    expected_blockers: Sequence[str],
    runs: int = 30,
    max_p95_ms: float = DEFAULT_MAX_P95_MS,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Run the exact CLI path and derive a fail-closed latency receipt."""
    config_path = Path(config_path).resolve(strict=True)
    scope = list(expected_scope)
    blockers = list(expected_blockers)
    if (runs < 1 or not math.isfinite(max_p95_ms) or max_p95_ms <= 0
            or not scope or len(scope) != len(set(scope))
            or any(isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0
                   for gpu in scope)
            or len(blockers) != len(set(blockers))
            or any(not isinstance(reason, str) or not reason
                   for reason in blockers)):
        raise ValueError("launch_policy_benchmark_contract_invalid")

    source_before = _source_manifest()
    config_sha256_before = _sha256(config_path)
    runtime_source_root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable, str(runtime_source_root / "orze/cli.py"),
        "-c", str(config_path),
        "--launch-status",
    ]
    env = dict(os.environ)
    # Do not inherit an import overlay: the child must execute the exact source
    # tree whose hashes are recorded below.
    env["PYTHONPATH"] = str(runtime_source_root)
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "none",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
    })
    samples = []
    observations = []
    failures = []
    for index in range(runs):
        started = clock()
        try:
            completed = runner(
                command,
                cwd=config_path.parent,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=10,
            )
            elapsed_ms = (clock() - started) * 1000.0
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(
                f"run_{index}:subprocess_{type(exc).__name__}")
            continue
        samples.append(elapsed_ms)
        stderr_sha256 = hashlib.sha256(
            completed.stderr.encode("utf-8"),
        ).hexdigest()
        try:
            payload = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError):
            failures.append(f"run_{index}:stdout_json_invalid")
            continue
        observations.append({
            "exit_code": completed.returncode,
            "status": payload.get("status") if isinstance(payload, dict)
            else None,
            "stderr_sha256": stderr_sha256,
        })
        if not isinstance(payload, dict) or set(payload) != _EXPECTED_RECEIPT_FIELDS:
            failures.append(f"run_{index}:receipt_fields_invalid")
            continue
        expected = {
            "schema_version": 1,
            "status": "BLOCKED",
            "launch_allowed_by_policy": False,
            "blockers": blockers,
            "configured_physical_scope": scope,
            "checks": {
                "stop_pause_policy": "complete",
                "full_preflight": "not_run",
            },
            "accelerator_access": "none",
            "accelerator_compute_access": "none",
        }
        if completed.returncode != 2:
            failures.append(f"run_{index}:exit_code_not_blocked")
        if payload != expected:
            failures.append(f"run_{index}:receipt_content_mismatch")

    source_after = _source_manifest()
    config_sha256_after = _sha256(config_path)
    if source_after != source_before:
        failures.append("benchmark_source_changed_during_run")
    if config_sha256_after != config_sha256_before:
        failures.append("config_changed_during_run")

    latency = None
    if samples:
        latency = {
            "samples": len(samples),
            "samples_ms": [round(value, 3) for value in samples],
            "median_ms": round(statistics.median(samples), 3),
            "p95_ms": round(_percentile(samples, 0.95), 3),
            "max_ms": round(max(samples), 3),
        }
    acceptance_scale = runs >= MIN_ACCEPTANCE_RUNS
    complete = len(samples) == runs and len(observations) == runs
    latency_passed = bool(latency and latency["p95_ms"] <= max_p95_ms)
    passed = acceptance_scale and complete and latency_passed and not failures
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "VERIFIED" if passed else "FAILED",
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc,
        ).isoformat(),
        "scope": {
            "config_path": str(config_path),
            "expected_physical_scope": scope,
            "expected_blockers": blockers,
            "requested_runs": runs,
            "minimum_acceptance_runs": MIN_ACCEPTANCE_RUNS,
            "acceptance_scale_met": acceptance_scale,
            "accelerator_access": "none",
            "accelerator_compute_access": "none",
            "full_preflight_performed": False,
        },
        "metrics": {"process_latency": latency},
        "target": {
            "blocked_launch_status_p95_ms_maximum": max_p95_ms,
            "actual_p95_ms": latency["p95_ms"] if latency else None,
            "passed": latency_passed,
            "provenance": (
                "operator-response engineering SLO adopted after the initial "
                "diagnostic; not a preregistered comparative claim"
            ),
        },
        "invariants": {
            "all_runs_observed": complete,
            "all_receipts_exact": not failures,
        },
        "failures": failures,
        "observations": observations,
        "input_identity": {
            "config_sha256": config_sha256_before,
            "runtime_source_root": str(runtime_source_root),
        },
        "source_manifest": source_before,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify blocked launch-policy feedback latency",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--physical-scope", type=int, nargs="+", required=True)
    parser.add_argument("--expected-blocker", action="append", required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--max-p95-ms", type=float, default=DEFAULT_MAX_P95_MS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = run_benchmark(
        args.config,
        expected_scope=args.physical_scope,
        expected_blockers=args.expected_blocker,
        runs=args.runs,
        max_p95_ms=args.max_p95_ms,
    )
    rendered = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    if args.output:
        atomic_write(args.output, rendered)
    print(rendered, end="")
    return 0 if receipt["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
