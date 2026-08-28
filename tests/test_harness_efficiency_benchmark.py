from orze.benchmarks.harness_efficiency import (
    DEFAULT_TARGETS_MS,
    evaluate_targets,
    run_benchmark,
)


def test_small_run_is_diagnostic_but_exercises_real_control_plane(tmp_path):
    receipt = run_benchmark(
        tmp_path,
        idea_count=100,
        queue_limit=50,
        iterations=3,
        claim_workers=4,
        targets={key: 10_000.0 for key in DEFAULT_TARGETS_MS},
    )

    assert receipt["status"] == "DIAGNOSTIC"
    assert receipt["scope"]["acceptance_scale_met"] is False
    assert receipt["scope"]["accelerator_access"] == "none"
    assert receipt["scope"]["model_or_evaluation_executed"] is False
    assert receipt["metrics"]["identity_coverage_count"] == 100
    assert receipt["metrics"]["identity_probe_matches"] == 64
    assert receipt["metrics"]["queue_count"] == 50
    assert receipt["metrics"]["queue_order_stable"] is True
    assert receipt["metrics"]["lifecycle_states_exact"] is True
    assert receipt["metrics"]["atomic_claim_successes"] == 1
    assert receipt["targets"]["passed"] is True
    assert receipt["targets"]["throughput"]["passed"] is True


def test_target_evaluation_fails_closed_on_latency_and_gpu_use():
    metric_names = {
        "cold_open": 1.0,
        "admitted_identity_lookup": 1.0,
        "queue_query": 1.0,
        "warm_queue_sync": 1.0,
        "steady_control_tick": 1.0,
        "lifecycle_round_trip": 1.0,
    }
    metrics = {
        name: {"p95_ms": value} for name, value in metric_names.items()
    }
    metrics.update({
        "bulk_insert_rows_per_second": 2_000.0,
        "idea_count": 10,
        "identity_coverage_count": 10,
        "identity_probe_count": 10,
        "identity_probe_matches": 10,
        "queue_limit": 10,
        "queue_count": 10,
        "queue_order_stable": True,
        "lifecycle_states_exact": True,
        "atomic_claim_successes": 1,
        "gpu_compute_requested": True,
    })
    targets = {key: 0.5 for key in DEFAULT_TARGETS_MS}

    result = evaluate_targets(metrics, targets)
    assert result["passed"] is False
    assert all(not item["passed"] for item in result["latency"].values())
    assert result["invariants"]["gpu_compute_requested"] == {
        "observed": True,
        "passed": False,
    }
