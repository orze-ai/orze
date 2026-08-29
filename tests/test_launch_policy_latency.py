import json
import subprocess

from orze.benchmarks.launch_policy_latency import run_benchmark


def _clock():
    value = -0.01

    def tick():
        nonlocal value
        value += 0.01
        return value

    return tick


def _runner(payload, exit_code=2):
    def run(command, **kwargs):
        assert command[1].endswith("/orze/src/orze/cli.py")
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["NVIDIA_VISIBLE_DEVICES"] == "none"
        assert kwargs["env"]["PYTHONPATH"].split(":", 1)[0].endswith(
            "/orze/src"
        )
        return subprocess.CompletedProcess(
            args=[], returncode=exit_code,
            stdout=json.dumps(payload), stderr="diagnostic log",
        )

    return run


def _payload():
    return {
        "schema_version": 1,
        "status": "BLOCKED",
        "launch_allowed_by_policy": False,
        "blockers": ["sentinel:.orze_disabled"],
        "configured_physical_scope": [4, 5, 6, 7],
        "checks": {
            "stop_pause_policy": "complete",
            "full_preflight": "not_run",
        },
        "accelerator_access": "none",
        "accelerator_compute_access": "none",
    }


def test_benchmark_verifies_exact_blocked_receipts_at_acceptance_scale(tmp_path):
    config = tmp_path / "orze.yaml"
    config.write_text("{}\n", encoding="utf-8")

    receipt = run_benchmark(
        config,
        expected_scope=[4, 5, 6, 7],
        expected_blockers=["sentinel:.orze_disabled"],
        runs=20,
        max_p95_ms=11.0,
        runner=_runner(_payload()),
        clock=_clock(),
    )

    assert receipt["status"] == "VERIFIED"
    assert receipt["metrics"]["process_latency"]["p95_ms"] == 10.0
    assert receipt["scope"]["accelerator_access"] == "none"
    assert receipt["scope"]["full_preflight_performed"] is False


def test_benchmark_fails_closed_on_one_false_receipt(tmp_path):
    config = tmp_path / "orze.yaml"
    config.write_text("{}\n", encoding="utf-8")
    payload = _payload()
    payload["status"] = "POLICY_ALLOWS_LAUNCH"

    receipt = run_benchmark(
        config,
        expected_scope=[4, 5, 6, 7],
        expected_blockers=["sentinel:.orze_disabled"],
        runs=20,
        max_p95_ms=11.0,
        runner=_runner(payload),
        clock=_clock(),
    )

    assert receipt["status"] == "FAILED"
    assert len(receipt["failures"]) == 20
    assert all(reason.endswith("receipt_content_mismatch")
               for reason in receipt["failures"])
