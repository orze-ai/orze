"""Cross-process GPU ownership must be exclusive and crash-safe."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import orze.lifecycle as lifecycle
from orze.core.gpu_lease import (
    GpuLeaseError,
    acquire_gpu_leases,
    gpu_execution_lease,
    safe_gpu_lease_reason,
)
from orze.engine.smoke_test import _find_free_gpu
from orze.engine import launcher
from orze.hardware.gpu import detect_all_gpus


def _gpu(offset: int) -> int:
    # Physical IDs are identifiers to the lease layer; no GPU is queried.
    return 900_000 + (os.getpid() % 10_000) * 10 + offset


def _attempt_in_fresh_process(gpu: int) -> subprocess.CompletedProcess:
    code = (
        "from orze.core.gpu_lease import acquire_gpu_leases; "
        f"lease=acquire_gpu_leases([{gpu}]); lease.close()"
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=_child_env())


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
    return env


def test_overlapping_controllers_fail_closed_but_disjoint_scopes_coexist():
    held = acquire_gpu_leases([_gpu(1), _gpu(2)])
    try:
        overlap = _attempt_in_fresh_process(_gpu(2))
        disjoint = _attempt_in_fresh_process(_gpu(3))
    finally:
        held.close()

    assert overlap.returncode != 0
    assert "gpu_lease_contended" in overlap.stderr
    assert disjoint.returncode == 0


def test_cli_lease_reason_never_echoes_arbitrary_exception_content():
    assert safe_gpu_lease_reason(GpuLeaseError(
        "gpu_lease_contended: physical_gpu=4")) == (
            "gpu_lease_contended: physical_gpu=4")
    assert safe_gpu_lease_reason(GpuLeaseError(
        "secret path and token")) == "gpu_lease_rejected"


def test_multi_gpu_acquisition_releases_partial_scope_on_contention():
    code = (
        "import sys; "
        "from orze.core.gpu_lease import acquire_gpu_leases; "
        f"lease=acquire_gpu_leases([{_gpu(2)}]); "
        "print('ready', flush=True); sys.stdin.buffer.read(1)"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True, env=_child_env())
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "ready"
    try:
        with pytest.raises(GpuLeaseError, match="gpu_lease_contended"):
            acquire_gpu_leases([_gpu(1), _gpu(2)])
        # The failed multi-GPU attempt must not retain its first device.
        probe = acquire_gpu_leases([_gpu(1)])
        probe.close()
    finally:
        assert holder.stdin is not None
        holder.stdin.write("x")
        holder.stdin.close()
        holder.wait(timeout=5)


def test_inherited_descriptor_keeps_lease_after_launcher_releases_parent_copy():
    with gpu_execution_lease(_gpu(4)) as lease_fds:
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
            stdin=subprocess.PIPE, pass_fds=lease_fds)

    try:
        overlap = _attempt_in_fresh_process(_gpu(4))
        assert overlap.returncode != 0
        assert "gpu_lease_contended" in overlap.stderr
    finally:
        assert child.stdin is not None
        child.stdin.write(b"x")
        child.stdin.close()
        child.wait(timeout=5)

    assert _attempt_in_fresh_process(_gpu(4)).returncode == 0


def test_smoke_gpu_probe_is_restricted_to_explicit_scope(monkeypatch):
    observed = {}

    def fake_run(command, **_kwargs):
        observed["command"] = command
        return subprocess.CompletedProcess(
            command, 0, stdout="4, 80000\n7, 79000\n", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    selected = _find_free_gpu({
        "_managed_gpu_ids": [4, 7],
        "gpu_mem_threshold": 40000,
    })

    assert selected == 4
    assert observed["command"][0:3] == ["nvidia-smi", "-i", "4,7"]


def test_smoke_gpu_probe_without_explicit_scope_uses_no_telemetry(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: pytest.fail("must not query implicit GPUs"),
    )
    assert _find_free_gpu({}) is None


def test_process_watchdog_telemetry_is_restricted_to_assigned_gpu(monkeypatch):
    calls = []
    responses = iter([
        subprocess.CompletedProcess([], 0, stdout="123, GPU-abc\n"),
        subprocess.CompletedProcess([], 0, stdout="GPU-abc, 17\n"),
    ])

    def fake_run(command, **_kwargs):
        calls.append(command)
        return next(responses)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    assert launcher._gpu_util_for_pid(123, 7) == 17
    assert len(calls) == 2
    assert all("--id=7" in command for command in calls)


def test_zombie_telemetry_is_restricted_to_assigned_gpu(
        tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, "_tree_cpu_jiffies", lambda pid: 0)

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    tp = SimpleNamespace(
        gpu=4,
        process=SimpleNamespace(pid=123),
        log_path=tmp_path / "absent.log",
    )

    assert launcher._detect_zombie(tp) is False
    assert calls[0][0:2] == ["nvidia-smi", "--id=4"]


def test_gpu_detection_can_be_restricted_to_explicit_scope(monkeypatch):
    observed = {}

    def fake_run(command, **_kwargs):
        observed["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="4\n7\n")

    monkeypatch.setattr("orze.hardware.gpu.subprocess.run", fake_run)

    assert detect_all_gpus([7, 4]) == [4, 7]
    assert "--id=4,7" in observed["command"]


def test_stop_orphan_probe_can_be_restricted_to_explicit_scope(monkeypatch):
    observed = {}

    def fake_run(command, **_kwargs):
        observed["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(lifecycle.subprocess, "run", fake_run)

    lifecycle._cleanup_gpu_orphans("/project", [7, 4])
    assert "--id=4,7" in observed["command"]
