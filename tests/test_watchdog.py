"""F3: zombie/stuck-training watchdog.

Two scenarios:
  (a) A subprocess that does nothing (sleeps) and produces no log/CPU/GPU
      activity — must be flagged as stuck after WATCHDOG_CONSECUTIVE samples.
  (b) A subprocess whose log keeps growing (batch lines) — must NEVER be
      flagged stuck.

We don't want to wait minutes in tests, so we shrink the watchdog
constants via env vars before importing the module-level code that
reads them, and monkeypatch the GPU + CPU samplers.
"""
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


# Shrink watchdog before module import so module-level constants are tiny.
os.environ["ORZE_WD_GRACE_MIN"] = "0"          # no grace
os.environ["ORZE_WD_CONSECUTIVE"] = "3"        # 3 consecutive samples
os.environ["ORZE_WD_GPU_UTIL"] = "5"
os.environ["ORZE_WD_CPU_DELTA_JIFFIES"] = "100"

# Force re-import so constants are re-read.
for _mod in list(sys.modules):
    if _mod.startswith("orze.engine.launcher"):
        del sys.modules[_mod]

from orze.engine import launcher as L  # noqa: E402


def _mk_tp(tmp_path: Path, pid: int = 12345):
    log = tmp_path / "train_output.log"
    log.write_text("")
    fake_proc = SimpleNamespace(pid=pid, poll=lambda: None)
    return SimpleNamespace(
        idea_id="idea-test",
        process=fake_proc,
        start_time=time.time() - 10.0,
        log_path=log,
        timeout=3600.0,
    )


def test_watchdog_kills_idle_process(tmp_path, monkeypatch):
    """Process with 0% GPU + frozen log + flat CPU is killed."""
    tp = _mk_tp(tmp_path)
    # Stub samplers: GPU idle, CPU flat.
    monkeypatch.setattr(L, "_gpu_util_for_pid", lambda pid: 0)
    monkeypatch.setattr(L, "_tree_cpu_jiffies", lambda pid: 100)

    # Sample N+1 times — the first establishes baseline, then N consecutive
    # bad samples must trigger.
    results = [L._watchdog_check(tp) for _ in range(L.WATCHDOG_CONSECUTIVE + 1)]
    assert results[0] is False  # baseline
    assert results[-1] is True   # stuck
    assert tp._wd_bad_count >= L.WATCHDOG_CONSECUTIVE


def test_watchdog_spares_active_process(tmp_path, monkeypatch):
    """Process with growing log is never killed."""
    tp = _mk_tp(tmp_path)
    monkeypatch.setattr(L, "_gpu_util_for_pid", lambda pid: 0)

    cpu = [100]
    def cpu_jiffies(pid):
        cpu[0] += 1000  # simulate CPU progress
        return cpu[0]
    monkeypatch.setattr(L, "_tree_cpu_jiffies", cpu_jiffies)

    for i in range(L.WATCHDOG_CONSECUTIVE * 3):
        # Touch log mtime forward (and append bytes for good measure).
        with tp.log_path.open("a") as f:
            f.write(f"batch {i}/100 loss=0.5\n")
        new_t = time.time() + i + 1
        os.utime(tp.log_path, (new_t, new_t))
        assert L._watchdog_check(tp) is False


def test_watchdog_grace_period_blocks_kill(tmp_path, monkeypatch):
    """If grace period not yet elapsed AND no first batch seen, watchdog
    cannot fire."""
    monkeypatch.setattr(L, "WATCHDOG_GRACE_MIN", 60)
    tp = _mk_tp(tmp_path)
    tp.start_time = time.time()  # just launched
    monkeypatch.setattr(L, "_gpu_util_for_pid", lambda pid: 0)
    monkeypatch.setattr(L, "_tree_cpu_jiffies", lambda pid: 100)
    for _ in range(L.WATCHDOG_CONSECUTIVE + 5):
        assert L._watchdog_check(tp) is False


def test_watchdog_first_batch_marker_activates(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "WATCHDOG_GRACE_MIN", 60)
    tp = _mk_tp(tmp_path)
    tp.start_time = time.time()  # just launched
    tp.log_path.write_text("epoch 1 step 5 loss=0.4\n")
    monkeypatch.setattr(L, "_gpu_util_for_pid", lambda pid: 0)
    monkeypatch.setattr(L, "_tree_cpu_jiffies", lambda pid: 100)
    # Now activated via first-batch marker.
    results = [L._watchdog_check(tp) for _ in range(L.WATCHDOG_CONSECUTIVE + 1)]
    assert results[0] is False  # baseline
    assert results[-1] is True


def test_first_batch_scanner(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("loading dataset...\n")
    assert L._scan_first_batch_marker(log) is False
    log.write_text("Epoch 0 batch 1/100 loss=0.5\n")
    assert L._scan_first_batch_marker(log) is True
    log.write_text("step 42 done\n")
    assert L._scan_first_batch_marker(log) is True


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc only")
def test_real_sleep_subprocess_marked_stuck(tmp_path, monkeypatch):
    """Integration-ish: spawn a real `sleep` subprocess; with stubbed
    GPU samplers (no real nvidia-smi), watchdog must mark it stuck.
    """
    proc = subprocess.Popen(
        ["sleep", "30"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        log = tmp_path / "log.txt"
        log.write_text("epoch 0 batch 1/10\n")  # marker -> activate
        tp = SimpleNamespace(
            idea_id="idea-sleep",
            process=proc,
            start_time=time.time() - 5,
            log_path=log,
            timeout=3600.0,
        )
        monkeypatch.setattr(L, "_gpu_util_for_pid", lambda pid: 0)
        # CPU jiffies flat (sleep doesn't burn CPU).
        for _ in range(L.WATCHDOG_CONSECUTIVE + 2):
            stuck = L._watchdog_check(tp)
        assert stuck is True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
