"""F3: zombie/stuck-training watchdog."""
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from orze.engine import launcher as L


@pytest.fixture(autouse=True)
def _shrink_watchdog(monkeypatch):
    monkeypatch.setattr(L, "WATCHDOG_GRACE_MIN", 0)
    monkeypatch.setattr(L, "WATCHDOG_CONSECUTIVE", 3)
    monkeypatch.setattr(L, "WATCHDOG_GPU_UTIL_THRESHOLD", 5)
    monkeypatch.setattr(L, "WATCHDOG_CPU_DELTA_JIFFIES", 100)


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
    tp = _mk_tp(tmp_path)
    monkeypatch.setattr(L, "_gpu_util_for_pid", lambda pid: 0)
    monkeypatch.setattr(L, "_tree_cpu_jiffies", lambda pid: 100)

    results = [L._watchdog_check(tp) for _ in range(L.WATCHDOG_CONSECUTIVE + 1)]
    assert results[0] is False
    assert results[-1] is True
    assert tp._wd_bad_count >= L.WATCHDOG_CONSECUTIVE


def test_watchdog_spares_active_process(tmp_path, monkeypatch):
    tp = _mk_tp(tmp_path)
    monkeypatch.setattr(L, "_gpu_util_for_pid", lambda pid: 0)

    cpu = [100]
    def cpu_jiffies(pid):
        cpu[0] += 1000
        return cpu[0]
    monkeypatch.setattr(L, "_tree_cpu_jiffies", cpu_jiffies)

    for i in range(L.WATCHDOG_CONSECUTIVE * 3):
        with tp.log_path.open("a") as f:
            f.write(f"batch {i}/100 loss=0.5\n")
        new_t = time.time() + i + 1
        os.utime(tp.log_path, (new_t, new_t))
        assert L._watchdog_check(tp) is False


def test_watchdog_grace_period_blocks_kill(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "WATCHDOG_GRACE_MIN", 60)
    tp = _mk_tp(tmp_path)
    tp.start_time = time.time()
    monkeypatch.setattr(L, "_gpu_util_for_pid", lambda pid: 0)
    monkeypatch.setattr(L, "_tree_cpu_jiffies", lambda pid: 100)
    for _ in range(L.WATCHDOG_CONSECUTIVE + 5):
        assert L._watchdog_check(tp) is False


def test_watchdog_first_batch_marker_activates(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "WATCHDOG_GRACE_MIN", 60)
    tp = _mk_tp(tmp_path)
    tp.start_time = time.time()
    tp.log_path.write_text("epoch 1 step 5 loss=0.4\n")
    monkeypatch.setattr(L, "_gpu_util_for_pid", lambda pid: 0)
    monkeypatch.setattr(L, "_tree_cpu_jiffies", lambda pid: 100)
    results = [L._watchdog_check(tp) for _ in range(L.WATCHDOG_CONSECUTIVE + 1)]
    assert results[0] is False
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
    proc = subprocess.Popen(
        ["sleep", "30"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        log = tmp_path / "log.txt"
        log.write_text("epoch 0 batch 1/10\n")
        tp = SimpleNamespace(
            idea_id="idea-sleep",
            process=proc,
            start_time=time.time() - 5,
            log_path=log,
            timeout=3600.0,
        )
        monkeypatch.setattr(L, "_gpu_util_for_pid", lambda pid: 0)
        stuck = False
        for _ in range(L.WATCHDOG_CONSECUTIVE + 2):
            stuck = L._watchdog_check(tp)
        assert stuck is True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_tree_cpu_jiffies_survives_vanished_pid(monkeypatch):
    """Regression (daemon crash 2026-06-20): a pid that exits mid-read makes
    /proc raise ProcessLookupError (ESRCH) — a sibling of FileNotFoundError under
    OSError, NOT caught by the old (FileNotFoundError, ...) handler. It must
    contribute 0 jiffies, never propagate."""
    import builtins
    real_open = builtins.open

    def boom(path, *a, **k):
        if isinstance(path, str) and path.startswith("/proc/"):
            raise ProcessLookupError(3, "No such process")
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", boom)
    assert L._tree_cpu_jiffies(1183080) == 0  # no exception raised


def test_detect_zombie_survives_proc_lookup_error(tmp_path, monkeypatch):
    """Regression: even if the CPU-tree probe raises ProcessLookupError, the
    zombie check degrades to 'assume alive' (False) instead of killing run()."""
    tp = _mk_tp(tmp_path)

    def raise_esrch(pid):
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(L, "_tree_cpu_jiffies", raise_esrch)
    assert L._detect_zombie(tp) is False
