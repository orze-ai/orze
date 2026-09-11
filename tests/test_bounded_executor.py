"""New legacy synchronous executor mechanisms, not native repair permission."""
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from orze.engine import bounded_executor as executor
from orze.engine.supervised_process import SupervisionUncertain, prepare_supervised


def run(tmp_path, source, **kwargs):
    return executor.run_bounded_executor([sys.executable, "-c", source],
        timeout=kwargs.pop("timeout", 5), env=dict(os.environ), cwd=tmp_path, **kwargs)


def test_real_utf8_output_and_private_ready_precede_any_worker_code(tmp_path):
    marker = tmp_path / "ran"
    events = []
    result = run(tmp_path,
        "import pathlib,sys; pathlib.Path('ran').write_text('yes'); "
        "sys.stdout.buffer.write('snow 雪\\r\\n'.encode()); sys.stderr.write('error\\n')",
        before_start=lambda: events.append(marker.exists()))
    assert events == [False]
    assert marker.read_text() == "yes"
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0 and result.output_complete is True and result.stopped is False
    assert result.stdout == "snow 雪\n" and result.stderr == "error\n"
    assert result.process_tree["event"] == "TREE_CLOSED"
    assert result.process_tree["wait_proof"] == "ECHILD_WALL"
    identity = result.process_tree["binding"]["identity"]
    assert identity["kind"] == "bounded_executor" and "attempt_ref" not in identity
    executor.require_executor_scope_clear(tmp_path)


@pytest.mark.parametrize("extra", [0, 1])
def test_output_limit_is_byte_exact_and_tail_marker_does_not_hide_truncation(tmp_path, extra):
    suffix = b"\nFIX_APPLIED\n"
    size = executor.MAX_OUTPUT_BYTES + extra
    result = run(tmp_path,
        f"import sys; sys.stdout.buffer.write(b'x'*{size - len(suffix)} + {suffix!r})")
    assert result.returncode == 0 and result.stopped is False
    assert result.stdout.endswith("\nFIX_APPLIED\n")
    assert result.stdout_bytes == size
    assert len(result.stdout.encode()) <= executor.MAX_OUTPUT_BYTES
    assert result.output_complete is (extra == 0)


def test_invalid_utf8_is_diagnostic_only_even_with_a_fix_marker(tmp_path):
    result = run(tmp_path, "import sys; sys.stdout.buffer.write(b'\\xff\\nFIX_APPLIED\\n')")
    assert result.returncode == 0 and "FIX_APPLIED" in result.stdout
    assert result.output_complete is False
    executor.require_executor_scope_clear(tmp_path)


@pytest.mark.parametrize("budget", [False, 0, float("nan")])
def test_invalid_budget_never_enters_prepare_or_poison_scope(tmp_path, budget):
    calls = []
    with pytest.raises(ValueError):
        run(tmp_path, "pass", timeout=budget, prepare=lambda *a, **k: calls.append(True))
    assert calls == []
    executor.require_executor_scope_clear(tmp_path)


def test_missing_executable_is_known_before_prepare_and_does_not_hold_scope(tmp_path):
    calls = []
    with pytest.raises(FileNotFoundError):
        executor.run_bounded_executor([str(tmp_path / "missing")], timeout=1,
            env={}, cwd=tmp_path, prepare=lambda *a, **k: calls.append(True))
    assert calls == []
    assert run(tmp_path, "pass").returncode == 0


def test_real_timeout_stop_zero_is_confirmed_once_but_not_a_success_result(tmp_path):
    stops, handles = [], []

    def prepare(*args, **kwargs):
        process = prepare_supervised(*args, **kwargs)
        handles.append(process)
        original = process.stop
        def stop(*a, **k):
            stops.append(True)
            return original(*a, **k)
        process.stop = stop
        return process

    try:
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            run(tmp_path,
                "import signal,time,sys; signal.signal(signal.SIGTERM,lambda *_:sys.exit(0)); "
                "print('FIX_APPLIED',flush=True); time.sleep(30)", timeout=0.3, prepare=prepare)
        assert caught.value.returncode == 0 and caught.value.stopped is True
        assert caught.value.process_tree["worker_returncode"] == 0
        assert caught.value.process_tree["stop_requested"] is True
        assert stops == [True]
        executor.require_executor_scope_clear(tmp_path)
    finally:
        for handle in handles:
            if handle.poll() is None:
                handle.stop()


def test_unknown_prepare_latches_canonical_path_even_after_directory_replacement(tmp_path):
    scope = tmp_path / "project"
    scope.mkdir()
    calls = []
    def uncertain(*args, **kwargs):
        calls.append(True)
        # Explicit unknown handoff fault; no fake closed-process fact.
        raise SupervisionUncertain("synthetic_handoff_unknown")
    with pytest.raises(executor.BoundedExecutorHOLD):
        run(scope, "pass", prepare=uncertain)
    original = scope.stat().st_ino
    scope.rename(tmp_path / "retired")
    scope.mkdir()
    assert scope.stat().st_ino != original
    with pytest.raises(executor.BoundedExecutorHOLD):
        run(scope, "pass", prepare=uncertain)
    with pytest.raises(executor.BoundedExecutorHOLD):
        executor.require_executor_scope_clear(scope)
    assert calls == [True]


def test_concurrent_scope_refusal_does_not_poison_the_confirmed_owner(tmp_path):
    ready, release = threading.Event(), threading.Event()
    results, errors = [], []
    def before_start():
        ready.set()
        assert release.wait(5)
    def owner():
        try:
            results.append(run(tmp_path, "print('first')", before_start=before_start))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=owner)
    thread.start()
    try:
        assert ready.wait(5)
        with pytest.raises(executor.BoundedExecutorHOLD):
            run(tmp_path, "raise AssertionError('must not run')")
    finally:
        release.set()
        thread.join(8)
    assert not thread.is_alive() and errors == []
    assert len(results) == 1 and results[0].returncode == 0
    assert run(tmp_path, "print('second')").stdout == "second\n"
