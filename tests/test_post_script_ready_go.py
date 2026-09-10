"""Independent C2d1 mechanisms through the real post-script public consumer.

The source event uses the shared explicit evaluation OS double with real
SQLite/receipts. Target workers, READY/GO, waits and STOP are real tiny CPU
processes; GPU allocation and policy/lease fault seams are explicit doubles.
The private-channel loss is injected on an actual supervisor socket, not a
fabricated successful wait. Cleanup uses only captured owned handles/pidfds.
These are new protocol mechanisms, not missing-API or old-release reds.
"""
from contextlib import contextmanager
import json
import time

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import evaluator
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.supervised_process import SupervisedProcess, SupervisionUncertain
from test_native_post_script_tree_completion import cpu_post_script, project
from test_stale_evaluation_completion import _lifecycle


def _script(c, tmp_path, *, pause=False, timeout=20):
    marker = tmp_path / "post-script-user-code-ran"
    c.script.write_text(
        "from pathlib import Path\nimport signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))\n"
        "Path(" + repr(str(marker)) + ").write_text('executed')\n"
        + ("while True: time.sleep(0.05)\n" if pause else ""), encoding="utf-8")
    c.cfg["post_scripts"] = [{"name": "owned-ready", "script": str(c.script),
                              "timeout": timeout}]
    return marker


def _run(c):
    return evaluator.run_post_scripts(c.folder.name, 0, c.results, c.cfg,
                                      lake=c.lake, source_event=c.event)


def _source(c):
    return (_lifecycle(c.lake),
            current_attempt(c.lake.conn, c.folder.name, c.event.attempt_ref.phase),
            (c.folder / "metrics.json").read_bytes())


def _capture(monkeypatch, handles, *, after_ready=None):
    actual = evaluator.prepare_supervised

    def prepare(*args, **kwargs):
        child = actual(*args, **kwargs)
        assert isinstance(child, SupervisedProcess)
        assert child.poll() is None
        assert child.binding["worker"]["pid"] == child.pid
        handles.append(child)
        if after_ready is not None:
            after_ready(child)
        return child

    monkeypatch.setattr(evaluator, "prepare_supervised", prepare)


def _started(c, child):
    row = current_attempt(c.lake.conn, c.folder.name, "post_script")
    assert row["state"] == "RUNNING"
    assert row["terminal"] is None
    assert row["binding"]["supervision"] == child.binding
    receipts = c.folder / "_compute_receipts" / row["attempt_id"]
    start = json.loads((receipts / "start.json").read_text())
    assert start["phase"] == "post_script"
    assert start["process_pid"] == child.pid
    assert not (receipts / "terminal.json").exists()
    return row


def _closed_stop(child):
    child.wait(timeout=5)
    receipt = child.closure_receipt()
    assert receipt["stop_requested"] is True
    assert receipt["wait_proof"] == "ECHILD_WALL"
    return receipt


def _wait_marker(marker):
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert marker.exists(), "actual user code must execute before the injected boundary"


@pytest.mark.parametrize("policy", ["runtime", "launch"])
def test_post_script_ready_requires_fresh_policy(cpu_post_script, tmp_path, monkeypatch, policy):
    c = cpu_post_script
    marker = _script(c, tmp_path)
    before = _source(c)
    handles, rejected = [], []
    _capture(monkeypatch, handles)
    attribute = ("_assert_controller_runtime_attested" if policy == "runtime"
                 else "_assert_launch_authorized")
    original = getattr(evaluator, attribute)

    def authorize(*args, **kwargs):
        if handles:
            rejected.append(True)
            assert not marker.exists()
            raise evaluator.LaunchIntegrityError("fixture post-script policy changed after READY")
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluator, attribute, authorize)
    try:
        with pytest.raises(AttemptEffectInDoubt, match="post_script_action_unconfirmed"):
            _run(c)
        assert rejected == [True]
        assert len(handles) == 1
        _closed_stop(handles[0])
        _started(c, handles[0])
        assert not marker.exists()
        assert _source(c) == before
        with pytest.raises(AttemptEffectBusy, match="post_script_previous_action_unclosed"):
            _run(c)
        assert len(handles) == 1
    finally:
        for child in handles:
            child.stop(timeout=3)


def test_post_script_lease_exit_failure_stops_blocked_ready_worker(cpu_post_script, tmp_path, monkeypatch):
    c = cpu_post_script
    marker = _script(c, tmp_path)
    before = _source(c)
    handles, exits = [], []
    _capture(monkeypatch, handles)
    original = evaluator.gpu_execution_lease

    @contextmanager
    def lease(*args, **kwargs):
        with original(*args, **kwargs) as descriptors:
            yield descriptors
        exits.append(True)
        assert len(handles) == 1
        assert not marker.exists()
        raise OSError("fixture lease exit could not be confirmed")

    monkeypatch.setattr(evaluator, "gpu_execution_lease", lease)
    try:
        with pytest.raises(AttemptEffectInDoubt, match="post_script_action_unconfirmed"):
            _run(c)
        assert exits == [True]
        _closed_stop(handles[0])
        _started(c, handles[0])
        assert not marker.exists()
        assert _source(c) == before
    finally:
        for child in handles:
            child.stop(timeout=3)


def test_post_script_timeout_stop_zero_is_not_success(cpu_post_script, tmp_path, monkeypatch):
    c = cpu_post_script
    marker = _script(c, tmp_path, pause=True, timeout=0.1)
    before = _source(c)
    handles = []

    def after_ready(child):
        start = child.start

        def go():
            start()
            _wait_marker(marker)

        monkeypatch.setattr(child, "start", go)

    _capture(monkeypatch, handles, after_ready=after_ready)
    try:
        _run(c)
        assert len(handles) == 1
        closure = _closed_stop(handles[0])
        assert closure["worker_returncode"] == 0
        row = current_attempt(c.lake.conn, c.folder.name, "post_script")
        assert row["state"] == "TERMINAL"
        assert row["terminal"]["outcome"] == "interrupted"
        assert row["terminal"]["reason_code"] == "post_script_timeout"
        assert row["terminal"]["return_code"] == 0
        assert row["terminal"]["process_tree"] == closure
        receipt_path = c.folder / "_compute_receipts" / row["attempt_id"] / "terminal.json"
        terminal = json.loads(receipt_path.read_text())
        assert terminal["outcome"] == "interrupted"
        assert terminal["return_code"] == 0
        assert terminal["process_pid"] == handles[0].pid
        assert _source(c) == before
    finally:
        for child in handles:
            child.stop(timeout=3)


def test_post_script_actual_channel_loss_keeps_pending_action(cpu_post_script, tmp_path, monkeypatch):
    c = cpu_post_script
    marker = _script(c, tmp_path, pause=True)
    before = _source(c)
    handles, stops = [], []

    def after_ready(child):
        start, stop = child.start, child.stop

        def go():
            start()
            _wait_marker(marker)
            child._channel.close()  # Actual private IO boundary, not fake return-code proof.

        def counted_stop(*args, **kwargs):
            stops.append(True)
            return stop(*args, **kwargs)

        monkeypatch.setattr(child, "start", go)
        monkeypatch.setattr(child, "stop", counted_stop)

    _capture(monkeypatch, handles, after_ready=after_ready)
    try:
        with pytest.raises(AttemptEffectInDoubt):
            _run(c)
        assert len(handles) == 1
        child = handles[0]
        assert child._uncertainty is not None
        with pytest.raises(SupervisionUncertain):
            child.closure_receipt()
        assert stops == []
        _started(c, child)
        assert not (c.folder / "_execution_stops").exists()
        assert _source(c) == before
        with pytest.raises(AttemptEffectBusy, match="post_script_previous_action_unclosed"):
            _run(c)
        assert len(handles) == 1
    finally:
        # The fixture owns worker/supervisor pidfds. Never STOP a latched
        # uncertain protocol or rediscover its process by PID/PGID.
        for child in handles:
            child._close_descriptors()


def test_post_script_ready_handoff_uncertainty_is_not_not_started(cpu_post_script, tmp_path, monkeypatch):
    c = cpu_post_script
    marker = _script(c, tmp_path)
    before = _source(c)
    handles = []

    def lost_return(child):
        assert not marker.exists()
        raise SupervisionUncertain("fixture post-script READY return lost", process=child)

    _capture(monkeypatch, handles, after_ready=lost_return)
    try:
        with pytest.raises(AttemptEffectInDoubt, match="post_script_action_unconfirmed"):
            _run(c)
        assert len(handles) == 1
        assert handles[0].poll() is None
        assert not marker.exists()
        row = current_attempt(c.lake.conn, c.folder.name, "post_script")
        assert row["state"] == "LAUNCHING"
        assert row["terminal"] is None
        assert not (c.folder / "_compute_receipts" / row["attempt_id"]).exists()
        assert _source(c) == before
    finally:
        # This injected lost-return exception did not corrupt the protocol;
        # the test still owns its captured real handle and can close it.
        for child in handles:
            child.stop(timeout=3)
