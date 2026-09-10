"""New C2b READY/GO mechanisms with actual blocked CPU workers.

These are protocol requirements, not old-API absence regressions. The shared
CPU fixture restores real Popen, process identity, reaper and supervision.
Only prepared-return and controller-attestation seams are interleaved here.
"""
import os
from pathlib import Path

import pytest

from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.engine import launcher
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.supervised_process import SupervisionUncertain
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_training_tree_completion import cpu_training, native_case


def _remember(c, handles, process):
    # Capture the blocked worker immediately, before injecting any failure.
    c.pidfds.append((process.pid, os.pidfd_open(process.pid)))
    handles.append(process)
    assert process.poll() is None


def _marker_script(path, marker):
    path.write_text("from pathlib import Path\nPath(" + repr(str(marker))
                    + ").write_text('executed')\n", encoding="utf-8")


def _launch_refused(c):
    result, rejected = None, None
    try:
        result = launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
        if result is not None:
            c.handles.append(result)
    except (launcher.LaunchIntegrityError, AttemptEffectBusy,
            AttemptEffectInDoubt, TerminationUnconfirmed, SupervisionUncertain) as exc:
        rejected = exc
    return result, rejected


@pytest.mark.parametrize("change", ["script", "runtime"],
                         ids=["changed-script-after-ready", "runtime-rejected-after-ready"])
def test_ready_does_not_authorize_changed_training_launch(cpu_training, tmp_path, monkeypatch, change):
    c = cpu_training
    script, marker = Path(c.cfg["train_script"]), tmp_path / "training-user-executed"
    if change == "script":
        script.write_text("# original authorized tiny worker\n", encoding="utf-8")
    else:
        _marker_script(script, marker)
    real_prepare = launcher.prepare_supervised
    real_attest = launcher._assert_controller_runtime_attested
    handles, runtime_rejections = [], []

    def prepare(*args, **kwargs):
        process = real_prepare(*args, **kwargs)
        _remember(c, handles, process)
        assert not marker.exists()
        if change == "script":
            _marker_script(script, marker)
        return process

    def attest(cfg):
        if handles and change == "runtime":
            runtime_rejections.append(True)
            raise launcher.LaunchIntegrityError("fixture runtime rejected after READY")
        return real_attest(cfg)

    monkeypatch.setattr(launcher, "prepare_supervised", prepare)
    monkeypatch.setattr(launcher, "_assert_controller_runtime_attested", attest)
    try:
        result, rejected = _launch_refused(c)
        assert len(handles) == 1, "must reach actual READY before the injected change"
        process = handles[0]
        process.wait(timeout=5)
        assert not marker.exists(), "GO executed code after its launch authority changed"
        closure = process.closure_receipt()
        assert closure["stop_requested"] is True
        assert closure["wait_proof"] == "ECHILD_WALL"
        assert result is None and rejected is not None
        if change == "runtime":
            assert runtime_rejections
        row = current_attempt(c.lake.conn, c.idea, "training")
        assert row["state"] != "NOT_STARTED"
        assert (row["terminal"] or {}).get("outcome") != "completed"
        assert not (c.folder / "metrics.json").exists()
        assert not list((Path(c.cfg["_orze_dir"]) / "artifacts").glob("*/content"))
    finally:
        for process in handles:
            process.stop(timeout=3)


def test_created_supervisor_error_cannot_be_relabelled_unstarted_or_release_reservation(cpu_training, tmp_path, monkeypatch):
    c = cpu_training
    marker = tmp_path / "uncertain-user-executed"
    _marker_script(Path(c.cfg["train_script"]), marker)
    real_prepare = launcher.prepare_supervised
    handles, reservation_before = [], {}
    registry = Path(c.cfg["_orze_dir"]) / "state" / "execution_identities"

    def prepare(*args, **kwargs):
        process = real_prepare(*args, **kwargs)
        _remember(c, handles, process)
        assert not marker.exists()
        reservation_before.update({path.name: path.read_bytes() for path in registry.glob("*.json")})
        assert len(reservation_before) == 1
        # The call raises before assignment to the launcher's local proc, but
        # the exception carries the real, already-created READY owner.
        raise SupervisionUncertain("fixture failure after actual READY", process=process)

    monkeypatch.setattr(launcher, "prepare_supervised", prepare)
    try:
        result, rejected = _launch_refused(c)
        assert len(handles) == 1, "must create the real blocked process first"
        assert result is None and rejected is not None
        assert not marker.exists()
        row = current_attempt(c.lake.conn, c.idea, "training")
        assert row["state"] != "NOT_STARTED"
        assert (row["terminal"] or {}).get("outcome") not in ("not_started", "completed")
        assert {path.name: path.read_bytes() for path in registry.glob("*.json")} == reservation_before
        assert not (c.folder / "metrics.json").exists()
        assert not list((Path(c.cfg["_orze_dir"]) / "artifacts").glob("*/content"))
        if row["state"] == "TERMINAL":
            closure = handles[0].closure_receipt()
            assert closure is not None and closure["stop_requested"] is True
            assert row["terminal"]["return_code"] == closure["worker_returncode"]
    finally:
        for process in handles:
            process.stop(timeout=3)
