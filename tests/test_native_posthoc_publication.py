"""C2c new native publication mechanisms with actual synthetic CPU workers.

No paid provider, GPU, model inference or host process scan. Launch, supervised
worker, adapter, Lake, compute, physical B1 snapshots and lifecycle stay real.
The original public baseline file and its OS-restoring fixture are unchanged.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import time

import pytest

from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.engine import launcher, posthoc_completion
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.attempt_effect_receipts import require_closed_effects
from orze.engine.failure import _reset_idea_for_retry
from orze.engine.scheduler import claim
from orze.engine.termination_hold import require_no_unconfirmed_stop, terminate_execution
from test_native_posthoc_tree_completion import (
    cpu_posthoc, cpu_training, native_case, _launch_posthoc, _alive, _wait_dead,
)


def _with_artifact(c):
    c.cfg["artifact_contract"] = {"version": 1, "outputs": {
        "prediction": {"path": "posthoc.bin", "max_bytes": 128}}}


def _row(c):
    return current_attempt(c.lake.conn, c.idea, "posthoc")


def _receipt(c, tp, event):
    return json.loads((c.folder / "_compute_receipts" / tp.attempt_id / (event + ".json")).read_bytes())


def _state(c):
    return {"row": _row(c), "global": c.lake.get_fsm_state(c.idea),
            "stage": c.lake.get_stage_state(c.idea, "training"),
            "files": {str(path.relative_to(c.folder)): path.read_bytes()
                      for path in c.folder.rglob("*") if path.is_file()}}


def _natural_close(c, tp):
    c.daemon_channel.sendall(b"Q")
    _wait_dead(c.daemon_pidfd)
    deadline = time.monotonic() + 5
    while tp.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert tp.process.poll() == 0
    closure = tp.process.closure_receipt()
    assert closure["wait_proof"] == "ECHILD_WALL"
    assert closure["stop_requested"] is False
    assert closure["forced_cleanup"] is False


def _assert_completed(c, tp, expected):
    row = _row(c)
    assert row["state"] == "TERMINAL"
    assert row["terminal"]["outcome"] == "completed"
    assert row["terminal"]["return_code"] == 0
    assert row["terminal"]["process_tree"] == tp.process.closure_receipt()
    assert c.lake.get_fsm_state(c.idea) == "COMPLETE"
    metrics = json.loads((c.folder / "metrics.json").read_bytes())
    assert metrics["status"] == "COMPLETED" and metrics["score"] == 0
    records = artifacts_for_attempt(c.lake.conn, tp.attempt_ref)
    assert len(records) == 1
    assert records[0]["producer"] == asdict(tp.attempt_ref)
    assert records[0]["logical_name"] == "prediction"
    assert row["terminal"]["artifact_ids"] == [records[0]["artifact_id"]]
    content = Path(records[0]["path"])
    assert content.read_bytes() == expected
    assert content.stat().st_ino != c.adapter_output.stat().st_ino
    assert _receipt(c, tp, "start")["process_pid"] == _receipt(c, tp, "terminal")["process_pid"] == tp.process.pid
    assert _receipt(c, tp, "terminal")["phase"] == "posthoc"
    require_closed_effects(c.folder)
    require_no_unconfirmed_stop(c.folder)
    return content


def test_closed_posthoc_parent_publishes_only_declared_current_work_artifact(cpu_posthoc, tmp_path):
    c = cpu_posthoc
    _with_artifact(c)
    (c.folder / "posthoc.bin").write_bytes(b"canonical-decoy")
    (c.folder / "old-predictions.npz").write_bytes(b"legacy-root-not-a-new-result")
    tp = _launch_posthoc(c, tmp_path, detached=False)
    assert tp.attempt_ref.phase == "posthoc"
    assert c.adapter_output.parent == c.folder / "_posthoc_attempts" / tp.attempt_id / "work"
    assert not (c.folder / "metrics.json").exists()
    assert _row(c)["state"] == "RUNNING"
    active, failures = {0: tp}, {}
    assert launcher.check_active(active, c.results, c.cfg, failures, lake=c.lake) == [(c.idea, 0)]
    assert active == {} and failures == {}
    _assert_completed(c, tp, b"synthetic-posthoc-output")
    assert (c.folder / "posthoc.bin").read_bytes() == b"canonical-decoy"
    assert (c.folder / "old-predictions.npz").read_bytes() == b"legacy-root-not-a-new-result"


def test_direct_posthoc_finish_waits_for_writer_then_snapshots_actual_late_bytes(cpu_posthoc, tmp_path):
    c = cpu_posthoc
    _with_artifact(c)
    tp = _launch_posthoc(c, tmp_path, detached=True)
    before = _state(c)
    with pytest.raises(AttemptEffectBusy, match="posthoc_process_tree_unclosed"):
        posthoc_completion.finish(c.lake, tp, 0, c.folder, c.cfg, 0, {})
    assert _state(c) == before
    assert _alive(c.daemon_pidfd)
    c.daemon_channel.sendall(b"W")
    assert c.daemon_channel.recv(1) == b"W"
    assert c.adapter_output.read_bytes() == b"late posthoc writer"
    _natural_close(c, tp)
    assert posthoc_completion.finish(c.lake, tp, 0, c.folder, c.cfg, 0, {}) == (c.idea, 0)
    content = _assert_completed(c, tp, b"late posthoc writer")
    c.adapter_output.write_bytes(b"later scratch change")
    assert content.read_bytes() == b"late posthoc writer"
    assert not any(was_live for _, was_live in c.publications)


def test_posthoc_owned_stop_with_worker_zero_is_failed_without_artifacts(cpu_posthoc, tmp_path):
    c = cpu_posthoc
    _with_artifact(c)
    tp = _launch_posthoc(c, tmp_path, detached=True)
    assert _alive(c.daemon_pidfd) and tp.process.poll() is None
    assert terminate_execution(tp, c.folder, phase="posthoc", reaper=launcher._terminate_and_reap) == 0
    assert not _alive(c.daemon_pidfd)
    active, failures = {0: tp}, {}
    assert launcher.check_active(active, c.results, c.cfg, failures, lake=c.lake) == [(c.idea, 0)]
    assert active == {} and failures == {c.idea: 1}
    row = _row(c)
    assert row["state"] == "TERMINAL" and row["terminal"]["outcome"] == "failed"
    assert row["terminal"]["return_code"] == 0
    assert row["terminal"]["reason_code"] == "posthoc_process_tree_stopped"
    assert row["terminal"]["process_tree"]["stop_requested"] is True
    assert row["terminal"]["artifact_ids"] == []
    assert artifacts_for_attempt(c.lake.conn, tp.attempt_ref) == []
    assert c.lake.get_fsm_state(c.idea) == "FAILED"
    terminal = _receipt(c, tp, "terminal")
    assert terminal["outcome"] == "failed" and terminal["return_code"] == 0
    assert terminal["process_pid"] == tp.process.pid
    require_closed_effects(c.folder)
    require_no_unconfirmed_stop(c.folder)


def test_lost_posthoc_supervisor_holds_active_and_all_publication(cpu_posthoc, tmp_path):
    c = cpu_posthoc
    _with_artifact(c)
    tp = _launch_posthoc(c, tmp_path, detached=True)
    owned = os.dup(tp.process.supervisor_pidfd)
    c.pidfds.append((tp.process.supervisor_pid, owned))
    before = _state(c)
    signal.pidfd_send_signal(owned, signal.SIGKILL)
    _wait_dead(owned)
    active, failures = {0: tp}, {}
    for _ in range(2):
        assert launcher.check_active(active, c.results, c.cfg, failures, lake=c.lake) == []
        assert active.get(0) is tp and failures == {}
        assert tp._termination_unconfirmed is True
        assert _state(c) == before
        assert _alive(c.daemon_pidfd)
    assert not any(was_live for _, was_live in c.publications)
    assert artifacts_for_attempt(c.lake.conn, tp.attempt_ref) == []


def test_old_and_duplicate_posthoc_callbacks_leave_new_attempt_and_publication_unchanged(cpu_posthoc, tmp_path):
    c = cpu_posthoc
    _with_artifact(c)
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    old = _launch_posthoc(c, first_dir, detached=False, code=7)
    failures = {}
    assert launcher.check_active({0: old}, c.results, c.cfg, failures, lake=c.lake) == [(c.idea, 0)]
    assert _row(c)["terminal"]["outcome"] == "failed"
    assert failures == {c.idea: 1}
    _reset_idea_for_retry(c.folder, release_claim=True, lake=c.lake)
    assert c.lake.set_status(c.idea, "queued")
    assert claim(c.idea, c.results, 0, lake=c.lake)
    fresh = _launch_posthoc(c, second_dir, detached=False)
    assert fresh.attempt_ref.generation > old.attempt_ref.generation
    assert fresh.attempt_id != old.attempt_id
    before = _state(c)
    assert posthoc_completion.finish(c.lake, old, 0, c.folder, c.cfg, 7, failures) is None
    assert _state(c) == before and failures == {c.idea: 1}
    assert launcher.check_active({0: fresh}, c.results, c.cfg, failures, lake=c.lake) == [(c.idea, 0)]
    _assert_completed(c, fresh, b"synthetic-posthoc-output")
    completed = _state(c)
    assert posthoc_completion.finish(c.lake, fresh, 0, c.folder, c.cfg, 0, failures) is None
    assert posthoc_completion.finish(c.lake, old, 0, c.folder, c.cfg, 7, failures) is None
    assert _state(c) == completed and failures == {c.idea: 1}
