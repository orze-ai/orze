"""A durable training replica request cannot authorize posthoc by file drift.

Source completion uses the existing explicit OS double and real B1/SQLite.
The target request, claim and launch are real; target process boundaries are
restored to real tiny CPU execution. No scientific replication is claimed.
"""
import contextlib
import copy
import os
import signal
import subprocess
import sys
import time

import pytest
import yaml

from orze.core.execution_attempts import current_attempt
from orze.core.replication_requests import ReplicationError, request_for_task
from orze.engine import launcher, process
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.supervised_process import prepare_supervised
from test_native_training_tree_completion import _alive, _wait_dead
from test_replication_execution_slots import source, case, _request, _select, _slot


REAL_POPEN = subprocess.Popen


def test_disk_kind_cannot_turn_authorized_training_replica_into_posthoc(source, monkeypatch):
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        pytest.skip("requires exact Linux pidfd cleanup")
    c = source
    request = _request(c, "posthoc-route-must-not-consume")
    _select(c, request)
    record = copy.deepcopy(request_for_task(c.lake.conn, c.idea))
    assert record["task_id"] == c.idea and record["source_ref"]["phase"] == "training"
    claim_before = (c.folder / "claim.json").read_bytes()
    source_row = copy.deepcopy(current_attempt(c.lake.conn, c.source_id, "training"))
    config = yaml.safe_load((c.folder / "idea_config.yaml").read_text())
    config.update(kind="posthoc_eval", adapter="null")
    (c.folder / "idea_config.yaml").write_text(yaml.safe_dump(config))
    c.cfg["python"] = sys.executable
    # Do not leave an inherited simulated supervisor or reaper at the target.
    monkeypatch.setattr(launcher.subprocess, "Popen", REAL_POPEN)
    monkeypatch.setattr(launcher, "capture_process_identity", process.capture_process_identity)
    monkeypatch.setattr(launcher, "_terminate_and_reap", process._terminate_and_reap)
    monkeypatch.setattr(launcher, "gpu_execution_lease", lambda *a, **k: contextlib.nullcontext(()))
    prepared, pidfds = [], []
    target = None

    def real_prepare(*args, **kwargs):
        handle = prepare_supervised(*args, **kwargs)
        prepared.append(handle)
        # READY blocks the real worker until both exact cleanup handles exist.
        pidfds.append(os.pidfd_open(handle.pid))
        pidfds.append(os.dup(handle.supervisor_pidfd))
        return handle

    monkeypatch.setattr(launcher, "prepare_supervised", real_prepare)
    try:
        with pytest.raises((ReplicationError, launcher.LaunchIntegrityError, AttemptEffectBusy)):
            target = launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
        assert prepared == [], "replica route must be rejected before preparing a posthoc worker"
        assert current_attempt(c.lake.conn, c.idea, "posthoc") is None
        assert current_attempt(c.lake.conn, c.idea, "training") is None
        assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
        assert (c.folder / "claim.json").read_bytes() == claim_before
        assert request_for_task(c.lake.conn, c.idea) == record
        assert current_attempt(c.lake.conn, c.source_id, "training") == source_row
        assert c.flat.read_bytes() == c.flat_bytes and not _slot(c, request).exists()
        assert not list(c.folder.glob("_compute_receipts/*/start.json"))
        assert not (c.folder / "_posthoc_attempts").exists()
    finally:
        # Only this null adapter can have run; no host scan or numeric PID kill.
        for handle in prepared:
            deadline = time.monotonic() + 5
            while handle.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            if handle.poll() is None:
                handle.stop(timeout=3)
        for fd in pidfds:
            if _alive(fd):
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            _wait_dead(fd)
            os.close(fd)
        for handle in prepared:
            handle._supervisor.wait(timeout=5)
        if target is not None:
            target.close_log()
