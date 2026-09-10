"""New native start/telemetry mechanisms; real blocked or escaped CPU trees."""
from dataclasses import asdict
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import accounting, launcher, process, training_attempts
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.supervised_process import prepare_supervised
from test_native_training_tree_completion import cpu_training, native_case, _launch, _alive

REAL_DETECT_ZOMBIE = launcher._detect_zombie
REAL_WATCHDOG = launcher._watchdog_check


def test_historical_launching_cannot_gain_supervision_or_compute_start(cpu_training, tmp_path):
    c = cpu_training
    claim = json.loads((c.folder / "claim.json").read_text())
    tp = SimpleNamespace(idea_id=c.idea, gpu=0, attempt_id=claim["attempt_id"],
                         execution_identity="a" * 64, process=None, start_time=time.time())
    tp.attempt_ref = training_attempts.begin(c.lake, tp, c.folder, c.cfg)
    row = current_attempt(c.lake.conn, c.idea, "training")
    binding = row["binding"]
    binding.pop("process_supervision_protocol")
    assert c.lake.conn.execute("UPDATE execution_attempts SET binding_json=? WHERE attempt_id=?",
                              (json.dumps(binding, sort_keys=True, separators=(",", ":")), tp.attempt_id)).rowcount == 1
    c.lake.conn.commit()
    marker = tmp_path / "must-not-execute"
    tp.process = prepare_supervised(
        [sys.executable, "-c", "from pathlib import Path; Path(" + repr(str(marker)) + ").touch()"],
        identity={"attempt_ref": asdict(tp.attempt_ref), "scope": str(c.folder.absolute())},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    c.pidfds.append((tp.process.pid, os.pidfd_open(tp.process.pid)))
    before = current_attempt(c.lake.conn, c.idea, "training")
    start = Mock(wraps=accounting.record_compute_start)
    try:
        with pytest.raises(AttemptEffectBusy, match="training_supervision_unbound"):
            training_attempts.record_ready_start(c.lake, tp, c.folder, start)
        with pytest.raises(AttemptEffectBusy, match="training_supervision_unbound"):
            training_attempts.started(c.lake, tp, c.folder,
                process.capture_process_identity(tp.process.pid), record_start=start)
        start.assert_not_called()
        assert current_attempt(c.lake.conn, c.idea, "training") == before
        assert not (c.folder / "_compute_receipts" / tp.attempt_id / "start.json").exists()
        assert json.loads((c.folder / "claim.json").read_text()) == claim
        assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
        assert not marker.exists()
    finally:
        tp.process.stop(timeout=3)


def test_worker_root_heuristics_do_not_misclassify_adopted_descendants(cpu_training, tmp_path, monkeypatch):
    c = cpu_training
    tp = _launch(c, tmp_path, detached=True)
    assert _alive(c.daemon_pidfd) and tp.process.poll() is None
    tp.start_time -= 7200
    tp._wd_first_batch = True
    monkeypatch.setattr(launcher, "_tree_cpu_jiffies",
                        lambda *a: pytest.fail("worker-root CPU sampling cannot cover adopted children"))
    monkeypatch.setattr(launcher, "_gpu_util_for_pid",
                        lambda *a: pytest.fail("worker-root GPU sampling cannot cover adopted children"))
    assert REAL_DETECT_ZOMBIE(tp) is False
    assert REAL_WATCHDOG(tp) is False
    assert _alive(c.daemon_pidfd) and tp.process.poll() is None
    assert not (c.folder / "_execution_stops").exists()
