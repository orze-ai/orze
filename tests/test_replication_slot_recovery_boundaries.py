"""Two bounded draft regressions: real requeue and exact slot release."""
import json

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import launcher
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from test_replication_execution_slots import source, case, _request, _select, _slot, _launch


def test_confirmed_native_resource_requeue_can_reuse_its_authorized_occurrence(source):
    c = source
    request = _request(c)
    _select(c, request)
    first = _launch(c)
    c.child.returncode = 0
    (c.folder / "metrics.json").write_text(json.dumps({
        "status": "FAILED", "error": "insufficient_vram: synthetic scheduler contention"}))
    active = {0: first}
    events = launcher.check_active(active, c.results, c.cfg, {}, lake=c.lake)
    assert len(events) == 1 and active == {}
    assert current_attempt(c.lake.conn, c.idea, "training")["terminal"]["outcome"] == "requeued"
    assert c.lake.get_fsm_state(c.idea) == "QUEUED"
    _select(c, request)
    second = _launch(c)
    try:
        assert second.attempt_id != first.attempt_id
        assert second.execution_identity == first.execution_identity
        assert json.loads(_slot(c, request).read_text())["attempt_id"] == second.attempt_id
        assert c.flat.read_bytes() == c.flat_bytes
    finally:
        second.close_log()


def test_known_no_launch_cannot_delete_same_bytes_replacement_slot(source, monkeypatch):
    c = source
    request = _request(c)
    _select(c, request)
    path = _slot(c, request)
    replacement = []
    def replace_then_no_gpu(*args, **kwargs):
        original = path.read_bytes()
        staged = path.with_suffix(".replacement")
        staged.write_bytes(original)
        old_inode = path.stat().st_ino
        staged.replace(path)
        replacement.append((original, path.stat().st_ino))
        assert replacement[-1][1] != old_inode
        raise launcher.GpuUnavailableError("synthetic unavailable after replacement")
    monkeypatch.setattr(launcher, "_verify_gpu_free", replace_then_no_gpu)
    with pytest.raises(AttemptEffectBusy, match="capture_changed"):
        _launch(c)
    assert c.popen_calls == [] and len(replacement) == 1
    assert path.read_bytes() == replacement[0][0]
    assert path.stat().st_ino == replacement[0][1]
    assert c.flat.read_bytes() == c.flat_bytes
