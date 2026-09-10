"""Borrowed claim cleanup cannot release its caller's uncertain file effects."""
import errno
import json
import os
from pathlib import Path
import socket

import pytest

from orze.engine import scheduler
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt, attempt_effect_lock


def test_cleanup_propagates_borrowed_lease_partial_effect_failure(tmp_path, monkeypatch):
    folder = tmp_path / "idea-borrowed-orphan"
    folder.mkdir()
    claim = folder / "claim.json"
    claim.write_text(json.dumps({"attempt_id": "legacy-A", "gpu": 0,
        "claimed_by": socket.gethostname(), "pid": 991991, "owner_start_ticks": 13}))
    os.utime(claim, (1, 1))
    (folder / "metrics.json").write_text('{"status":"PARTIAL","step":3}')
    monkeypatch.setattr(scheduler, "process_is_running", lambda *args: False)
    real_replace = os.replace

    def fail_claim_archive(source, target, *args, **kwargs):
        if Path(source) == claim:
            raise OSError(errno.EIO, "synthetic claim archive failure after metrics moved")
        return real_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_claim_archive)
    returned = False
    with pytest.raises(AttemptEffectInDoubt):
        with attempt_effect_lock(folder) as lease:
            scheduler.cleanup_orphans(tmp_path, 1, effect_lease=lease)
            returned = True
    assert not returned, "the outer owner must not commit or release after uncertain cleanup"
    assert (folder / "_attempt_effect.lock").is_dir()
    assert claim.exists()
    assert not (folder / "metrics.json").exists()
    archives = list(folder.glob("metrics.orphan.*.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text()) == {"status": "PARTIAL", "step": 3}
