"""All reset publication modes confirm their own directory changes."""
from contextlib import nullcontext
import errno
import os

import pytest

from orze.engine import failure, scheduler
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt, attempt_effect_lock


@pytest.mark.parametrize("mode", ["release", "renew", "no_claim"])
@pytest.mark.parametrize("sync_failure", [False, True])
def test_reset_modes_acknowledge_durability_before_return(tmp_path, monkeypatch, mode, sync_failure):
    folder = tmp_path / "idea-reset-mode"
    if mode == "no_claim":
        folder.mkdir()
    else:
        assert scheduler.claim(folder.name, tmp_path, 0)
    (folder / "metrics.json").write_text('{"status":"FAILED"}')
    (folder / "train_output.log").write_text("old log")
    info = folder.stat()
    fsync = os.fsync
    probe = {"armed": False, "seen": []}

    def sync(fd):
        actual = os.fstat(fd)
        if probe["armed"] and (actual.st_dev, actual.st_ino) == (info.st_dev, info.st_ino):
            probe["seen"].append(True)
            if sync_failure:
                raise OSError(errno.EIO, "synthetic reset directory sync failure")
        return fsync(fd)

    monkeypatch.setattr(os, "fsync", sync)
    returned = False
    at_return = []
    expectation = pytest.raises(AttemptEffectInDoubt) if sync_failure else nullcontext()
    with expectation:
        with attempt_effect_lock(folder) as lease:
            probe["armed"] = True
            failure._reset_idea_for_retry(folder, release_claim=mode == "release", effect_lease=lease)
            returned = True
            at_return = list(probe["seen"])
    if sync_failure:
        assert not returned, "a caller must not continue after unconfirmed reset effects"
        assert (folder / "_attempt_effect.lock").is_dir()
    else:
        assert returned and at_return, "reset cannot borrow its outer owner's later fsync"
        assert not (folder / "metrics.json").exists()
        assert not (folder / "train_output.log").exists()
        assert (folder / "claim.json").exists() is (mode == "renew")
