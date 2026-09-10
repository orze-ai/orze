"""New effect-lock mechanism: real source-lock cleanup fault boundaries.

Only filesystem unlink/close are fault-injected.  Acquisition, owner checks,
and source-lock release run normally; these are not old-API regressions.
"""
import errno
import os
from pathlib import Path

import pytest

from orze.engine.attempt_effect_lock import (
    AttemptEffectInDoubt, attempt_effect_lock,
)


@pytest.mark.parametrize("body_fails", [False, True], ids=["normal-exit", "body-error"])
def test_metadata_unlink_failure_is_always_a_dedicated_hold(
        tmp_path, monkeypatch, body_fails):
    folder = tmp_path / "idea-effects"
    metadata = folder / "_attempt_effect.lock" / "lock.json"
    original_unlink = Path.unlink
    attempted = []

    def fail_metadata_unlink(path, *args, **kwargs):
        if path == metadata:
            attempted.append(path)
            raise OSError(errno.EIO, "fixture lock metadata unlink failure")
        return original_unlink(path, *args, **kwargs)

    observed = None
    with monkeypatch.context() as faults:
        faults.setattr(Path, "unlink", fail_metadata_unlink)
        try:
            with attempt_effect_lock(folder):
                if body_fails:
                    raise ValueError("fixture validation error before effects")
        except Exception as exc:
            observed = exc

    assert attempted == [metadata]
    assert metadata.is_file(), "failed cleanup must not erase unresolved ownership"
    assert isinstance(observed, AttemptEffectInDoubt), (
        f"cleanup uncertainty escaped into ordinary failure handling: {observed!r}"
    )


def test_parent_directory_close_failure_after_release_has_dedicated_classification(
        tmp_path, monkeypatch):
    folder = tmp_path / "idea-effects"
    metadata = folder / "_attempt_effect.lock" / "lock.json"
    original_unlink = Path.unlink
    original_close = os.close
    release_started = []
    failed_closes = []

    def observe_metadata_unlink(path, *args, **kwargs):
        result = original_unlink(path, *args, **kwargs)
        if path == metadata:
            release_started.append(True)
        return result

    def fail_release_directory_close(fd):
        info = os.fstat(fd)
        parent = folder.stat()
        target = (release_started and not failed_closes
                  and (info.st_dev, info.st_ino) == (parent.st_dev, parent.st_ino))
        original_close(fd)
        if target:
            failed_closes.append(True)
            raise OSError(errno.EIO, "fixture release directory close failure")

    observed = None
    with monkeypatch.context() as faults:
        faults.setattr(Path, "unlink", observe_metadata_unlink)
        faults.setattr(os, "close", fail_release_directory_close)
        try:
            with attempt_effect_lock(folder):
                pass
        except Exception as exc:
            observed = exc

    assert release_started == [True]
    assert failed_closes == [True]
    # Directory removal precedes this fault: only classification is promised,
    # not retained ownership or atomic filesystem/SQLite publication.
    assert not metadata.parent.exists()
    assert isinstance(observed, AttemptEffectInDoubt), (
        f"release uncertainty escaped into ordinary failure handling: {observed!r}"
    )
