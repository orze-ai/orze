"""C2c descriptor binding: real file descriptors, injected open boundary.

The wrong-inode case does not claim a reproduced path rename/ABA race. The
named file stays unchanged while an injected open returns another real file;
the reader must bind its actual descriptor to the previously captured path.
"""
import os

import pytest

from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.native_posthoc import _configuration


@pytest.mark.parametrize("wrong_inode", [True, False], ids=["wrong-opened-inode", "matching-opened-inode"])
def test_configuration_requires_captured_path_descriptor(tmp_path, monkeypatch, wrong_inode):
    named = tmp_path / "idea_config.yaml"
    other = tmp_path / "different.yaml"
    named.write_bytes(b"kind: posthoc_eval\nadapter: good\n")
    other.write_bytes(b"kind: posthoc_eval\nadapter: evil\n")
    before = named.stat()
    original = named.read_bytes()
    assert before.st_ino != other.stat().st_ino
    real_open, real_close = os.open, os.close
    descriptors, closed = [], []

    def open_file(path, flags, *args, **kwargs):
        if os.fspath(path) == str(named):
            descriptor = real_open(other if wrong_inode else named, flags, *args, **kwargs)
            descriptors.append(descriptor)
            assert (os.fstat(descriptor).st_ino != before.st_ino) is wrong_inode
            return descriptor
        return real_open(path, flags, *args, **kwargs)

    def close_file(descriptor):
        if descriptor in descriptors:
            closed.append(descriptor)
        return real_close(descriptor)

    monkeypatch.setattr(os, "open", open_file)
    monkeypatch.setattr(os, "close", close_file)
    result, rejected = None, None
    try:
        result = _configuration(named, {}, "posthoc_eval")
    except AttemptEffectBusy as exc:
        rejected = exc
    assert len(descriptors) == 1
    assert closed == descriptors
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    after = named.stat()
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    assert named.read_bytes() == original
    if wrong_inode:
        assert rejected is not None, ("reader accepted bytes from an unrelated opened inode", result)
        assert result is None
    else:
        assert rejected is None
        assert result == {"kind": "posthoc_eval", "adapter": "good"}
