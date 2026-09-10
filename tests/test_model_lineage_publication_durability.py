"""The short publisher must acknowledge its parent-directory durability."""
import errno
import os

import pytest

from orze.core import model_lineage as lineage
from test_model_lineage_prepared_finalization import prepared_project, _api


def test_parent_fsync_failure_after_lineage_write_is_not_acknowledged(prepared_project, monkeypatch):
    p = prepared_project
    prepare, publish = _api()
    prepared = prepare(p.tp, p.folder, p.cfg)
    parent = p.folder.stat()
    fsync = os.fsync
    seen = []

    def fail_parent(fd):
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == (parent.st_dev, parent.st_ino):
            seen.append("parent_fsync")
            raise OSError(errno.ENOSPC, "test-only parent durability failure")
        return fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_parent)
    with pytest.raises(lineage.ModelLineageError):
        publish(prepared, p.tp, p.folder, p.cfg)
    assert seen == ["parent_fsync"]
    # An already-written envelope is an uncertain effect, not permission to
    # roll back the caller's intent or retry publication automatically.
    assert (p.folder / lineage.LINEAGE_FILE).is_file()
