"""New in-process scope identity requirement, not durable native authority."""
import os
import sys

import pytest

from orze.engine import bounded_executor as executor
from orze.engine.supervised_process import SupervisionUncertain


def test_unknown_scope_rename_cannot_reenter_the_same_directory_inode(tmp_path):
    scope = tmp_path / "project"
    scope.mkdir()
    calls = []
    def uncertain(*args, **kwargs):
        calls.append(True)
        raise SupervisionUncertain("synthetic_unknown_prepare")
    def invoke(path):
        return executor.run_bounded_executor([sys.executable, "-c", "pass"],
            timeout=1, env=dict(os.environ), cwd=path, prepare=uncertain)
    with pytest.raises(executor.BoundedExecutorHOLD):
        invoke(scope)
    old = scope.stat()
    renamed = tmp_path / "renamed"
    scope.rename(renamed)
    current = renamed.stat()
    assert (old.st_dev, old.st_ino) == (current.st_dev, current.st_ino)
    with pytest.raises(executor.BoundedExecutorHOLD):
        invoke(renamed)
    assert calls == [True]
    with pytest.raises(executor.BoundedExecutorHOLD):
        executor.require_executor_scope_clear(renamed)
