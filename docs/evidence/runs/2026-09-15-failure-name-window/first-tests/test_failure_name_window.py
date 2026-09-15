"""Preserve the selected failures while releasing discarded directory names."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

@pytest.mark.parametrize("limit", [1, 17, 2000])
def test_directory_names_are_released_beyond_selected_tail(tmp_path, monkeypatch, limit):
    from orze.engine.failure import load_recent_failures as reader
    count = 5000
    root = tmp_path / "results"
    root.mkdir()
    tail = f"idea-{count - 1:05d}"
    folder = root / tail
    folder.mkdir()
    (folder / "metrics.json").write_text(json.dumps({"status": "FAILED", "error": "Preserve the tail counterexample"}))

    class Name(str):
        alive = 0
        peak = 0

        def __new__(cls, value):
            instance = super().__new__(cls, value)
            cls.alive += 1
            cls.peak = max(cls.peak, cls.alive)
            return instance

        def __del__(self):
            type(self).alive -= 1

    observed = []

    @contextmanager
    def entries(path):
        assert path == root
        def values():
            for index in range(count):
                name = Name(f"idea-{index:05d}")
                observed.append(index)  # Integers only; do not keep name objects.
                yield SimpleNamespace(name=name, is_dir=lambda *, follow_symlinks: not follow_symlinks)
        iterator = values()
        try:
            yield iterator
        finally:
            iterator.close()

    with monkeypatch.context() as local:
        local.setattr(os, "scandir", entries)
        result = reader(root, limit)
    assert observed == list(range(count)), "all names must still be enumerated"
    assert [row["idea_id"] for rows in result.values() for row in rows] == [tail]
    assert Name.peak <= limit + 4, f"kept {Name.peak} names for a tail of {limit}"
