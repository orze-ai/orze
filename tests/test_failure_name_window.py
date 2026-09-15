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


class IntegerSubclass(int):
    pass


def reader():
    from orze.engine.failure import load_recent_failures as implementation
    return implementation


def selected(result):
    return [row["idea_id"] for rows in result.values() for row in rows]


def failures(root, names=("idea-a", "idea-b", "idea-c")):
    root.mkdir()
    for name in names:
        folder = root / name
        folder.mkdir()
        (folder / "metrics.json").write_text(json.dumps({"status": "FAILED", "error": "Preserve this counterexample"}))
    return root


@pytest.mark.parametrize("limit,expected", [
    (0, ["idea-a", "idea-b", "idea-c"]),
    (-1, ["idea-b", "idea-c"]),
    (-4, []),
    (True, ["idea-c"]),
    (False, ["idea-a", "idea-b", "idea-c"]),
    (4097, ["idea-a", "idea-b", "idea-c"]),
    (5000.0, ["idea-a", "idea-b", "idea-c"]),
    (IntegerSubclass(1), ["idea-c"]),
])
def test_legacy_limit_selection_is_preserved(tmp_path, limit, expected):
    root = failures(tmp_path / "results")
    assert selected(reader()(root, limit)) == expected


@pytest.mark.parametrize("limit", [None, "2", 1.0])
def test_legacy_invalid_slices_still_raise(tmp_path, limit):
    root = failures(tmp_path / "results")
    with pytest.raises(TypeError):
        reader()(root, limit)


def test_links_and_non_directories_do_not_displace_selected_failures(tmp_path):
    root = failures(tmp_path / "results", ["idea-a"])
    outside = failures(tmp_path / "outside", ["idea-external"])
    (root / "idea-z").symlink_to(outside / "idea-external", target_is_directory=True)
    (root / "idea-y").write_text("not a directory")
    (root / "other-directory").mkdir()
    assert selected(reader()(root, 1)) == ["idea-a"]


def test_late_enumeration_error_discards_the_whole_selection(tmp_path, monkeypatch):
    root = failures(tmp_path / "results")
    implementation = reader()
    reads = []
    original = Path.read_text

    def observe(path, *args, **kwargs):
        reads.append(path)
        return original(path, *args, **kwargs)

    @contextmanager
    def entries(path):
        assert path == root
        def values():
            for name in ("idea-a", "idea-b", "idea-c"):
                yield SimpleNamespace(name=name, is_dir=lambda *, follow_symlinks: not follow_symlinks)
            raise OSError("late enumeration failure")
        yield values()

    with monkeypatch.context() as local:
        local.setattr(os, "scandir", entries)
        local.setattr(Path, "read_text", observe)
        result = implementation(root, 1)
    assert selected(result) == []
    assert reads == [], "failed enumeration must not publish a partial scan"
