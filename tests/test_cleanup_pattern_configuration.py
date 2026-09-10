"""Public cleanup validates the complete destructive pattern batch first."""

import pytest

from orze.engine.scheduler import run_cleanup


@pytest.mark.parametrize("bad_patterns", [
    "*.tmp",
    ["*.tmp", 7],
    ["*.tmp", "../../outside.txt"],
    ["*.tmp", "/absolute/not-a-cleanup-target"],
])
def test_invalid_pattern_batch_preserves_disposable_files(tmp_path, bad_patterns):
    results = tmp_path / "results"
    idea = results / "idea-configuration"
    idea.mkdir(parents=True)
    disposable = idea / "scratch.tmp"
    disposable.write_bytes(b"do not partially apply an invalid batch")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"project source stand-in")

    # Periodic maintenance must not raise or partially delete valid matches
    # before discovering an invalid later pattern. No GC/custom script here.
    run_cleanup(results, {"cleanup": {"patterns": bad_patterns}, "gc": {}})

    assert disposable.read_bytes() == b"do not partially apply an invalid batch"
    assert outside.read_bytes() == b"project source stand-in"


def test_empty_cleanup_configuration_is_non_destructive(tmp_path):
    results = tmp_path / "results"
    idea = results / "idea-default"
    idea.mkdir(parents=True)
    disposable = idea / "scratch.tmp"
    disposable.write_bytes(b"default is opt out")

    run_cleanup(results, {})

    assert disposable.read_bytes() == b"default is opt out"


def test_valid_pattern_batch_cleans_legacy_disposable(tmp_path):
    results = tmp_path / "results"
    idea = results / "idea-legacy"
    idea.mkdir(parents=True)
    disposable = idea / "scratch.tmp"
    disposable.write_bytes(b"explicit disposable")
    keep = idea / "notes.txt"
    keep.write_bytes(b"not matched")

    run_cleanup(results, {"cleanup": {"patterns": ["*.tmp"]}})

    assert not disposable.exists()
    assert keep.read_bytes() == b"not matched"
