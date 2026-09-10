"""Public cleanup path regressions; all deletion is limited to pytest tmp paths."""
from pathlib import Path

import pytest

from orze.engine.scheduler import run_cleanup


def _tree(tmp_path):
    results = tmp_path / "results"
    task = results / "idea-local"
    task.mkdir(parents=True)
    return results, task


def test_upward_pattern_cannot_delete_project_source(tmp_path):
    results, task = _tree(tmp_path)
    source = tmp_path / "train.py"
    source.write_bytes(b"project source must remain\n")
    before = source.read_bytes()
    run_cleanup(results, {"cleanup": {"patterns": ["../../train.py"]}})
    assert source.exists(), "cleanup escaped task and deleted project source"
    assert source.read_bytes() == before


def test_task_directory_symlink_cannot_delete_another_projects_artifact(tmp_path):
    results, task = _tree(tmp_path)
    other = tmp_path / "other_project" / "idea-other"
    other.mkdir(parents=True)
    artifact = other / "weights.pt"
    artifact.write_bytes(b"other project artifact")
    (results / "idea-linked").symlink_to(other, target_is_directory=True)
    run_cleanup(results, {"cleanup": {"patterns": ["*.pt"]}})
    assert artifact.exists(), "cleanup followed a task directory link"
    assert artifact.read_bytes() == b"other project artifact"


def test_intermediate_symlink_cannot_delete_another_tasks_artifact(tmp_path):
    results, task = _tree(tmp_path)
    other = results / "idea-peer"
    other.mkdir()
    artifact = other / "weights.pt"
    artifact.write_bytes(b"peer task artifact")
    (task / "borrowed").symlink_to(other, target_is_directory=True)
    run_cleanup(results, {"cleanup": {"patterns": ["borrowed/*.pt"]}})
    assert artifact.exists(), "cleanup followed a nested directory link"
    assert artifact.read_bytes() == b"peer task artifact"


def test_broad_glob_cannot_delete_native_routing_and_publication_records(tmp_path):
    results, task = _tree(tmp_path)
    paths = [task / "_execution_catalog.json",
             task / "_execution_effects" / "attempt" / "prepared.json",
             task / "_execution_stops" / "attempt" / "requested.json",
             task / "_evaluation_attempts" / "attempt" / "input_manifest.json"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{"retained":"framework evidence"}')
    run_cleanup(results, {"cleanup": {"patterns": ["**/*.json"]}})
    assert all(path.exists() for path in paths), "broad pattern deleted native evidence"
    assert all(path.read_bytes() == b'{"retained":"framework evidence"}' for path in paths)


def test_legacy_local_disposable_files_are_still_cleaned(tmp_path):
    results, task = _tree(tmp_path)
    nested = task / "scratch" / "cache.json"
    nested.parent.mkdir()
    nested.write_bytes(b"temporary")
    flat = task / "scratch.log"
    flat.write_bytes(b"temporary")
    assert run_cleanup(results, {"cleanup": {"patterns": ["**/*.json", "*.log"]}}) is None
    assert not nested.exists() and not flat.exists()


def test_default_disabled_cleanup_does_not_open_authority_or_change_files(tmp_path):
    results, task = _tree(tmp_path)
    scratch = task / "scratch.json"
    scratch.write_bytes(b"unchanged")
    run_cleanup(results, {"idea_lake_db": str(tmp_path / "missing.db")})
    assert scratch.read_bytes() == b"unchanged"
    assert not (tmp_path / "missing.db").exists()
    assert sorted(path.name for path in task.iterdir()) == ["scratch.json"]
