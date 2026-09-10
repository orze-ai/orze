"""Frozen low-level GC containment; destructive fixtures are pytest tmp only.

These existing APIs have no project-authority argument on the old release.
An explicit safe refusal is acceptable; blind deletion is not. No missing
new API is used as a regression assertion.
"""
from pathlib import Path

from orze.agents.orze_gc import gc_checkpoints, gc_results, archive_to_cold_storage


def refuse_or_run(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except ValueError:
        return None


def test_checkpoint_api_cannot_treat_project_source_as_disposable(tmp_path):
    source = tmp_path / "idea-project-source" / "train.py"
    source.parent.mkdir()
    source.write_bytes(b"project source")
    refuse_or_run(gc_checkpoints, tmp_path, set())
    assert source.is_file(), "unscoped checkpoint API removed project source"
    assert source.read_bytes() == b"project source"


def test_checkpoint_root_redirect_cannot_delete_other_tree(tmp_path):
    foreign = tmp_path / "other-project" / "checkpoints" / "idea-owner"
    foreign.mkdir(parents=True)
    model = foreign / "weights.pt"
    model.write_bytes(b"other owner's artifact")
    redirected = tmp_path / "checkpoints"
    redirected.symlink_to(foreign.parent, target_is_directory=True)
    refuse_or_run(gc_checkpoints, redirected, set())
    assert model.is_file(), "checkpoint root link redirected deletion"
    assert model.read_bytes() == b"other owner's artifact"


def test_results_api_cannot_delete_unscoped_declared_input(tmp_path):
    task = tmp_path / "idea-source"
    task.mkdir()
    artifact = task / "dataset.bin"
    artifact.write_bytes(b"project input, not a disposable measurement")
    refuse_or_run(gc_results, tmp_path, set())
    assert artifact.is_file(), "unscoped results API removed arbitrary input"
    assert artifact.read_bytes() == b"project input, not a disposable measurement"


def test_cold_archive_cannot_overwrite_existing_destination(tmp_path):
    results, archive = tmp_path / "results", tmp_path / "archive"
    source = results / "idea-local" / "weights.pt"
    target = archive / "idea-local" / "weights.pt"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    source.write_bytes(b"new but unaccepted source")
    target.write_bytes(b"existing unrelated archive")
    refuse_or_run(archive_to_cold_storage, results, archive, set())
    assert target.read_bytes() == b"existing unrelated archive", "archive overwrote existing artifact"
    assert source.read_bytes() == b"new but unaccepted source"


def test_cold_archive_cannot_move_redirected_overlay(tmp_path):
    results, archive = tmp_path / "results", tmp_path / "archive"
    task = results / "idea-local"
    task.mkdir(parents=True)
    source = tmp_path / "other-project"
    source.mkdir()
    (source / "train.py").write_bytes(b"other project")
    overlay = task / "overlays"
    overlay.symlink_to(source, target_is_directory=True)
    refuse_or_run(archive_to_cold_storage, results, archive, set())
    assert overlay.is_symlink(), "archive moved a redirected source"
    assert (source / "train.py").read_bytes() == b"other project"


def test_dry_run_remains_nonmutating_without_scope(tmp_path):
    results, checkpoints, archive = tmp_path / "results", tmp_path / "checkpoints", tmp_path / "archive"
    for root in (results, checkpoints):
        task = root / "idea-local"
        task.mkdir(parents=True)
        (task / "weights.pt").write_bytes(b"unchanged")
    before = {str(path.relative_to(tmp_path)): path.read_bytes()
              for path in tmp_path.rglob("*") if path.is_file()}
    refuse_or_run(gc_checkpoints, checkpoints, {"idea-local"}, dry_run=True)
    refuse_or_run(gc_results, results, set(), dry_run=True)
    refuse_or_run(archive_to_cold_storage, results, archive, set(), dry_run=True)
    after = {str(path.relative_to(tmp_path)): path.read_bytes()
             for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before
    assert not archive.exists()
