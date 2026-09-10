"""Public GC path regressions, restricted to temporary fixture files.

The old public call runs first. When the new optional cfg parameter exists,
the same physical assertions are also exercised with actual project paths:
a no-cfg refusal alone is not evidence that the scoped implementation is safe.
New-parameter absence is never the reason for an old-source failure.
"""
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from orze.agents import orze_gc as gc


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    results = root / "results"
    task = results / "idea-local"
    task.mkdir(parents=True)
    (task / "metrics.json").write_text('{"status":"FAILED"}', encoding="utf-8")
    (task / "idea_config.yaml").write_text("seed: 13\n", encoding="utf-8")
    script = root / "src" / "train.py"
    script.parent.mkdir()
    script.write_bytes(b"# real declared project source\n")
    ideas = root / "ideas.md"
    ideas.write_bytes(b"# Empty inbox\n")
    cfg_path = root / "orze.yaml"
    cfg = {
        "_project_root": str(root), "_config_path": str(cfg_path),
        "_orze_dir": str(root / ".orze"), "results_dir": str(results),
        "train_script": str(script), "ideas_file": str(ideas),
        "report": {"primary_metric": "quality", "sort": "descending", "columns": []},
        "gc": {"keep_top": 0, "keep_recent": 0},
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    checkpoints = tmp_path / "external-checkpoints"
    checkpoints.mkdir()
    return SimpleNamespace(root=root, results=results, task=task,
                           script=script, cfg=cfg, checkpoints=checkpoints,
                           scratch=tmp_path)


def _calls(p, **options):
    arguments = dict(
        results_dir=p.results, checkpoints_dir=None, primary_metric="quality",
        keep_top=0, keep_recent=0, min_free_gb=0,
    )
    arguments.update(options)
    yield gc.run_gc(**arguments)
    if "cfg" in inspect.signature(gc.run_gc).parameters:
        yield gc.run_gc(**arguments, cfg=p.cfg)


def _checkpoint(p, name="idea-local", body=b"disposable checkpoint"):
    folder = p.checkpoints / name
    folder.mkdir()
    (folder / "model.bin").write_bytes(body)
    return folder


@pytest.mark.parametrize("root_kind", ["results", "project-ancestor"])
def test_gc_checkpoint_root_cannot_erase_results_or_project_inputs(project, root_kind):
    p = project
    before = {path: path.read_bytes() for path in (
        p.task / "metrics.json", p.task / "idea_config.yaml", p.script,
    )}
    checkpoint_root = p.results if root_kind == "results" else p.root

    for _ in _calls(p, checkpoints_dir=checkpoint_root):
        missing = [str(path) for path in before if not path.is_file()]
        assert not missing, "GC erased a protected results/source tree: " + repr(missing)
        assert {path: path.read_bytes() for path in before} == before


def test_gc_archive_cannot_publish_into_declared_source_directory(project):
    p = project
    source = p.task / "model.bin"
    source.write_bytes(b"must not relocate into project source")
    archive = p.script.parent
    original_script = p.script.read_bytes()

    for _ in _calls(p, archive_dir=archive):
        assert source.is_file(), "GC moved a result into its project source namespace"
        assert source.read_bytes() == b"must not relocate into project source"
        assert p.script.read_bytes() == original_script
        assert not (archive / p.task.name).exists()


def test_gc_does_not_delete_same_name_directory_replaced_after_scan(project, monkeypatch):
    p = project
    original = _checkpoint(p, body=b"scanned original")
    retired = p.scratch / "retired-scanned-directory"
    actual_scandir = os.scandir
    root_stat = p.checkpoints.stat()
    swapped = []

    def is_checkpoint_root(path):
        if type(path) is int:
            info = os.fstat(path)
            return (info.st_dev, info.st_ino) == (root_stat.st_dev, root_stat.st_ino)
        return Path(path).absolute() == p.checkpoints

    class Scan:
        def __init__(self, iterator):
            self.iterator = iterator

        def __iter__(self):
            return self

        def __next__(self):
            try:
                entry = next(self.iterator)
            except StopIteration:
                if not swapped:
                    original.rename(retired)
                    original.mkdir()
                    (original / "model.bin").write_bytes(b"new directory owner")
                    swapped.append(True)
                raise
            # Capture the real DirEntry metadata before the concurrent rename.
            # No fabricated identity and no mock of the GC domain decision.
            entry.stat(follow_symlinks=False)
            return entry

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def close(self):
            self.iterator.close()

    def scandir(path):
        iterator = actual_scandir(path)
        return Scan(iterator) if is_checkpoint_root(path) and not swapped else iterator

    monkeypatch.setattr(gc.os, "scandir", scandir)

    for _ in _calls(p, checkpoints_dir=p.checkpoints):
        if swapped:
            assert original.is_dir(), "GC deleted the replacement directory"
            assert (original / "model.bin").read_bytes() == b"new directory owner"
            assert (retired / "model.bin").read_bytes() == b"scanned original"
    assert swapped, "the scoped call must reach the real directory scan"


def test_gc_does_not_follow_checkpoint_directory_symlink(project):
    p = project
    foreign = p.scratch / "foreign-project" / "idea-foreign"
    foreign.mkdir(parents=True)
    artifact = foreign / "weights.pt"
    artifact.write_bytes(b"another project's data")
    link = p.checkpoints / "idea-linked"
    link.symlink_to(foreign, target_is_directory=True)

    for _ in _calls(p, checkpoints_dir=p.checkpoints):
        assert link.is_symlink()
        assert artifact.read_bytes() == b"another project's data"


def test_scoped_gc_keeps_selected_and_removes_only_independent_legacy_checkpoint(project):
    p = project
    disposable = _checkpoint(p)
    keep = _checkpoint(p, "idea-keep", b"explicitly retained")
    for _ in _calls(p, checkpoints_dir=p.checkpoints, extra_keep_ids={"idea-keep"}):
        pass

    assert not disposable.exists(), "a blanket no-cfg refusal is not a passing control"
    assert (keep / "model.bin").read_bytes() == b"explicitly retained"
    assert json.loads((p.task / "metrics.json").read_text()) == {"status": "FAILED"}
    assert p.script.read_bytes() == b"# real declared project source\n"
    assert not list(p.root.rglob("*.db"))
