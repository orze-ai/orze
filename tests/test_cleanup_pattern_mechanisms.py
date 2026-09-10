"""Small new cleanup-helper mechanisms; API absence is not an old red."""
from pathlib import Path

import pytest

from orze.engine.cleanup_patterns import cleanup_pattern_files


def tree(tmp_path):
    results = tmp_path / "results"
    task = results / "idea-disposable"
    task.mkdir(parents=True)
    scratch = task / "scratch.tmp"
    scratch.write_bytes(b"disposable")
    return results, task, scratch


@pytest.mark.parametrize("database_state", ["missing", "malformed"])
def test_unavailable_explicit_catalog_never_becomes_legacy_cleanup(tmp_path, database_state):
    results, task, scratch = tree(tmp_path)
    database = tmp_path / "authority.db"
    if database_state == "malformed":
        database.write_bytes(b"not a database")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = cleanup_pattern_files(results, {"idea_lake_db": str(database),
                                            "cleanup": {"patterns": ["*.tmp"]}})
    assert report["status"] == "partial" and report["deleted"] == 0
    assert report["skipped_tasks"] == 1 and report["errors"]
    assert scratch.read_bytes() == b"disposable"
    assert all(path.read_bytes() == content for path, content in before.items())
    if database_state == "missing":
        assert not database.exists()
    assert not list(tmp_path.glob("authority.db-*"))


def test_configured_sources_control_tree_and_report_sources_are_not_disposable(tmp_path):
    results, task, scratch = tree(tmp_path)
    paths = [task / name for name in ("train.py", "base.yaml", "inbox.md", "eval.py", "measurements.json",
                                     "cleanup.py", "post.py", "sealed.py", "bundle.py", "pre.py")]
    for path in paths:
        path.write_bytes(b"declared input or evidence")
    control = task / "control" / "artifacts" / "artifact" / "content"
    control.parent.mkdir(parents=True)
    control.write_bytes(b"framework controlled bytes")
    cfg = {"_project_root": str(tmp_path), "_orze_dir": str(task / "control"),
           "train_script": str(paths[0]), "base_config": str(paths[1]),
           "ideas_file": str(paths[2]), "eval_script": str(paths[3]),
           "pre_script": str(paths[9]), "post_scripts": [{"script": str(paths[6])}],
           "sealed_hashes": {str(paths[7]): "a" * 64},
           "evaluation_bundle": {"enabled": True, "files": [str(paths[8])]},
           "report": {"columns": [{"key": "score", "source": "measurements.json:score"}]},
           "cleanup": {"patterns": ["**/*"], "script": str(paths[5])}}
    before = {path: path.read_bytes() for path in paths + [control]}
    report = cleanup_pattern_files(results, cfg)
    assert report["deleted"] == 1 and report["status"] == "completed"
    assert not scratch.exists()
    assert {path: path.read_bytes() for path in before} == before


def test_regular_leaf_requirement_rejects_symlinks_and_hardlinks(tmp_path):
    import os
    results, task, scratch = tree(tmp_path)
    outside = tmp_path / "foreign.bin"
    outside.write_bytes(b"foreign bytes")
    (task / "symlink.tmp").symlink_to(outside)
    os.link(outside, task / "hardlink.tmp")
    report = cleanup_pattern_files(results, {"cleanup": {"patterns": ["*.tmp"]}})
    assert report["deleted"] == 1 and not scratch.exists()
    assert outside.read_bytes() == b"foreign bytes"
    assert (task / "symlink.tmp").is_symlink()
    assert (task / "hardlink.tmp").exists()


def test_refused_unlink_is_not_counted_as_success(tmp_path, monkeypatch):
    import os
    results, task, scratch = tree(tmp_path)
    original = os.unlink
    def denied(path, *args, **kwargs):
        if path == "scratch.tmp":
            raise PermissionError("selected disposable denied")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "unlink", denied)
    report = cleanup_pattern_files(results, {"cleanup": {"patterns": ["*.tmp"]}})
    assert report["status"] == "partial" and report["deleted"] == 0
    assert report["skipped_tasks"] == 1 and report["errors"]
    assert scratch.read_bytes() == b"disposable"


def test_disabled_batch_never_opens_or_initializes_authority(tmp_path, monkeypatch):
    import orze.engine.cleanup_patterns as module
    results, task, scratch = tree(tmp_path)
    def tripwire(*args, **kwargs):
        pytest.fail("disabled cleanup must not inspect authority")
    monkeypatch.setattr(module, "_open_authoritative_lifecycle", tripwire)
    report = cleanup_pattern_files(results, {"idea_lake_db": str(tmp_path / "missing.db"),
                                            "cleanup": {"patterns": []}})
    assert report == {"status": "disabled", "deleted": 0, "skipped_tasks": 0, "errors": []}
    assert scratch.read_bytes() == b"disposable"
    assert list(task.iterdir()) == [scratch]
