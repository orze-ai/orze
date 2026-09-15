"""Real source admission, bounded prefix hints, and fresh precedence after edits."""
import os
from pathlib import Path

import pytest

from orze.engine import idea_ingress, sidecar_prefix
from test_idea_ingress_contract import engine, _block


def populate(source, count=300):
    side = source.parent / "ideas.d"
    side.mkdir()
    for index in range(count):
        (side / f"{index:04d}.md").write_text(_block(f"idea-side-{index:04d}", index))
    return side


def test_continuation_does_not_reread_preceding_sidecar_payloads(engine, monkeypatch):
    instance, cfg, source = engine
    side = populate(source)
    reads = []
    original = idea_ingress._read_source
    def read(path):
        if path.parent == side:
            reads.append(path.name)
        return original(path)
    monkeypatch.setattr(idea_ingress, "_read_source", read)
    assert len(idea_ingress.ingest_ideas_source(instance, cfg)[1]) == 128
    reads.clear()
    raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert inserted == [f"idea-side-{i:04d}" for i in range(128, 256)]
    assert len(raw) == 128
    assert len(reads) <= 129 and all(name >= "0128.md" for name in reads)
    idea_ingress.ingest_ideas_source(instance, cfg)
    assert instance.lake.get("idea-side-0299") is not None


@pytest.mark.parametrize("change", ["rewrite_restore_mtime", "replace", "unlink", "hardlink", "symlink", "namespace", "primary"])
def test_prefix_change_forces_current_source_read_before_later_admission(engine, monkeypatch, change):
    instance, cfg, source = engine
    side = populate(source)
    idea_ingress.ingest_ideas_source(instance, cfg)
    first = side / "0000.md"
    old = first.stat()
    if change == "rewrite_restore_mtime":
        first.write_text(_block("idea-side-0000", 9))
        os.utime(first, ns=(old.st_atime_ns, old.st_mtime_ns))
    elif change == "replace":
        replacement = side / "replacement"
        replacement.write_text(first.read_text())
        replacement.replace(first)
    elif change == "unlink":
        first.unlink()
    elif change == "hardlink":
        os.link(first, source.parent / "linked.md")
    elif change == "symlink":
        target = source.parent / "redirected.md"
        first.rename(target)
        first.symlink_to(target)
    elif change == "namespace":
        (side / "-new.md").write_text(_block("idea-new-prefix", 1000))
    else:
        source.write_text("# Changed primary\n")
    reads = []
    original = idea_ingress._read_source
    def read(path):
        if path.parent == side:
            reads.append(path.name)
        return original(path)
    monkeypatch.setattr(idea_ingress, "_read_source", read)
    raw, _ = idea_ingress.ingest_ideas_source(instance, cfg)
    assert reads and min(reads) < "0128.md"
    assert raw and all(value is not None for value in raw.values())
    # Previously admitted source identity still cannot be overwritten.
    assert instance.lake.get("idea-side-0000")["config"] == "seed: 0\n"


def test_changed_prefix_during_page_read_rejects_batch_before_any_insert(engine, monkeypatch):
    instance, cfg, source = engine
    side = populate(source)
    idea_ingress.ingest_ideas_source(instance, cfg)
    original = idea_ingress._read_source
    changed = []
    def read(path):
        value = original(path)
        if path == side / "0128.md" and not changed:
            (side / "0000.md").write_text(_block("idea-side-0000", 9))
            changed.append(True)
        return value
    monkeypatch.setattr(idea_ingress, "_read_source", read)
    raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert changed == [True] and not raw and not inserted
    assert instance.lake.get("idea-side-0128") is None
    assert instance._idea_sidecar_prefix.files == []


def test_duplicate_after_cached_prefix_cannot_override_first_valid_definition(engine):
    instance, cfg, source = engine
    side = populate(source)
    (side / "0256.md").write_text(_block("idea-side-0000", 999) + _block("idea-last", 1000))
    for _ in range(3):
        idea_ingress.ingest_ideas_source(instance, cfg)
    assert instance.lake.get("idea-side-0000")["config"] == "seed: 0\n"
    assert instance.lake.get("idea-last")["config"] == "seed: 1000\n"


@pytest.mark.parametrize("limit,value", [("MAX_FILES", 2), ("MAX_IDS", 2), ("MAX_ID_BYTES", 1)])
def test_bounded_hints_fall_back_without_losing_tail(engine, monkeypatch, limit, value):
    instance, cfg, source = engine
    populate(source)
    monkeypatch.setattr(sidecar_prefix, limit, value)
    for _ in range(3):
        idea_ingress.ingest_ideas_source(instance, cfg)
        prefix = instance._idea_sidecar_prefix
        assert len(prefix.files) <= sidecar_prefix.MAX_FILES
        assert prefix.id_count <= sidecar_prefix.MAX_IDS
        assert prefix.id_bytes <= sidecar_prefix.MAX_ID_BYTES
    assert instance.lake.get("idea-side-0299") is not None


def test_partial_multirecord_file_is_read_fresh_and_cached_ids_do_not_hide_new_content(engine):
    instance, cfg, source = engine
    side = source.parent / "ideas.d"
    side.mkdir()
    path = side / "all.md"
    path.write_text("".join(_block(f"idea-side-{i:04d}", i) for i in range(300)))
    idea_ingress.ingest_ideas_source(instance, cfg)
    assert instance._idea_sidecar_prefix.files == []
    path.write_text(path.read_text().replace("seed: 128\n", "seed: 999\n"))
    idea_ingress.ingest_ideas_source(instance, cfg)
    assert instance.lake.get("idea-side-0128")["config"] == "seed: 999\n"


def test_directory_windows_preserve_glob_order_with_bounded_name_lists(tmp_path, monkeypatch):
    for name in [f"{i:04d}.md" for i in range(33)] + [".hidden.md", "ignored.txt", "case.MD"]:
        (tmp_path / name).touch()
    (tmp_path / "directory.md").mkdir()
    monkeypatch.setattr(sidecar_prefix, "NAME_WINDOW", 5)
    real = sidecar_prefix.heapq.nsmallest
    windows = []
    def bounded(n, values):
        result = real(n, values)
        windows.append(len(result))
        assert len(result) <= n == 5
        return result
    monkeypatch.setattr(sidecar_prefix.heapq, "nsmallest", bounded)
    assert list(sidecar_prefix._names(tmp_path)) == [p.name for p in sorted(tmp_path.glob("*.md"))]
    assert len(windows) > 1


def test_new_cycle_discards_prefix_and_retries_current_sidecar_definition(engine):
    instance, cfg, source = engine
    side = populate(source, 129)
    for _ in range(2):
        idea_ingress.ingest_ideas_source(instance, cfg)
    assert instance._idea_ingress_cursor[2] == 0
    (side / "0000.md").write_text(_block("idea-new-cycle", 999))
    _, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert inserted == ["idea-new-cycle"]


def test_parser_exception_discards_partial_prefix_before_retry(engine, monkeypatch):
    instance, cfg, source = engine
    populate(source)
    idea_ingress.ingest_ideas_source(instance, cfg)
    original = sidecar_prefix._iter_sidecar_text
    def failing(text, seen):
        if "seed: 140\n" in text:
            raise TypeError("injected parser failure")
        yield from original(text, seen)
    with monkeypatch.context() as scoped:
        scoped.setattr(sidecar_prefix, "_iter_sidecar_text", failing)
        assert idea_ingress.ingest_ideas_source(instance, cfg) == ({}, [])
    assert instance._idea_sidecar_prefix.files == []
    _, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert inserted == [f"idea-side-{i:04d}" for i in range(128, 256)]


def test_changed_pid_rejects_publication_of_existing_prefix(engine, monkeypatch):
    instance, cfg, source = engine
    populate(source)
    idea_ingress.ingest_ideas_source(instance, cfg)
    prefix = instance._idea_sidecar_prefix
    previous = os.getpid()
    monkeypatch.setattr(sidecar_prefix.os, "getpid", lambda: previous + 1)
    with pytest.raises(ValueError, match="sidecar_prefix_changed"):
        prefix.verify()
    assert prefix.files == []
