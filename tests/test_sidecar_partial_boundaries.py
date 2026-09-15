"""Invalidate partial-file IDs under current-source and cleanup boundaries."""
import os

import pytest

from orze.engine import idea_ingress, sidecar_prefix
from test_idea_ingress_contract import engine, _block
from test_sidecar_partial_prefix import populate


@pytest.mark.parametrize("change", ["rewrite_restore_mtime", "replace", "hardlink", "symlink", "unlink", "namespace", "primary"])
def test_partial_source_changes_require_fresh_parsing(engine, monkeypatch, change):
    instance, cfg, source = engine
    path = populate(source)
    idea_ingress.ingest_ideas_source(instance, cfg)
    old = path.stat()
    if change == "rewrite_restore_mtime":
        path.write_text(path.read_text().replace("seed: 128\n", "seed: 999\n"))
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    elif change == "replace":
        replacement = path.with_suffix(".replacement")
        replacement.write_text(path.read_text().replace("seed: 128\n", "seed: 999\n"))
        replacement.replace(path)
    elif change == "hardlink":
        os.link(path, source.parent / "linked.md")
    elif change == "symlink":
        target = source.parent / "redirected.md"
        path.rename(target)
        path.symlink_to(target)
    elif change == "unlink":
        path.unlink()
    elif change == "namespace":
        (path.parent / "0.md").write_text(_block("idea-before", 999))
    else:
        source.write_text("# Changed primary\n")
    seen_sizes = []
    original = sidecar_prefix._iter_sidecar_text
    def parse(text, seen):
        seen_sizes.append(len(seen))
        yield from original(text, seen)
    monkeypatch.setattr(sidecar_prefix, "_iter_sidecar_text", parse)
    raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert not seen_sizes or seen_sizes[0] == 0
    assert instance.lake.get("idea-side-0000")["config"] == "seed: 0\n"
    if change in ("rewrite_restore_mtime", "replace"):
        assert instance.lake.get("idea-side-0128")["config"] == "seed: 999\n"
    elif change in ("hardlink", "symlink", "unlink"):
        assert not raw and not inserted and instance.lake.get("idea-side-0128") is None
    elif change == "namespace":
        assert inserted[0] == "idea-side-0128" and inserted[-1] == "idea-side-0254"
    else:
        assert list(raw)[0] == "idea-side-0000" and not inserted


@pytest.mark.parametrize("when", ["after_read", "during_parse", "after_verify"])
def test_partial_changes_before_close_reject_before_writer(engine, monkeypatch, when):
    instance, cfg, source = engine
    path = populate(source)
    idea_ingress.ingest_ideas_source(instance, cfg)
    changed = []
    def change():
        if not changed:
            path.write_text(path.read_text().replace("seed: 128\n", "seed: 999\n"))
            changed.append(True)
    if when == "after_read":
        original = idea_ingress._read_source
        def read(p):
            value = original(p)
            if p == path:
                change()
            return value
        monkeypatch.setattr(idea_ingress, "_read_source", read)
    elif when == "during_parse":
        original = sidecar_prefix._iter_sidecar_text
        def parse(text, seen):
            for pair in original(text, seen):
                change()
                yield pair
        monkeypatch.setattr(sidecar_prefix, "_iter_sidecar_text", parse)
    else:
        prefix = instance._idea_sidecar_prefix
        original = prefix.verify
        checks = []
        def verify():
            original()
            checks.append(True)
            if len(checks) == 2:
                change()
        monkeypatch.setattr(prefix, "verify", verify)
    writes = []
    original_insert = instance.lake.insert
    def insert(*args, **kwargs):
        writes.append(args[0])
        return original_insert(*args, **kwargs)
    monkeypatch.setattr(instance.lake, "insert", insert)
    assert idea_ingress.ingest_ideas_source(instance, cfg) == ({}, [])
    assert changed and not writes
    prefix = instance._idea_sidecar_prefix
    assert prefix.partial is None and prefix.files == []
    assert instance.lake.get("idea-side-0128") is None


def test_partial_parser_exception_clears_hint_and_allows_fresh_retry(engine, monkeypatch):
    instance, cfg, source = engine
    populate(source)
    idea_ingress.ingest_ideas_source(instance, cfg)
    original = sidecar_prefix._iter_sidecar_text
    def parse(text, seen):
        for index, pair in enumerate(original(text, seen)):
            if index == 4:
                raise TypeError("injected partial parser failure")
            yield pair
    with monkeypatch.context() as scoped:
        scoped.setattr(sidecar_prefix, "_iter_sidecar_text", parse)
        assert idea_ingress.ingest_ideas_source(instance, cfg) == ({}, [])
    prefix = instance._idea_sidecar_prefix
    assert prefix.partial is None and prefix.files == []
    _, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert inserted == [f"idea-side-{i:04d}" for i in range(128, 256)]


def test_backward_offset_discards_partial_hint_and_returns_current_records(engine):
    instance, cfg, source = engine
    populate(source)
    idea_ingress.ingest_ideas_source(instance, cfg)
    batch, raw, offset = idea_ingress._batch(source, source.read_text(), [], {}, 64,
                                            prefix=instance._idea_sidecar_prefix)
    assert [row[0] for row in batch] == [f"idea-side-{i:04d}" for i in range(64, 192)]
    assert all(value is not None for value in raw.values()) and offset == 192


def test_raw_byte_boundary_does_not_cache_unconsumed_lookahead(tmp_path, monkeypatch):
    source = tmp_path / "ideas.md"
    primary = _block("idea-primary", 999) + "x" * 250
    source.write_text(primary)
    side = tmp_path / "ideas.d"
    side.mkdir()
    blocks = [_block(f"idea-side-{i:04d}", i) for i in range(3)]
    (side / "all.md").write_text("".join(blocks))
    # _iter_sidecar_text strips heading from raw; obtain the actual boundary.
    records = list(sidecar_prefix._iter_sidecar_text("".join(blocks), set()))
    limit = len(primary.encode()) + sum(len(v["raw"].encode()) for _, v in records[:2])
    monkeypatch.setattr(idea_ingress, "_MAX_SOURCE_BYTES", limit)
    prefix = sidecar_prefix.SidecarPrefix()
    candidates = [("idea-primary", 0, len(primary))]
    batch, raw, offset = idea_ingress._batch(source, primary, candidates, {"idea-primary": 1}, 0, prefix=prefix)
    assert [row[0] for row in batch] == ["idea-primary", "idea-side-0000", "idea-side-0001"]
    assert offset == 3 and prefix.partial[2] == ("idea-side-0000", "idea-side-0001")
    batch, raw, offset = idea_ingress._batch(source, primary, candidates, {"idea-primary": 1}, offset, prefix=prefix)
    assert [row[0] for row in batch] == ["idea-side-0002"] and offset == 0
    assert raw["idea-side-0002"]["config"] == {"seed": 2}


@pytest.mark.parametrize("limit,value", [("MAX_FILES", 0), ("MAX_IDS", 2), ("MAX_ID_BYTES", 1)])
def test_partial_capacity_fallback_preserves_complete_tail(engine, monkeypatch, limit, value):
    instance, cfg, source = engine
    populate(source)
    monkeypatch.setattr(sidecar_prefix, limit, value)
    for _ in range(3):
        idea_ingress.ingest_ideas_source(instance, cfg)
        prefix = instance._idea_sidecar_prefix
        assert prefix.partial is None
    assert instance.lake.get("idea-side-0299")["config"] == "seed: 299\n"


def test_partial_hint_does_not_cross_an_unconfirmed_preceding_file(engine):
    instance, cfg, source = engine
    path = populate(source)
    (path.parent / "0.md").write_text("")
    idea_ingress.ingest_ideas_source(instance, cfg)
    assert instance._idea_sidecar_prefix.partial is None
    assert instance._idea_sidecar_prefix.files == []
    for _ in range(2):
        idea_ingress.ingest_ideas_source(instance, cfg)
    assert instance.lake.get("idea-side-0299")["config"] == "seed: 299\n"
