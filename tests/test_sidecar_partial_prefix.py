"""Real ingress checks for consumed IDs inside a freshly read partial file."""
from orze.engine import idea_ingress, sidecar_prefix
from test_idea_ingress_contract import engine, _block


def populate(source):
    side = source.parent / "ideas.d"
    side.mkdir()
    path = side / "a.md"
    path.write_text("".join(_block(f"idea-side-{i:04d}", i) for i in range(300)))
    return path


def test_partial_resume_does_not_reparse_consumed_yaml(engine, monkeypatch):
    instance, cfg, source = engine
    populate(source)
    idea_ingress.ingest_ideas_source(instance, cfg)
    calls = []
    original = sidecar_prefix._iter_sidecar_text
    def observe(text, seen):
        calls.append(len(seen))
        yield from original(text, seen)
    monkeypatch.setattr(sidecar_prefix, "_iter_sidecar_text", observe)
    raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert calls == [128]
    assert inserted == [f"idea-side-{i:04d}" for i in range(128, 256)]
    assert all(value is not None for value in raw.values())


def test_read_failure_does_not_grant_precedence_to_cached_ids(engine, monkeypatch):
    instance, cfg, source = engine
    path = populate(source)
    (path.parent / "b.md").write_text(_block("idea-tail", 999))
    idea_ingress.ingest_ideas_source(instance, cfg)
    original = idea_ingress._read_sidecar
    monkeypatch.setattr(idea_ingress, "_read_sidecar", lambda p: "" if p == path else original(p))
    # Fresh parsing sees only one record, before the inspection offset 128.
    # A failed selected-file read cannot supply 128 authoritative prior IDs.
    assert idea_ingress.ingest_ideas_source(instance, cfg) == ({}, [])
    assert instance.lake.get("idea-tail") is None


def test_unconsumed_lookahead_is_not_a_skippable_hint(engine):
    instance, cfg, source = engine
    populate(source)
    idea_ingress.ingest_ideas_source(instance, cfg)
    prefix = instance._idea_sidecar_prefix
    assert prefix.partial is not None and len(prefix.partial[2]) == 128
    assert prefix.partial[2][-1] == "idea-side-0127"
    assert prefix.files == []
    _, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert inserted[0] == "idea-side-0128"


def test_clear_before_close_cannot_republish_partial_hint(tmp_path):
    source = tmp_path / "ideas.md"
    source.write_text("")
    populate(source)
    prefix = sidecar_prefix.SidecarPrefix()
    stream = prefix.stream(source, set(), 0, read_text=lambda p: p.read_text())
    next(stream)
    next(stream)
    prefix.clear()
    stream.close()
    assert prefix.partial is None and prefix.files == [] and prefix.scope is None


def test_partial_hint_shares_id_capacity_with_complete_prefix(engine, monkeypatch):
    instance, cfg, source = engine
    path = populate(source)
    (path.parent / "0.md").write_text(_block("idea-prefix", 999))
    monkeypatch.setattr(sidecar_prefix, "MAX_IDS", 128)
    for _ in range(3):
        idea_ingress.ingest_ideas_source(instance, cfg)
        prefix = instance._idea_sidecar_prefix
        assert prefix.id_count + (len(prefix.partial[2]) if prefix.partial else 0) <= 128
    assert instance.lake.get("idea-side-0299") is not None
