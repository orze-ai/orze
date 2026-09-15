"""Real immutable ingress: cache hints never decide admission, bounded sidecars."""
import pytest

from orze.engine import idea_ingress
from test_idea_ingress_contract import engine, _block


@pytest.mark.parametrize("status,stale_config", [("completed", True), ("failed", False), ("archived", False)])
def test_stale_config_cache_does_not_suppress_a_legal_proposal(engine, status, stale_config):
    instance, cfg, source = engine
    instance.lake.insert("idea-history", "History", "seed: 2\n" if stale_config else "seed: 1\n",
                         "", status=status)
    instance._save_config_hash("idea-history", {"seed": 1})
    source.write_text(_block("idea-new", 1))
    _, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert inserted == ["idea-new"]
    assert source.read_text() == ""
    assert instance.lake.get("idea-new")["status"] == "queued"


def test_large_global_cache_is_never_loaded_but_real_duplicate_is_retained(engine, monkeypatch):
    instance, cfg, source = engine
    instance.lake.insert("idea-history", "History", "seed: 1\n", "", status="completed")
    instance._save_config_hash("idea-history", {"seed": 1})
    monkeypatch.setattr(instance, "_load_config_hashes", lambda: pytest.fail("global cache read"))
    content = _block("idea-duplicate", 1) + _block("idea-new", 2)
    source.write_text(content)
    _, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert inserted == ["idea-new"]
    assert source.read_text() == _block("idea-duplicate", 1)
    assert instance.lake.get("idea-duplicate") is None


def test_sidecars_only_parse_the_requested_window_and_reach_tail(engine, monkeypatch):
    import orze.core.ideas as ideas
    instance, cfg, source = engine
    side = source.parent / "ideas.d"
    side.mkdir()
    for i in range(300):
        (side / f"{i:04d}.md").write_text(_block(f"idea-side-{i:04d}", i))
    calls = []
    original = ideas.yaml.safe_load

    def count(value, *args, **kwargs):
        calls.append(value)
        return original(value, *args, **kwargs)

    # Count only sidecar parsing before the first Lake membership lookup.
    seen = []
    lookup = instance.lake.find_existing_ids

    def lookup_after_read(ids):
        seen.append(len(calls))
        return lookup(ids)

    monkeypatch.setattr(ideas.yaml, "safe_load", count)
    monkeypatch.setattr(instance.lake, "find_existing_ids", lookup_after_read)
    first = idea_ingress.ingest_ideas_source(instance, cfg)
    assert first[1] == [f"idea-side-{i:04d}" for i in range(128)]
    assert seen[0] <= 129
    assert len(first[0]) == 128
    for _ in range(2):
        idea_ingress.ingest_ideas_source(instance, cfg)
    assert instance.lake.get("idea-side-0299") is not None
    assert len(list(side.glob("*.md"))) == 300
    assert all(p.read_text() for p in side.glob("*.md"))


def test_full_primary_window_does_not_read_sidecars(engine, monkeypatch):
    from pathlib import Path
    instance, cfg, source = engine
    source.write_text("".join(_block(f"idea-main-{i:04d}", i) for i in range(129)))
    side = source.parent / "ideas.d"
    side.mkdir()
    candidate = side / "tail.md"
    candidate.write_text(_block("idea-tail", 999))
    real = Path.read_text

    def read(path, *args, **kwargs):
        if path == candidate:
            pytest.fail("off-page sidecar read")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    fresh = idea_ingress._read_source
    def fresh_primary_only(path):
        assert path != candidate, "off-page sidecar fresh read"
        return fresh(path)
    monkeypatch.setattr(idea_ingress, "_read_source", fresh_primary_only)
    assert len(idea_ingress.ingest_ideas_source(instance, cfg)[1]) == 128
    assert "idea-main-0128" in source.read_text()


def test_duplicate_sidecars_keep_first_valid_definition_and_primary_priority(engine):
    instance, cfg, source = engine
    source.write_text(_block("idea-main", 1))
    side = source.parent / "ideas.d"
    side.mkdir()
    (side / "a.md").write_text("## idea-duplicate: Broken\n```yaml\nseed: [\n```\n")
    (side / "b.md").write_text(_block("idea-duplicate", 2) + _block("idea-main", 8))
    (side / "c.md").write_text(_block("idea-duplicate", 3))
    _, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert inserted == ["idea-main", "idea-duplicate"]
    assert instance.lake.get("idea-main")["config"] == "seed: 1\n"
    assert instance.lake.get("idea-duplicate")["config"] == "seed: 2\n"
    assert (side / "a.md").exists() and (side / "c.md").exists()


def test_large_sidecar_batch_is_split_without_losing_the_delayed_entry(engine):
    instance, cfg, source = engine
    side = source.parent / "ideas.d"
    side.mkdir()
    # Each valid source is smaller than the primary source limit, but the
    # combined raw payload must not grow with the number of sidecars.
    for i in range(3):
        (side / f"{i}.md").write_text(_block(f"idea-large-{i}", i) + "n" * (2 * 1024 * 1024))
    first, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert len(inserted) == 1
    assert sum(len(v["raw"].encode()) for v in first.values()) <= 4 * 1024 * 1024
    for _ in range(2):
        idea_ingress.ingest_ideas_source(instance, cfg)
    assert instance.lake.get("idea-large-2") is not None


def test_oversized_sidecar_is_retained_and_does_not_hide_valid_other_file(engine):
    instance, cfg, source = engine
    side = source.parent / "ideas.d"
    side.mkdir()
    oversized = side / "a.md"
    oversized.write_text(_block("idea-too-large", 1) + "x" * (4 * 1024 * 1024))
    (side / "b.md").write_text(_block("idea-valid", 2))
    _, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert inserted == ["idea-valid"]
    assert instance.lake.get("idea-too-large") is None
    assert oversized.exists()


@pytest.mark.parametrize("changed_to_duplicate", [False, True])
def test_config_change_after_lookup_is_decided_by_current_transaction(engine, monkeypatch, changed_to_duplicate):
    from orze.idea_lake import IdeaLake
    instance, cfg, source = engine
    instance.lake.insert("idea-peer", "Peer", "seed: 2\n" if changed_to_duplicate else "seed: 1\n",
                         "", status="completed")
    source.write_text(_block("idea-new", 1))
    peer = IdeaLake(cfg["idea_lake_db"])
    original = instance.lake.find_admitted_config_hashes
    changed = []
    def lookup(identities):
        result = original(identities)
        peer.conn.execute("UPDATE ideas SET config=? WHERE idea_id='idea-peer'",
                          ("seed: 1\n" if changed_to_duplicate else "seed: 2\n",))
        peer.conn.commit()
        changed.append(True)
        return result
    monkeypatch.setattr(instance.lake, "find_admitted_config_hashes", lookup)
    try:
        _, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
        assert changed == [True]
        assert inserted == ([] if changed_to_duplicate else ["idea-new"])
        assert source.read_text() == (_block("idea-new", 1) if changed_to_duplicate else "")
    finally:
        peer.close()


@pytest.mark.parametrize("kind", ["symlink", "changed_during_read"])
def test_unconfirmed_sidecar_is_not_admitted_or_consumed(engine, monkeypatch, kind):
    instance, cfg, source = engine
    side = source.parent / "ideas.d"
    side.mkdir()
    candidate = side / "a.md"
    original = _block("idea-unconfirmed", 1)
    candidate.write_text(original)
    if kind == "symlink":
        foreign = source.parent / "foreign.md"
        candidate.rename(foreign)
        candidate.symlink_to(foreign)
    else:
        real_read = idea_ingress._read_source
        def refuse(path):
            if path == candidate:
                raise ValueError("idea_source_changed_during_read")
            return real_read(path)
        monkeypatch.setattr(idea_ingress, "_read_source", refuse)
    (side / "b.md").write_text(_block("idea-valid", 2))
    _, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert inserted == ["idea-valid"]
    assert instance.lake.get("idea-unconfirmed") is None
    assert candidate.read_text() == original
