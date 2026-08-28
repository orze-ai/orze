import hashlib

import yaml

from orze.core.integrity import hash_config
from orze.engine.phases import OrzePhaseMixin
from orze.idea_lake import IdeaLake


def test_insert_persists_exact_config_identity(tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    source = "strategy: train\nlr: 0.001\npriority: high\n"
    lake.insert(
        "idea-001", "one", source, "", status="queued",
    )

    row = lake.conn.execute(
        "SELECT config_hash, config_source_sha256 FROM ideas "
        "WHERE idea_id = 'idea-001'"
    ).fetchone()
    assert row["config_hash"] == hash_config(yaml.safe_load(source))
    assert row["config_source_sha256"] == hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()
    assert lake.get_admitted_config_hashes() == {
        row["config_hash"]: "idea-001"
    }
    lake.close()


def test_legacy_identity_is_backfilled_once_and_source_edits_invalidate(
        tmp_path, monkeypatch):
    lake = IdeaLake(tmp_path / "lake.db")
    original = "strategy: train\nlr: 0.001\n"
    changed = "strategy: train\nlr: 0.002\n"
    lake.insert("idea-001", "one", original, "", status="queued")
    lake.conn.execute(
        "UPDATE ideas SET config_hash = NULL, config_source_sha256 = NULL "
        "WHERE idea_id = 'idea-001'"
    )
    lake.conn.commit()

    real_safe_load = yaml.safe_load
    calls = []

    def counted_safe_load(value):
        calls.append(value)
        return real_safe_load(value)

    monkeypatch.setattr("orze.idea_lake.yaml.safe_load", counted_safe_load)
    first = lake.get_admitted_config_hashes()
    assert first == {hash_config(real_safe_load(original)): "idea-001"}
    assert calls == [original]

    # A steady-state read trusts the source-bound stored identity and performs
    # no YAML parsing or write.
    calls.clear()
    assert lake.get_admitted_config_hashes() == first
    assert calls == []

    # Simulate an unsupported direct SQL config edit that forgot to update the
    # derived columns. The source digest must force recomputation.
    lake.conn.execute(
        "UPDATE ideas SET config = ? WHERE idea_id = 'idea-001'", (changed,)
    )
    lake.conn.commit()
    calls.clear()
    updated = lake.get_admitted_config_hashes()
    assert updated == {hash_config(real_safe_load(changed)): "idea-001"}
    assert calls == [changed]
    lake.close()


def test_admitted_identity_preserves_first_row_dedup_semantics(tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    lake.insert(
        "idea-001", "first", "lr: 0.001\npriority: high\n", "",
        status="completed",
    )
    lake.insert(
        "idea-002", "second", "priority: low\nlr: 0.001\n", "",
        status="queued",
    )

    identity = hash_config({"lr": 0.001})
    assert lake.get_admitted_config_hashes()[identity] == "idea-001"
    assert lake.find_admitted_config_hashes([identity]) == {
        identity: "idea-001"
    }
    assert lake.find_admitted_config_hashes([
        hash_config({"lr": 123.0})
    ]) == {}
    lake.close()


def test_empty_identity_batch_is_constant_time_and_does_not_touch_database(
        tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    lake.close()
    # A normal tick with no new proposals must not scan or even access the
    # lake. A closed connection makes accidental SQL observable.
    assert lake.find_admitted_config_hashes([]) == {}


class _PhaseHarness(OrzePhaseMixin):
    pass


def test_queue_parse_cache_is_exact_invalidating_and_mutation_safe(monkeypatch):
    owner = _PhaseHarness()
    real_safe_load = yaml.safe_load
    calls = []

    def counted_safe_load(value):
        calls.append(value)
        return real_safe_load(value)

    monkeypatch.setattr("orze.engine.phases.yaml.safe_load", counted_safe_load)
    source = "strategy: train\nnested:\n  width: 32\n"
    first = owner._parse_lake_queue_config("idea-001", source)
    first["nested"]["width"] = 999
    second = owner._parse_lake_queue_config("idea-001", source)
    assert second["nested"]["width"] == 32
    assert calls == [source]

    changed = source.replace("32", "64")
    assert owner._parse_lake_queue_config(
        "idea-001", changed
    )["nested"]["width"] == 64
    assert calls == [source, changed]

    owner._parse_lake_queue_config("idea-002", source)
    owner._prune_lake_queue_config_cache(["idea-002"])
    assert set(owner._queue_config_parse_cache) == {"idea-002"}
