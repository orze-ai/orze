"""Real ingress preserves preparation and admission without building an unused map.

Historical rows are imported metadata; no worker or provider runs in these tests.
Fault observers delegate to the real preparation or admission where applicable.
"""
import hashlib
import sqlite3

import pytest

from orze.core.integrity import hash_config
from orze.engine import idea_ingress
from orze.idea_lake import IdeaLake
from test_idea_ingress_contract import engine, _block


def history(lake, count, *, dense=False, missing=False):
    rows = []
    for index in range(count):
        seed = 13 if dense or index == 0 else 1000 + index
        config = f"seed: {seed}\n"
        rows.append((f"idea-old-{index:05d}", config,
                     ("queued", "PENDING", "running", "COMPLETED")[index % 4],
                     None if missing else hash_config({"seed": seed}),
                     None if missing else hashlib.sha256(config.encode()).hexdigest()))
    lake.conn.executemany(
        "INSERT INTO ideas(idea_id,title,config,raw_markdown,status,config_hash,config_source_sha256) "
        "VALUES (?,'Imported metadata',?,'',?,?,?)", rows)
    lake.conn.commit()


def mapping_queries(statements):
    return [s for s in statements if " ".join(s.lower().split()).startswith(
        "select config_hash, idea_id from ideas")]


@pytest.mark.parametrize("dense", [False, True])
def test_ingress_avoids_unused_owner_mapping_with_same_real_result(engine, dense):
    instance, cfg, source = engine
    history(instance.lake, 5000, dense=dense)
    original = _block("idea-copy", 13)
    source.write_text(original)
    before = list(instance.lake.conn.iterdump())
    statements = []
    instance.lake.conn.set_trace_callback(statements.append)
    raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    instance.lake.conn.set_trace_callback(None)
    assert set(raw) == {"idea-copy"} and inserted == []
    assert source.read_text() == original
    assert list(instance.lake.conn.iterdump()) == before
    assert statements.count("BEGIN IMMEDIATE") == statements.count("ROLLBACK") == 1
    assert not mapping_queries(statements), "ingress discarded this full matching-history result"


def test_missing_history_is_prepared_before_legal_normal_admission(engine):
    instance, cfg, source = engine
    history(instance.lake, 1500, missing=True)
    assert instance.lake.insert("idea-new", "Proposal", "seed: -1\n", "", if_absent=True)[
        "reason"] == "proposal_dedup_capacity"
    source.write_text(_block("idea-new", -1))
    statements = []
    instance.lake.conn.set_trace_callback(statements.append)
    raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    instance.lake.conn.set_trace_callback(None)
    assert set(raw) == {"idea-new"} and inserted == ["idea-new"]
    assert source.read_text() == "" and instance.lake.get("idea-new") is not None
    assert instance.lake.conn.execute("SELECT COUNT(*) FROM ideas WHERE config_hash IS NULL "
                                      "OR config_source_sha256 IS NULL").fetchone()[0] == 0
    assert not mapping_queries(statements)


@pytest.mark.parametrize("identities", [[], [None, True, 1, "x"], ["A" * 64], "a" * 64])
def test_preparation_without_valid_identity_never_touches_closed_database(tmp_path, identities):
    lake = IdeaLake(tmp_path / "lake.db")
    lake.close()
    assert lake.prepare_admitted_config_hashes(iter(identities)) is None


def test_preparation_consumes_entire_input_before_repair(engine):
    instance, _, _ = engine
    history(instance.lake, 2, missing=True)
    statements = []
    instance.lake.conn.set_trace_callback(statements.append)

    def broken_input():
        yield hash_config({"seed": 13})
        raise RuntimeError("input_iteration_failed")

    with pytest.raises(RuntimeError, match="input_iteration_failed"):
        instance.lake.prepare_admitted_config_hashes(broken_input())
    assert statements == []


def test_preparation_keeps_public_lookup_first_owner_and_source_edit_semantics(engine):
    instance, _, _ = engine
    history(instance.lake, 2, dense=True, missing=True)
    identity = hash_config({"seed": 13})
    assert instance.lake.prepare_admitted_config_hashes([None, identity, identity]) is None
    assert instance.lake.find_admitted_config_hashes([identity]) == {identity: "idea-old-00000"}
    instance.lake.conn.execute("UPDATE ideas SET config='seed: 17\n' WHERE idea_id='idea-old-00000'")
    instance.lake.conn.commit()
    instance.lake.prepare_admitted_config_hashes([identity])
    assert instance.lake.find_admitted_config_hashes([identity]) == {identity: "idea-old-00001"}
    assert instance.lake.find_admitted_config_hashes([hash_config({"seed": 17})]) == {
        hash_config({"seed": 17}): "idea-old-00000"}


@pytest.mark.parametrize("capability", ["absent", "noncallable"])
def test_legacy_adapter_falls_back_to_original_lookup_and_real_writer(engine, capability):
    instance, cfg, source = engine
    actual = instance.lake
    history(actual, 2, missing=True)
    original = _block("idea-copy", 13) + _block("idea-new", -1)
    source.write_text(original)
    calls = []

    class Adapter:
        def __getattr__(self, name):
            if name == "prepare_admitted_config_hashes":
                if capability == "absent":
                    raise AttributeError(name)
                return None
            return getattr(actual, name)

        def find_admitted_config_hashes(self, identities):
            calls.append(set(identities))
            return actual.find_admitted_config_hashes(identities)

    instance.lake = Adapter()
    raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert calls == [{hash_config({"seed": 13}), hash_config({"seed": -1})}]
    assert set(raw) == {"idea-copy", "idea-new"} and inserted == ["idea-new"]
    assert source.read_text() == _block("idea-copy", 13)
    assert actual.get("idea-copy") is None and actual.get("idea-new") is not None


@pytest.mark.parametrize("count,seed,expected", [(2, -1, "inserted"), (2, 13, "config_duplicate"),
                                                 (1500, -1, "rejected")])
def test_preparation_failure_still_uses_normal_admission_authority(engine, monkeypatch, count, seed, expected):
    instance, cfg, source = engine
    history(instance.lake, count, missing=True)
    original = _block("idea-proposed", seed)
    source.write_text(original)
    attempted, outcomes = [], []
    real_insert = instance.lake.insert

    def unavailable(identities):
        attempted.append(set(identities))
        raise sqlite3.OperationalError("injected preparation failure")

    def insert(*args, **kwargs):
        result = real_insert(*args, **kwargs)
        outcomes.append(result)
        return result

    monkeypatch.setattr(instance.lake, "prepare_admitted_config_hashes", unavailable)
    monkeypatch.setattr(instance.lake, "insert", insert)
    raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert attempted == [{hash_config({"seed": seed})}]
    assert len(outcomes) == 1 and outcomes[0]["status"] == expected
    assert set(raw) == {"idea-proposed"}
    assert inserted == (["idea-proposed"] if expected == "inserted" else [])
    assert source.read_text() == ("" if expected == "inserted" else original)
    if expected == "rejected":
        assert outcomes[0]["reason"] == "proposal_dedup_capacity"


def test_known_same_id_skips_legacy_preparation_but_still_acknowledges_exact_content(engine):
    instance, cfg, source = engine
    source.write_text(_block("idea-known", -1))
    assert idea_ingress.ingest_ideas_source(instance, cfg)[1] == ["idea-known"]
    history(instance.lake, 2, missing=True)
    source.write_text(_block("idea-known", -1))
    statements = []
    instance.lake.conn.set_trace_callback(statements.append)
    assert idea_ingress.ingest_ideas_source(instance, cfg)[1] == []
    instance.lake.conn.set_trace_callback(None)
    assert source.read_text() == ""
    assert not any("FROM ideas WHERE status" in s for s in statements)
    assert instance.lake.conn.execute("SELECT COUNT(*) FROM ideas WHERE config_hash IS NULL").fetchone()[0] == 2
