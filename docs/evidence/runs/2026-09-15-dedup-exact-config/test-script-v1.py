"""Current configuration and conservative limits across repeated duplicates."""
import pytest

from orze.core import proposal_admission
from orze.engine import idea_ingress
from orze.idea_lake import IdeaLake
from test_idea_ingress_contract import engine, _block
from test_proposal_admission import lake, _admit, _snapshot


@pytest.mark.parametrize("change", ["config", "status", "kind"])
def test_repeated_duplicate_rechecks_peer_change_inside_one_ingress_batch(engine, monkeypatch, change):
    instance, cfg, source = engine
    instance.lake.insert("idea-owner", "Owner", "seed: 13\n", "", status="completed")
    original = _block("idea-copy-a", 13) + _block("idea-copy-b", 13)
    source.write_text(original)
    peer = IdeaLake(cfg["idea_lake_db"])
    insert = instance.lake.insert
    observed = []

    def insert_then_change(*args, **kwargs):
        result = insert(*args, **kwargs)
        observed.append(result["status"])
        if len(observed) == 1:
            assert result["status"] == "config_duplicate"
            value = {"config": "seed: 99\n", "status": "failed", "kind": "native_cpu_action"}[change]
            peer.conn.execute(f"UPDATE ideas SET {change}=? WHERE idea_id='idea-owner'", (value,))
            peer.conn.commit()
        return result

    monkeypatch.setattr(instance.lake, "insert", insert_then_change)
    try:
        raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
        assert len(raw) == 2 and inserted == ["idea-copy-b"]
        assert observed == ["config_duplicate", "inserted"]
        assert source.read_text() == _block("idea-copy-a", 13)
        assert instance.lake.get("idea-copy-a") is None
        assert instance.lake.get("idea-copy-b")["status"] == "queued"
    finally:
        peer.close()


@pytest.mark.parametrize("stored", ["seed: 13\n", "{seed: 13} # same meaning\n"])
def test_exact_and_differently_formatted_configs_choose_first_current_owner(lake, stored):
    lake.insert("idea-first", "First", stored, "", status="completed")
    lake.insert("idea-second", "Second", "seed: 13\n", "", status="completed")
    before = _snapshot(lake)
    result = _admit(lake)
    assert result == {"status": "config_duplicate", "reason": "proposal_config_duplicate",
                      "idea_id": "idea-a", "existing_id": "idea-first"}
    assert _snapshot(lake) == before


@pytest.mark.parametrize("count,expected", [(1024, "config_duplicate"), (1025, "rejected")])
def test_exact_match_does_not_skip_candidate_capacity_refusal(lake, count, expected):
    lake.conn.executemany(
        "INSERT INTO ideas(idea_id,title,config,status) VALUES (?,'History','seed: 13\n','completed')",
        ((f"idea-history-{i}",) for i in range(count)),
    )
    lake.conn.commit()
    before = _snapshot(lake)
    result = _admit(lake)
    assert result["status"] == expected
    if expected == "rejected":
        assert result["reason"] == "proposal_dedup_capacity"
    else:
        assert result["existing_id"] == "idea-history-0"
    assert _snapshot(lake) == before


@pytest.mark.parametrize("unavailable", [b"seed: 13\n", "seed: 13\n#" + "x" * 65536])
def test_unavailable_preceding_config_is_not_hidden_by_later_exact_match(lake, unavailable):
    lake.conn.execute(
        "INSERT INTO ideas(idea_id,title,config,status) VALUES ('idea-unknown','Unknown',?,'completed')",
        (unavailable,),
    )
    lake.conn.commit()
    lake.insert("idea-exact", "Exact", "seed: 13\n", "", status="completed")
    before = _snapshot(lake)
    assert _admit(lake)["reason"] == "proposal_dedup_config_unavailable"
    assert _snapshot(lake) == before


def test_caller_owned_writer_rejects_forged_identity_even_with_exact_current_text(lake):
    lake.insert("idea-owner", "Owner", "seed: 13\n", "", status="completed")
    prepared = lake.prepare_proposal("idea-copy", "Copy", "seed: 13\n", "")
    prepared["config_hash"] = "0" * 64
    before = _snapshot(lake)
    lake.conn.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(proposal_admission.ProposalAdmissionError, match="proposal_config_identity_mismatch"):
            proposal_admission.admit_proposal_in_tx(lake, prepared)
        assert lake.conn.in_transaction
        assert _snapshot(lake) == before
    finally:
        lake.conn.rollback()
