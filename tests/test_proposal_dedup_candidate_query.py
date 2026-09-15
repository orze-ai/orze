"""Admission outcomes across indexed and missing-identity candidate groups."""
import hashlib
import sqlite3

import pytest

from orze.core.integrity import hash_config
from orze.core.proposal_admission import admit_proposal_in_tx
from test_proposal_admission import lake, _admit, _snapshot

MATCH = hash_config({"seed": 13})
SOURCE = hashlib.sha256(b"seed: 13\n").hexdigest()


def _history(lake, idea_id, config="seed: 13\n", fingerprint=MATCH, source=SOURCE,
             status="completed", kind="train"):
    # Deliberate imported metadata, including missing/corrupt derived fields.
    lake.conn.execute(
        "INSERT INTO ideas(idea_id,title,config,raw_markdown,status,kind,config_hash,config_source_sha256) "
        "VALUES (?,'Historical proposal',?,'',?,?,?,?)",
        (idea_id, config, status, kind, fingerprint, source),
    )


@pytest.mark.parametrize("first_missing", [False, True])
@pytest.mark.parametrize("kind", ["train", "native_cpu_action"])
def test_first_semantic_owner_keeps_row_order_across_candidate_groups(lake, first_missing, kind):
    _history(lake, "idea-z-first", "{seed: 13}\n", None if first_missing else MATCH,
             None if first_missing else SOURCE, status="CoMpLeTeD", kind=kind)
    _history(lake, "idea-a-second", fingerprint=MATCH if first_missing else None,
             source=SOURCE if first_missing else None, kind=kind)
    lake.conn.commit()
    before = _snapshot(lake)
    assert _admit(lake, kind=kind)["existing_id"] == "idea-z-first"
    assert _snapshot(lake) == before


@pytest.mark.parametrize("fingerprint,source,first_eligible", [
    (MATCH, SOURCE, True), (MATCH, None, True), (None, SOURCE, True),
    (None, None, True), ("wrong", None, True), (b"wrong", None, True),
    ("wrong", SOURCE, False), (b"wrong", SOURCE, False),
])
def test_missing_identity_union_keeps_null_and_typed_hash_behavior(lake, fingerprint, source, first_eligible):
    _history(lake, "idea-first", fingerprint=fingerprint, source=source)
    _history(lake, "idea-second")
    lake.conn.commit()
    before = _snapshot(lake)
    assert _admit(lake)["existing_id"] == ("idea-first" if first_eligible else "idea-second")
    assert _snapshot(lake) == before


@pytest.mark.parametrize("count", [1024, 1025])
@pytest.mark.parametrize("mixed", [False, True])
def test_overlap_is_counted_once_and_capacity_applies_to_combined_candidates(lake, count, mixed):
    for i in range(count):
        # A matching hash with absent source identity belongs in the result
        # once, even though both old OR predicates match that same row.
        _history(lake, f"idea-row-{i:04d}", fingerprint=None if mixed and i % 2 else MATCH,
                 source=None)
    lake.conn.commit()
    before = _snapshot(lake)
    result = _admit(lake)
    assert result["status"] == ("config_duplicate" if count == 1024 else "rejected")
    if count == 1024:
        assert result["existing_id"] == "idea-row-0000"
    else:
        assert result["reason"] == "proposal_dedup_capacity"
    assert _snapshot(lake) == before


@pytest.mark.parametrize("config", [b"seed: 13\n", "seed: 13\n#" + "x" * 65536], ids=["blob", "oversized"])
@pytest.mark.parametrize("first_missing", [False, True])
def test_unavailable_first_candidate_cannot_be_reordered_behind_valid_match(lake, config, first_missing):
    _history(lake, "idea-first", config, None if first_missing else MATCH, source=None)
    _history(lake, "idea-second", fingerprint=MATCH if first_missing else None, source=None)
    lake.conn.commit()
    before = _snapshot(lake)
    assert _admit(lake)["reason"] == "proposal_dedup_config_unavailable"
    assert _snapshot(lake) == before


@pytest.mark.parametrize("status", [None, "failed", "archived", "unknown"])
def test_ineligible_matching_and_missing_rows_do_not_block_a_legal_proposal(lake, status):
    _history(lake, "idea-indexed", status=status)
    _history(lake, "idea-missing", fingerprint=None, source=None, status=status)
    assert _admit_after_commit(lake)["status"] == "inserted"


def _admit_after_commit(lake, **kwargs):
    lake.conn.commit()
    return _admit(lake, **kwargs)


def test_nonmatching_current_yaml_cannot_be_declared_duplicate_by_derived_hash(lake):
    _history(lake, "idea-stale", "seed: 99\n")
    _history(lake, "idea-missing", "seed: 99\n", fingerprint=None, source=None)
    assert _admit_after_commit(lake)["status"] == "inserted"


@pytest.mark.parametrize("drop", [("idx_status_config_hash_nocase",), ("idx_missing_config_identity",),
                                  ("idx_status_config_hash_nocase", "idx_missing_config_identity")])
def test_missing_optional_index_changes_cost_not_admission_semantics(lake, drop):
    _history(lake, "idea-first", fingerprint=None, source=None)
    _history(lake, "idea-second")
    for name in drop:
        lake.conn.execute(f"DROP INDEX {name}")
    lake.conn.commit()
    before = _snapshot(lake)
    assert _admit(lake)["existing_id"] == "idea-first"
    assert _snapshot(lake) == before


def test_caller_writer_observes_config_invalidation_and_retains_rollback_ownership(lake):
    _history(lake, "idea-owner")
    lake.conn.commit()
    before = _snapshot(lake)
    lake.conn.execute("BEGIN IMMEDIATE")
    try:
        first = lake.prepare_proposal("idea-first", "First", "seed: 13\n", "")
        assert admit_proposal_in_tx(lake, first)["status"] == "config_duplicate"
        lake.conn.execute("UPDATE ideas SET config='seed: 99\n' WHERE idea_id='idea-owner'")
        assert lake.conn.execute("SELECT config_hash,config_source_sha256 FROM ideas").fetchone()[:] == (None, None)
        second = lake.prepare_proposal("idea-second", "Second", "seed: 13\n", "")
        assert admit_proposal_in_tx(lake, second)["status"] == "inserted"
        assert lake.conn.in_transaction
    finally:
        lake.conn.rollback()
    assert _snapshot(lake) == before


@pytest.mark.parametrize("caller_owned", [False, True])
def test_peer_cannot_change_candidates_between_the_two_current_reads(lake, caller_owned):
    _history(lake, "idea-owner")
    _history(lake, "idea-missing", "seed: 99\n", fingerprint=None, source=None)
    lake.conn.commit()
    before = _snapshot(lake)
    peer = sqlite3.connect(lake.db_path, timeout=0)
    observed = []

    def interleave(statement):
        if statement.startswith("SELECT rowid AS candidate_order") and "config_hash !=" in statement:
            observed.append(lake.conn.in_transaction)
            try:
                peer.execute("UPDATE ideas SET config='seed: 99\n' WHERE idea_id='idea-owner'")
                peer.commit()
                observed.append("committed")
            except sqlite3.OperationalError as exc:
                observed.append("blocked" if "locked" in str(exc) else str(exc))
                peer.rollback()

    try:
        lake.conn.set_trace_callback(interleave)
        if caller_owned:
            lake.conn.execute("BEGIN IMMEDIATE")
            prepared = lake.prepare_proposal("idea-a", "Proposal", "seed: 13\n", "raw")
            result = admit_proposal_in_tx(lake, prepared)
            assert lake.conn.in_transaction
            lake.conn.rollback()
        else:
            result = _admit(lake)
        lake.conn.set_trace_callback(None)
        assert observed == [True, "blocked"], "exercise the inter-read peer write under the actual writer"
        assert result["existing_id"] == "idea-owner"
        assert _snapshot(lake) == before
        # Once that writer closes the peer can commit; the next invocation
        # must see the changed configuration and admit the now-legal proposal.
        peer.execute("UPDATE ideas SET config='seed: 99\n' WHERE idea_id='idea-owner'")
        peer.commit()
        assert _admit(lake)["status"] == "inserted"
    finally:
        lake.conn.set_trace_callback(None)
        if lake.conn.in_transaction:
            lake.conn.rollback()
        peer.close()
