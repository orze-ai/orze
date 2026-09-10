"""New queue-review mechanism contracts, not old-API behavioral red claims."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from threading import Barrier

import pytest

import orze.core.queue_review as review
from orze.idea_lake import IdeaLake


@pytest.fixture
def lake(tmp_path):
    store = IdeaLake(tmp_path / "authority.db")
    yield store
    store.close()


def _add(lake, idea_id="idea-one", **kwargs):
    lake.insert(idea_id, kwargs.pop("title", "A bounded task"),
                kwargs.pop("config", "seed: 13\nnested: {value: 2}"), "",
                hypothesis="A test hypothesis", status=kwargs.pop("status", "queued"),
                **kwargs)
    return idea_id


def _decision(idea_id="idea-one", decision="APPROVE", **kwargs):
    return {"idea_id": idea_id, "decision": decision, "reason": "operational review", **kwargs}


def _snapshot(lake):
    return {table: [tuple(row) for row in lake.conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
            for table in ("ideas", "idea_state", "idea_stage_state", "idea_transitions",
                          "idea_stage_transitions")}


def _receipts(lake):
    present = lake.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='idea_review_decisions'").fetchone()
    return [] if present is None else lake.conn.execute(
        "SELECT * FROM idea_review_decisions ORDER BY rowid").fetchall()


def test_read_is_bounded_closed_readonly_and_preserves_kind_config(lake, monkeypatch):
    _add(lake, kind="posthoc_eval")
    before = _snapshot(lake)
    original = review._open_authoritative_lifecycle
    connections, sql = [], []

    def observe(path):
        connection, reason = original(path)
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        connection.set_trace_callback(sql.append)
        connections.append(connection)
        return connection, reason

    monkeypatch.setattr(review, "_open_authoritative_lifecycle", observe)
    monkeypatch.setattr(IdeaLake, "__init__", lambda *a, **k: pytest.fail("observer bootstrap"))
    batch = review.review_batch(lake.db_path)

    assert len(batch) == 1
    assert batch[0]["config"] == {"seed": 13, "nested": {"value": 2}}
    assert batch[0]["kind"] == "posthoc_eval"
    assert len(batch[0]["revision"]) == 64
    assert _snapshot(lake) == before and _receipts(lake) == []
    assert any(statement == "BEGIN" for statement in sql)
    assert not any(statement.lstrip().upper().startswith(("UPDATE", "INSERT", "CREATE", "DELETE"))
                   for statement in sql)
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")


@pytest.mark.parametrize("limit", [0, 21, True, None])
def test_invalid_limits_do_not_create_authority(tmp_path, limit):
    with pytest.raises(review.QueueReviewError):
        review.review_batch(tmp_path / "absent.db", limit)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("unsafe", ["missing", "corrupt", "symlink", "hardlink"])
def test_unavailable_or_redirected_database_is_never_created_or_repaired(tmp_path, unsafe):
    path = tmp_path / "selected.db"
    if unsafe == "corrupt":
        path.write_bytes(b"not sqlite")
    elif unsafe in {"symlink", "hardlink"}:
        original = tmp_path / "original.db"
        store = IdeaLake(original)
        store.close()
        if unsafe == "symlink":
            path.symlink_to(original)
        else:
            path.hardlink_to(original)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(review.QueueReviewError):
        review.review_batch(path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_read_only_includes_three_way_unstarted_queue(lake):
    for idea_id in ("idea-good", "idea-pending", "idea-running", "idea-mirror", "idea-missing", "idea-stage"):
        _add(lake, idea_id)
    lake.conn.execute("UPDATE ideas SET status='pending' WHERE idea_id='idea-pending'")
    lake.conn.commit()
    assert lake.record_state_transition("idea-running", "QUEUED", "CLAIMED", "test claim")
    lake.conn.execute("UPDATE ideas SET status='completed' WHERE idea_id='idea-mirror'")
    lake.conn.execute("DELETE FROM idea_state WHERE idea_id='idea-missing'")
    lake.conn.execute("INSERT INTO idea_stage_state (idea_id,stage,current_state,updated_at) "
                      "VALUES ('idea-stage','training','COMPLETE',CURRENT_TIMESTAMP)")
    lake.conn.commit()

    assert {row["id"] for row in review.review_batch(lake.db_path)} == {"idea-good", "idea-pending"}


@pytest.mark.parametrize("field,value", [
    ("config", "x" * 65537), ("title", "x" * 4097), ("hypothesis", "x" * 16385),
    ("config", "[not, a, mapping]"), ("config", "recursive: &cycle [*cycle]"),
])
def test_undisplayable_metadata_is_omitted_not_condemned(lake, field, value):
    _add(lake, "idea-hidden")
    _add(lake, "idea-visible")
    lake.conn.execute(f"UPDATE ideas SET {field}=? WHERE idea_id='idea-hidden'", (value,))
    lake.conn.commit()
    before = _snapshot(lake)

    batch = review.review_batch(lake.db_path)

    assert [row["id"] for row in batch] == ["idea-visible"]
    assert _snapshot(lake) == before


def test_approved_prefix_cannot_starve_later_queue_rows(lake):
    for index in range(70):
        _add(lake, f"idea-{index:03d}")
    for _ in range(3):
        batch = review.review_batch(lake.db_path)
        assert len(batch) == 20
        review.apply_review_decisions(lake.db_path, batch,
                                      [_decision(row["id"]) for row in batch])

    assert [row["id"] for row in review.review_batch(lake.db_path, 1)] == ["idea-060"]
    assert len(review.review_batch(lake.db_path)) == 10


def test_approve_preserves_all_idea_and_lifecycle_fields_and_replay_is_rejected(lake):
    _add(lake)
    batch = review.review_batch(lake.db_path)
    before = _snapshot(lake)
    result = review.apply_review_decisions(lake.db_path, batch, [_decision()])

    assert result == [{**_decision(), "revision": batch[0]["revision"]}]
    assert _snapshot(lake) == before
    assert len(_receipts(lake)) == 1
    assert review.review_batch(lake.db_path) == []
    with pytest.raises(review.QueueReviewError, match="already_applied"):
        review.apply_review_decisions(lake.db_path, batch, [_decision()])
    assert _snapshot(lake) == before and len(_receipts(lake)) == 1


def test_prioritize_changes_only_priority_and_does_not_review_its_own_new_revision(lake):
    _add(lake)
    batch = review.review_batch(lake.db_path)
    before = _snapshot(lake)
    original = dict(lake.conn.execute("SELECT * FROM ideas").fetchone())

    result = review.apply_review_decisions(lake.db_path, batch, [_decision(decision="PRIORITIZE")],
                                          allow_prioritize=True)

    changed = dict(lake.conn.execute("SELECT * FROM ideas").fetchone())
    assert changed == {**original, "priority": "critical"}
    after = _snapshot(lake)
    assert all(after[key] == before[key] for key in after if key != "ideas")
    assert len(result) == 1 and review.review_batch(lake.db_path) == []


def test_skip_atomically_updates_mirror_fsm_and_audit_without_touching_results(lake, tmp_path):
    _add(lake, kind="agg_search")
    artifact = tmp_path / "results" / "idea-one" / "metrics.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'{"status":"COMPLETED","keep":"existing artifact"}')
    before_bytes = artifact.read_bytes()
    original = dict(lake.conn.execute("SELECT * FROM ideas").fetchone())
    result = review.apply_review_decisions(
        lake.db_path, review.review_batch(lake.db_path), [_decision(decision="SKIP")], allow_skip=True)

    assert dict(lake.conn.execute("SELECT * FROM ideas").fetchone()) == {**original, "status": "skipped"}
    assert lake.get_fsm_state("idea-one") == "SKIPPED"
    transitions = lake.conn.execute("SELECT from_state,to_state,sop_type FROM idea_transitions").fetchall()
    assert [tuple(row) for row in transitions] == [("QUEUED", "SKIPPED", "queue_review")]
    assert len(result) == len(_receipts(lake)) == 1
    assert artifact.read_bytes() == before_bytes


@pytest.mark.parametrize("field", ["config", "title", "hypothesis", "kind", "priority", "queued_at", "ABA"])
def test_each_bound_revision_change_rejects_the_whole_selected_batch(lake, field):
    _add(lake, "idea-one")
    _add(lake, "idea-two")
    batch = review.review_batch(lake.db_path)
    if field == "ABA":
        assert lake.record_state_transition("idea-two", "QUEUED", "CLAIMED", "test claim")
        assert lake.record_state_transition("idea-two", "CLAIMED", "QUEUED", "test return")
    elif field == "queued_at":
        lake.conn.execute("UPDATE idea_state SET queued_at='changed-clock' WHERE idea_id='idea-two'")
        lake.conn.commit()
    else:
        value = {"config": "seed: 99", "kind": "audit", "priority": "low"}.get(field, "changed")
        lake.conn.execute(f"UPDATE ideas SET {field}=? WHERE idea_id='idea-two'", (value,))
        lake.conn.commit()
    before = _snapshot(lake)

    with pytest.raises(review.QueueReviewError, match="stale"):
        review.apply_review_decisions(lake.db_path, batch,
                                      [_decision("idea-one", "SKIP"), _decision("idea-two")], allow_skip=True)
    assert _snapshot(lake) == before and _receipts(lake) == []


def test_reentering_queue_after_review_creates_new_reviewable_version(lake):
    _add(lake)
    batch = review.review_batch(lake.db_path)
    review.apply_review_decisions(lake.db_path, batch, [_decision()])
    assert lake.record_state_transition("idea-one", "QUEUED", "CLAIMED", "test claim")
    assert lake.record_state_transition("idea-one", "CLAIMED", "QUEUED", "test return")

    new = review.review_batch(lake.db_path)

    assert new[0]["revision"] != batch[0]["revision"]


@pytest.mark.parametrize("mutation", ["claim", "mirror", "stage", "missing_fsm"])
def test_current_queue_eligibility_is_rechecked_at_commit(lake, mutation):
    _add(lake)
    batch = review.review_batch(lake.db_path)
    if mutation == "claim":
        assert lake.record_state_transition("idea-one", "QUEUED", "CLAIMED", "concurrent claim")
    elif mutation == "mirror":
        lake.conn.execute("UPDATE ideas SET status='running' WHERE idea_id='idea-one'")
    elif mutation == "stage":
        lake.conn.execute("INSERT INTO idea_stage_state (idea_id,stage,current_state,updated_at) "
                          "VALUES ('idea-one','evaluation','IN_PROGRESS',CURRENT_TIMESTAMP)")
    else:
        lake.conn.execute("DELETE FROM idea_state WHERE idea_id='idea-one'")
    lake.conn.commit()
    before = _snapshot(lake)

    with pytest.raises(review.QueueReviewError, match="stale"):
        review.apply_review_decisions(lake.db_path, batch, [_decision(decision="SKIP")], allow_skip=True)
    assert _snapshot(lake) == before and _receipts(lake) == []


@pytest.mark.parametrize("decision", ["SKIP", "PRIORITIZE", "skip", "DELETE"])
def test_unpermitted_or_unknown_decision_never_writes(lake, decision):
    _add(lake)
    before = _snapshot(lake)
    with pytest.raises(review.QueueReviewError):
        review.apply_review_decisions(lake.db_path, review.review_batch(lake.db_path),
                                      [_decision(decision=decision)])
    assert _snapshot(lake) == before and _receipts(lake) == []


@pytest.mark.parametrize("bad", ["foreign", "unsafe", "duplicate", "revision"])
def test_unsubmitted_unsafe_duplicate_or_wrong_revision_output_rejects(lake, bad):
    _add(lake)
    batch = review.review_batch(lake.db_path)
    decisions = [_decision()]
    if bad == "duplicate":
        decisions.append(_decision())
    elif bad == "revision":
        decisions[0]["revision"] = "0" * 64
    else:
        decisions[0]["idea_id"] = "idea-unsubmitted" if bad == "foreign" else "../other"
    before = _snapshot(lake)
    with pytest.raises(review.QueueReviewError):
        review.apply_review_decisions(lake.db_path, batch, decisions)
    assert _snapshot(lake) == before and _receipts(lake) == []


def test_unselected_batch_row_change_does_not_block_selected_decision(lake):
    _add(lake, "idea-one")
    _add(lake, "idea-two")
    batch = review.review_batch(lake.db_path)
    assert lake.record_state_transition("idea-two", "QUEUED", "CLAIMED", "test claim")

    assert len(review.apply_review_decisions(lake.db_path, batch, [_decision()])) == 1
    assert lake.get_fsm_state("idea-two") == "CLAIMED"


def test_receipt_failure_rolls_back_all_prior_state_and_mirror_mutations(lake):
    _add(lake)
    _add(lake, "idea-two")
    batch = review.review_batch(lake.db_path)
    lake.conn.execute(review._SCHEMA)
    lake.conn.execute("CREATE TRIGGER reject_review BEFORE INSERT ON idea_review_decisions "
                      "WHEN NEW.idea_id='idea-two' "
                      "BEGIN SELECT RAISE(ABORT, 'synthetic receipt fault'); END")
    lake.conn.commit()
    before = _snapshot(lake)

    with pytest.raises(review.QueueReviewError, match="commit_failed"):
        review.apply_review_decisions(lake.db_path, batch,
                                      [_decision(decision="SKIP"), _decision("idea-two", "SKIP")],
                                      allow_skip=True)
    assert _snapshot(lake) == before and _receipts(lake) == []


def test_two_concurrent_approvals_commit_exactly_one_revision_receipt(lake):
    _add(lake)
    batch = review.review_batch(lake.db_path)
    barrier = Barrier(2)

    def submit():
        barrier.wait()
        try:
            return len(review.apply_review_decisions(lake.db_path, deepcopy(batch), [_decision()]))
        except review.QueueReviewError as error:
            assert str(error) == "queue_review_already_applied"
            return 0

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(lambda _: submit(), range(2))) == [0, 1]
    assert len(_receipts(lake)) == 1
    assert lake.get_fsm_state("idea-one") == "QUEUED"
