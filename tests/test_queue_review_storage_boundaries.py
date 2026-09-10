"""Silent SQLite write suppression cannot be reported as committed review.

These three fault-injection cases target the initial K1 queue-review mechanism,
not an old-release API. Real SQLite RAISE(IGNORE) differs from RAISE(ABORT): it
returns normally and requires an explicit postcondition to reject the batch.
"""

import pytest

from orze.core import queue_review
from orze.idea_lake import IdeaLake


@pytest.fixture
def lake(tmp_path):
    store = IdeaLake(tmp_path / "authority.db")
    store.insert("idea-silent-write", "Synthetic queued task", "seed: 11", "",
                 status="queued", priority="medium")
    yield store
    store.close()


def _snapshot(lake):
    return {
        table: [tuple(row) for row in lake.conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        for table in ("ideas", "idea_state", "idea_stage_state", "idea_transitions",
                      "idea_stage_transitions", "idea_review_decisions")
    }


@pytest.mark.parametrize("write", ["priority", "receipt", "transition"])
def test_silently_ignored_review_write_rejects_and_rolls_back_every_effect(lake, write):
    batch = queue_review.review_batch(lake.db_path)
    lake.conn.execute(queue_review._SCHEMA)
    if write == "priority":
        lake.conn.execute(
            "CREATE TRIGGER ignore_priority BEFORE UPDATE OF priority ON ideas "
            "BEGIN SELECT RAISE(IGNORE); END")
        decision = "PRIORITIZE"
    elif write == "receipt":
        lake.conn.execute(
            "CREATE TRIGGER ignore_receipt BEFORE INSERT ON idea_review_decisions "
            "BEGIN SELECT RAISE(IGNORE); END")
        decision = "SKIP"
    else:
        lake.conn.execute(
            "CREATE TRIGGER ignore_transition BEFORE INSERT ON idea_transitions "
            "BEGIN SELECT RAISE(IGNORE); END")
        decision = "SKIP"
    lake.conn.commit()
    before = _snapshot(lake)

    with pytest.raises(queue_review.QueueReviewError):
        queue_review.apply_review_decisions(
            lake.db_path, batch,
            [{"idea_id": "idea-silent-write", "decision": decision,
              "reason": "Synthetic operational decision"}],
            allow_skip=True, allow_prioritize=True)

    assert _snapshot(lake) == before
    assert lake.get_fsm_state("idea-silent-write") == "QUEUED"
    assert lake.get("idea-silent-write")["priority"] == "medium"
