"""Draft training import must honor the real claim's explicit catalog scope."""
import json
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine.accounting import record_compute_start
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.scheduler import claim
from orze.engine.training_completion import finish
from orze.idea_lake import IdeaLake


@pytest.fixture
def case(tmp_path):
    actual = IdeaLake(tmp_path / "actual.db")
    other = IdeaLake(tmp_path / "other.db")
    idea = "idea-bound-catalog"
    results = tmp_path / "results"
    for lake in (actual, other):
        lake.insert(idea, "Catalog boundary", "{}", "", status="queued")
    assert claim(idea, results, 0, lake=actual)
    assert actual.record_state_transition(idea, "CLAIMED", "IN_PROGRESS")
    assert other.record_state_transition(idea, "QUEUED", "CLAIMED")
    assert other.record_state_transition(idea, "CLAIMED", "IN_PROGRESS")
    folder = results / idea
    claim_value = json.loads((folder / "claim.json").read_text())
    tp = SimpleNamespace(idea_id=idea, attempt_id=claim_value["attempt_id"],
                         attempt_ref=None, gpu=0, start_time=1000.0,
                         process=SimpleNamespace(pid=12345))
    record_compute_start(tp, folder)
    (folder / "metrics.json").write_text('{"status":"COMPLETED","score":0}')
    try:
        yield actual, other, folder, tp
    finally:
        other.close()
        actual.close()


def _files(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def test_legacy_completion_cannot_import_claim_bound_to_a_different_catalog(case):
    actual, other, folder, tp = case
    before = _files(folder)
    history = {lake.db_path: lake.get_fsm_history(tp.idea_id) for lake in (actual, other)}
    failures = {}
    with pytest.raises((AttemptEffectBusy, AttemptEffectInDoubt)):
        finish(other, tp, 0, folder, {}, 0, failures)
    assert failures == {}
    assert _files(folder) == before
    for lake in (actual, other):
        assert lake.get_fsm_history(tp.idea_id) == history[lake.db_path]
        assert current_attempt(lake.conn, tp.idea_id, "training") is None


def test_legacy_completion_with_its_exact_bound_catalog_remains_compatible(case):
    actual, other, folder, tp = case
    result = finish(actual, tp, 0, folder, {}, 0, {})
    assert result == (tp.idea_id, 0)
    assert actual.get_fsm_state(tp.idea_id) == "COMPLETE"
    assert other.get_fsm_state(tp.idea_id) == "IN_PROGRESS"
    assert current_attempt(actual.conn, tp.idea_id, "training")["state"] == "TERMINAL"
    assert current_attempt(other.conn, tp.idea_id, "training") is None
