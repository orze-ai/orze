"""V1-02D read-only-design diagnostics, outside repository collection.

A stale process handle is an explicit precondition, not a claim that normal
orphan cleanup creates overlapping live trainers. Admission, retry, state and
compute receipts use real Core APIs; only child processes are synthetic.
"""
import json
import time

import pytest

from orze.engine.accounting import (
    ComputeAccountingError, record_compute_start, record_compute_terminal,
)
from orze.engine.failure import _reset_idea_for_retry
from orze.engine.launcher import check_active
from orze.engine.process import TrainingProcess
from orze.engine.scheduler import claim
from orze.idea_lake import IdeaLake


class FinishedProcess:
    def __init__(self, pid, returncode):
        self.pid = pid
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def retry_case(tmp_path, request):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "lake.db"))
    idea_id = "idea-retry"
    lake.insert(idea_id, "Task", "seed: 13\n", "proposal", status="queued")
    assert claim(idea_id, results, 0, lake=lake)
    idea_dir = results / idea_id
    a_claim = json.loads((idea_dir / "claim.json").read_text())
    assert lake.record_state_transition(idea_id, "CLAIMED", "IN_PROGRESS", "attempt_a_start", pid=10001)
    old_exit = getattr(request, "param", 1)
    old = TrainingProcess(
        idea_id, 0, FinishedProcess(10001, old_exit), time.time() - 1,
        idea_dir / "train_output.log", 60, attempt_id=a_claim["attempt_id"],
    )
    record_compute_start(old, idea_dir)
    record_compute_terminal(
        old, idea_dir, "failed",
        "trainer_declared_failed" if old_exit == 0 else "process_exit_nonzero",
        return_code=old_exit,
    )
    assert lake.record_state_transition(idea_id, "IN_PROGRESS", "FAILED", "attempt_a_failed", pid=10001)
    (idea_dir / "metrics.json").write_text('{"status":"FAILED","error":"old attempt"}')
    _reset_idea_for_retry(idea_dir, release_claim=True)
    assert lake.record_state_transition(idea_id, "FAILED", "QUEUED", "explicit_retry")
    assert claim(idea_id, results, 0, lake=lake)
    b_claim = json.loads((idea_dir / "claim.json").read_text())
    assert b_claim["attempt_id"] != old.attempt_id
    assert lake.record_state_transition(idea_id, "CLAIMED", "IN_PROGRESS", "attempt_b_start", pid=10002)
    current = TrainingProcess(
        idea_id, 0, FinishedProcess(10002, None), time.time(),
        idea_dir / "train_output.log", 60, attempt_id=b_claim["attempt_id"],
    )
    record_compute_start(current, idea_dir)
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": "IN_PROGRESS", "attempt_id": current.attempt_id, "step": 7}),
        encoding="utf-8",
    )
    try:
        yield lake, results, idea_dir, old, current
    finally:
        lake.close()


@pytest.mark.parametrize("retry_case", [0, 1], indirect=True)
def test_stale_process_completion_cannot_fail_current_attempt_or_rewrite_its_metrics(retry_case):
    lake, results, idea_dir, old, current = retry_case
    # Exit zero with trainer-declared failure is another legitimate failed
    # allocation. The receipt stores the first terminal facts; a delayed poll
    # must not interpret the next attempt's partial metrics as that old output.
    metrics_before = (idea_dir / "metrics.json").read_bytes()
    claim_before = (idea_dir / "claim.json").read_bytes()
    history_before = lake.get_fsm_history(old.idea_id)
    failures = {}
    check_active(
        {0: old}, results,
        {"max_fix_attempts": 0, "sops": {"failure_feedback": False}},
        failures, lake=lake,
    )
    assert (idea_dir / "claim.json").read_bytes() == claim_before
    assert lake.get_fsm_state(old.idea_id) == "IN_PROGRESS", "stale A must not fail running B"
    assert lake.get_fsm_history(old.idea_id) == history_before
    assert (idea_dir / "metrics.json").read_bytes() == metrics_before
    assert failures == {}


def test_generic_state_name_cas_is_not_an_attempt_fence(retry_case):
    lake, _, _, old, current = retry_case
    assert old.attempt_id != current.attempt_id
    applied = lake.record_state_transition(
        old.idea_id, "IN_PROGRESS", "COMPLETE", "late_attempt_a_completion", pid=old.process.pid,
    )
    # This existing generic API treats pid as audit metadata, not ownership.
    # Positive control: it does what it declares, but cannot substitute for a
    # worker-facing expected-attempt/generation compare-and-swap boundary.
    assert applied is True
    assert lake.get_fsm_state(old.idea_id) == "COMPLETE"


def test_existing_claim_blocks_a_second_scheduler_claim(retry_case):
    lake, results, idea_dir, _, _ = retry_case
    before = (idea_dir / "claim.json").read_bytes()
    assert not claim("idea-retry", results, 0, lake=lake)
    assert (idea_dir / "claim.json").read_bytes() == before


def test_compute_receipt_rejects_conflicting_outcome_for_the_same_attempt(retry_case):
    _, _, idea_dir, old, _ = retry_case
    with pytest.raises(ComputeAccountingError, match="conflicting_receipt"):
        record_compute_terminal(old, idea_dir, "completed", "trainer_completed", return_code=0)


def test_fsm_rejects_a_truly_different_expected_state(retry_case):
    lake, _, _, old, _ = retry_case
    before = lake.get_fsm_history(old.idea_id)
    assert not lake.record_state_transition(old.idea_id, "CLAIMED", "FAILED", "stale_state")
    assert lake.get_fsm_history(old.idea_id) == before
