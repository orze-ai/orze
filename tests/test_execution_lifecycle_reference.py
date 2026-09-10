"""New controller-action lifecycle reference contract acceptance."""
import pytest

from orze.core.execution_attempts import create_attempt, finish_attempt, mark_running
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.execution_authority import execution_transaction, lifecycle_fence
from test_execution_authority import case


@pytest.mark.parametrize("late_mutation", [False, True])
def test_controller_terminal_validates_explicit_training_lifecycle(case, late_mutation):
    c = case
    if late_mutation:
        c.conn.execute("CREATE TRIGGER corrupt_action AFTER UPDATE ON execution_attempts "
                       "WHEN NEW.phase='launch_failure_report' AND NEW.state='TERMINAL' "
                       "BEGIN UPDATE idea_state SET current_state='IN_PROGRESS'; END")
        c.conn.commit()

    def publish():
        with execution_transaction(c.lake, c.folder) as tx:
            ref = create_attempt(c.conn, c.ref.task_id, "launch_failure_report", "action-A", {})
            mark_running(c.conn, ref)
            digest = tx.prepare(ref, {"operation": "controller_report"})
            assert c.lake._record_state_transition_in_tx(c.ref.task_id, "IN_PROGRESS", "FAILED")
            assert finish_attempt(c.conn, ref, {
                "effect_receipt_sha256": digest, "lifecycle_phase": "training",
                "lifecycle": lifecycle_fence(c.lake, c.ref.task_id, "training"),
            }) == "committed"

    if late_mutation:
        with pytest.raises(AttemptEffectInDoubt):
            publish()
        assert c.lock.is_dir()
        assert c.lake.get_fsm_state(c.ref.task_id) == "IN_PROGRESS"
    else:
        publish()
        assert c.lake.get_fsm_state(c.ref.task_id) == "FAILED"
        assert not c.lock.exists()


@pytest.mark.parametrize("phase", [None, True, "../training"])
def test_explicit_lifecycle_reference_rejects_invalid_tokens(case, phase):
    c = case
    with pytest.raises(AttemptEffectInDoubt):
        with execution_transaction(c.lake, c.folder) as tx:
            digest = tx.prepare(c.ref, {"operation": "test_invalid_reference"})
            assert finish_attempt(c.conn, c.ref, {
                "effect_receipt_sha256": digest, "lifecycle_phase": phase,
                "lifecycle": lifecycle_fence(c.lake, c.ref.task_id, "training"),
            }) == "committed"
    assert c.lock.is_dir()
