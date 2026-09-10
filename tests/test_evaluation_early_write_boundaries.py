"""Private fallback/early rejection cannot bypass a native effect owner.

Actual launch, SQLite transitions, file publication and durable prepared
intents are used. Only the existing GPU/process fixture is simulated.
These are D2 draft behavior regressions, not new-API absence failures.
"""
import pytest

from orze.engine import evaluator
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.execution_authority import execution_transaction
from orze.engine.termination_hold import TerminationUnconfirmed

from test_stale_evaluation_completion import (
    project, _prepare, _launch, _files, _lifecycle,
)


def _hold_after_prepare(p, folder, ep):
    with pytest.raises(AttemptEffectInDoubt):
        with execution_transaction(p.lake, folder) as tx:
            tx.prepare(ep.attempt_ref, {"operation": "fixture interrupted publication"})
            raise ValueError("fixture controller stopped after durable intent")
    assert (folder / "_execution_effects" / ep.attempt_id / "prepared.json").exists()
    assert (folder / "_attempt_effect.lock" / "lock.json").exists()


def test_tokenless_private_failure_cannot_publish_before_native_fsm_write_is_rejected(project):
    p = project
    folder = _prepare(p, "idea-private-native")
    _launch(p, folder.name)
    # A real zero-row SQL write makes the old fallback's split outcome
    # visible: it publishes FAILED output, then ignores the FSM rejection.
    p.lake.conn.execute(
        "CREATE TRIGGER reject_failure BEFORE UPDATE ON idea_state "
        "WHEN NEW.current_state='FAILED' BEGIN SELECT RAISE(IGNORE); END")
    p.lake.conn.commit()
    before_files, before_state = _files(folder), _lifecycle(p.lake)

    with pytest.raises(TerminationUnconfirmed):
        evaluator._write_eval_failure_marker(
            p.results, folder.name, "assessment.json", "old tokenless callback",
            lake=p.lake)

    assert _files(folder) == before_files
    assert _lifecycle(p.lake) == before_state
    assert p.lake.get_fsm_state(folder.name) == "IN_PROGRESS"


def test_private_failure_without_lake_cannot_write_through_real_prepared_hold(project):
    p = project
    folder = _prepare(p, "idea-private-hold")
    ep = _launch(p, folder.name)
    _hold_after_prepare(p, folder, ep)
    before_files, before_state = _files(folder), _lifecycle(p.lake)

    with pytest.raises(TerminationUnconfirmed):
        evaluator._write_eval_failure_marker(
            p.results, folder.name, "assessment.json", "late no-Lake failure")

    assert _files(folder) == before_files
    assert _lifecycle(p.lake) == before_state


@pytest.mark.parametrize("authority", ["launching", "prepared_hold"])
def test_launch_rejection_cannot_append_audit_before_checking_existing_native_authority(project, authority):
    p = project
    folder = _prepare(p, "idea-early-audit")
    if authority == "launching":
        original_popen = p.popen.side_effect

        def interrupt_before_creation(*args, **kwargs):
            p.logs.append(kwargs["stdout"])
            raise KeyboardInterrupt("fixture loss before Popen returned")

        p.popen.side_effect = interrupt_before_creation
        with pytest.raises(KeyboardInterrupt):
            evaluator.launch_eval(folder.name, 0, p.results, p.cfg, lake=p.lake)
        p.popen.side_effect = original_popen
        for handle in p.logs:
            handle.close()
    else:
        ep = _launch(p, folder.name)
        _hold_after_prepare(p, folder, ep)
    (folder / "metrics.json").write_bytes(b'{"status":"FAILED","quality":0}')
    before_files, before_state = _files(folder), _lifecycle(p.lake)
    calls = p.popen.call_count

    with pytest.raises(TerminationUnconfirmed):
        evaluator.launch_eval(folder.name, 0, p.results, p.cfg, lake=p.lake)

    assert p.popen.call_count == calls
    assert _files(folder) == before_files
    assert _lifecycle(p.lake) == before_state
