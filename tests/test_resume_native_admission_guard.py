"""D2 draft: legacy checkpoint admission must not erase native ownership."""
import pytest

from orze.core.execution_attempts import create_attempt, mark_running
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.execution_authority import execution_transaction
from orze.engine.execution_catalog import bind_catalog
from orze.engine.resume import admit_resume, ResumeValidationError
from orze.engine.termination_hold import TerminationUnconfirmed
from orze.idea_lake import IdeaLake
from test_resume import resume_case, _write_valid_receipt
from test_stale_evaluation_completion import _files, _lifecycle


@pytest.mark.parametrize("state", ["LAUNCHING", "RUNNING", "PARTIAL_EFFECT"])
def test_legacy_resume_cannot_requeue_or_archive_a_native_owned_task(resume_case, state):
    project, results, folder, checkpoint, cfg, _ = resume_case
    _write_valid_receipt(resume_case)
    (folder / "metrics.json").write_text('{"status":"FAILED"}')
    (folder / "claim.json").write_text('{"trainer_pid":999999999}')
    lake = IdeaLake(project / "lake.db")
    cfg["idea_lake_db"] = str(lake.db_path)
    lake.insert(folder.name, "native analysis action", "{}", "", status="queued")
    assert lake.record_state_transition(folder.name, "QUEUED", "CLAIMED")
    assert lake.record_state_transition(folder.name, "CLAIMED", "IN_PROGRESS")
    try:
        with execution_transaction(lake, folder) as tx:
            bind_catalog(lake, folder, tx.lease)
            ref = create_attempt(tx.conn, folder.name, "analyze", "analysis-active",
                                 {"origin": "controller_action"})
            if state != "LAUNCHING":
                mark_running(tx.conn, ref)
            tx.watch_attempt(ref)
        if state == "PARTIAL_EFFECT":
            with pytest.raises(AttemptEffectInDoubt):
                with execution_transaction(lake, folder) as tx:
                    tx.prepare(ref, {"operation": "analysis_report"})
                    raise RuntimeError("simulated interruption after effect preparation")
        files, lifecycle = _files(folder), _lifecycle(lake)
        rejected = False
        try:
            admit_resume(folder.name, results, cfg, str(checkpoint))
        except (ResumeValidationError, TerminationUnconfirmed):
            rejected = True
        assert rejected
        assert _files(folder) == files
        assert _lifecycle(lake) == lifecycle
    finally:
        lake.close()
