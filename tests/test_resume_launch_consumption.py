"""Existing resume preparation/consumption must not create fresh authority."""
import json

import pytest

from orze.core.execution_attempts import create_attempt, mark_running
from orze.engine.execution_authority import execution_transaction
from orze.engine.execution_catalog import bind_catalog
from orze.engine.resume import admit_resume, prepare_resume_launch, mark_resume_launched, ResumeValidationError
from orze.engine.termination_hold import TerminationUnconfirmed
from orze.idea_lake import IdeaLake
from test_resume import resume_case, _write_valid_receipt
from test_stale_evaluation_completion import _files


def _prepare(case):
    project, results, folder, checkpoint, cfg, _ = case
    _write_valid_receipt(case)
    admit_resume(folder.name, results, cfg, str(checkpoint))
    context = prepare_resume_launch(folder.name, results, cfg)
    (folder / "claim.json").write_text('{"trainer_pid":999999999,"attempt_id":"attempt-current"}')
    return context


def test_native_private_resume_marker_requires_explicit_effect_lease(resume_case):
    context = _prepare(resume_case)
    project, _, folder, _, _, _ = resume_case
    lake = IdeaLake(project / "ideas.db")
    try:
        lake.insert(folder.name, "Native", "{}", "", status="queued")
        with execution_transaction(lake, folder) as tx:
            bind_catalog(lake, folder, tx.lease)
            ref = create_attempt(tx.conn, folder.name, "training", "attempt-current", {})
            mark_running(tx.conn, ref)
        before = _files(folder)
        with pytest.raises((ResumeValidationError, TerminationUnconfirmed)):
            mark_resume_launched(context, folder / "claim.json")
        assert _files(folder) == before
    finally:
        lake.close()


def test_request_replaced_after_actual_prepare_cannot_be_consumed(resume_case):
    context = _prepare(resume_case)
    folder = resume_case[2]
    request = folder / "resume_request.json"
    changed = json.loads(request.read_text())
    changed["created_at"] = "different request generation"
    request.write_text(json.dumps(changed))
    before = _files(folder)
    with pytest.raises((ResumeValidationError, TerminationUnconfirmed)):
        mark_resume_launched(context, folder / "claim.json")
    assert _files(folder) == before


def test_prepared_resume_cannot_write_another_task_claim(resume_case):
    context = _prepare(resume_case)
    _, results, folder, _, _, _ = resume_case
    other = results / "idea-other"
    other.mkdir()
    claim = other / "claim.json"
    claim.write_text('{"trainer_pid":999999998}')
    before, other_before = _files(folder), _files(other)
    with pytest.raises((ResumeValidationError, TerminationUnconfirmed)):
        mark_resume_launched(context, claim)
    assert _files(folder) == before and _files(other) == other_before
