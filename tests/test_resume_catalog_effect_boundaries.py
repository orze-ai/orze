"""Uncommitted resume publisher scope/close-failure boundary regressions."""
import json

import pytest

from orze.engine.attempt_effect_lock import AttemptEffectInDoubt, attempt_effect_lock
from orze.engine.execution_catalog import bind_catalog
from orze.engine.resume import admit_resume, prepare_resume_launch, mark_resume_launched, ResumeValidationError
from orze.engine.termination_hold import TerminationUnconfirmed
from orze.idea_lake import IdeaLake
from test_resume import resume_case, _write_valid_receipt
from test_stale_evaluation_completion import _files


def test_owned_marker_rejects_declaration_conflicting_with_context_and_claim(resume_case):
    project, results, folder, checkpoint, cfg, _ = resume_case
    _write_valid_receipt(resume_case)
    admit_resume(folder.name, results, cfg, str(checkpoint))
    declared = IdeaLake(project / "declared.db")
    other = IdeaLake(project / "other.db")
    cfg["idea_lake_db"] = str(other.db_path)
    try:
        claim = folder / "claim.json"
        claim.write_text(json.dumps({"trainer_pid":999999999, "attempt_id":"attempt-current",
                                     "lifecycle_db":str(other.db_path)}))
        context = prepare_resume_launch(folder.name, results, cfg)
        with attempt_effect_lock(folder) as lease:
            bind_catalog(declared, folder, lease)
            before = _files(folder)
            with pytest.raises((ResumeValidationError, TerminationUnconfirmed)):
                mark_resume_launched(context, claim, effect_lease=lease)
            assert _files(folder) == before
    finally:
        declared.close()
        other.close()


def test_admission_catalog_close_failure_is_hold_after_real_publication(resume_case, monkeypatch):
    project, results, folder, checkpoint, cfg, _ = resume_case
    _write_valid_receipt(resume_case)
    lake = IdeaLake(project / "ideas.db")
    cfg["idea_lake_db"] = str(lake.db_path)
    lake.insert(folder.name, "Legacy", "{}", "", status="failed")
    from orze.engine import resume_publication
    real_open = resume_publication.open_existing_lake
    closed = []

    def open_then_fail_close(path):
        opened = real_open(path)
        close = opened.close
        def fail_close():
            close()
            closed.append(True)
            raise OSError("synthetic close response failure")
        opened.close = fail_close
        return opened

    monkeypatch.setattr(resume_publication, "open_existing_lake", open_then_fail_close)
    try:
        with pytest.raises(AttemptEffectInDoubt):
            admit_resume(folder.name, results, cfg, str(checkpoint))
        assert closed == [True]
        assert (folder / "resume_request.json").exists()
        assert (folder / "_attempt_effect.lock").is_dir()
    finally:
        lake.close()
