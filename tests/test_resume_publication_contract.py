"""New resume publication mechanisms; not old-API-absence behavior reds."""
import json
import os
import stat
import time
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import accounting, scheduler, training_attempts
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.resume import admit_resume, prepare_resume_launch, mark_resume_launched, ResumeValidationError
from orze.idea_lake import IdeaLake
from test_resume import resume_case, _write_valid_receipt
from test_resume_launch_consumption import _prepare
from test_stale_evaluation_completion import _files


def test_actual_native_started_consumes_prepared_request_inside_its_attempt_transaction(resume_case):
    project, results, folder, checkpoint, cfg, _ = resume_case
    _write_valid_receipt(resume_case)
    admit_resume(folder.name, results, cfg, str(checkpoint))
    lake = IdeaLake(project / "ideas.db")
    cfg["idea_lake_db"] = str(lake.db_path)
    try:
        lake.insert(folder.name, "Native resumed launch", "{}", "", status="queued")
        assert scheduler.claim(folder.name, results, 4, lake=lake)
        context = prepare_resume_launch(folder.name, results, cfg)
        claim = json.loads((folder / "claim.json").read_text())
        tp = SimpleNamespace(idea_id=folder.name, attempt_id=claim["attempt_id"], gpu=4,
                             start_time=time.time(), process=SimpleNamespace(pid=987654321),
                             execution_identity="a" * 64)
        tp.attempt_ref = training_attempts.begin(lake, tp, folder)
        accounting.record_compute_start(tp, folder)
        training_attempts.started(lake, tp, folder,
                                  {"pid": tp.process.pid, "pgid": tp.process.pid, "start_ticks": 1234},
                                  resume_context=context)
        assert current_attempt(lake.conn, folder.name, "training")["state"] == "RUNNING"
        assert lake.get_fsm_state(folder.name) == "IN_PROGRESS"
        assert not (folder / "resume_request.json").exists()
        assert (folder / "resume_request.consumed.json").exists()
        assert json.loads((folder / "claim.json").read_text())["resume_checkpoint"] == str(checkpoint)
        assert not (folder / "_attempt_effect.lock").exists()
    finally:
        lake.close()


def test_configured_missing_catalog_is_not_created_or_silently_downgraded(resume_case):
    project, results, folder, checkpoint, cfg, _ = resume_case
    _write_valid_receipt(resume_case)
    cfg["idea_lake_db"] = str(project / "missing" / "ideas.db")
    before = _files(folder)
    with pytest.raises(ResumeValidationError, match="catalog_unavailable"):
        admit_resume(folder.name, results, cfg, str(checkpoint))
    assert _files(folder) == before
    assert not (project / "missing").exists()


def test_legacy_catalog_admission_does_not_bootstrap_or_migrate(resume_case, monkeypatch):
    project, results, folder, checkpoint, cfg, _ = resume_case
    _write_valid_receipt(resume_case)
    lake = IdeaLake(project / "ideas.db")
    cfg["idea_lake_db"] = str(lake.db_path)
    lake.insert(folder.name, "Legacy failed", "{}", "", status="failed")
    try:
        def forbidden_init(*args, **kwargs):
            pytest.fail("publication may only open existing authority, never bootstrap")
        monkeypatch.setattr(IdeaLake, "__init__", forbidden_init)
        admit_resume(folder.name, results, cfg, str(checkpoint))
        assert lake.get_fsm_state(folder.name) == "QUEUED"
    finally:
        lake.close()


def test_consumption_parent_fsync_failure_retains_hold(resume_case, monkeypatch):
    context = _prepare(resume_case)
    folder = resume_case[2]
    original = os.fsync
    seen = []
    identity = folder.stat()

    def fail_directory(fd):
        info = os.fstat(fd)
        if (stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino)
                and (folder / "resume_request.consumed.json").exists()):
            seen.append(True)
            raise OSError("synthetic parent fsync failure")
        return original(fd)

    monkeypatch.setattr(os, "fsync", fail_directory)
    with pytest.raises(AttemptEffectInDoubt):
        mark_resume_launched(context, folder / "claim.json")
    assert seen
    assert (folder / "resume_request.consumed.json").exists()
    assert (folder / "_attempt_effect.lock").is_dir()


def test_prepared_context_checkpoint_cannot_diverge_from_request(resume_case):
    context = _prepare(resume_case)
    folder = resume_case[2]
    context["checkpoint"] = str(folder / "a-different-checkpoint")
    before = _files(folder)
    with pytest.raises(ResumeValidationError, match="checkpoint_context"):
        mark_resume_launched(context, folder / "claim.json")
    assert _files(folder) == before
