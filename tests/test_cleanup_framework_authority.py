"""Independent C1 public cleanup regressions against actual native evidence.

Fixtures run real claim/launch/finish/replication/evaluation transactions and
filesystem publication. Only process/GPU boundaries are test substitutes.
The assertions use the existing cfg-only run_cleanup API, not a new result
schema. No standalone GC or custom cleanup program is invoked.
"""
import logging
from pathlib import Path

import pytest

from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.engine import scheduler
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.evaluation_retry import request_evaluation_retry
from test_native_training_caller_boundaries import case
from test_native_artifact_publication import _declare, _launch, _complete
from test_replication_execution_slots import source, _request, _select, _slot
from test_observation_snapshot_contract import (
    project as evaluation_project, artifact_project, native_case,
    _launch as launch_evaluation, _finish as finish_evaluation, _observations,
)


def _files(root):
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _assert_preserved(before):
    missing = [str(path) for path in before if not path.is_file()]
    assert not missing, "cleanup deleted protected files: " + repr(missing)
    assert {path: path.read_bytes() for path in before} == before


def _framework(folder, declared=()):
    selected = {}
    for path, body in _files(folder).items():
        relative = path.relative_to(folder)
        if (relative.parts[0].startswith("_") or path.name in {
                "idea_config.yaml", "metrics.json", "claim.json", *declared}):
            selected[path] = body
    assert folder / "_execution_catalog.json" in selected
    assert any("_execution_effects" in path.parts for path in selected)
    return selected


def _cleanup(c, patterns):
    cfg = dict(c.cfg, results_dir=str(c.results), idea_lake_db=str(c.lake.db_path),
               cleanup={"patterns": patterns}, gc={"enabled": False})
    before = tuple(c.lake.conn.iterdump())
    result = scheduler.run_cleanup(c.results, cfg)
    assert result is None, "legacy public return contract stays None"
    assert tuple(c.lake.conn.iterdump()) == before, "cleanup must not rewrite authority"


def test_completed_replica_broad_glob_preserves_real_framework_and_declared_outputs(source):
    c = source
    source_folder = c.folder
    request = _request(c, "cleanup-real-replica")
    _select(c, request)
    tp = _launch(c)
    (c.folder / "model.bin").write_bytes(b"same-seed-output")
    assert len(_complete(c, tp)) == 1
    row = current_attempt(c.lake.conn, c.idea, "training")
    assert row["state"] == "TERMINAL" and row["binding"]["replication"]
    before = _framework(source_folder, ("model.bin",))
    before.update(_framework(c.folder, ("model.bin",)))
    before[c.flat] = c.flat.read_bytes()
    slot = _slot(c, request)
    before[slot] = slot.read_bytes()
    for record in c.source_records + artifacts_for_attempt(c.lake.conn, tp.attempt_ref):
        path = Path(record["path"])
        before[path] = path.read_bytes()
    disposable = c.folder / "scratch.tmp"
    disposable.write_bytes(b"closed task disposable")

    _cleanup(c, ["**/*"])

    _assert_preserved(before)
    assert not disposable.exists()


def test_completed_evaluation_keeps_real_retry_and_attempt_protocol_history(evaluation_project):
    c = evaluation_project
    first, _, _, first_output = launch_evaluation(c)
    finish_evaluation(c, first, first_output, _observations(), code=1)
    assert request_evaluation_retry(c.idea, c.results, c.cfg, c.lake)["status"] == "evaluation_retry_pending"
    second, _, _, output = launch_evaluation(c)
    row = finish_evaluation(c, second, output, _observations())
    assert row["state"] == "TERMINAL" and len(row["terminal"]["observation_ids"]) == 2
    before = _framework(c.folder, ("best_model.pt", "summary.txt"))
    assert any("_evaluation_retries" in path.parts for path in before)
    assert any("_evaluation_attempts" in path.parts for path in before)
    for record in c.training_records + artifacts_for_attempt(c.lake.conn, second.attempt_ref):
        path = Path(record["path"])
        before[path] = path.read_bytes()
    disposable = c.folder / "scratch.tmp"
    disposable.write_bytes(b"closed evaluation disposable")

    _cleanup(c, ["**/*"])

    _assert_preserved(before)
    assert not disposable.exists()


def test_running_native_attempt_refuses_scratch_and_checkpoint_cleanup(case, caplog):
    c = case
    _declare(c)
    tp = _launch(c)
    try:
        (c.folder / "model.bin").write_bytes(b"live checkpoint")
        (c.folder / "scratch.tmp").write_bytes(b"live worker scratch")
        assert current_attempt(c.lake.conn, c.idea, "training")["state"] == "RUNNING"
        before = _files(c.folder)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            _cleanup(c, ["**/*"])
        _assert_preserved(before)
        assert not any("Cleanup: deleted" in record.message for record in caplog.records)
    finally:
        tp.close_log()


def test_real_partial_effect_hold_does_not_delete_or_report_success(case, caplog):
    c = case
    _declare(c)
    tp = _launch(c)
    try:
        (c.folder / "model.bin").write_bytes(b"unaccepted checkpoint")
        (c.folder / "scratch.tmp").write_bytes(b"must remain during uncertain commit")
        c.lake.conn.execute(
            "CREATE TRIGGER reject_cleanup_terminal BEFORE UPDATE ON execution_attempts "
            "WHEN NEW.state='TERMINAL' BEGIN SELECT RAISE(IGNORE); END")
        c.lake.conn.commit()
        with pytest.raises(AttemptEffectInDoubt):
            _complete(c, tp)
        assert (c.folder / "_attempt_effect.lock").is_dir()
        prepared = list(c.folder.glob("_execution_effects/*/prepared.json"))
        assert prepared and any(not path.with_name("committed.json").exists() for path in prepared)
        assert current_attempt(c.lake.conn, c.idea, "training")["state"] == "RUNNING"
        before = _files(c.folder)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            _cleanup(c, ["**/*"])
        _assert_preserved(before)
        assert not any("Cleanup: deleted" in record.message for record in caplog.records)
    finally:
        tp.close_log()


def test_confirmed_closed_native_attempt_allows_disposable_pattern(case):
    c = case
    tp = _launch(c)
    assert len(_complete(c, tp)) == 1
    assert current_attempt(c.lake.conn, c.idea, "training")["state"] == "TERMINAL"
    before = _framework(c.folder)
    disposable = c.folder / "scratch.tmp"
    disposable.write_bytes(b"safe disposable")
    _cleanup(c, ["*.tmp"])
    assert not disposable.exists()
    _assert_preserved(before)


def test_genuine_legacy_disposable_cleanup_does_not_create_native_authority(tmp_path):
    results = tmp_path / "results"
    folder = results / "idea-offline-legacy"
    folder.mkdir(parents=True)
    metrics = folder / "metrics.json"
    metrics.write_bytes(b'{"status":"COMPLETED"}')
    disposable = folder / "scratch.tmp"
    disposable.write_bytes(b"ordinary legacy disposable")
    assert scheduler.run_cleanup(results, {"cleanup": {"patterns": ["*.tmp"]}}) is None
    assert not disposable.exists() and metrics.read_bytes() == b'{"status":"COMPLETED"}'
    assert not list(tmp_path.rglob("*.db"))
    assert not (folder / "_execution_catalog.json").exists()
