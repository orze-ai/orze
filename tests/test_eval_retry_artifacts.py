"""Artifact preservation and recoverable preparation for explicit eval retry.

This tests a new mechanism; an absent evaluation_retry module is not an old
behavioral regression. All evidence, ledger, SQLite, and IO faults are real and
temporary. Only the alias-launch control doubles external GPU/Popen boundaries.
"""

import hashlib
import json
import os
import sqlite3
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.core.benchmark_contract import prepare_benchmark_evaluation
from orze.engine import evaluator
from orze.engine.evaluation_retry import (
    EvaluationRetryError,
    request_evaluation_retry,
)
from orze.engine.sealed import write_sealed_manifest
from orze.idea_lake import IdeaLake


def _put(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    folder = results / "idea-retry-artifacts"
    folder.mkdir(parents=True)
    script = tmp_path / "eval.py"
    script.write_bytes(b"# never executed by these tests\n")
    script_hash = hashlib.sha256(script.read_bytes()).hexdigest()
    cfg = {
        "_project_root": str(tmp_path), "results_dir": str(results),
        "eval_script": "eval.py", "eval_output": "assessment.json",
        "eval_checkpoint": "checkpoint.pt", "eval_timeout": 60,
        "sealed_files": [str(folder / "sealed-policy.json")],
        "report": {
            "primary_metric": "quality", "sort": "ascending",
            "columns": [{"key": "quality", "source": "assessment.json:quality"}],
            "benchmark_contract": {
                "benchmark_id": "test/generic", "revision": "a" * 40,
                "view": "development", "required_metrics": ["quality"],
                "receipt": "benchmark_receipt.json",
                "model_form": "single_model_single_pass", "aggregate": "macro_mean",
                "evidence_scope": "development_proxy", "selection_mode": "adaptive",
                "prior_exposures": 0, "max_evaluations": 10,
                "evaluator_sha256": script_hash,
                "dataset_manifest_sha256": "b" * 64, "scorer_sha256": "c" * 64,
            },
        },
    }
    protected = {
        "metrics.json": b'{"status":"COMPLETED","quality":0}',
        "claim.json": b'{"attempt_id":"training-original"}',
        "checkpoint.pt": b"original-checkpoint\x00\xff",
        "_model_lineage.json": b'{"training_lineage":"preserve"}',
        "_compute_receipts/training/start.json": b'{"event":"start","phase":"training"}',
        "_compute_receipts/training/terminal.json": b'{"event":"terminal","outcome":"completed"}',
        "_compute_receipts/old-eval/terminal.json": b'{"event":"terminal","outcome":"failed"}',
        "_eval_audit.jsonl": b'{"action":"old-evaluation-audit"}\n',
        "train_output.log": b"successful training log\n",
        "training_access.jsonl": b'{"access":"training-history"}\n',
        "_evaluation_bundle/pinned/config.json": b'{"immutable_bundle":"keep"}',
        "sealed-policy.json": b'{"sealed_policy":"keep"}',
    }
    for relative, payload in protected.items():
        _put(folder / relative, payload)
    write_sealed_manifest(results, {
        str(folder / "sealed-policy.json"): hashlib.sha256(
            protected["sealed-policy.json"]).hexdigest()})
    # A real prior reservation establishes history without running a scorer.
    prepare_benchmark_evaluation(folder, cfg)
    ledger = tmp_path / ".orze" / "_benchmark_exposures.jsonl"
    legacy_ledger = results / "_benchmark_exposures.jsonl"
    legacy_ledger.write_bytes(ledger.read_bytes())
    external = tmp_path / "external-checkpoint.pt"
    external.write_bytes(b"external checkpoint must never move")
    analysis = b'{"category":"historical-training-or-evaluation-analysis"}'
    (folder / "failure_analysis.json").write_bytes(analysis)
    movable = {
        "assessment.json": b'{"status":"FAILED","quality":999}',
        "benchmark_receipt.json": b'{"evaluation_nonce":"old-failed-attempt"}',
        "eval_output.log": b"old failed evaluation output\n",
    }
    provenance = (folder / "_benchmark_evaluation.json").read_bytes()
    for relative, payload in movable.items():
        _put(folder / relative, payload)
    lake = IdeaLake(tmp_path / "ideas.db")
    cfg["idea_lake_db"] = str(lake.db_path)
    lake.insert(folder.name, "retain metadata", "seed: 7", "original notes", status="queued")
    assert lake.reconcile_training_complete(folder.name, "reconcile_test_training")
    assert lake.record_stage_transition(
        folder.name, "evaluation", "PENDING", "IN_PROGRESS", "evaluation_launched")
    assert lake.record_state_transition(
        folder.name, "IN_PROGRESS", "FAILED", "evaluation_failed")
    failed_id = lake.conn.execute(
        "SELECT MAX(id) FROM idea_transitions WHERE idea_id=? AND to_state='FAILED'",
        (folder.name,)).fetchone()[0]
    extra_protected = {
        ledger: ledger.read_bytes(), legacy_ledger: legacy_ledger.read_bytes(),
        external: external.read_bytes(), script: script.read_bytes(),
        results / ".sealed_hashes": (results / ".sealed_hashes").read_bytes(),
    }
    try:
        yield SimpleNamespace(
            root=tmp_path, results=results, folder=folder, idea_id=folder.name,
            cfg=cfg, lake=lake, failed_id=failed_id, protected=protected,
            extra_protected=extra_protected, movable=movable, analysis=analysis,
            provenance=provenance,
            archive=folder / "_evaluation_retries" / str(failed_id), external=external)
    finally:
        lake.close()


def _request(p):
    result = request_evaluation_retry(p.idea_id, p.results, p.cfg, p.lake)
    assert result["status"] == "evaluation_retry_pending"
    assert result["idea_id"] == p.idea_id
    assert str(result["retry_id"])
    return result


def _db_snapshot(p):
    return {table: [tuple(row) for row in p.lake.conn.execute(
        f"SELECT * FROM {table} ORDER BY rowid")]
        for table in ("ideas", "idea_state", "idea_stage_state",
                      "idea_transitions", "idea_stage_transitions")}


def _assert_protected(p):
    for relative, payload in p.protected.items():
        assert (p.folder / relative).read_bytes() == payload, relative
    for path, payload in p.extra_protected.items():
        assert path.read_bytes() == payload, str(path)
    assert (p.folder / "failure_analysis.json").read_bytes() == p.analysis
    # Current-root provenance keeps the previous exposure linked to its
    # ledger until prepare reserves the next look and publishes a new nonce.
    assert (p.folder / "_benchmark_evaluation.json").read_bytes() == p.provenance
    assert p.lake.get_stage_state(p.idea_id, "training") == "COMPLETE"


def _archive_payloads(p):
    return [path.read_bytes() for path in p.archive.rglob("*")
            if path.is_file() and path.name != "manifest.json"]


def _assert_prepared(p):
    manifest = json.loads((p.archive / "manifest.json").read_text(encoding="utf-8"))
    manifest_text = json.dumps(manifest, sort_keys=True)
    archived = _archive_payloads(p)
    for relative, payload in p.movable.items():
        assert not (p.folder / relative).exists(), relative
        assert archived.count(payload) == 1, relative
        assert hashlib.sha256(payload).hexdigest() in manifest_text
    assert archived.count(p.analysis) == 1
    assert hashlib.sha256(p.analysis).hexdigest() in manifest_text
    assert archived.count(p.provenance) == 1
    assert hashlib.sha256(p.provenance).hexdigest() in manifest_text
    _assert_protected(p)


def _filesystem_snapshot(p):
    # Never follow a redirected path, even within this temporary fixture.
    snapshot = {}
    for path in p.root.rglob("*"):
        if path == Path(p.lake.db_path) or path.name.startswith("ideas.db-"):
            continue
        relative = str(path.relative_to(p.root))
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            snapshot[relative] = ("file", path.read_bytes())
        elif path.is_dir():
            snapshot[relative] = ("directory",)
    return snapshot


def test_retry_archives_only_owned_outputs_and_copies_shared_failure_analysis(project):
    p = project

    _request(p)

    assert p.lake.get_fsm_state(p.idea_id) == "IN_PROGRESS"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "PENDING"
    _assert_prepared(p)


def test_repeated_pending_request_preserves_same_archive_and_database_history(project):
    p = project
    first = _request(p)
    before_files, before_db = _filesystem_snapshot(p), _db_snapshot(p)

    second = _request(p)

    assert second == first
    assert _filesystem_snapshot(p) == before_files
    assert _db_snapshot(p) == before_db


def test_database_failure_after_preparation_keeps_archive_and_resumes_same_request(project):
    p = project
    before_db = _db_snapshot(p)
    p.lake.conn.execute(
        "CREATE TRIGGER reject_retry_state BEFORE UPDATE ON idea_state "
        "WHEN NEW.current_state='IN_PROGRESS' "
        "BEGIN SELECT RAISE(ABORT, 'injected post-prepare DB failure'); END")
    p.lake.conn.commit()

    with pytest.raises((EvaluationRetryError, sqlite3.DatabaseError)):
        _request(p)

    assert _db_snapshot(p) == before_db
    _assert_prepared(p)
    preserved_archive = {str(path.relative_to(p.archive)): path.read_bytes()
                         for path in p.archive.rglob("*") if path.is_file()}
    p.lake.conn.execute("DROP TRIGGER reject_retry_state")
    p.lake.conn.commit()

    _request(p)

    assert p.lake.get_fsm_state(p.idea_id) == "IN_PROGRESS"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "PENDING"
    _assert_prepared(p)
    assert {str(path.relative_to(p.archive)): path.read_bytes()
            for path in p.archive.rglob("*") if path.is_file()} == preserved_archive


def test_second_file_move_failure_keeps_manifest_and_resumes_remaining_files(project, monkeypatch):
    p = project
    before_db = _db_snapshot(p)
    moved = []
    originals = {p.folder / name for name in p.movable}

    def inject(real):
        def move(source, destination, *args, **kwargs):
            if Path(source) in originals:
                assert (p.archive / "manifest.json").is_file()
                json.loads((p.archive / "manifest.json").read_text(encoding="utf-8"))
                if moved:
                    raise OSError("injected second artifact move failure")
                result = real(source, destination, *args, **kwargs)
                moved.append(Path(source))
                return result
            return real(source, destination, *args, **kwargs)
        return move

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "rename", inject(os.rename))
        patcher.setattr(os, "replace", inject(os.replace))
        with pytest.raises((EvaluationRetryError, OSError)):
            _request(p)

    assert len(moved) == 1, "Exercise a real partial archive, not a preflight rejection"
    assert _db_snapshot(p) == before_db
    assert (p.archive / "manifest.json").is_file()
    _assert_protected(p)

    _request(p)

    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "PENDING"
    _assert_prepared(p)


def test_metrics_output_alias_is_preserved_and_pending_retry_really_launches_evaluator(
        project, monkeypatch):
    p = project
    p.cfg["eval_output"] = "./metrics.json"
    p.cfg["report"]["columns"] = [{"key": "quality", "source": "metrics.json:quality"}]
    p.movable.pop("assessment.json")  # Not the configured output; leave untouched.
    old_assessment = (p.folder / "assessment.json").read_bytes()

    _request(p)

    _assert_prepared(p)
    assert (p.folder / "assessment.json").read_bytes() == old_assessment
    process = SimpleNamespace(pid=None, returncode=None, poll=lambda: None)
    popen = Mock(return_value=process)
    lease = Mock(side_effect=lambda *args, **kwargs: nullcontext(()))
    gpu_check = Mock()
    monkeypatch.setattr(evaluator.subprocess, "Popen", popen)
    monkeypatch.setattr(evaluator, "gpu_execution_lease", lease)
    monkeypatch.setattr(evaluator, "_verify_gpu_free", gpu_check)

    ep = evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)

    assert ep is not None
    try:
        popen.assert_called_once()
        lease.assert_called_once_with(0, require_idle=True)
        gpu_check.assert_called_once()
        assert p.lake.get_stage_state(p.idea_id, "training") == "COMPLETE"
        assert p.lake.get_stage_state(p.idea_id, "evaluation") == "IN_PROGRESS"
        assert (p.folder / "metrics.json").read_bytes() == p.protected["metrics.json"]
        assert (p.folder / "checkpoint.pt").read_bytes() == p.protected["checkpoint.pt"]
    finally:
        ep.close_log()


@pytest.mark.parametrize("unsafe", [
    "parent-traversal", "absolute", "symlink", "hardlink", "archive-symlink",
])
def test_unsafe_source_or_archive_path_rejects_before_any_artifact_or_state_operation(project, unsafe):
    p = project
    if unsafe == "parent-traversal":
        p.cfg["eval_output"] = "../../external-checkpoint.pt"
    elif unsafe == "absolute":
        p.cfg["eval_output"] = str(p.external)
    elif unsafe == "symlink":
        (p.folder / "redirected.json").symlink_to(p.external)
        p.cfg["eval_output"] = "redirected.json"
    elif unsafe == "hardlink":
        os.link(p.external, p.folder / "redirected.json")
        p.cfg["eval_output"] = "redirected.json"
    else:
        destination = p.root / "external-archive"
        destination.mkdir()
        (p.folder / "_evaluation_retries").symlink_to(destination, target_is_directory=True)
    before_files, before_db = _filesystem_snapshot(p), _db_snapshot(p)

    with pytest.raises(EvaluationRetryError):
        _request(p)

    assert _filesystem_snapshot(p) == before_files
    assert _db_snapshot(p) == before_db


@pytest.mark.parametrize("field,relative", [
    ("eval_output", "claim.json"),
    ("eval_output", "checkpoint.pt"),
    ("eval_output", "_eval_audit.jsonl"),
    ("receipt", "_model_lineage.json"),
    ("receipt", "_compute_receipts/training/terminal.json"),
    ("receipt", "_evaluation_bundle/pinned/config.json"),
    ("eval_output", "sealed-policy.json"),
])
def test_configured_output_or_receipt_cannot_authorize_moving_protected_evidence(
        project, field, relative):
    p = project
    if field == "receipt":
        p.cfg["report"]["benchmark_contract"]["receipt"] = relative
    else:
        p.cfg[field] = relative
    before_files, before_db = _filesystem_snapshot(p), _db_snapshot(p)

    with pytest.raises(EvaluationRetryError):
        _request(p)

    assert _filesystem_snapshot(p) == before_files
    assert _db_snapshot(p) == before_db
