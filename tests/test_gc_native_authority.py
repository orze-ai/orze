"""C1b public native GC regressions plus explicit-scope acceptance.

Four historical cases first call the original public API and assert real
files survive. Their old failures occur before any new cfg keyword is used.
On fixed code each repeats through an explicitly configured real GC scope,
so cfg=None refusal alone cannot pass the native safety assertions. The last
case is a new destructive-API contract, excluded from old-source replay.
Only existing native process/GPU fixtures substitute execution boundaries.
"""
import json
from pathlib import Path

import pytest

from orze.agents import orze_gc
from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from test_native_training_caller_boundaries import case
from test_native_artifact_publication import _declare, _launch, _complete
from test_replication_execution_slots import source


def _run(results, checkpoints, *, database=None, cfg=None, explicit=False,
         archive=None, dry_run=False):
    options = dict(results_dir=results, checkpoints_dir=checkpoints,
                   primary_metric="score", lake_db_path=database,
                   keep_top=0, keep_recent=0, min_free_gb=0,
                   gc_results_enabled=True, archive_dir=archive, dry_run=dry_run)
    if explicit:
        options["cfg"] = cfg
    return orze_gc.run_gc(**options)


def _checkpoint(root, idea):
    path = root / idea / "checkpoint.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"checkpoint must remain until GC is authorized")
    return path


def _contents(root):
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _unchanged(before):
    missing = [str(path) for path in before if not path.is_file()]
    assert not missing, "GC removed protected native files: " + repr(missing)
    assert {path: path.read_bytes() for path in before} == before


def _no_success(report):
    for section in ("checkpoints", "results", "archive"):
        assert all(report.get(section, {}).get(key, 0) == 0
                   for key in ("deleted", "deleted_files", "archived_files", "freed_bytes"))


def _tree_identity(root):
    result = {}
    for path in (root, *sorted(root.rglob("*"))):
        info = path.lstat()
        result[str(path.relative_to(root))] = (
            info.st_mode, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            path.read_bytes() if path.is_file() else None)
    return result


def test_native_running_with_metrics_cannot_lose_checkpoints_or_result_files(case):
    c = case
    _declare(c)
    tp = _launch(c)
    try:
        (c.folder / "metrics.json").write_text(json.dumps({"status": "COMPLETED", "score": 0}))
        (c.folder / "model.bin").write_bytes(b"writer is still running")
        checkpoint_root = c.results.parent / "checkpoints"
        checkpoint = _checkpoint(checkpoint_root, c.idea)
        before = _contents(c.folder)
        before[checkpoint] = checkpoint.read_bytes()
        row = current_attempt(c.lake.conn, c.idea, "training")
        assert row["state"] == "RUNNING" and c.lake.get_fsm_state(c.idea) == "IN_PROGRESS"
        # The first assertion is an old-API behavioral red. Fixed code must
        # then exercise actual explicitly configured authority, not stop here.
        for explicit in (False, True):
            report = _run(c.results, checkpoint_root, database=Path(c.lake.db_path),
                          cfg=c.cfg, explicit=explicit)
            _unchanged(before)
            _no_success(report)
            assert current_attempt(c.lake.conn, c.idea, "training") == row
    finally:
        tp.close_log()


def test_partial_native_effect_hold_blocks_checkpoint_delete_and_cold_archive(case):
    c = case
    _declare(c)
    tp = _launch(c)
    try:
        (c.folder / "model.bin").write_bytes(b"uncertain terminal model")
        overlays = c.folder / "overlays"
        overlays.mkdir()
        (overlays / "preview.bin").write_bytes(b"retained under HOLD")
        c.lake.conn.execute(
            "CREATE TRIGGER reject_gc_terminal BEFORE UPDATE ON execution_attempts "
            "WHEN NEW.state='TERMINAL' BEGIN SELECT RAISE(IGNORE); END")
        c.lake.conn.commit()
        with pytest.raises(AttemptEffectInDoubt):
            _complete(c, tp)
        assert (c.folder / "_attempt_effect.lock").is_dir()
        assert any(not path.with_name("committed.json").exists()
                   for path in c.folder.glob("_execution_effects/*/prepared.json"))
        checkpoint_root = c.results.parent / "checkpoints"
        checkpoint = _checkpoint(checkpoint_root, c.idea)
        before = _contents(c.folder)
        before[checkpoint] = checkpoint.read_bytes()
        archive = c.results.parent / "cold"
        for explicit in (False, True):
            report = _run(c.results, checkpoint_root, database=Path(c.lake.db_path),
                          cfg=c.cfg, explicit=explicit, archive=archive)
            _unchanged(before)
            _no_success(report)
            assert not archive.exists(), "refused GC must not stage a cold archive"
            assert current_attempt(c.lake.conn, c.idea, "training")["state"] == "RUNNING"
    finally:
        tp.close_log()


def test_closed_native_declared_source_and_accepted_artifact_remain_retained(source):
    c = source
    records = artifacts_for_attempt(c.lake.conn, c.source_tp.attempt_ref)
    before = {c.folder / "model.bin": (c.folder / "model.bin").read_bytes()}
    for record in records:
        path = Path(record["path"])
        before[path] = path.read_bytes()
    row = current_attempt(c.lake.conn, c.idea, "training")
    assert row["state"] == "TERMINAL" and row["terminal"]["artifact_ids"]
    for explicit in (False, True):
        report = _run(c.results, None, database=Path(c.lake.db_path),
                      cfg=c.cfg, explicit=explicit)
        _unchanged(before)
        _no_success(report)
        assert current_attempt(c.lake.conn, c.idea, "training") == row
        assert artifacts_for_attempt(c.lake.conn, c.source_tp.attempt_ref) == records


def test_unreadable_requested_catalog_cannot_downgrade_to_legacy_gc(tmp_path):
    results = tmp_path / "results"
    folder = results / "idea-unverified"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_bytes(b'{"status":"COMPLETED"}')
    (folder / "old-model.pt").write_bytes(b"not authorized for deletion")
    checkpoints = tmp_path / "checkpoints"
    _checkpoint(checkpoints, folder.name)
    database = tmp_path / "requested.db"
    database.write_bytes(b"unreadable SQLite authority")
    before = _contents(tmp_path)
    for explicit in (False, True):
        report = _run(results, checkpoints, database=database, cfg={}, explicit=explicit)
        _unchanged(before)
        _no_success(report)
        assert not list(tmp_path.glob("requested.db-*"))


def test_legacy_api_dry_run_creates_no_marker_archive_or_changed_bytes(source):
    c = source
    checkpoints = c.results.parent / "checkpoints"
    _checkpoint(checkpoints, c.idea)
    archive = c.results.parent / "cold-not-created"
    before = _tree_identity(c.results.parent)
    _run(c.results, checkpoints, database=Path(c.lake.db_path),
         archive=archive, dry_run=True)
    assert _tree_identity(c.results.parent) == before
    assert not archive.exists()
    assert not list(checkpoints.rglob(".orze_protected"))


def test_explicit_legacy_declaration_dry_run_then_real_disposable_cleanup(tmp_path):
    # New destructive-API safety contract: cfg={} explicitly declares a
    # legacy scope. This case is excluded from old-source behavioral replay.
    results = tmp_path / "results"
    folder = results / "idea-legacy-disposable"
    folder.mkdir(parents=True)
    metrics = folder / "metrics.json"
    metrics.write_bytes(b'{"status":"COMPLETED"}')
    model = folder / "disposable.pt"
    model.write_bytes(b"explicit legacy disposable")
    checkpoints = tmp_path / "checkpoints"
    checkpoint = _checkpoint(checkpoints, folder.name)
    before = _tree_identity(tmp_path)
    _run(results, checkpoints, cfg={}, explicit=True, dry_run=True)
    assert _tree_identity(tmp_path) == before
    report = _run(results, checkpoints, cfg={}, explicit=True)
    assert not checkpoint.parent.exists() and not model.exists()
    assert metrics.read_bytes() == b'{"status":"COMPLETED"}'
    assert report["checkpoints"]["deleted"] == 1
    assert report["results"]["deleted_files"] == 1
    assert not list(tmp_path.rglob("*.db"))
