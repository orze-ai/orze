"""New two-phase lineage mechanism; API absence is not an old behavior red."""
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.core import model_lineage as lineage
from orze.engine.accounting import record_compute_start
from test_model_lineage import _config, _tp, _write_boundary


@pytest.fixture
def prepared_project(tmp_path):
    cfg = _config(tmp_path)
    folder = tmp_path / "results" / "idea-lineage"
    folder.mkdir(parents=True)
    (folder / "model.bin").write_bytes(b"stable model\x00\xff")
    tp = _tp()
    _write_boundary(folder, cfg, tp)
    record_compute_start(tp, folder, phase="training")
    return SimpleNamespace(root=tmp_path, folder=folder, cfg=cfg, tp=tp)


def _api():
    if not hasattr(lineage, "prepare_model_lineage_finalization"):
        pytest.skip("new preparation API absent; not an old behavior red")
    return lineage.prepare_model_lineage_finalization, lineage.publish_model_lineage_finalization


def _snapshot(root):
    return {str(path.relative_to(root)): (path.is_dir(),
            None if path.is_dir() else path.read_bytes(),
            path.stat().st_mtime_ns, path.stat().st_ctime_ns)
            for path in root.rglob("*")}


def test_disabled_preparation_and_publication_do_not_touch_missing_project(tmp_path, monkeypatch):
    prepare, publish = _api()
    folder = tmp_path / "missing" / "idea-disabled"
    tp = SimpleNamespace()
    tripwire = Mock(side_effect=AssertionError("disabled lineage touched storage"))
    monkeypatch.setattr(lineage, "ensure_data_separation", tripwire)
    monkeypatch.setattr(lineage, "_artifact_digest", tripwire)
    monkeypatch.setattr(Path, "mkdir", tripwire)
    prepared = prepare(tp, folder, {})
    assert publish(prepared, tp, folder, {}) == {"status": "disabled"}
    assert not folder.parent.exists()
    tripwire.assert_not_called()


def test_preparation_is_readonly_and_never_reaudits_separation(prepared_project, monkeypatch):
    p = prepared_project
    prepare, _ = _api()
    before = _snapshot(p.root)
    tripwire = Mock(side_effect=AssertionError("preparation attempted a write/audit"))
    monkeypatch.setattr(lineage, "ensure_data_separation", tripwire)
    monkeypatch.setattr(Path, "mkdir", tripwire)
    prepare(p.tp, p.folder, p.cfg)
    assert _snapshot(p.root) == before
    tripwire.assert_not_called()


def test_publication_is_compact_and_never_rehashes_model_or_manifests(prepared_project, monkeypatch):
    p = prepared_project
    prepare, publish = _api()
    prepared = prepare(p.tp, p.folder, p.cfg)
    tripwire = Mock(side_effect=AssertionError("short publication performed slow validation"))
    for name in ("_artifact_digest", "_hash_stable_file", "ensure_data_separation",
                 "read_data_separation_receipt"):
        monkeypatch.setattr(lineage, name, tripwire)
    result = publish(prepared, p.tp, p.folder, p.cfg)
    assert result["artifact_kind"] == "file"
    assert result["artifact_files"] == 1
    assert result["rank_claim_proven"] is False
    assert (p.folder / lineage.LINEAGE_FILE).is_file()
    tripwire.assert_not_called()


@pytest.mark.parametrize("change", [
    "artifact_inplace", "artifact_replaced", "boundary", "start",
    "separation", "train_manifest", "evaluation_manifest",
])
def test_publication_rejects_every_changed_bound_file_without_publishing(prepared_project, change):
    p = prepared_project
    prepare, publish = _api()
    prepared = prepare(p.tp, p.folder, p.cfg)
    receipt_dir = p.folder / "_compute_receipts" / p.tp.attempt_id
    paths = {"artifact_inplace": p.folder / "model.bin",
             "artifact_replaced": p.folder / "model.bin",
             "boundary": receipt_dir / "boundary.json", "start": receipt_dir / "start.json",
             "separation": p.root / ".orze" / "state" / "data_separation.json",
             "train_manifest": Path(p.cfg["data_separation"]["train_manifest"]),
             "evaluation_manifest": Path(p.cfg["data_separation"]["evaluation_manifest"])}
    target = paths[change]
    if change == "artifact_replaced":
        other = p.folder / "replacement.bin"
        other.write_bytes(target.read_bytes())
        os.replace(other, target)
    else:
        before = target.stat()
        target.write_bytes(target.read_bytes() + b" ")
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(lineage.ModelLineageError):
        publish(prepared, p.tp, p.folder, p.cfg)
    assert not (p.folder / lineage.LINEAGE_FILE).exists()


@pytest.mark.parametrize("change", ["policy", "attempt"])
def test_publication_requires_same_policy_and_attempt(prepared_project, change):
    p = prepared_project
    prepare, publish = _api()
    prepared = prepare(p.tp, p.folder, p.cfg)
    if change == "policy":
        p.cfg["model_lineage"]["max_bytes"] += 1
    else:
        p.tp.attempt_id = "another-attempt"
    with pytest.raises(lineage.ModelLineageError):
        publish(prepared, p.tp, p.folder, p.cfg)
    assert not (p.folder / lineage.LINEAGE_FILE).exists()


def test_receipt_change_during_slow_hash_rejects_preparation(prepared_project, monkeypatch):
    p = prepared_project
    prepare, _ = _api()
    original = lineage._artifact_digest

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        boundary = p.folder / "_compute_receipts" / p.tp.attempt_id / "boundary.json"
        boundary.write_bytes(boundary.read_bytes() + b" ")
        return result

    monkeypatch.setattr(lineage, "_artifact_digest", changed)
    with pytest.raises(lineage.ModelLineageError):
        prepare(p.tp, p.folder, p.cfg)
    assert not (p.folder / lineage.LINEAGE_FILE).exists()


def test_missing_separation_is_unavailable_not_recreated(prepared_project, monkeypatch):
    p = prepared_project
    prepare, _ = _api()
    receipt = p.root / ".orze" / "state" / "data_separation.json"
    receipt.unlink()
    before = _snapshot(p.root)
    tripwire = Mock(side_effect=AssertionError("missing receipt was reaudit permission"))
    monkeypatch.setattr(lineage, "ensure_data_separation", tripwire)
    with pytest.raises(lineage.ModelLineageError):
        prepare(p.tp, p.folder, p.cfg)
    assert _snapshot(p.root) == before
    tripwire.assert_not_called()


def test_short_publication_explicitly_does_not_support_directory_artifacts(prepared_project, monkeypatch):
    p = prepared_project
    prepare, _ = _api()
    artifact = p.folder / "model.bin"
    artifact.unlink()
    artifact.mkdir()
    (artifact / "weights").write_bytes(b"weights")
    tripwire = Mock(side_effect=AssertionError("unsupported directory was hashed"))
    monkeypatch.setattr(lineage, "_artifact_digest", tripwire)
    with pytest.raises(lineage.ModelLineagePublicationUnsupported):
        prepare(p.tp, p.folder, p.cfg)
    assert not (p.folder / lineage.LINEAGE_FILE).exists()
    tripwire.assert_not_called()


def test_changed_input_after_envelope_write_is_not_acknowledged(prepared_project, monkeypatch):
    p = prepared_project
    prepare, publish = _api()
    prepared = prepare(p.tp, p.folder, p.cfg)
    original = lineage._write_envelope_once

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        (p.folder / "model.bin").write_bytes(b"new model after publication")
        return result

    monkeypatch.setattr(lineage, "_write_envelope_once", changed)
    with pytest.raises(lineage.ModelLineageError):
        publish(prepared, p.tp, p.folder, p.cfg)
    # The coordinator must HOLD this partial effect; the publisher does not
    # pretend the already-written envelope never happened or auto-retry it.
    assert (p.folder / lineage.LINEAGE_FILE).exists()


def test_legacy_finalizer_keeps_directory_artifact_behavior(prepared_project):
    p = prepared_project
    artifact = p.folder / "model.bin"
    artifact.unlink()
    artifact.mkdir()
    (artifact / "weights").write_bytes(b"weights")
    result = lineage.finalize_model_lineage(p.tp, p.folder, p.cfg)
    assert result["artifact_kind"] == "directory_tree_v1"
    assert result["artifact_files"] == 1
