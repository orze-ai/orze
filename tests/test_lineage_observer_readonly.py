"""Existing observer entry points must not repair or re-audit lineage evidence.

The fixture creates one synthetic train fingerprint, one different evaluation
fingerprint, a tiny model, and real completed compute/Lake receipts. Its pipe
attestation is local test evidence, not a real training or GPU execution claim.
Spies wrap real audit/storage operations; qualification itself is never mocked.
"""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import orze.core.data_separation as separation_module
from orze.core.benchmark_contract import (
    prepare_benchmark_evaluation,
    validate_benchmark_receipt,
)
from orze.core.managed_run import ManagedRunError, verify_managed_idea_outcome
from orze.core.model_lineage import audit_campaign_model_lineage
from orze.idea_lake import IdeaLake
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence,
    qualify_local_report_evidence,
)
from test_benchmark_contract import (
    _config as benchmark_config,
    _write_receipt as write_benchmark_receipt,
)
from test_model_lineage import _completed_lineage


@pytest.fixture
def case(tmp_path, monkeypatch):
    def forbidden_process(*args, **kwargs):
        pytest.fail("observer fixture must not launch any subprocess")

    monkeypatch.setattr(subprocess, "Popen", forbidden_process)
    monkeypatch.setattr(subprocess, "run", forbidden_process)
    cfg, idea_dir, tp, _ = _completed_lineage(tmp_path)
    cfg.update({
        "results_dir": str(idea_dir.parent),
        "idea_lake_db": str(tmp_path / "authority.db"),
        "managed_run": {"require_data_separation": True},
        "report": {
            "primary_metric": "score", "sort": "ascending",
            "columns": [{"key": "score", "source": "metrics.json:score"}],
        },
    })
    metrics = {"status": "COMPLETED", "score": 0.5,
               "tainted_leakage": False}
    (idea_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8")
    lake = IdeaLake(cfg["idea_lake_db"])
    try:
        lake.insert(tp.idea_id, "Synthetic observer case", "seed: 1", "",
                    status="completed", eval_metrics=metrics)
    finally:
        lake.close()
    return SimpleNamespace(
        root=tmp_path, cfg=cfg, idea_dir=idea_dir, idea_id=tp.idea_id,
        attempt_id=tp.attempt_id,
        receipt=tmp_path / ".orze" / "state" / "data_separation.json",
    )


def _observe(case, observer):
    if observer == "local":
        _, _, value, reason = qualify_local_report_evidence(
            case.idea_dir, case.cfg)
        return value is not None, reason
    if observer == "shared":
        ids, reason = authoritative_completed_idea_ids(
            Path(case.cfg["idea_lake_db"]))
        assert reason == "authoritative_lifecycle_loaded"
        _, _, value, reason = qualify_authoritative_report_evidence(
            case.idea_id, case.idea_dir.parent, case.cfg, ids)
        return value is not None, reason
    if observer == "campaign":
        report = audit_campaign_model_lineage(
            case.idea_dir.parent, case.cfg, idea_ids=[case.idea_id],
            artifact_relation="any")
        return report["status"] == "VERIFIED", report
    if observer == "managed":
        try:
            report = verify_managed_idea_outcome(case.cfg, case.idea_id)
        except ManagedRunError as exc:
            return False, str(exc)
        return report["completed"] is True, report
    if observer == "benchmark":
        return validate_benchmark_receipt(case.idea_dir, case.cfg)
    raise AssertionError(observer)


def _tree(root):
    """Compare durable directories and exact bytes, not read-induced atime."""
    return {
        path.relative_to(root).as_posix(): (
            None if path.is_dir() else path.read_bytes())
        for path in sorted(root.rglob("*"))
    }


def _audit_spies(monkeypatch):
    spies = {}
    for name in ("_read_manifest", "atomic_write", "_fs_lock"):
        spy = Mock(wraps=getattr(separation_module, name))
        monkeypatch.setattr(separation_module, name, spy)
        spies[name] = spy
    spy = Mock(wraps=separation_module.tempfile.TemporaryDirectory)
    monkeypatch.setattr(separation_module.tempfile, "TemporaryDirectory", spy)
    spies["temporary_index_directory"] = spy
    return spies


def _assert_readonly(case, before, spies):
    changed = _tree(case.root) != before
    calls = {name: spy.call_count for name, spy in spies.items()}
    assert not changed and not any(calls.values()), {
        "durable_tree_or_bytes_changed": changed, "audit_calls": calls,
    }


def _invalidate_separation(case, mutation):
    if mutation == "missing":
        case.receipt.unlink()  # Only this test's synthetic receipt.
    elif mutation == "corrupt":
        case.receipt.write_bytes(b"{broken synthetic receipt\n")
    elif mutation == "metadata":
        manifest = Path(case.cfg["data_separation"]["train_manifest"])
        stat = manifest.stat()
        os.utime(manifest, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    elif mutation == "missing_state":
        case.receipt.parent.rename(case.root / "saved-separation-state")
    else:
        raise AssertionError(mutation)


def _check_invalid_readonly(case, observer, monkeypatch):
    before = _tree(case.root)
    spies = _audit_spies(monkeypatch)
    accepted, reason = _observe(case, observer)
    # Check side effects first: a broad except rejecting after a full scan/write
    # does not satisfy this contract.
    _assert_readonly(case, before, spies)
    assert not accepted, reason


def test_valid_receipt_observers_remain_readonly_and_successful(case, monkeypatch):
    before = _tree(case.root)
    spies = _audit_spies(monkeypatch)
    for observer in ("local", "shared", "campaign", "managed"):
        accepted, reason = _observe(case, observer)
        assert accepted, (observer, reason)
    _assert_readonly(case, before, spies)


@pytest.mark.parametrize("mutation", [
    "missing", "corrupt", "metadata", "missing_state",
])
def test_local_qualification_never_repairs_separation(case, monkeypatch, mutation):
    assert _observe(case, "local")[0]
    _invalidate_separation(case, mutation)
    _check_invalid_readonly(case, "local", monkeypatch)


def test_shared_qualification_cannot_recreate_deleted_receipt(case, monkeypatch):
    assert _observe(case, "shared")[0]
    _invalidate_separation(case, "missing")
    _check_invalid_readonly(case, "shared", monkeypatch)


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "metadata"])
def test_campaign_audit_never_repairs_separation(case, monkeypatch, mutation):
    assert _observe(case, "campaign")[0]
    _invalidate_separation(case, mutation)
    _check_invalid_readonly(case, "campaign", monkeypatch)


@pytest.mark.parametrize("mutation", ["missing", "metadata"])
def test_benchmark_verification_never_repairs_separation(case, monkeypatch, mutation):
    # The real benchmark producer reserves a look and publishes provenance;
    # no evaluator is executed. The real helper writes a synthetic receipt.
    benchmark = benchmark_config(case.root)
    case.cfg.update(benchmark)
    case.cfg["report"]["benchmark_contract"]["managed_model_lineage"] = True
    env = prepare_benchmark_evaluation(case.idea_dir, case.cfg)
    write_benchmark_receipt(
        case.idea_dir, case.cfg, env["ORZE_BENCHMARK_EVALUATION_NONCE"])
    assert _observe(case, "benchmark")[0]
    _invalidate_separation(case, mutation)
    _check_invalid_readonly(case, "benchmark", monkeypatch)


@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_managed_outcome_direct_separation_check_never_repairs(
        case, monkeypatch, mutation):
    # Isolate managed_run's own direct separation check rather than relying on
    # an earlier lineage check to reject before that branch is reached.
    case.cfg["model_lineage"]["enabled"] = False
    assert _observe(case, "managed")[0]
    _invalidate_separation(case, mutation)
    _check_invalid_readonly(case, "managed", monkeypatch)


@pytest.mark.parametrize("observer", ["local", "campaign"])
def test_missing_compute_receipt_directory_is_not_recreated(
        case, monkeypatch, observer):
    assert _observe(case, observer)[0]
    receipt_dir = case.idea_dir / "_compute_receipts" / case.attempt_id
    receipt_dir.rename(case.root / "saved-compute-attempt")
    _check_invalid_readonly(case, observer, monkeypatch)


@pytest.mark.parametrize("mutation", ["artifact", "terminal", "policy"])
def test_lineage_drift_still_rejects_without_repair(case, monkeypatch, mutation):
    assert _observe(case, "local")[0]
    if mutation == "artifact":
        (case.idea_dir / "model.bin").write_bytes(b"changed tiny model")
    elif mutation == "terminal":
        terminal = (case.idea_dir / "_compute_receipts" / case.attempt_id /
                    "terminal.json")
        payload = json.loads(terminal.read_text(encoding="utf-8"))
        payload["outcome"] = "failed"
        terminal.write_text(json.dumps(payload), encoding="utf-8")
    else:
        case.cfg["data_boundaries"]["watch_paths"] = [str(case.root / "watch")]
    before = _tree(case.root)
    spies = _audit_spies(monkeypatch)
    for observer in ("local", "campaign"):
        accepted, reason = _observe(case, observer)
        assert not accepted, (observer, reason)
    _assert_readonly(case, before, spies)
