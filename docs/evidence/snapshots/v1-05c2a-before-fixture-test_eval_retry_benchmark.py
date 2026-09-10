"""A retry is a new benchmark look, never a history reset or receipt replay.

Preparation, nonce publication, receipts, exposure ledger and Lake are real.
The evaluator process and GPU boundaries are doubled; no scorer is executed.
"""

import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.core import benchmark_contract as benchmark
from orze.engine import evaluator
from orze.engine.evaluation_retry import EvaluationRetryError, request_evaluation_retry
from orze.idea_lake import IdeaLake


def _receipt(provenance):
    return dict(provenance, model_form="single_model_single_pass",
                component_model_count=1, inference_passes_per_sample=1,
                dataset_specific_routing=False, model_artifact_sha256="d" * 64,
                decoding_config_sha256="e" * 64, metric_keys=["fold"])


@pytest.fixture
def project(tmp_path, monkeypatch, request):
    results = tmp_path / "results"
    folder = results / "idea-benchmark-retry"
    folder.mkdir(parents=True)
    script = tmp_path / "eval.py"
    script.write_bytes(b"# sealed test evaluator, never executed\n")
    cfg = {
        "_project_root": str(tmp_path), "results_dir": str(results),
        "eval_script": "eval.py", "eval_output": "assessment.json",
        "eval_checkpoint": "checkpoint.pt",
        "report": {
            "primary_metric": "quality", "sort": "ascending",
            "columns": [{"key": key, "source": f"assessment.json:{key}"}
                        for key in ("quality", "fold")],
            "benchmark_contract": {
                "benchmark_id": "test/retry", "revision": "a" * 40,
                "view": "development", "required_metrics": ["fold"],
                "receipt": "benchmark_receipt.json", "aggregate": "macro_mean",
                "model_form": "single_model_single_pass", "evidence_scope": "development_proxy",
                "selection_mode": "adaptive", "prior_exposures": 0,
                "max_evaluations": getattr(request, "param", 5),
                "evaluator_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "dataset_manifest_sha256": "b" * 64, "scorer_sha256": "c" * 64,
            },
        },
    }
    metrics = b'{"status":"COMPLETED","quality":999,"fold":999}'
    checkpoint = b"successful training checkpoint\x00\xff"
    (folder / "metrics.json").write_bytes(metrics)
    (folder / "checkpoint.pt").write_bytes(checkpoint)
    benchmark.prepare_benchmark_evaluation(folder, cfg)
    provenance_path = folder / benchmark.PROVENANCE_FILE
    old_provenance = provenance_path.read_bytes()
    old_receipt = json.dumps(_receipt(json.loads(old_provenance))).encode()
    (folder / "benchmark_receipt.json").write_bytes(old_receipt)
    (folder / "assessment.json").write_bytes(b'{"status":"FAILED","quality":999,"fold":999}')
    ledger = benchmark.benchmark_exposure_ledger_path(cfg)
    lake = IdeaLake(tmp_path / "ideas.db")
    cfg["idea_lake_db"] = str(lake.db_path)
    lake.insert(folder.name, "benchmark retry", "seed: 7", "", status="queued")
    assert lake.reconcile_training_complete(folder.name, "reconcile_test_training")
    assert lake.record_stage_transition(
        folder.name, "evaluation", "PENDING", "IN_PROGRESS", "evaluation_launched")
    assert lake.record_state_transition(folder.name, "IN_PROGRESS", "FAILED", "evaluation_failed")
    p = SimpleNamespace(
        results=results, folder=folder, cfg=cfg, lake=lake, idea_id=folder.name,
        metrics=metrics, checkpoint=checkpoint, ledger=ledger,
        old_ledger=ledger.read_bytes(), old_provenance=old_provenance,
        old_receipt=old_receipt, replay_old=False, child_env=None)

    class ScorerProcess:
        pid = None
        returncode = None

        def poll(self):
            if self.returncode is None:
                current = json.loads(provenance_path.read_text())
                (folder / "assessment.json").write_text(
                    '{"status":"COMPLETED","quality":0,"fold":0}', encoding="utf-8")
                receipt = p.old_receipt if p.replay_old else json.dumps(_receipt(current)).encode()
                (folder / "benchmark_receipt.json").write_bytes(receipt)
                self.returncode = 0
            return self.returncode

    def popen(*args, **kwargs):
        p.child_env = kwargs["env"]
        return ScorerProcess()

    p.popen = Mock(side_effect=popen)
    p.gpu_check = Mock()
    p.lease = Mock(side_effect=lambda *args, **kwargs: nullcontext(()))
    monkeypatch.setattr(evaluator.subprocess, "Popen", p.popen)
    monkeypatch.setattr(evaluator, "_verify_gpu_free", p.gpu_check)
    monkeypatch.setattr(evaluator, "gpu_execution_lease", p.lease)
    try:
        yield p
    finally:
        lake.close()


def _request(p):
    result = request_evaluation_retry(p.idea_id, p.results, p.cfg, p.lake)
    assert result["status"] == "evaluation_retry_pending"
    return p.folder / "_evaluation_retries" / str(result["retry_id"])


def _assert_training(p):
    assert p.lake.get_stage_state(p.idea_id, "training") == "COMPLETE"
    assert (p.folder / "metrics.json").read_bytes() == p.metrics
    assert (p.folder / "checkpoint.pt").read_bytes() == p.checkpoint


def test_request_preserves_current_provenance_and_ledger_and_archives_old_receipt(project):
    p = project

    archive = _request(p)

    assert p.ledger.read_bytes() == p.old_ledger
    assert (p.folder / benchmark.PROVENANCE_FILE).read_bytes() == p.old_provenance
    assert not (p.folder / "benchmark_receipt.json").exists()
    archived = [path.read_bytes() for path in archive.rglob("*") if path.is_file()]
    assert p.old_receipt in archived
    assert p.old_provenance in archived
    p.popen.assert_not_called()
    _assert_training(p)


@pytest.mark.parametrize("replay_old,expected", [(True, "FAILED"), (False, "COMPLETE")])
def test_real_relaunch_reserves_new_nonce_and_only_new_receipt_can_complete(project, replay_old, expected):
    p = project
    p.replay_old = replay_old
    archive = _request(p)
    assert p.ledger.read_bytes() == p.old_ledger
    assert (p.folder / benchmark.PROVENANCE_FILE).read_bytes() == p.old_provenance

    ep = evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)

    assert ep is not None
    new_provenance = json.loads((p.folder / benchmark.PROVENANCE_FILE).read_text())
    old_provenance = json.loads(p.old_provenance)
    assert new_provenance["evaluation_nonce"] != old_provenance["evaluation_nonce"]
    assert new_provenance["exposure_ordinal"] == old_provenance["exposure_ordinal"] + 1 == 2
    assert p.child_env["ORZE_BENCHMARK_EVALUATION_NONCE"] == new_provenance["evaluation_nonce"]
    assert p.ledger.read_bytes().startswith(p.old_ledger)
    assert len(p.ledger.read_text().splitlines()) == 2
    assert evaluator.check_active_evals(
        {0: ep}, p.results, p.cfg, lake=p.lake) == [(p.idea_id, 0)]

    assert p.lake.get_fsm_state(p.idea_id) == expected
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == expected
    ok, reason = benchmark.validate_benchmark_receipt(
        p.folder, p.cfg, values={"quality": 0, "fold": 0})
    assert ok is (not replay_old)
    if replay_old:
        assert reason == "benchmark_receipt_nonce_mismatch"
    terminal = json.loads((p.folder / "_compute_receipts" / ep.attempt_id / "terminal.json").read_text())
    assert terminal["outcome"] == ("failed" if replay_old else "completed")
    assert terminal["return_code"] == 0
    assert p.old_receipt in [path.read_bytes() for path in archive.rglob("*") if path.is_file()]
    p.popen.assert_called_once()
    p.gpu_check.assert_called_once()
    p.lease.assert_called_once_with(0, require_idle=True)
    _assert_training(p)


@pytest.mark.parametrize("project", [1], indirect=True)
def test_exhausted_budget_rejects_retry_without_archiving_or_refunding(project):
    p = project

    with pytest.raises(EvaluationRetryError, match="budget_exhausted"):
        _request(p)

    assert p.ledger.read_bytes() == p.old_ledger
    assert (p.folder / benchmark.PROVENANCE_FILE).read_bytes() == p.old_provenance
    assert (p.folder / "benchmark_receipt.json").read_bytes() == p.old_receipt
    assert not (p.folder / "_evaluation_retries").exists()
    assert p.lake.get_fsm_state(p.idea_id) == "FAILED"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "FAILED"
    p.popen.assert_not_called()
    _assert_training(p)


def test_corrupt_ledger_rejects_retry_without_repair_or_archiving(project):
    p = project
    damaged = p.old_ledger + b"{broken\n"
    p.ledger.write_bytes(damaged)

    with pytest.raises(EvaluationRetryError, match="history_invalid"):
        _request(p)

    assert p.ledger.read_bytes() == damaged
    assert (p.folder / benchmark.PROVENANCE_FILE).read_bytes() == p.old_provenance
    assert (p.folder / "benchmark_receipt.json").read_bytes() == p.old_receipt
    assert not (p.folder / "_evaluation_retries").exists()
    assert p.lake.get_fsm_state(p.idea_id) == "FAILED"
    p.popen.assert_not_called()
    _assert_training(p)


def test_provenance_left_in_place_detects_ledger_deletion_after_retry_admission(project):
    p = project
    _request(p)
    p.ledger.unlink()  # Corrupt only this fixture's real project history.

    assert evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake) is None

    p.popen.assert_not_called()
    p.lease.assert_not_called()
    p.gpu_check.assert_not_called()
    assert not p.ledger.exists(), "Never turn deleted history into a fresh zero-look ledger"
    assert (p.folder / benchmark.PROVENANCE_FILE).read_bytes() == p.old_provenance
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "PENDING"
    _assert_training(p)


def test_silent_provenance_publication_failure_prevents_launch_but_keeps_reserved_look(
        project, monkeypatch):
    p = project
    _request(p)
    publish = Mock()  # Exercise the documented atomic_write silent-no-write failure.
    monkeypatch.setattr(benchmark, "atomic_write", publish)

    assert evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake) is None

    publish.assert_called_once()
    p.popen.assert_not_called()
    p.lease.assert_not_called()
    p.gpu_check.assert_not_called()
    assert (p.folder / benchmark.PROVENANCE_FILE).read_bytes() == p.old_provenance
    assert p.ledger.read_bytes().startswith(p.old_ledger)
    assert len(p.ledger.read_text().splitlines()) == 2
    assert not (p.folder / "_compute_receipts").exists()
    assert p.lake.get_fsm_state(p.idea_id) == "IN_PROGRESS"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "PENDING"
    _assert_training(p)
