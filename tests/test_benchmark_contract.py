import hashlib
import json
import os
from pathlib import Path

import pytest

from orze.core.benchmark_contract import (
    BenchmarkContractError,
    PROVENANCE_FILE,
    prepare_benchmark_evaluation,
    validate_benchmark_contract_config,
    validate_benchmark_receipt,
)
from orze.reporting.leaderboard import update_report
from orze.engine import evaluator as evaluator_module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path) -> dict:
    evaluator = tmp_path / "eval_exact.py"
    evaluator.write_text("# sealed evaluator\n", encoding="utf-8")
    digest = _sha(evaluator)
    return {
        "_project_root": str(tmp_path),
        "eval_script": "eval_exact.py",
        "eval_output": "eval_report.json",
        "sealed_hashes": {"eval_exact.py": digest},
        "report": {
            "title": "Exact benchmark",
            "primary_metric": "avg_score",
            "sort": "ascending",
            "min_datasets": 2,
            "columns": [
                {"key": "avg_score", "label": "Average"},
                {"key": "metric_a", "label": "A"},
                {"key": "metric_b", "label": "B"},
            ],
            "benchmark_contract": {
                "benchmark_id": "owner/benchmark",
                "revision": "a" * 40,
                "view": "default",
                "required_metrics": ["metric_a", "metric_b"],
                "receipt": "benchmark_receipt.json",
                "model_form": "single_model_single_pass",
                "aggregate": "macro_mean",
                "aggregate_tolerance": 1e-9,
                "evaluator_sha256": digest,
                "dataset_manifest_sha256": "d" * 64,
                "scorer_sha256": "e" * 64,
            },
        },
    }


def _write_receipt(idea_dir: Path, cfg: dict, nonce: str) -> None:
    contract = cfg["report"]["benchmark_contract"]
    (idea_dir / contract["receipt"]).write_text(json.dumps({
        "schema_version": 1,
        "benchmark_id": contract["benchmark_id"],
        "benchmark_revision": contract["revision"],
        "benchmark_view": contract["view"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "dataset_manifest_sha256": contract["dataset_manifest_sha256"],
        "scorer_sha256": contract["scorer_sha256"],
        "evaluation_nonce": nonce,
        "model_form": "single_model_single_pass",
        "component_model_count": 1,
        "inference_passes_per_sample": 1,
        "dataset_specific_routing": False,
        "model_artifact_sha256": "b" * 64,
        "decoding_config_sha256": "c" * 64,
        "metric_keys": ["metric_a", "metric_b"],
    }), encoding="utf-8")


def test_valid_contract_pins_exact_evaluator_and_report_columns(tmp_path):
    assert validate_benchmark_contract_config(_config(tmp_path)) == []


@pytest.mark.parametrize("mutation, expected", [
    (lambda c: c["report"]["benchmark_contract"].update(revision="main"),
     "revision"),
    (lambda c: c["report"]["benchmark_contract"].update(
        evaluator_sha256="c" * 64), "does not match"),
    (lambda c: c.update(sealed_hashes={}), "sealed_hashes"),
    (lambda c: c["report"].update(min_datasets=1), "min_datasets"),
    (lambda c: c["report"]["benchmark_contract"].update(
        required_metrics=["metric_a", "private_metric"]),
     "missing from report.columns"),
    (lambda c: c["report"]["columns"][1].update(
        source="../borrowed.json:metric_a"), "inside the idea directory"),
])
def test_invalid_contracts_fail_closed(tmp_path, mutation, expected):
    cfg = _config(tmp_path)
    mutation(cfg)
    errors = validate_benchmark_contract_config(cfg)
    assert any(expected in error for error in errors), errors


def test_fresh_receipt_proves_exact_single_model_result(tmp_path):
    cfg = _config(tmp_path)
    idea_dir = tmp_path / "results" / "idea-valid"
    idea_dir.mkdir(parents=True)

    env = prepare_benchmark_evaluation(idea_dir, cfg)
    nonce = env["ORZE_BENCHMARK_EVALUATION_NONCE"]
    _write_receipt(idea_dir, cfg, nonce)

    assert validate_benchmark_receipt(
        idea_dir, cfg,
        values={"metric_a": 2.0, "metric_b": 4.0, "avg_score": 3.0},
    ) == (True, "benchmark_contract_verified")

    receipt_path = idea_dir / "benchmark_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["component_model_count"] = 3
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    ok, reason = validate_benchmark_receipt(idea_dir, cfg)
    assert ok is False
    assert reason == "benchmark_receipt_component_count_mismatch"


def test_preexisting_receipt_is_rejected_before_evaluation(tmp_path):
    cfg = _config(tmp_path)
    idea_dir = tmp_path / "results" / "idea-prefabricated"
    idea_dir.mkdir(parents=True)
    (idea_dir / "benchmark_receipt.json").write_text("{}", encoding="utf-8")

    with pytest.raises(BenchmarkContractError, match="existed before evaluation"):
        prepare_benchmark_evaluation(idea_dir, cfg)
    assert not (idea_dir / PROVENANCE_FILE).exists()


def test_wrong_coverage_or_aggregate_is_unrankable(tmp_path):
    cfg = _config(tmp_path)
    idea_dir = tmp_path / "results" / "idea-invalid"
    idea_dir.mkdir(parents=True)
    env = prepare_benchmark_evaluation(idea_dir, cfg)
    _write_receipt(
        idea_dir, cfg, env["ORZE_BENCHMARK_EVALUATION_NONCE"])

    ok, reason = validate_benchmark_receipt(
        idea_dir, cfg,
        values={"metric_a": 2.0, "metric_b": 4.0, "avg_score": 2.5},
    )
    assert ok is False
    assert reason == "benchmark_primary_metric_not_macro_mean"

    receipt_path = idea_dir / "benchmark_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["metric_keys"] = ["metric_a"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    ok, reason = validate_benchmark_receipt(idea_dir, cfg)
    assert ok is False
    assert reason == "benchmark_receipt_metric_coverage_mismatch"

    receipt["metric_keys"] = [{"unhashable": True}]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    ok, reason = validate_benchmark_receipt(idea_dir, cfg)
    assert ok is False
    assert reason == "benchmark_receipt_metric_coverage_mismatch"


@pytest.mark.parametrize("field, value, expected", [
    ("evaluation_nonce", "wrong", "benchmark_receipt_nonce_mismatch"),
    ("dataset_manifest_sha256", "f" * 64,
     "benchmark_receipt_dataset_manifest_sha256_mismatch"),
    ("scorer_sha256", "f" * 64,
     "benchmark_receipt_scorer_sha256_mismatch"),
    ("model_form", "ensemble", "benchmark_receipt_model_form_mismatch"),
    ("inference_passes_per_sample", 2,
     "benchmark_receipt_inference_pass_count_mismatch"),
    ("dataset_specific_routing", True,
     "benchmark_receipt_dataset_routing_not_disabled"),
    ("model_artifact_sha256", "not-a-hash",
     "benchmark_receipt_model_artifact_sha256_invalid"),
    ("decoding_config_sha256", "not-a-hash",
     "benchmark_receipt_decoding_config_sha256_invalid"),
])
def test_receipt_identity_and_single_pass_fields_fail_closed(
        tmp_path, field, value, expected):
    cfg = _config(tmp_path)
    idea_dir = tmp_path / "results" / "idea-invalid-receipt"
    idea_dir.mkdir(parents=True)
    env = prepare_benchmark_evaluation(idea_dir, cfg)
    _write_receipt(
        idea_dir, cfg, env["ORZE_BENCHMARK_EVALUATION_NONCE"])
    receipt_path = idea_dir / "benchmark_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert validate_benchmark_receipt(idea_dir, cfg) == (False, expected)


def test_report_ranks_only_contract_verified_rows_and_labels_scope(tmp_path):
    cfg = _config(tmp_path)
    results = tmp_path / "results"
    ideas = {
        "idea-valid": {"title": "Valid"},
        "idea-unproven": {"title": "Unproven"},
    }
    for idea_id, average in (("idea-valid", 3.0), ("idea-unproven", 2.0)):
        idea_dir = results / idea_id
        idea_dir.mkdir(parents=True)
        (idea_dir / "metrics.json").write_text(json.dumps({
            "status": "COMPLETED",
            "avg_score": average,
            "metric_a": average - 1,
            "metric_b": average + 1,
            "training_time": 100,
        }), encoding="utf-8")
    valid_dir = results / "idea-valid"
    env = prepare_benchmark_evaluation(valid_dir, cfg)
    _write_receipt(
        valid_dir, cfg, env["ORZE_BENCHMARK_EVALUATION_NONCE"])

    completed = update_report(results, ideas, cfg)

    assert [row["id"] for row in completed] == ["idea-valid"]
    report = (results / "report.md").read_text(encoding="utf-8")
    assert "| Local Rank |" in report
    assert "not an official leaderboard rank" in report
    assert "idea-unproven" in report
    assert "benchmark_provenance_missing" in report
    cache = json.loads((results / "_leaderboard.json").read_text())
    assert cache["rank_scope"] == "local"
    assert [row["idea_id"] for row in cache["top"]] == ["idea-valid"]

    # Preserve timestamps while changing same-width receipt content. Contract
    # cache invalidation must follow bytes, not metadata that can be restored.
    receipt_path = valid_dir / "benchmark_receipt.json"
    original_stat = receipt_path.stat()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["component_model_count"] = 2
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    os.utime(receipt_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert update_report(results, ideas, cfg) == []
    cache = json.loads((results / "_leaderboard.json").read_text())
    assert cache["top"] == []


def test_eval_launcher_rejects_prefabricated_receipt_without_spawning(
        tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    results = tmp_path / "results"
    idea_dir = results / "idea-prefabricated"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED"}), encoding="utf-8")
    (idea_dir / "benchmark_receipt.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluator_module, "_assert_launch_authorized",
                        lambda *args: None)
    monkeypatch.setattr(evaluator_module, "_assert_gpu_authorized",
                        lambda *args: None)
    monkeypatch.setattr(evaluator_module, "_verify_gpu_free",
                        lambda *args: None)

    def _must_not_spawn(*args, **kwargs):
        raise AssertionError("evaluation process should not be spawned")

    monkeypatch.setattr(evaluator_module.subprocess, "Popen", _must_not_spawn)
    assert evaluator_module.launch_eval(
        "idea-prefabricated", 4, results, cfg) is None
    audit = (idea_dir / "_eval_audit.jsonl").read_text(encoding="utf-8")
    assert "benchmark_contract_preflight_failed" in audit


class _FinishedProcess:
    returncode = 0

    def poll(self):
        return 0


class _FinishedEval:
    def __init__(self, idea_id: str, log_path: Path):
        self.idea_id = idea_id
        self.process = _FinishedProcess()
        self.start_time = 0.0
        self.log_path = log_path
        self.timeout = 10_000_000

    def close_log(self):
        pass


def test_finished_eval_without_contract_receipt_is_failed(tmp_path):
    cfg = _config(tmp_path)
    results = tmp_path / "results"
    idea_dir = results / "idea-no-receipt"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(json.dumps({
        "status": "COMPLETED",
        "avg_score": 3.0,
        "metric_a": 2.0,
        "metric_b": 4.0,
    }), encoding="utf-8")
    log_path = idea_dir / "eval_output.log"
    log_path.write_text("done\n", encoding="utf-8")
    ep = _FinishedEval("idea-no-receipt", log_path)

    assert evaluator_module.check_active_evals(
        {4: ep}, results, cfg) == [("idea-no-receipt", 4)]
    audit = (idea_dir / "_eval_audit.jsonl").read_text(encoding="utf-8")
    assert "benchmark_contract_validation_failed" in audit
    marker = json.loads((idea_dir / "eval_report.json").read_text())
    assert "benchmark_provenance_missing" in marker["reason"]
    terminal = json.loads(next(
        (idea_dir / "_compute_receipts").glob("*/terminal.json")
    ).read_text())
    assert terminal["outcome"] == "failed"
