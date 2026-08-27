import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from orze.core.benchmark_contract import (
    BenchmarkContractError,
    EXPOSURE_LEDGER_FILE,
    EXPOSURE_LOCK_DIR,
    PROVENANCE_FILE,
    benchmark_exposure_ledger_path,
    benchmark_exposure_summary,
    prepare_benchmark_evaluation,
    validate_benchmark_contract_config,
    validate_benchmark_receipt,
)
from orze.reporting.leaderboard import update_report
from orze.reporting.search_path import build_from_lake
from orze.reporting.state import write_status_json
from orze.engine import evaluator as evaluator_module
from orze.idea_lake import IdeaLake


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
                "evidence_scope": "local_reproduction",
                "selection_mode": "adaptive",
                "prior_exposures": 2,
                "max_evaluations": 100,
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
    provenance = json.loads(
        (idea_dir / PROVENANCE_FILE).read_text(encoding="utf-8"))
    (idea_dir / contract["receipt"]).write_text(json.dumps({
        "schema_version": 1,
        "benchmark_id": contract["benchmark_id"],
        "benchmark_revision": contract["revision"],
        "benchmark_view": contract["view"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "dataset_manifest_sha256": contract["dataset_manifest_sha256"],
        "scorer_sha256": contract["scorer_sha256"],
        "evidence_scope": contract["evidence_scope"],
        "selection_mode": contract["selection_mode"],
        "prior_exposures": contract["prior_exposures"],
        "max_evaluations": contract["max_evaluations"],
        "exposure_ordinal": provenance["exposure_ordinal"],
        "exposure_record_sha256": provenance["exposure_record_sha256"],
        "evaluation_nonce": nonce,
        "model_form": "single_model_single_pass",
        "component_model_count": 1,
        "inference_passes_per_sample": 1,
        "dataset_specific_routing": False,
        "model_artifact_sha256": "b" * 64,
        "decoding_config_sha256": "c" * 64,
        "metric_keys": ["metric_a", "metric_b"],
    }), encoding="utf-8")


def _ledger(cfg: dict) -> Path:
    return benchmark_exposure_ledger_path(cfg)


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
    (lambda c: c["report"]["benchmark_contract"].update(
        selection_mode="confirmation", prior_exposures=1),
     "already adaptive/benchmark-fitted"),
    (lambda c: c["report"]["benchmark_contract"].update(
        selection_mode="confirmation", prior_exposures=0,
        max_evaluations=2), "additional looks must be labeled adaptive"),
    (lambda c: c["report"]["benchmark_contract"].update(
        evidence_scope="official"), "evidence_scope"),
    (lambda c: c["report"]["benchmark_contract"].update(
        prior_exposures=101, max_evaluations=100),
     "prior_exposures cannot exceed max_evaluations"),
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


def test_evo_score_counts_only_current_contract_verified_evidence(tmp_path):
    cfg = _config(tmp_path)
    results = tmp_path / "results"
    cfg["_env_ORZE_RESULTS_DIR"] = str(results)
    db_path = tmp_path / "ideas.db"
    metrics = {
        "status": "COMPLETED",
        "metric_a": 2.0,
        "metric_b": 4.0,
        "avg_score": 3.0,
    }
    lake = IdeaLake(str(db_path))
    for idea_id in ("idea-valid", "idea-unproven"):
        idea_dir = results / idea_id
        idea_dir.mkdir(parents=True)
        (idea_dir / "metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8")
        lake.insert(
            idea_id, idea_id, "{}", "", eval_metrics=metrics,
            status="completed",
        )
    lake.close()

    valid_dir = results / "idea-valid"
    env = prepare_benchmark_evaluation(valid_dir, cfg)
    _write_receipt(
        valid_dir, cfg, env["ORZE_BENCHMARK_EVALUATION_NONCE"])

    evidence = build_from_lake(str(db_path), cfg)
    qualification = evidence["evidence_qualification"]
    assert evidence["stats"]["n_scored"] == 1
    assert qualification["mode"] == "benchmark_contract"
    assert qualification["accepted"] == 1
    assert qualification["rejected"] == {"benchmark_provenance_missing": 1}
    assert evidence["research_efficiency"][
        "evidence_qualification"] == qualification


def test_preexisting_receipt_is_rejected_before_evaluation(tmp_path):
    cfg = _config(tmp_path)
    idea_dir = tmp_path / "results" / "idea-prefabricated"
    idea_dir.mkdir(parents=True)
    (idea_dir / "benchmark_receipt.json").write_text("{}", encoding="utf-8")

    with pytest.raises(BenchmarkContractError, match="existed before evaluation"):
        prepare_benchmark_evaluation(idea_dir, cfg)
    assert not (idea_dir / PROVENANCE_FILE).exists()
    assert not _ledger(cfg).exists()


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
    ("evidence_scope", "development_proxy",
     "benchmark_receipt_evidence_scope_mismatch"),
    ("selection_mode", "confirmation",
     "benchmark_receipt_selection_mode_mismatch"),
    ("exposure_ordinal", 99,
     "benchmark_receipt_exposure_ordinal_mismatch"),
    ("exposure_record_sha256", "f" * 64,
     "benchmark_receipt_exposure_record_mismatch"),
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
    assert "Evidence scope: `local_reproduction`" in report
    assert "Selection mode: `adaptive`" in report
    assert "3/100 used (97 remaining)" in report
    assert "benchmark-fitted adaptive evidence" in report
    assert "idea-unproven" in report
    assert "benchmark_provenance_missing" in report
    cache = json.loads((results / "_leaderboard.json").read_text())
    assert cache["rank_scope"] == "local"
    assert cache["benchmark_exposure"]["total_exposures"] == 3
    assert cache["benchmark_exposure"]["benchmark_fitted"] is True
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


def test_confirmation_budget_is_reserved_atomically_and_fails_closed(tmp_path):
    cfg = _config(tmp_path)
    contract = cfg["report"]["benchmark_contract"]
    contract.update({
        "selection_mode": "confirmation",
        "prior_exposures": 0,
        "max_evaluations": 1,
    })
    first = tmp_path / "results" / "idea-first"
    second = tmp_path / "results" / "idea-second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    env = prepare_benchmark_evaluation(first, cfg)
    assert env["ORZE_BENCHMARK_EXPOSURE_ORDINAL"] == "1"
    summary = benchmark_exposure_summary(first.parent, cfg)
    assert summary == {
        "enabled": True,
        "valid": True,
        "evidence_scope": "local_reproduction",
        "selection_mode": "confirmation",
        "prior_exposures": 0,
        "managed_exposures": 1,
        "total_exposures": 1,
        "max_evaluations": 1,
        "remaining": 0,
        "benchmark_fitted": False,
    }
    write_status_json(
        first.parent, iteration=1, active={}, free_gpus=[], queue_depth=0,
        completed_count=0, failed_count=0, skipped_count=0, top_results=[],
        cfg=cfg,
    )
    status = json.loads((first.parent / "status.json").read_text())
    assert status["benchmark_exposure"] == summary

    with pytest.raises(
            BenchmarkContractError,
            match="benchmark_exposure_budget_exhausted:1/1"):
        prepare_benchmark_evaluation(second, cfg)
    assert not (second / PROVENANCE_FILE).exists()


def test_concurrent_budget_reservations_cannot_oversubscribe(tmp_path):
    cfg = _config(tmp_path)
    cfg["report"]["benchmark_contract"].update({
        "selection_mode": "confirmation",
        "prior_exposures": 0,
        "max_evaluations": 1,
    })
    ideas = [tmp_path / "results" / f"idea-{index}" for index in range(2)]
    for idea_dir in ideas:
        idea_dir.mkdir(parents=True)

    def reserve(idea_dir):
        try:
            return "ok", prepare_benchmark_evaluation(idea_dir, cfg)
        except BenchmarkContractError as exc:
            return "error", str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(reserve, ideas))

    assert [kind for kind, _ in outcomes].count("ok") == 1
    error = next(value for kind, value in outcomes if kind == "error")
    assert error in {
        "benchmark_exposure_ledger_locked",
        "benchmark_exposure_budget_exhausted:1/1",
    }
    ledger = _ledger(cfg)
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1


def test_corrupt_or_tampered_exposure_ledger_invalidates_receipt(tmp_path):
    cfg = _config(tmp_path)
    idea_dir = tmp_path / "results" / "idea-ledger"
    idea_dir.mkdir(parents=True)
    env = prepare_benchmark_evaluation(idea_dir, cfg)
    _write_receipt(
        idea_dir, cfg, env["ORZE_BENCHMARK_EVALUATION_NONCE"])
    ledger = _ledger(cfg)

    ledger.write_text("not-json\n", encoding="utf-8")
    ok, reason = validate_benchmark_receipt(idea_dir, cfg)
    assert ok is False
    assert reason == "benchmark_exposure_ledger_corrupt:1"


def test_exposure_record_content_hash_is_verified(tmp_path):
    cfg = _config(tmp_path)
    idea_dir = tmp_path / "results" / "idea-ledger-hash"
    idea_dir.mkdir(parents=True)
    env = prepare_benchmark_evaluation(idea_dir, cfg)
    _write_receipt(
        idea_dir, cfg, env["ORZE_BENCHMARK_EVALUATION_NONCE"])
    ledger = _ledger(cfg)
    record = json.loads(ledger.read_text(encoding="utf-8"))
    record["pid"] += 1
    ledger.write_text(json.dumps(record) + "\n", encoding="utf-8")

    ok, reason = validate_benchmark_receipt(idea_dir, cfg)
    assert ok is False
    assert reason == "benchmark_exposure_ledger_integrity_invalid:1"


def test_deleted_exposure_record_is_detected_from_provenance(tmp_path):
    cfg = _config(tmp_path)
    idea_dir = tmp_path / "results" / "idea-ledger-delete"
    idea_dir.mkdir(parents=True)
    env = prepare_benchmark_evaluation(idea_dir, cfg)
    _write_receipt(
        idea_dir, cfg, env["ORZE_BENCHMARK_EVALUATION_NONCE"])
    _ledger(cfg).unlink()

    ok, reason = validate_benchmark_receipt(idea_dir, cfg)
    assert ok is False
    assert reason == "benchmark_exposure_record_deleted:idea-ledger-delete"
    summary = benchmark_exposure_summary(idea_dir.parent, cfg)
    assert summary["valid"] is False
    assert summary["reason"] == reason


def test_exposure_budget_cannot_be_rewritten_after_first_look(tmp_path):
    cfg = _config(tmp_path)
    idea_dir = tmp_path / "results" / "idea-policy-drift"
    idea_dir.mkdir(parents=True)
    env = prepare_benchmark_evaluation(idea_dir, cfg)
    _write_receipt(
        idea_dir, cfg, env["ORZE_BENCHMARK_EVALUATION_NONCE"])

    cfg["report"]["benchmark_contract"]["max_evaluations"] = 101
    ok, reason = validate_benchmark_receipt(idea_dir, cfg)
    assert ok is False
    assert reason == "benchmark_provenance_max_evaluations_mismatch"
    summary = benchmark_exposure_summary(idea_dir.parent, cfg)
    assert summary["valid"] is False
    assert summary["reason"] == "benchmark_exposure_policy_drift"


@pytest.mark.parametrize("mutation", [
    {"selection_mode": "confirmation", "prior_exposures": 0,
     "max_evaluations": 1},
    {"evidence_scope": "development_proxy"},
    {"benchmark_id": "renamed-benchmark"},
    {"view": "renamed-view"},
])
def test_same_dataset_history_cannot_be_reset_by_relabeling(
        tmp_path, mutation):
    cfg = _config(tmp_path)
    idea_dir = tmp_path / "results" / "idea-before-relabel"
    idea_dir.mkdir(parents=True)
    prepare_benchmark_evaluation(idea_dir, cfg)

    cfg["report"]["benchmark_contract"].update(mutation)
    next_idea = idea_dir.parent / "idea-after-relabel"
    next_idea.mkdir()
    with pytest.raises(
            BenchmarkContractError,
            match="benchmark_exposure_policy_drift"):
        prepare_benchmark_evaluation(next_idea, cfg)

    summary = benchmark_exposure_summary(idea_dir.parent, cfg)
    assert summary["valid"] is False
    assert summary["reason"] == "benchmark_exposure_policy_drift"


def test_confirmation_budget_survives_results_directory_change(tmp_path):
    cfg = _config(tmp_path)
    cfg["report"]["benchmark_contract"].update({
        "selection_mode": "confirmation",
        "prior_exposures": 0,
        "max_evaluations": 1,
    })
    first = tmp_path / "results-first" / "idea-first"
    second = tmp_path / "results-second" / "idea-second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    prepare_benchmark_evaluation(first, cfg)
    assert _ledger(cfg).parent == tmp_path / ".orze"
    assert _ledger(cfg).exists()
    assert not (first.parent / EXPOSURE_LEDGER_FILE).exists()

    with pytest.raises(
            BenchmarkContractError,
            match="benchmark_exposure_budget_exhausted:1/1"):
        prepare_benchmark_evaluation(second, cfg)
    summary = benchmark_exposure_summary(second.parent, cfg)
    assert summary["total_exposures"] == 1
    assert summary["remaining"] == 0


def test_result_local_ledger_is_migrated_before_cross_root_reservation(
        tmp_path):
    cfg = _config(tmp_path)
    cfg["report"]["benchmark_contract"].update({
        "selection_mode": "confirmation",
        "prior_exposures": 0,
        "max_evaluations": 1,
    })
    first = tmp_path / "results-old" / "idea-first"
    second = tmp_path / "results-new" / "idea-second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    prepare_benchmark_evaluation(first, cfg)

    project_ledger = _ledger(cfg)
    legacy_ledger = first.parent / EXPOSURE_LEDGER_FILE
    original = project_ledger.read_bytes()
    project_ledger.replace(legacy_ledger)
    assert not project_ledger.exists()
    assert benchmark_exposure_summary(second.parent, cfg)[
        "total_exposures"] == 1

    with pytest.raises(
            BenchmarkContractError,
            match="benchmark_exposure_budget_exhausted:1/1"):
        prepare_benchmark_evaluation(second, cfg)
    assert project_ledger.read_bytes() == original
    assert legacy_ledger.read_bytes() == original


def test_divergent_result_local_history_fails_closed(tmp_path):
    cfg = _config(tmp_path)
    idea_dir = tmp_path / "results" / "idea-project-history"
    idea_dir.mkdir(parents=True)
    prepare_benchmark_evaluation(idea_dir, cfg)
    record = json.loads(_ledger(cfg).read_text(encoding="utf-8"))
    record["idea_id"] = "conflicting-history"
    canonical = dict(record)
    canonical.pop("record_sha256")
    record["record_sha256"] = hashlib.sha256(json.dumps(
        canonical, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    legacy = idea_dir.parent / EXPOSURE_LEDGER_FILE
    legacy.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    summary = benchmark_exposure_summary(idea_dir.parent, cfg)
    assert summary["valid"] is False
    assert summary["reason"] == "benchmark_exposure_legacy_ledger_conflict"
    next_idea = idea_dir.parent / "idea-after-conflict"
    next_idea.mkdir()
    with pytest.raises(
            BenchmarkContractError,
            match="benchmark_exposure_legacy_ledger_conflict"):
        prepare_benchmark_evaluation(next_idea, cfg)


def test_exposure_control_directory_cannot_be_redirected(tmp_path):
    cfg = _config(tmp_path)
    cfg["_orze_dir"] = str(tmp_path / "fresh-history")
    idea_dir = tmp_path / "results" / "idea-redirect"
    idea_dir.mkdir(parents=True)

    with pytest.raises(
            BenchmarkContractError,
            match="benchmark_exposure_control_directory_drift"):
        prepare_benchmark_evaluation(idea_dir, cfg)


def test_invalid_exposure_control_path_renders_unrankable_report(tmp_path):
    cfg = _config(tmp_path)
    cfg["_orze_dir"] = str(tmp_path / "fresh-history")
    results = tmp_path / "results"
    idea_dir = results / "idea-invalid-control"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(json.dumps({
        "status": "COMPLETED",
        "avg_score": 3.0,
        "metric_a": 2.0,
        "metric_b": 4.0,
    }), encoding="utf-8")

    assert update_report(
        results, {"idea-invalid-control": {"title": "Invalid"}}, cfg,
    ) == []
    report = (results / "report.md").read_text(encoding="utf-8")
    assert "unrankable: exposure evidence is invalid" in report
    assert "benchmark_exposure_control_directory_drift" in report


@pytest.mark.parametrize("target_kind, expected", [
    ("ledger", "benchmark_exposure_ledger_symlink_forbidden"),
    ("lock", "benchmark_exposure_lock_symlink_forbidden"),
])
def test_exposure_control_paths_reject_symlinks(
        tmp_path, target_kind, expected):
    cfg = _config(tmp_path)
    results = tmp_path / "results"
    idea_dir = results / "idea-symlink"
    idea_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("do not touch", encoding="utf-8")
    ledger = _ledger(cfg)
    ledger.parent.mkdir(parents=True)
    target = (
        ledger if target_kind == "ledger"
        else ledger.parent / EXPOSURE_LOCK_DIR
    )
    target.symlink_to(outside)

    with pytest.raises(BenchmarkContractError, match=expected):
        prepare_benchmark_evaluation(idea_dir, cfg)
    assert outside.read_text(encoding="utf-8") == "do not touch"


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
