import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from orze.core.config import _validate_config
from orze.core.benchmark_contract import (
    PROVENANCE_FILE,
    prepare_benchmark_evaluation,
    validate_benchmark_receipt,
)
from orze.core.evaluation_bundle import (
    EvaluationBundleError,
    stage_evaluation_bundle,
    verify_evaluation_bundle,
)
from orze.engine import evaluator as evaluator_module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project(tmp_path: Path) -> dict:
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True)
    evaluator = scripts / "eval.py"
    child = scripts / "child.py"
    evaluator.write_text(
        "from pathlib import Path\n"
        "import os, subprocess, sys, time\n"
        "root = Path(os.environ['ORZE_EVALUATION_BUNDLE_ROOT'])\n"
        "time.sleep(0.25)\n"
        "value = subprocess.check_output(\n"
        "    [sys.executable, str(root / 'scripts/child.py')], text=True)\n"
        "Path(sys.argv[1]).write_text(value, encoding='utf-8')\n",
        encoding="utf-8",
    )
    child.write_text("print('frozen-v1')\n", encoding="utf-8")
    return {
        "_project_root": str(tmp_path),
        "python": sys.executable,
        "eval_script": "scripts/eval.py",
        "eval_output": "eval_report.json",
        "sealed_hashes": {
            "scripts/eval.py": _sha(evaluator),
            "scripts/child.py": _sha(child),
        },
        "evaluation_bundle": {
            "enabled": True,
            "files": ["scripts/eval.py", "scripts/child.py"],
        },
    }


def test_bundle_survives_concurrent_worktree_edit(tmp_path):
    cfg = _project(tmp_path)
    idea_dir = tmp_path / "results" / "idea-frozen"
    idea_dir.mkdir(parents=True)
    bundle = stage_evaluation_bundle(idea_dir, cfg)

    (tmp_path / "scripts/child.py").write_text(
        "print('mutated-v2')\n", encoding="utf-8")
    output = idea_dir / "observed.txt"
    env = os.environ.copy()
    env.update(bundle.environment(tmp_path))
    import subprocess
    subprocess.run(
        [sys.executable, str(bundle.entrypoint), str(output)],
        env=env, check=True,
    )

    assert output.read_text(encoding="utf-8").strip() == "frozen-v1"
    assert verify_evaluation_bundle(idea_dir, cfg).sha256 == bundle.sha256


def test_bundle_paths_are_absolute_for_relative_idea_directory(
        tmp_path, monkeypatch):
    cfg = _project(tmp_path)
    monkeypatch.chdir(tmp_path)
    relative = Path("results/idea-relative")
    relative.mkdir(parents=True)

    bundle = stage_evaluation_bundle(relative, cfg)

    assert bundle.root.is_absolute()
    assert bundle.entrypoint.is_absolute()
    assert bundle.manifest_path.is_absolute()
    assert verify_evaluation_bundle(relative, cfg).root == bundle.root


def test_bundle_rejects_source_drift_before_copy(tmp_path):
    cfg = _project(tmp_path)
    (tmp_path / "scripts/child.py").write_text(
        "print('changed-before-stage')\n", encoding="utf-8")
    idea_dir = tmp_path / "results" / "idea-drift"
    idea_dir.mkdir(parents=True)

    with pytest.raises(EvaluationBundleError, match="source_hash_drift"):
        stage_evaluation_bundle(idea_dir, cfg)
    assert not any((idea_dir / "_evaluation_bundle").iterdir())


def test_bundle_rejects_tampered_copy(tmp_path):
    cfg = _project(tmp_path)
    idea_dir = tmp_path / "results" / "idea-tampered"
    idea_dir.mkdir(parents=True)
    bundle = stage_evaluation_bundle(idea_dir, cfg)
    child = bundle.root / "scripts/child.py"
    child.chmod(0o644)
    child.write_text("print('tampered')\n", encoding="utf-8")

    with pytest.raises(EvaluationBundleError, match="file_hash_mismatch"):
        verify_evaluation_bundle(idea_dir, cfg)


def test_bundle_config_requires_entrypoint_and_sealed_files(tmp_path):
    cfg = _project(tmp_path)
    cfg["evaluation_bundle"]["files"] = ["scripts/child.py"]
    errors, _ = _validate_config(cfg)
    assert "evaluation_bundle.files: must include eval_script" in errors

    cfg = _project(tmp_path / "second")
    del cfg["sealed_hashes"]["scripts/child.py"]
    errors, _ = _validate_config(cfg)
    assert any("child.py must have a sealed SHA-256" in error
               for error in errors)


def test_bundle_rejects_symlinked_source(tmp_path):
    cfg = _project(tmp_path)
    real = tmp_path / "scripts/child.py"
    outside = tmp_path / "outside.py"
    real.replace(outside)
    real.symlink_to(outside)
    cfg["sealed_hashes"]["scripts/child.py"] = _sha(outside)
    idea_dir = tmp_path / "results" / "idea-symlink"
    idea_dir.mkdir(parents=True)

    with pytest.raises(EvaluationBundleError, match="source_symlink_forbidden"):
        stage_evaluation_bundle(idea_dir, cfg)


def test_eval_launcher_executes_frozen_entrypoint_and_child(
        tmp_path, monkeypatch):
    cfg = _project(tmp_path)
    results = tmp_path / "results"
    idea_dir = results / "idea-launch"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED"}), encoding="utf-8")
    observed = idea_dir / "observed.txt"
    cfg["eval_args"] = [str(observed)]
    cfg["gpu_scheduling"] = {"allowed_gpus": [4], "reserved_gpus": []}

    monkeypatch.setattr(
        evaluator_module, "_assert_launch_authorized", lambda *args: None)
    monkeypatch.setattr(
        evaluator_module, "_assert_controller_runtime_attested",
        lambda *args: None)
    monkeypatch.setattr(
        evaluator_module, "_verify_gpu_free", lambda *args: None)

    @contextmanager
    def fake_lease(_gpu):
        yield ()

    monkeypatch.setattr(evaluator_module, "gpu_execution_lease", fake_lease)

    ep = evaluator_module.launch_eval("idea-launch", 4, results, cfg)
    assert ep is not None
    (tmp_path / "scripts/eval.py").write_text(
        "raise SystemExit('mutated evaluator')\n", encoding="utf-8")
    (tmp_path / "scripts/child.py").write_text(
        "print('mutated child')\n", encoding="utf-8")
    ep.process.wait(timeout=10)
    ep.close_log()

    assert ep.process.returncode == 0
    assert observed.read_text(encoding="utf-8").strip() == "frozen-v1"
    bundle = verify_evaluation_bundle(idea_dir, cfg)
    assert Path(ep.process.args[1]).resolve() == bundle.entrypoint


def test_benchmark_receipt_is_bound_to_exact_bundle(tmp_path):
    cfg = _project(tmp_path)
    evaluator_digest = cfg["sealed_hashes"]["scripts/eval.py"]
    cfg["report"] = {
        "primary_metric": "avg_score",
        "min_datasets": 2,
        "columns": [
            {"key": "avg_score"},
            {"key": "metric_a"},
            {"key": "metric_b"},
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
            "prior_exposures": 0,
            "max_evaluations": 1,
            "aggregate": "macro_mean",
            "aggregate_tolerance": 1e-9,
            "evaluator_sha256": evaluator_digest,
            "dataset_manifest_sha256": "d" * 64,
            "scorer_sha256": "e" * 64,
        },
    }
    idea_dir = tmp_path / "results" / "idea-receipt"
    idea_dir.mkdir(parents=True)
    bundle = stage_evaluation_bundle(idea_dir, cfg)
    env = prepare_benchmark_evaluation(idea_dir, cfg)
    provenance = json.loads(
        (idea_dir / PROVENANCE_FILE).read_text(encoding="utf-8"))
    assert provenance["evaluation_bundle_sha256"] == bundle.sha256
    assert env["ORZE_EVALUATION_BUNDLE_SHA256"] == bundle.sha256

    contract = cfg["report"]["benchmark_contract"]
    receipt = {
        "schema_version": 1,
        "benchmark_id": contract["benchmark_id"],
        "benchmark_revision": contract["revision"],
        "benchmark_view": contract["view"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "dataset_manifest_sha256": contract["dataset_manifest_sha256"],
        "scorer_sha256": contract["scorer_sha256"],
        "evidence_scope": contract["evidence_scope"],
        "selection_mode": contract["selection_mode"],
        "prior_exposures": 0,
        "max_evaluations": 1,
        "exposure_ordinal": provenance["exposure_ordinal"],
        "exposure_record_sha256": provenance["exposure_record_sha256"],
        "evaluation_nonce": provenance["evaluation_nonce"],
        "model_form": "single_model_single_pass",
        "component_model_count": 1,
        "inference_passes_per_sample": 1,
        "dataset_specific_routing": False,
        "model_artifact_sha256": "b" * 64,
        "decoding_config_sha256": "c" * 64,
        "metric_keys": ["metric_a", "metric_b"],
    }
    receipt_path = idea_dir / "benchmark_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert validate_benchmark_receipt(idea_dir, cfg) == (
        False, "benchmark_receipt_evaluation_bundle_mismatch")

    receipt["evaluation_bundle_sha256"] = bundle.sha256
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert validate_benchmark_receipt(
        idea_dir, cfg,
        values={"metric_a": 2.0, "metric_b": 4.0, "avg_score": 3.0},
    ) == (True, "benchmark_contract_verified")
