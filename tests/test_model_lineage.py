"""Fail-closed managed training-to-evaluation lineage tests."""

import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import orze.core.model_lineage as lineage_module
import orze.data_boundaries.wrap as boundary_wrap
from orze.core.config import _validate_config
from orze.core.data_separation import ensure_data_separation
from orze.core.model_lineage import (
    audit_campaign_model_lineage,
    ModelLineageError,
    _artifact_digest,
    finalize_model_lineage,
    prepare_model_lineage_launch,
    receive_model_lineage_attestation,
    validate_model_lineage_for_evaluation,
)
from orze.engine.accounting import (
    record_compute_start,
    record_compute_terminal,
)
from orze.engine.evaluator import launch_eval
from orze.engine.launcher import check_active, launch
from orze.reporting.leaderboard import update_report


_KEY = b"model-lineage-test-key"
_NAMESPACE = "a" * 64
_NORMALIZATION = "b" * 64


def _fp(value: str) -> str:
    return hmac.new(_KEY, value.encode(), hashlib.sha256).hexdigest()


def _manifest(path: Path, role: str, sample: str) -> str:
    values = [{
        "schema_version": 1,
        "role": role,
        "fingerprint_algorithm": "hmac-sha256",
        "fingerprint_namespace_sha256": _NAMESPACE,
        "normalization_contract_sha256": _NORMALIZATION,
        "fields": ["sample"],
    }, {"sample": _fp(sample)}]
    content = "\n".join(json.dumps(
        value, sort_keys=True, separators=(",", ":"))
        for value in values) + "\n"
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def _config(tmp_path: Path) -> dict:
    held_out = tmp_path / "held-out"
    held_out.mkdir(exist_ok=True)
    train_manifest = tmp_path / "train.jsonl"
    evaluation_manifest = tmp_path / "evaluation.jsonl"
    train_sha = _manifest(train_manifest, "train", "train-sample")
    evaluation_sha = _manifest(
        evaluation_manifest, "evaluation", "evaluation-sample")
    return {
        "_project_root": str(tmp_path),
        "_orze_dir": str(tmp_path / ".orze"),
        "data_boundaries": {
            "forbidden_in_training": [str(held_out)],
            "watch_paths": [],
            "training_network": "deny",
        },
        "data_separation": {
            "enabled": True,
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": train_sha,
            "evaluation_manifest": str(evaluation_manifest),
            "evaluation_manifest_sha256": evaluation_sha,
            "fingerprint_namespace_sha256": _NAMESPACE,
            "normalization_contract_sha256": _NORMALIZATION,
            "fields": ["sample"],
            "max_overlap": {"sample": 0},
            "max_records": 100,
            "max_bytes": 1024 * 1024,
            "max_line_bytes": 4096,
        },
        "model_lineage": {
            "enabled": True,
            "artifact": "model.bin",
            "max_files": 100,
            "max_bytes": 1024 * 1024,
            "attestation_timeout": 1,
        },
    }


def _tp(idea_id="idea-lineage", attempt_id="attempt-1"):
    return SimpleNamespace(
        idea_id=idea_id,
        attempt_id=attempt_id,
        execution_identity="c" * 64,
        process=SimpleNamespace(pid=os.getpid()),
        gpu=4,
        start_time=time.time(),
    )


def _write_boundary(idea_dir: Path, cfg: dict, tp) -> None:
    separation = ensure_data_separation(cfg)
    context = prepare_model_lineage_launch(
        idea_id=tp.idea_id,
        attempt_id=tp.attempt_id,
        execution_identity=tp.execution_identity,
        idea_dir=idea_dir,
        cfg=cfg,
        separation_receipt=separation,
    )
    os.write(
        context["write_fd"], (context["nonce"] + "\n").encode("ascii"))
    receive_model_lineage_attestation(context, process_pid=os.getpid())


def _completed_lineage(
    tmp_path: Path,
    *,
    cfg=None,
    idea_id="idea-lineage",
    attempt_id="attempt-1",
    artifact=b"one standalone model",
):
    cfg = _config(tmp_path) if cfg is None else cfg
    idea_dir = tmp_path / "results" / idea_id
    idea_dir.mkdir(parents=True)
    (idea_dir / "model.bin").write_bytes(artifact)
    tp = _tp(idea_id=idea_id, attempt_id=attempt_id)
    _write_boundary(idea_dir, cfg, tp)
    record_compute_start(tp, idea_dir, phase="training")
    finalized = finalize_model_lineage(tp, idea_dir, cfg)
    record_compute_terminal(
        tp, idea_dir, "completed", "trainer_completed",
        phase="training", return_code=0)
    return cfg, idea_dir, tp, finalized


@pytest.mark.parametrize("mutation, expected", [
    (lambda cfg: cfg["model_lineage"].update(enabled="yes"),
     "model_lineage.enabled"),
    (lambda cfg: cfg["model_lineage"].update(artifact="../model.bin"),
     "model_lineage.artifact"),
    (lambda cfg: cfg["data_boundaries"].update(
        forbidden_in_training=[]), "non-empty hard forbidden"),
    (lambda cfg: cfg["data_boundaries"].update(
        training_network="inherit"), "training_network"),
    (lambda cfg: cfg["data_separation"].update(enabled=False),
     "data_separation.enabled"),
])
def test_model_lineage_policy_requires_hard_boundaries(
        tmp_path, mutation, expected):
    cfg = _config(tmp_path)
    mutation(cfg)
    errors, _ = _validate_config(cfg)
    assert any(expected in error for error in errors), errors


def test_wrapper_attests_only_after_kernel_marker_and_strips_secret_env(
        tmp_path, monkeypatch):
    read_fd, write_fd = os.pipe()
    nonce = "d" * 64
    monkeypatch.setenv("ORZE_REQUIRE_KERNEL_BOUNDARY", "1")
    monkeypatch.setenv("ORZE_KERNEL_BOUNDARY_ACTIVE", "1")
    monkeypatch.setenv("ORZE_BOUNDARY_ATTEST_FD", str(write_fd))
    monkeypatch.setenv("ORZE_BOUNDARY_ATTEST_NONCE", nonce)
    monkeypatch.setattr(boundary_wrap, "activate", lambda: None)
    monkeypatch.setattr(boundary_wrap, "is_active", lambda: False)
    observed = {}

    def fake_run_path(*args, **kwargs):
        observed["fd"] = os.environ.get("ORZE_BOUNDARY_ATTEST_FD")
        observed["nonce"] = os.environ.get("ORZE_BOUNDARY_ATTEST_NONCE")

    monkeypatch.setattr(boundary_wrap.runpy, "run_path", fake_run_path)
    monkeypatch.setattr(
        boundary_wrap.sys, "argv", ["wrap", str(tmp_path / "train.py")])

    boundary_wrap.main()

    assert os.read(read_fd, 128) == (nonce + "\n").encode("ascii")
    os.close(read_fd)
    assert observed == {"fd": None, "nonce": None}


def test_parent_rejects_forged_boundary_attestation(tmp_path):
    cfg = _config(tmp_path)
    idea_dir = tmp_path / "results" / "idea-lineage"
    idea_dir.mkdir(parents=True)
    tp = _tp()
    context = prepare_model_lineage_launch(
        idea_id=tp.idea_id,
        attempt_id=tp.attempt_id,
        execution_identity=tp.execution_identity,
        idea_dir=idea_dir,
        cfg=cfg,
        separation_receipt=ensure_data_separation(cfg),
    )
    os.write(context["write_fd"], b"0" * 64 + b"\n")

    with pytest.raises(
            ModelLineageError, match="boundary_attestation_invalid"):
        receive_model_lineage_attestation(
            context, process_pid=os.getpid())
    assert not (idea_dir / "_compute_receipts" / tp.attempt_id /
                "boundary.json").exists()


def test_completed_lineage_binds_artifact_attempt_and_policy_without_rank_claim(
        tmp_path):
    cfg, idea_dir, tp, finalized = _completed_lineage(tmp_path)

    lineage, lineage_sha = validate_model_lineage_for_evaluation(
        idea_dir, cfg)

    assert lineage == finalized
    assert len(lineage_sha) == 64
    assert lineage["artifact_sha256"] == hashlib.sha256(
        b"one standalone model").hexdigest()
    assert lineage["managed_training"] is True
    assert lineage["rank_claim_proven"] is False
    durable = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            idea_dir / "_model_lineage.json",
            idea_dir / "_compute_receipts" / tp.attempt_id / "boundary.json",
        ])
    assert str(idea_dir / "model.bin") not in durable
    assert _fp("train-sample") not in durable


def test_publication_manifest_is_derived_from_same_validated_artifact(tmp_path):
    cfg, idea_dir, _, finalized = _completed_lineage(tmp_path)

    lineage, lineage_sha, manifest = validate_model_lineage_for_evaluation(
        idea_dir, cfg, include_artifact_manifest=True)

    expected_core = {
        "schema_version": 1,
        "hash_method": "sha256_bytes_v1",
        "files": [{
            "path": "model.bin",
            "size": len(b"one standalone model"),
            "sha256": hashlib.sha256(
                b"one standalone model").hexdigest(),
        }],
    }
    assert lineage == finalized
    assert len(lineage_sha) == 64
    assert manifest == {
        **expected_core,
        "manifest_sha256": lineage_module._canonical_hash(expected_core),
    }


def test_lineage_audit_rejects_compute_execution_identity_mismatch(tmp_path):
    cfg, idea_dir, tp, _ = _completed_lineage(tmp_path)
    replacement_identity = "d" * 64
    receipt_dir = idea_dir / "_compute_receipts" / tp.attempt_id

    boundary_path = receipt_dir / "boundary.json"
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    boundary["payload"]["execution_identity_sha256"] = replacement_identity
    boundary["payload_sha256"] = lineage_module._canonical_hash(
        boundary["payload"])
    boundary_path.write_text(
        json.dumps(boundary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    lineage_path = idea_dir / "_model_lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["payload"]["execution_identity_sha256"] = replacement_identity
    lineage["payload"]["boundary_receipt_sha256"] = boundary[
        "payload_sha256"]
    lineage["payload_sha256"] = lineage_module._canonical_hash(
        lineage["payload"])
    lineage_path.write_text(
        json.dumps(lineage, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    audit = audit_campaign_model_lineage(
        idea_dir.parent,
        cfg,
        idea_ids=[tp.idea_id],
        artifact_relation="any",
    )

    assert json.loads(
        (receipt_dir / "start.json").read_text(encoding="utf-8")
    )["execution_identity_sha256"] == "c" * 64
    assert lineage["payload"][
        "execution_identity_sha256"] == replacement_identity
    assert audit["execution_identity_sha256_by_idea"] == {}
    assert audit["status"] == "UNVERIFIED"
    assert audit["invalid_idea_ids"] == [tp.idea_id]


def test_lineage_finalization_rejects_compute_start_identity_mismatch(
        tmp_path):
    cfg = _config(tmp_path)
    idea_dir = tmp_path / "results" / "idea-lineage"
    idea_dir.mkdir(parents=True)
    (idea_dir / "model.bin").write_bytes(b"one standalone model")
    lineage_tp = _tp()
    lineage_tp.execution_identity = "d" * 64
    _write_boundary(idea_dir, cfg, lineage_tp)
    compute_tp = _tp()
    record_compute_start(compute_tp, idea_dir, phase="training")

    with pytest.raises(
            ModelLineageError, match="model_lineage_compute_start_invalid"):
        finalize_model_lineage(lineage_tp, idea_dir, cfg)


def test_campaign_lineage_audit_proves_identical_replication_artifacts(
        tmp_path):
    cfg, _, _, _ = _completed_lineage(
        tmp_path, idea_id="idea-replica-a", attempt_id="attempt-a"
    )
    _completed_lineage(
        tmp_path, cfg=cfg, idea_id="idea-replica-b", attempt_id="attempt-b"
    )

    audit = audit_campaign_model_lineage(
        tmp_path / "results",
        cfg,
        idea_ids=["idea-replica-a", "idea-replica-b"],
        artifact_relation="identical",
    )

    assert audit["status"] == "VERIFIED"
    assert audit["verified_lineage_count"] == 2
    assert audit["unique_artifact_count"] == 1
    assert audit["attempt_id_by_idea"] == {
        "idea-replica-a": "attempt-a",
        "idea-replica-b": "attempt-b",
    }
    assert audit["artifact_relation_passed"] is True
    assert audit["rank_claim_proven"] is False


def test_campaign_lineage_audit_rejects_wrong_artifact_relation(tmp_path):
    cfg, _, _, _ = _completed_lineage(
        tmp_path, idea_id="idea-distinct-a", attempt_id="attempt-a",
        artifact=b"model-a",
    )
    _completed_lineage(
        tmp_path, cfg=cfg, idea_id="idea-distinct-b", attempt_id="attempt-b",
        artifact=b"model-b",
    )

    audit = audit_campaign_model_lineage(
        tmp_path / "results",
        cfg,
        idea_ids=["idea-distinct-a", "idea-distinct-b"],
        artifact_relation="identical",
    )

    assert audit["status"] == "UNVERIFIED"
    assert audit["unique_artifact_count"] == 2
    assert audit["artifact_relation_passed"] is False


def test_directory_artifact_hash_is_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "b.bin").write_bytes(b"b")
    (first / "a.bin").write_bytes(b"a")
    (second / "a.bin").write_bytes(b"a")
    (second / "b.bin").write_bytes(b"b")

    left = _artifact_digest(first, 10, 100)
    right = _artifact_digest(second, 10, 100)

    assert left == right
    assert left["artifact_kind"] == "directory_tree_v1"


@pytest.mark.parametrize("kind, expected", [
    ("root_symlink", "artifact_redirected"),
    ("nested_symlink", "artifact_redirected"),
    ("empty", "artifact_empty"),
    ("too_many", "artifact_limit_exceeded"),
    ("too_large", "artifact_limit_exceeded"),
])
def test_artifact_hash_rejects_redirects_empty_and_limits(
        tmp_path, kind, expected):
    artifact = tmp_path / "artifact"
    if kind == "root_symlink":
        target = tmp_path / "target"
        target.write_bytes(b"model")
        artifact.symlink_to(target)
    else:
        artifact.mkdir()
        if kind == "nested_symlink":
            target = tmp_path / "target"
            target.write_bytes(b"model")
            (artifact / "link").symlink_to(target)
        elif kind == "too_many":
            (artifact / "one").write_bytes(b"1")
            (artifact / "two").write_bytes(b"2")
        elif kind == "too_large":
            (artifact / "large").write_bytes(b"12")
    max_files = 1 if kind == "too_many" else 10
    max_bytes = 1 if kind == "too_large" else 100

    with pytest.raises(ModelLineageError, match=expected):
        _artifact_digest(artifact, max_files, max_bytes)


def test_evaluation_rejects_redirected_compute_start_receipt(tmp_path):
    cfg, idea_dir, tp, _ = _completed_lineage(tmp_path)
    start_path = (
        idea_dir / "_compute_receipts" / tp.attempt_id / "start.json"
    )
    redirected = tmp_path / "redirected-start.json"
    redirected.write_bytes(start_path.read_bytes())
    start_path.unlink()
    start_path.symlink_to(redirected)

    with pytest.raises(
            ModelLineageError, match="model_lineage_compute_start_invalid"):
        validate_model_lineage_for_evaluation(idea_dir, cfg)


def test_evaluation_rejects_tainted_training_access_log(tmp_path):
    cfg, idea_dir, _, _ = _completed_lineage(tmp_path)
    cfg["managed_run"] = {"require_clean_training_access_log": True}
    (idea_dir / "_access_log.tsv").write_text(
        "WATCH\t/private/eval\t/private/eval/sample.arrow\n",
        encoding="utf-8",
    )

    with pytest.raises(
            ModelLineageError,
            match="model_lineage_training_access_log_not_clean"):
        validate_model_lineage_for_evaluation(idea_dir, cfg)


def test_single_file_rewrite_during_hash_is_detected(tmp_path, monkeypatch):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"a" * (1024 * 1024 + 1))
    original_read = lineage_module.os.read
    calls = 0

    def rewriting_read(fd, size):
        nonlocal calls
        chunk = original_read(fd, size)
        calls += 1
        if calls == 1:
            artifact.write_bytes(b"b" * (1024 * 1024 + 1))
        return chunk

    monkeypatch.setattr(lineage_module.os, "read", rewriting_read)

    with pytest.raises(
            ModelLineageError, match="artifact_changed_during_hash"):
        _artifact_digest(artifact, 10, 2 * 1024 * 1024)


def test_evaluation_rejects_artifact_drift_and_boundary_substitution(tmp_path):
    cfg, idea_dir, tp, _ = _completed_lineage(tmp_path)
    (idea_dir / "model.bin").write_bytes(b"substituted standalone model")
    with pytest.raises(ModelLineageError, match="artifact_drift"):
        validate_model_lineage_for_evaluation(idea_dir, cfg)

    (idea_dir / "model.bin").write_bytes(b"one standalone model")
    boundary = (
        idea_dir / "_compute_receipts" / tp.attempt_id / "boundary.json")
    document = json.loads(boundary.read_text(encoding="utf-8"))
    document["payload"]["training_network_denied"] = False
    boundary.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ModelLineageError, match="receipt_invalid"):
        validate_model_lineage_for_evaluation(idea_dir, cfg)


def test_launch_wires_parent_attestation_before_compute_start(
        tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    results = tmp_path / "results"
    idea_dir = results / "idea-lineage"
    idea_dir.mkdir(parents=True)
    train = tmp_path / "train.py"
    base = tmp_path / "base.yaml"
    ideas = tmp_path / "ideas.md"
    train.write_text("# trainer\n", encoding="utf-8")
    base.write_text("{}\n", encoding="utf-8")
    ideas.write_text("", encoding="utf-8")
    cfg.update({
        "train_script": str(train),
        "base_config": str(base),
        "ideas_file": str(ideas),
        "python": "python3",
    })
    monkeypatch.setattr(
        "orze.engine.launcher._probe_kernel_boundary", lambda **kwargs: None)
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free", lambda *args: None)

    class RunningProcess:
        pid = 987654321

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        # The trainer inherits both the physical-GPU lease and the lineage
        # attestation pipe; the latter is appended last by the launcher.
        assert len(kwargs["pass_fds"]) == 2
        fd = kwargs["pass_fds"][-1]
        os.write(fd, (kwargs["env"]["ORZE_BOUNDARY_ATTEST_NONCE"]
                      + "\n").encode("ascii"))
        return RunningProcess()

    monkeypatch.setattr("orze.engine.launcher.subprocess.Popen", fake_popen)

    tp = launch("idea-lineage", 4, results, cfg)
    tp.close_log()

    assert len(tp.execution_identity) == 64
    attempt = idea_dir / "_compute_receipts" / tp.attempt_id
    assert (attempt / "boundary.json").is_file()
    assert (attempt / "start.json").is_file()


def test_launch_accounts_then_terminates_unattested_child(
        tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    results = tmp_path / "results"
    idea_dir = results / "idea-lineage"
    idea_dir.mkdir(parents=True)
    train = tmp_path / "train.py"
    base = tmp_path / "base.yaml"
    ideas = tmp_path / "ideas.md"
    train.write_text("# trainer\n", encoding="utf-8")
    base.write_text("{}\n", encoding="utf-8")
    ideas.write_text("", encoding="utf-8")
    cfg.update({
        "train_script": str(train),
        "base_config": str(base),
        "ideas_file": str(ideas),
        "python": "python3",
    })
    monkeypatch.setattr(
        "orze.engine.launcher._probe_kernel_boundary", lambda **kwargs: None)
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free", lambda *args: None)
    terminated = []

    class RunningProcess:
        pid = 987654322
        returncode = None

        def poll(self):
            return self.returncode

    process = RunningProcess()

    def fake_popen(command, **kwargs):
        assert len(kwargs["pass_fds"]) == 2
        fd = kwargs["pass_fds"][-1]
        os.write(fd, b"0" * 64 + b"\n")
        return process

    def fake_terminate(proc, *args, **kwargs):
        terminated.append(proc.pid)
        proc.returncode = -15

    monkeypatch.setattr("orze.engine.launcher.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "orze.engine.launcher._terminate_and_reap", fake_terminate)

    with pytest.raises(
            ModelLineageError, match="boundary_attestation_invalid"):
        launch("idea-lineage", 4, results, cfg)

    assert terminated == [process.pid]
    receipt_dirs = list((idea_dir / "_compute_receipts").iterdir())
    assert len(receipt_dirs) == 1
    start = json.loads((receipt_dirs[0] / "start.json").read_text())
    assert start["phase"] == "training"
    assert start["process_pid"] == process.pid
    terminal = json.loads((receipt_dirs[0] / "terminal.json").read_text())
    assert terminal["outcome"] == "failed"
    assert terminal["reason_code"] == "training_launch_initialization_failed"


def test_completed_metrics_fail_closed_when_lineage_finalization_fails(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_dir = results / "idea-lineage"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED"}), encoding="utf-8")

    class FinishedProcess:
        pid = os.getpid()

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    tp = _tp()
    tp.process = FinishedProcess()
    tp.log_path = idea_dir / "train_output.log"
    tp.close_log = lambda: None
    record_compute_start(tp, idea_dir)
    monkeypatch.setattr(
        lineage_module, "finalize_model_lineage",
        lambda *args: (_ for _ in ()).throw(
            ModelLineageError("model_lineage_artifact_missing")))

    assert check_active(
        {4: tp}, results,
        {"model_lineage": {"enabled": True},
         "sops": {"failure_feedback": False}}, {},
    ) == [("idea-lineage", 4)]

    metrics = json.loads((idea_dir / "metrics.json").read_text())
    terminal = json.loads((
        idea_dir / "_compute_receipts" / tp.attempt_id / "terminal.json"
    ).read_text())
    assert metrics["status"] == "FAILED"
    assert metrics["error"] == "model_lineage_validation_failed"
    assert terminal["outcome"] == "failed"
    assert terminal["reason_code"] == "model_lineage_invalid"
    assert list(idea_dir.glob("metrics.lineage_invalid.*.json"))


def test_eval_rejects_missing_lineage_before_gpu_inspection(
        tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg.update({"eval_script": "eval.py", "eval_output": "eval.json"})
    results = tmp_path / "results"
    idea_dir = results / "idea-lineage"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED"}), encoding="utf-8")
    inspected = []
    monkeypatch.setattr(
        "orze.engine.evaluator._verify_gpu_free",
        lambda *args: inspected.append(True))

    assert launch_eval("idea-lineage", 4, results, cfg) is None
    assert inspected == []
    assert "training_model_lineage_invalid" in (
        idea_dir / "_eval_audit.jsonl").read_text(encoding="utf-8")


def test_local_report_cache_reuses_unchanged_lineage_and_rejects_drift(
        tmp_path, monkeypatch):
    cfg, idea_dir, _, _ = _completed_lineage(tmp_path)
    cfg["report"] = {
        "title": "Lineage-qualified results",
        "primary_metric": "score",
        "sort": "descending",
        "columns": [{"key": "score", "label": "Score"}],
    }
    (idea_dir / "metrics.json").write_text(json.dumps({
        "status": "COMPLETED", "score": 1.0,
    }), encoding="utf-8")
    results = idea_dir.parent
    ideas = {"idea-lineage": {"title": "Lineage"}}
    assert len(update_report(results, ideas, cfg)) == 1

    calls = 0
    original_digest = lineage_module._artifact_digest

    def counted_digest(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_digest(*args, **kwargs)

    monkeypatch.setattr(lineage_module, "_artifact_digest", counted_digest)
    assert len(update_report(results, ideas, cfg)) == 1
    assert calls == 0

    artifact = idea_dir / "model.bin"
    original_stat = artifact.stat()
    artifact.write_bytes(b"x" * len(b"one standalone model"))
    os.utime(
        artifact,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert update_report(results, ideas, cfg) == []
    assert calls == 1
    assert "local_model_lineage_invalid" in (
        results / "report.md").read_text(encoding="utf-8")
