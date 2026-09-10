"""Recoverable, allowlisted file preparation for an evaluation-only retry.

CALLING SPEC:
    prepare_retry_files(idea_dir, cfg, failure_id) -> retry_id
        Called while the lifecycle writer holds BEGIN IMMEDIATE, before its
        transition. A durable immutable manifest precedes every file move.
        Repeating after filesystem/DB failure resumes that same manifest;
        conflicting/new bytes are never overwritten or silently discarded.
        No compute, lifecycle writes, recursive deletion, or ledger refund.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from orze.core.fs import atomic_create
from orze.reporting.evidence import _evidence_path_unsafe, _safe_source_path


class EvaluationRetryError(ValueError):
    """Stable, content-free rejection of an evaluation retry request."""


_PROTECTED_NAMES = {
    "metrics.json", "claim.json", "_model_lineage.json", "_access_log.tsv",
    "_eval_audit.jsonl", "train_output.log", "interruption.json",
    "resume_request.json", "_benchmark_exposures.jsonl", ".sealed_hashes",
    "config.json", "config.yaml", "config.yml",
    "idea_config.yaml", "resolved_config.yaml",
}
_PROTECTED_TREES = {
    "_compute_receipts", "_evaluation_bundle", "_evaluation_retries", "_execution_stops",
    "_execution_effects", "_attempt_effect.lock", "_attempt_effect.lock.source-lock",
    "_execution_catalog.json",
    "_evaluation_attempts", "artifacts", "research_artifacts",
    "checkpoints", "checkpoint", "models",
}


def safe_file(idea_dir: Path, name: str) -> Path:
    path = _safe_source_path(idea_dir, name)
    if path is None or _evidence_path_unsafe(path):
        raise EvaluationRetryError("evaluation_retry_artifact_path_invalid")
    return path


def _safe_directory(path: Path) -> None:
    current = Path(path.absolute().anchor)
    for part in path.absolute().parts[1:]:
        current /= part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise EvaluationRetryError("evaluation_retry_directory_invalid")


def file_hash(path: Path) -> str:
    if _evidence_path_unsafe(path):
        raise EvaluationRetryError("evaluation_retry_artifact_path_invalid")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_once(source: Path, destination: Path) -> None:
    """Publish a byte-exact diagnostic copy; no partial destination appears."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=".copy-", delete=False) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as original:
                for chunk in iter(lambda: original.read(1024 * 1024), b""):
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    _sync_directory(destination.parent)


def _protected(path: Path, idea_dir: Path, cfg: dict) -> bool:
    relative = path.relative_to(idea_dir)
    if (relative.name in _PROTECTED_NAMES
            or any(part in _PROTECTED_TREES for part in relative.parts)
            or relative.suffix.lower() in {
                ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin"}):
        return True
    declared = [cfg.get("eval_checkpoint") or "best_model.pt"]
    lineage = cfg.get("model_lineage") or {}
    if isinstance(lineage, dict) and lineage.get("artifact"):
        declared.append(lineage["artifact"])
    for name in declared:
        target = Path(str(name))
        target = target if target.is_absolute() else idea_dir / target
        if path == target or target in path.parents:
            return True
    for name in cfg.get("sealed_files") or []:
        target = Path(name)
        if not target.is_absolute():
            target = Path(cfg.get("_project_root") or Path.cwd()) / target
        if path.absolute() == target.absolute():
            return True
    return False


def retry_file_policy(idea_dir: Path, cfg: dict) -> dict[str, str]:
    """Only these named outputs may move; domain metric sources are not owned."""
    from orze.core.benchmark_contract import get_benchmark_contract
    from orze.core.observation_contract import get_observation_contract
    if get_observation_contract(cfg) is not None:
        return {}  # Each evaluation owns a separate immutable history directory.

    output = str(cfg.get("eval_output") or "eval_report.json")
    candidates = {output: "move", "eval_output.log": "move"}
    contract = get_benchmark_contract(cfg)
    if contract:
        candidates[str(contract["receipt"])] = "move"
    # Current provenance continues to pin the exposure ledger until the next
    # prepare validates it, reserves a NEW look and publishes new provenance.
    copies = {"_benchmark_evaluation.json", "failure_analysis.json"}
    policy = {}
    for name, action in candidates.items():
        path = safe_file(idea_dir, name)
        normalized = str(path.relative_to(idea_dir))
        if path == idea_dir / "metrics.json" and name == output:
            continue  # Legacy in-place evaluator: never move training output.
        if _protected(path, idea_dir, cfg) or normalized in copies:
            raise EvaluationRetryError("evaluation_retry_protected_artifact")
        policy[normalized] = action
    for name in copies:
        safe_file(idea_dir, name)
        policy[name] = "copy"
    return policy


def _policy_hash(cfg: dict, policy: dict) -> str:
    fields = (
        "eval_script", "eval_args", "eval_output", "eval_checkpoint",
        "report", "metric_validation", "model_lineage", "data_boundaries",
        "data_separation", "managed_run", "sealed_files", "evaluation_bundle",
        "python", "train_extra_env", "eval_timeout",
    )
    payload = {key: cfg.get(key) for key in fields}
    payload["file_policy"] = policy
    try:
        data = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise EvaluationRetryError("evaluation_retry_policy_invalid") from exc
    return hashlib.sha256(data).hexdigest()


def _training_evidence(idea_dir: Path, cfg: dict) -> dict:
    """Pin declared completed-generation inputs, including explicit absence."""
    names = {"metrics.json", "claim.json", "idea_config.yaml",
             "resolved_config.yaml", "_model_lineage.json"}
    names.add(str(cfg.get("eval_checkpoint") or "best_model.pt"))
    evidence = {}
    for name in sorted(names):
        path = Path(name)
        if path.is_absolute():
            if _evidence_path_unsafe(path):
                raise EvaluationRetryError("evaluation_retry_training_path_invalid")
        else:
            path = safe_file(idea_dir, name)
        evidence[name] = file_hash(path) if path.exists() else None
    return evidence


def prepare_retry_files(idea_dir: Path, cfg: dict, failure_id: int) -> str:
    idea_dir = Path(idea_dir)
    if isinstance(failure_id, bool) or not isinstance(failure_id, int) or failure_id < 1:
        raise EvaluationRetryError("evaluation_retry_failure_identity_invalid")
    policy = retry_file_policy(idea_dir, cfg)
    training_hash = file_hash(safe_file(idea_dir, "metrics.json"))
    training_evidence = _training_evidence(idea_dir, cfg)
    fingerprint = _policy_hash(cfg, policy)
    retry_id = str(failure_id)
    archive = idea_dir / "_evaluation_retries" / retry_id
    _safe_directory(archive)
    manifest_path = safe_file(archive, "manifest.json")
    if not manifest_path.exists():
        entries = []
        for name, action in sorted(policy.items()):
            path = safe_file(idea_dir, name)
            entries.append({"path": name, "action": action,
                            "sha256": file_hash(path) if path.exists() else None})
        manifest = {
            "schema_version": 1, "idea_id": idea_dir.name, "retry_id": retry_id,
            "failed_transition_id": failure_id, "policy_sha256": fingerprint,
            "training_metrics_sha256": training_hash, "files": entries,
            "training_evidence": training_evidence,
        }
        atomic_create(manifest_path, json.dumps(manifest, sort_keys=True, indent=2))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationRetryError("evaluation_retry_manifest_invalid") from exc
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("idea_id") != idea_dir.name
            or manifest.get("retry_id") != retry_id
            or manifest.get("failed_transition_id") != failure_id
            or manifest.get("policy_sha256") != fingerprint
            or manifest.get("training_metrics_sha256") != training_hash
            or manifest.get("training_evidence") != training_evidence):
        raise EvaluationRetryError("evaluation_retry_manifest_identity_conflict")
    entries = manifest.get("files")
    if (not isinstance(entries, list) or len(entries) != len(policy)
            or any(not isinstance(entry, dict) for entry in entries)
            or {entry.get("path"): entry.get("action") for entry in entries} != policy):
        raise EvaluationRetryError("evaluation_retry_manifest_files_invalid")

    # Validate the entire remaining move set before mutating any file. A
    # previous partial preparation may have source OR destination, not new
    # conflicting bytes at either location.
    pending = []
    for entry in entries:
        name, action, expected = entry["path"], entry["action"], entry.get("sha256")
        source = safe_file(idea_dir, name)
        target = archive / "artifacts" / name
        safe_file(archive, str(target.relative_to(archive)))
        if expected is None:
            if source.exists() or target.exists():
                raise EvaluationRetryError("evaluation_retry_unexpected_artifact")
            continue
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise EvaluationRetryError("evaluation_retry_manifest_hash_invalid")
        if target.exists():
            if file_hash(target) != expected:
                raise EvaluationRetryError("evaluation_retry_archive_conflict")
            if action == "move" and source.exists():
                raise EvaluationRetryError("evaluation_retry_source_reappeared")
            if action == "copy" and (not source.exists() or file_hash(source) != expected):
                raise EvaluationRetryError("evaluation_retry_source_changed")
            continue
        if not source.exists() or file_hash(source) != expected:
            raise EvaluationRetryError("evaluation_retry_source_changed")
        pending.append((source, target, action, expected))
    for source, target, action, expected in pending:
        _safe_directory(target.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        if action == "move":
            if target.exists() or file_hash(source) != expected:
                raise EvaluationRetryError("evaluation_retry_move_conflict")
            os.rename(source, target)
            _sync_directory(source.parent)
            _sync_directory(target.parent)
        else:
            _copy_once(source, target)
        if file_hash(target) != expected:
            raise EvaluationRetryError("evaluation_retry_archive_changed")
    if _training_evidence(idea_dir, cfg) != training_evidence:
        raise EvaluationRetryError("evaluation_retry_training_changed")
    return retry_id


def verify_retry_prepared(idea_dir: Path, cfg: dict, failure_id: int) -> None:
    """Read-only launch/replay check; never finish partial preparation here."""
    policy = retry_file_policy(idea_dir, cfg)
    archive = idea_dir / "_evaluation_retries" / str(failure_id)
    _safe_directory(archive)
    manifest_path = safe_file(archive, "manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationRetryError("evaluation_retry_manifest_invalid") from exc
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("idea_id") != idea_dir.name
            or manifest.get("retry_id") != str(failure_id)
            or manifest.get("failed_transition_id") != failure_id
            or manifest.get("policy_sha256") != _policy_hash(cfg, policy)
            or manifest.get("training_evidence") != _training_evidence(idea_dir, cfg)
            or manifest.get("training_metrics_sha256") != file_hash(
                safe_file(idea_dir, "metrics.json"))):
        raise EvaluationRetryError("evaluation_retry_manifest_identity_conflict")
    entries = manifest.get("files")
    if (not isinstance(entries, list) or len(entries) != len(policy)
            or any(not isinstance(entry, dict) for entry in entries)
            or {entry.get("path"): entry.get("action") for entry in entries} != policy):
        raise EvaluationRetryError("evaluation_retry_manifest_files_invalid")
    for entry in entries:
        name, expected = entry["path"], entry.get("sha256")
        target = safe_file(archive, "artifacts/" + name)
        if expected is None:
            if target.exists():
                raise EvaluationRetryError("evaluation_retry_archive_conflict")
        elif not target.exists() or file_hash(target) != expected:
            raise EvaluationRetryError("evaluation_retry_archive_conflict")
        # Copies pin historical diagnostics. Current provenance may advance
        # after a legitimate pre-launch exposure reservation, so do not demand
        # that it still contains the previous nonce after admission.
        if entry["action"] == "move" and safe_file(idea_dir, name).exists():
            raise EvaluationRetryError("evaluation_retry_source_reappeared")
