"""Fail-closed interruption receipts and explicit checkpoint re-admission.

Orze never guesses that a checkpoint is complete. A trainer must publish a
cooperative progress manifest, and the project must opt into a resume contract.
On interruption Orze hashes the checkpoint, progress manifest, launch inputs,
and declared immutable inputs. Re-admission and launch both recompute every
hash; drift prevents GPU allocation.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Iterable, Optional

from orze.core.fs import atomic_write
from orze.engine.process import process_is_running


class ResumeValidationError(RuntimeError):
    """Raised when interruption or resume evidence is incomplete or changed."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ResumeValidationError(f"invalid_json:{path.name}") from exc
    if not isinstance(value, dict):
        raise ResumeValidationError(f"expected_mapping:{path.name}")
    return value, raw


def _inside(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.resolve(strict=False)
    for root in roots:
        try:
            resolved.relative_to(root.resolve(strict=False))
            return True
        except ValueError:
            continue
    return False


def _reject_symlink_components(path: Path) -> None:
    """Reject an existing symlink anywhere in a path before resolving it."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            if current.is_symlink():
                raise ResumeValidationError("symlink_artifact_forbidden")
        except OSError as exc:
            raise ResumeValidationError("artifact_path_unreadable") from exc


def _resolve_path(raw: str, base: Path, roots: Iterable[Path], label: str
                  ) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ResumeValidationError(f"{label}_path_missing")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    _reject_symlink_components(path)
    path = path.resolve(strict=False)
    if not _inside(path, roots):
        raise ResumeValidationError(f"{label}_outside_allowed_roots")
    if not path.exists():
        raise ResumeValidationError(f"{label}_missing")
    return path


def _hash_path(path: Path, max_files: int = 10000,
               max_bytes: int = 0) -> dict:
    """Hash a file or directory without following symlinks."""
    _reject_symlink_components(path)
    path = path.resolve(strict=False)
    if path.is_file():
        before = path.stat()
        size = before.st_size
        if max_bytes and size > max_bytes:
            raise ResumeValidationError("artifact_byte_limit_exceeded")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns)):
            raise ResumeValidationError("artifact_changed_during_hash")
        return {
            "kind": "file", "sha256": digest.hexdigest(),
            "size_bytes": size, "file_count": 1,
        }
    if not path.is_dir():
        raise ResumeValidationError("artifact_not_regular")

    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    candidates = sorted(path.rglob("*"), key=lambda item: str(item))
    relative_files = []
    for candidate in candidates:
        if candidate.is_symlink():
            raise ResumeValidationError("symlink_artifact_forbidden")
        if not candidate.is_file():
            continue
        file_count += 1
        if max_files and file_count > max_files:
            raise ResumeValidationError("artifact_file_limit_exceeded")
        before = candidate.stat()
        size = before.st_size
        total_bytes += size
        if max_bytes and total_bytes > max_bytes:
            raise ResumeValidationError("artifact_byte_limit_exceeded")
        file_digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(block)
        after = candidate.stat()
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns)):
            raise ResumeValidationError("artifact_changed_during_hash")
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        relative_files.append(relative.decode("utf-8"))
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_digest.hexdigest()))
    if file_count == 0:
        raise ResumeValidationError("checkpoint_empty")
    current_files = []
    for candidate in sorted(path.rglob("*"), key=lambda item: str(item)):
        if candidate.is_symlink():
            raise ResumeValidationError("symlink_artifact_forbidden")
        if candidate.is_file():
            current_files.append(candidate.relative_to(path).as_posix())
    if current_files != relative_files:
        raise ResumeValidationError("artifact_changed_during_hash")
    return {
        "kind": "directory", "sha256": digest.hexdigest(),
        "size_bytes": total_bytes, "file_count": file_count,
    }


def _project_root(cfg: dict, results_dir: Path) -> Path:
    return Path(cfg.get("_project_root") or results_dir.parent).resolve()


def _idea_dir(idea_id: str, results_dir: Path) -> Path:
    if (not isinstance(idea_id, str) or not idea_id
            or Path(idea_id).parts != (idea_id,)
            or idea_id in (".", "..")):
        raise ResumeValidationError("idea_id_invalid")
    return (Path(results_dir) / idea_id).resolve(strict=False)


def _configured_roots(cfg: dict, key: str, project_root: Path) -> list[Path]:
    roots = []
    for raw in (cfg.get("resume") or {}).get(key, []) or []:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = project_root / path
        roots.append(path.resolve(strict=False))
    return roots


def _display_path(path: Path, project_root: Path) -> dict:
    try:
        return {"scope": "project", "path": path.relative_to(project_root).as_posix()}
    except ValueError:
        return {"scope": "absolute", "path": str(path)}


def _stored_path(value: dict, project_root: Path) -> Path:
    if not isinstance(value, dict):
        raise ResumeValidationError("stored_path_invalid")
    raw = value.get("path")
    if not isinstance(raw, str) or not raw:
        raise ResumeValidationError("stored_path_invalid")
    path = Path(raw)
    if value.get("scope") == "project":
        path = project_root / path
    elif value.get("scope") != "absolute":
        raise ResumeValidationError("stored_path_scope_invalid")
    _reject_symlink_components(path)
    return path.resolve(strict=False)


def _contract_file(path: Path, project_root: Path) -> dict:
    if not path.is_file():
        raise ResumeValidationError(f"contract_file_missing:{path.name}")
    hashed = _hash_path(path)
    return {"location": _display_path(path, project_root), **hashed}


def _train_script_for_idea(idea_config: Path, cfg: dict,
                           project_root: Path) -> Path:
    raw = cfg.get("train_script", "")
    try:
        import yaml
        idea = yaml.safe_load(idea_config.read_text(encoding="utf-8")) or {}
        if isinstance(idea, dict) and idea.get("train_script"):
            raw = idea["train_script"]
    except (OSError, ValueError, TypeError):
        pass
    path = Path(str(raw))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def _resume_args(cfg: dict) -> list[str]:
    args = (cfg.get("resume") or {}).get(
        "args", ["--resume-from", "{checkpoint}"])
    if not isinstance(args, list) or not any(
            "{checkpoint}" in str(arg) for arg in args):
        raise ResumeValidationError("resume_args_missing_checkpoint_placeholder")
    return [str(arg) for arg in args]


def _receipt_contract(idea_id: str, results_dir: Path, cfg: dict,
                      progress: dict, progress_raw: bytes,
                      train_script_override: Optional[str] = None) -> dict:
    project_root = _project_root(cfg, results_dir)
    idea_dir = _idea_dir(idea_id, results_dir)
    resume_cfg = cfg.get("resume") or {}
    if not resume_cfg.get("enabled", False):
        raise ResumeValidationError("resume_policy_disabled")
    if progress.get("schema_version") != 1:
        raise ResumeValidationError("progress_schema_unsupported")
    if progress.get("resume_eligible") is not True:
        raise ResumeValidationError("trainer_declared_non_resumable")
    completed_step = progress.get("last_completed_step")
    if (isinstance(completed_step, bool)
            or not isinstance(completed_step, int)
            or completed_step < 0):
        raise ResumeValidationError("last_completed_step_invalid")

    checkpoint_roots = [idea_dir]
    checkpoint_roots.extend(
        _configured_roots(cfg, "checkpoint_roots", project_root))
    checkpoint = _resolve_path(
        progress.get("checkpoint_path"), idea_dir, checkpoint_roots,
        "checkpoint")
    max_files = int(resume_cfg.get("max_files", 10000))
    max_bytes = int(resume_cfg.get("max_bytes", 0))
    checkpoint_hash = _hash_path(checkpoint, max_files, max_bytes)

    idea_config = idea_dir / "idea_config.yaml"
    if train_script_override:
        train_script = Path(train_script_override)
        if not train_script.is_absolute():
            train_script = project_root / train_script
        train_script = train_script.resolve(strict=False)
    else:
        train_script = _train_script_for_idea(idea_config, cfg, project_root)

    declared_inputs = list(resume_cfg.get("immutable_inputs") or [])
    extra_inputs = progress.get("immutable_inputs") or []
    if not isinstance(extra_inputs, list):
        raise ResumeValidationError("progress_immutable_inputs_not_list")
    declared_inputs.extend(extra_inputs)
    if not declared_inputs:
        raise ResumeValidationError("immutable_inputs_missing")
    input_roots = [project_root]
    input_roots.extend(_configured_roots(cfg, "input_roots", project_root))
    immutable = []
    seen = set()
    for raw in declared_inputs:
        path = _resolve_path(str(raw), project_root, input_roots,
                             "immutable_input")
        if str(path) in seen:
            continue
        seen.add(str(path))
        immutable.append({
            "location": _display_path(path, project_root),
            **_hash_path(path, max_files, max_bytes),
        })

    return {
        "last_completed_step": progress["last_completed_step"],
        "progress_sha256": _sha256_bytes(progress_raw),
        "checkpoint": {
            "location": _display_path(checkpoint, project_root),
            **checkpoint_hash,
        },
        "idea_config": _contract_file(idea_config, project_root),
        "train_script": _contract_file(train_script, project_root),
        "immutable_inputs": immutable,
        "resume_args": _resume_args(cfg),
    }


def write_interruption_receipt(tp, results_dir: Path, cfg: dict, reason: str,
                               terminating_signal: str,
                               return_code: Optional[int] = None) -> dict:
    """Write a non-secret receipt; never infer resumability on errors."""
    results_dir = Path(results_dir)
    idea_dir = _idea_dir(tp.idea_id, results_dir)
    idea_dir.mkdir(parents=True, exist_ok=True)
    progress_name = str((cfg.get("resume") or {}).get(
        "progress_file", "progress.json"))
    unresolved_progress = idea_dir / progress_name
    receipt = {
        "schema_version": 1,
        "idea_id": tp.idea_id,
        "interrupted_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "reason": str(reason),
        "terminating_signal": str(terminating_signal),
        "return_code": return_code,
        "gpu": tp.gpu,
        # This is allocated-slot wall time, not a utilization estimate.
        "allocated_gpu_seconds": round(max(0, time.time() - tp.start_time), 3),
        "trainer_pid": getattr(tp.process, "pid", None),
        "resume_eligible": False,
    }
    try:
        if not (cfg.get("resume") or {}).get("enabled", False):
            raise ResumeValidationError("resume_policy_disabled")
        _reject_symlink_components(unresolved_progress)
        progress_path = unresolved_progress.resolve(strict=False)
        if not _inside(progress_path, [idea_dir]):
            raise ResumeValidationError(
                "progress_manifest_outside_idea_directory")
        progress, progress_raw = _read_json(progress_path)
        contract = _receipt_contract(
            tp.idea_id, results_dir, cfg, progress, progress_raw,
            train_script_override=getattr(tp, "train_script", None),
        )
        receipt.update(contract)
        receipt["resume_eligible"] = True
        receipt["resume_reason"] = "verified"
    except ResumeValidationError as exc:
        receipt["resume_reason"] = str(exc)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        receipt["resume_reason"] = f"validation_error:{type(exc).__name__}"
    atomic_write(
        idea_dir / "interruption.json",
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    return receipt


def _verify_hashed_path(record: dict, project_root: Path, roots: list[Path],
                        max_files: int, max_bytes: int, label: str) -> Path:
    if not isinstance(record, dict):
        raise ResumeValidationError(f"{label}_record_invalid")
    path = _stored_path(record.get("location"), project_root)
    if not _inside(path, roots):
        raise ResumeValidationError(f"{label}_outside_allowed_roots")
    current = _hash_path(path, max_files, max_bytes)
    for field in ("kind", "sha256", "size_bytes", "file_count"):
        if current.get(field) != record.get(field):
            raise ResumeValidationError(f"{label}_hash_mismatch")
    return path


def validate_resume_evidence(idea_id: str, results_dir: Path, cfg: dict,
                             checkpoint_override: Optional[str] = None
                             ) -> tuple[dict, Path, str]:
    """Revalidate an interruption receipt and return checkpoint + receipt hash."""
    results_dir = Path(results_dir)
    idea_dir = _idea_dir(idea_id, results_dir)
    project_root = _project_root(cfg, results_dir)
    resume_cfg = cfg.get("resume") or {}
    if not resume_cfg.get("enabled", False):
        raise ResumeValidationError("resume_policy_disabled")
    receipt_path = idea_dir / "interruption.json"
    receipt, receipt_raw = _read_json(receipt_path)
    if receipt.get("schema_version") != 1 or receipt.get("idea_id") != idea_id:
        raise ResumeValidationError("interruption_receipt_identity_invalid")
    if receipt.get("resume_eligible") is not True:
        raise ResumeValidationError(
            f"interruption_not_resumable:{receipt.get('resume_reason', 'unknown')}")
    if receipt.get("resume_args") != _resume_args(cfg):
        raise ResumeValidationError("resume_args_changed")
    if not isinstance(receipt.get("immutable_inputs"), list) or not receipt[
            "immutable_inputs"]:
        raise ResumeValidationError("immutable_inputs_missing")
    max_files = int(resume_cfg.get("max_files", 10000))
    max_bytes = int(resume_cfg.get("max_bytes", 0))

    checkpoint_roots = [idea_dir]
    checkpoint_roots.extend(
        _configured_roots(cfg, "checkpoint_roots", project_root))
    checkpoint = _verify_hashed_path(
        receipt["checkpoint"], project_root, checkpoint_roots,
        max_files, max_bytes, "checkpoint")
    if checkpoint_override:
        override = _resolve_path(
            checkpoint_override, project_root, checkpoint_roots,
            "resume_override")
        if override != checkpoint:
            raise ResumeValidationError("resume_override_not_attested_checkpoint")

    _verify_hashed_path(
        receipt["idea_config"], project_root, [idea_dir],
        max_files, max_bytes, "idea_config")
    _verify_hashed_path(
        receipt["train_script"], project_root, [project_root],
        max_files, max_bytes, "train_script")
    input_roots = [project_root]
    input_roots.extend(_configured_roots(cfg, "input_roots", project_root))
    for index, record in enumerate(receipt.get("immutable_inputs") or []):
        _verify_hashed_path(
            record, project_root, input_roots, max_files, max_bytes,
            f"immutable_input_{index}")

    progress_name = str(resume_cfg.get("progress_file", "progress.json"))
    unresolved_progress = idea_dir / progress_name
    _reject_symlink_components(unresolved_progress)
    progress_path = unresolved_progress.resolve(strict=False)
    if not _inside(progress_path, [idea_dir]):
        raise ResumeValidationError("progress_manifest_outside_idea_directory")
    try:
        progress_raw = progress_path.read_bytes()
    except OSError as exc:
        raise ResumeValidationError("progress_manifest_missing") from exc
    if _sha256_bytes(progress_raw) != receipt.get("progress_sha256"):
        raise ResumeValidationError("progress_manifest_hash_mismatch")
    return receipt, checkpoint, _sha256_bytes(receipt_raw)


def admit_resume(idea_id: str, results_dir: Path, cfg: dict,
                 checkpoint_override: str) -> dict:
    """Explicitly re-admit one attested checkpoint without deleting evidence."""
    results_dir = Path(results_dir)
    idea_dir = _idea_dir(idea_id, results_dir)
    receipt, checkpoint, receipt_sha = validate_resume_evidence(
        idea_id, results_dir, cfg, checkpoint_override)
    claim_path = idea_dir / "claim.json"
    if claim_path.exists():
        try:
            claim, _ = _read_json(claim_path)
            pid = int(claim.get("trainer_pid") or 0)
            if pid <= 0:
                raise ResumeValidationError("claim_identity_missing")
            start_ticks = claim.get("trainer_start_ticks")
            if pid and process_is_running(pid, start_ticks):
                raise ResumeValidationError("trainer_still_running")
        except ResumeValidationError:
            raise
        except (TypeError, ValueError):
            raise ResumeValidationError("claim_identity_invalid")

    request = {
        "schema_version": 1,
        "idea_id": idea_id,
        "created_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "created_by_host": socket.gethostname(),
        "checkpoint": _display_path(checkpoint, _project_root(cfg, results_dir)),
        "checkpoint_sha256": receipt["checkpoint"]["sha256"],
        "interruption_receipt_sha256": receipt_sha,
    }
    # Requeue in the audited DB before releasing filesystem admission. A crash
    # before the request is written remains fail-closed because claim/metrics
    # still block the scheduler.
    lake = None
    try:
        from orze.idea_lake import IdeaLake
        db_path = cfg.get("idea_lake_db")
        if db_path and Path(db_path).exists():
            lake = IdeaLake(str(db_path))
            if not lake.set_status(idea_id, "queued"):
                raise ResumeValidationError("idea_lake_requeue_failed")
    finally:
        if lake is not None:
            lake.close()

    atomic_write(
        idea_dir / "resume_request.json",
        json.dumps(request, indent=2, sort_keys=True) + "\n",
    )

    stamp = int(time.time())
    for source, label in ((idea_dir / "metrics.json", "metrics"),
                          (claim_path, "claim")):
        if source.exists():
            target = idea_dir / f"{label}.interrupted.{stamp}.json"
            suffix = 1
            while target.exists():
                target = idea_dir / f"{label}.interrupted.{stamp}.{suffix}.json"
                suffix += 1
            os.replace(source, target)
    return request


def prepare_resume_launch(idea_id: str, results_dir: Path, cfg: dict
                          ) -> Optional[dict]:
    """Validate an explicit request again and format its launch arguments."""
    request_path = _idea_dir(idea_id, Path(results_dir)) / "resume_request.json"
    if not request_path.exists():
        return None
    request, _ = _read_json(request_path)
    receipt, checkpoint, receipt_sha = validate_resume_evidence(
        idea_id, results_dir, cfg, str(_stored_path(
            request.get("checkpoint"), _project_root(cfg, Path(results_dir)))))
    if request.get("idea_id") != idea_id:
        raise ResumeValidationError("resume_request_identity_invalid")
    if request.get("interruption_receipt_sha256") != receipt_sha:
        raise ResumeValidationError("resume_request_receipt_hash_mismatch")
    if request.get("checkpoint_sha256") != receipt["checkpoint"]["sha256"]:
        raise ResumeValidationError("resume_request_checkpoint_hash_mismatch")
    args = [
        str(arg).replace("{checkpoint}", str(checkpoint))
        for arg in _resume_args(cfg)
    ]
    return {
        "args": args,
        "checkpoint": str(checkpoint),
        "receipt_sha256": receipt_sha,
        "request_path": request_path,
    }


def mark_resume_launched(context: dict, claim_path: Path) -> None:
    """Consume a request only after trainer identity is durably recorded."""
    request_path = Path(context["request_path"])
    consumed = request_path.with_name("resume_request.consumed.json")
    if consumed.exists():
        consumed = request_path.with_name(
            f"resume_request.consumed.{int(time.time())}.json")
    if not claim_path.exists():
        raise ResumeValidationError("resume_launch_claim_missing")
    claim, _ = _read_json(claim_path)
    claim.update({
        "resume_checkpoint": context["checkpoint"],
        "resume_receipt_sha256": context["receipt_sha256"],
    })
    atomic_write(claim_path, json.dumps(claim, indent=2) + "\n")
    os.replace(request_path, consumed)
