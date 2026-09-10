"""Read-only interruption preparation and bounded, short publication.

Caller contract: prepare outside the effect lock. Publish only after acquiring
the current execution's effect lease and durable intent; this value is NOT an
authorization capability. Any uncertain publication error requires caller HOLD.
Regular-file metadata bounds the second phase. Directories are deliberately
non-resumable here; the legacy resume wrapper retains full directory hashing.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from orze.engine import accounting, resume
from orze.engine.execution_authority import canonical_identity_equal
from orze.engine.resume import ResumeValidationError
from orze.engine.termination_hold import require_no_unconfirmed_stop

MAX_INPUT_FILES = 64
MAX_PARENT_BINDINGS = 512
MAX_RECEIPT_BYTES = 65536


@dataclass(frozen=True)
class PreparedInterruption:
    payload_json: str
    payload_sha256: str
    idea_dir: str
    project_root: str
    policy_sha256: str
    subject: tuple
    files: tuple
    parents: tuple
    outputs: tuple


def _encoded(payload) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def _policy(cfg) -> str:
    if not isinstance(cfg, dict) or not isinstance(cfg.get("resume", {}), dict):
        raise ResumeValidationError("interruption_policy_invalid")
    return hashlib.sha256(_encoded({"resume": cfg.get("resume", {}),
                                   "train_script": str(cfg.get("train_script", ""))})).hexdigest()


def _subject(tp) -> tuple:
    attempt = getattr(tp, "attempt_id", None)
    if not isinstance(attempt, str) or not attempt or attempt in (".", ".."):
        raise ResumeValidationError("interruption_attempt_missing")
    try:
        accounting._token(attempt, "attempt_id")
        phase = "posthoc" if getattr(tp, "is_posthoc", False) else "training"
        accounting._base(tp, phase, "terminal", "interrupted")
        started = float(tp.start_time)
        if not math.isfinite(started) or started < 0:
            raise ValueError("invalid start")
    except (accounting.ComputeAccountingError, TypeError, ValueError) as exc:
        raise ResumeValidationError("interruption_subject_invalid") from exc
    return (tp.idea_id, attempt, getattr(tp, "execution_identity", None), phase,
            tp.gpu, getattr(tp.process, "pid", None), started,
            str(getattr(tp, "train_script", "")))


def _scope(tp, results_dir, cfg) -> tuple[Path, Path]:
    results = Path(results_dir).absolute()
    resume._reject_symlink_components(results)
    # Validate the ID before constructing a lexical child path.
    idea_dir = resume._idea_dir(tp.idea_id, results)
    resume._reject_symlink_components(results / tp.idea_id)
    return idea_dir, resume._project_root(cfg, results)


def _identity(path: Path, *, missing=False):
    try:
        value = path.lstat()
    except FileNotFoundError:
        if missing:
            return None
        raise
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ResumeValidationError("interruption_file_not_regular")
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _parents(paths) -> tuple:
    parents = {}
    for path in paths:
        for parent in (path.parent, *path.parent.parents):
            if str(parent) in parents:
                continue
            if len(parents) >= MAX_PARENT_BINDINGS:
                raise ResumeValidationError("native_resume_parent_limit_unsupported")
            try:
                info = parent.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(info.st_mode):
                raise ResumeValidationError("interruption_parent_not_directory")
            # Effect-intent/lock siblings may legitimately be created meanwhile.
            parents[str(parent)] = (info.st_dev, info.st_ino, info.st_mode)
    return tuple(sorted(parents.items()))


def _verify_bindings(files, parents):
    for raw, identity in files:
        if _identity(Path(raw)) != identity:
            raise ResumeValidationError("interruption_input_changed")
    for raw, identity in parents:
        info = Path(raw).lstat()
        if (info.st_dev, info.st_ino, info.st_mode) != identity:
            raise ResumeValidationError("interruption_parent_changed")


def _regular_inputs(tp, results, cfg, idea_dir, project_root):
    """Preflight before _receipt_contract can hash any directory or long list."""
    policy = cfg.get("resume", {})
    if not policy.get("enabled", False):
        raise ResumeValidationError("resume_policy_disabled")
    progress_path = idea_dir / str(policy.get("progress_file", "progress.json"))
    resume._reject_symlink_components(progress_path)
    if not resume._inside(progress_path, [idea_dir]):
        raise ResumeValidationError("progress_manifest_outside_idea_directory")
    progress_path = progress_path.resolve()
    progress_identity = _identity(progress_path)
    if progress_identity[4] > MAX_RECEIPT_BYTES:
        raise ResumeValidationError("native_resume_progress_limit_unsupported")
    progress, raw = resume._read_json(progress_path)
    declared = policy.get("immutable_inputs") or []
    extra = progress.get("immutable_inputs") or []
    if not isinstance(declared, list) or not isinstance(extra, list):
        raise ResumeValidationError("progress_immutable_inputs_not_list")
    if len(declared) + len(extra) + 4 > MAX_INPUT_FILES:
        raise ResumeValidationError("native_resume_file_limit_unsupported")
    checkpoint = resume._resolve_path(progress.get("checkpoint_path"), idea_dir,
        [idea_dir, *resume._configured_roots(cfg, "checkpoint_roots", project_root)], "checkpoint")
    idea_config = idea_dir / "idea_config.yaml"
    config_identity = _identity(idea_config)
    override = getattr(tp, "train_script", None)
    if override:
        script = Path(override)
    else:
        import yaml
        idea = yaml.safe_load(idea_config.read_text(encoding="utf-8")) or {}
        if not isinstance(idea, dict):
            idea = {}
        script = Path(str(idea.get("train_script") or cfg.get("train_script", "")))
    if not script.is_absolute():
        script = project_root / script
    resume._reject_symlink_components(script)
    script = script.resolve()
    roots = [project_root, *resume._configured_roots(cfg, "input_roots", project_root)]
    inputs = [resume._resolve_path(str(raw), project_root, roots, "immutable_input")
              for raw in (*declared, *extra)]
    paths = tuple(sorted({progress_path, checkpoint, idea_config, script, *inputs}, key=str))
    for path in paths:
        resume._reject_symlink_components(path)
        if stat.S_ISDIR(path.lstat().st_mode):
            raise ResumeValidationError("native_resume_directory_unsupported")
    files = tuple((str(path), _identity(path)) for path in paths)
    parents = _parents(paths)
    if dict(files)[str(progress_path)] != progress_identity or dict(files)[str(idea_config)] != config_identity:
        raise ResumeValidationError("interruption_input_changed")
    contract = resume._receipt_contract(tp.idea_id, results, cfg, progress, raw,
                                       train_script_override=str(script))
    _verify_bindings(files, parents)
    return contract, files, parents


def prepare_interruption(tp, results_dir: Path, cfg: dict, reason: str,
                         signal: str, return_code=None) -> PreparedInterruption:
    """Hash supported inputs read-only; unsupported layouts are non-resumable."""
    idea_dir, project_root = _scope(tp, results_dir, cfg)
    require_no_unconfirmed_stop(idea_dir)
    subject, policy = _subject(tp), _policy(cfg)
    if return_code is not None and (type(return_code) is not int):
        raise ResumeValidationError("interruption_return_code_invalid")
    payload = resume._interruption_payload(tp, reason, signal, return_code)
    files, parents = (), ()
    try:
        contract, files, parents = _regular_inputs(tp, Path(results_dir), cfg, idea_dir, project_root)
        payload.update(contract)
        payload.update(resume_eligible=True, resume_reason="verified")
    except ResumeValidationError as exc:
        payload["resume_reason"] = str(exc)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        payload["resume_reason"] = f"validation_error:{type(exc).__name__}"
    output_paths = (idea_dir / "interruption.json",
                    idea_dir / "_compute_receipts" / subject[1] / "terminal.json")
    outputs = tuple((str(path), _identity(path, missing=True)) for path in output_paths)
    parents = tuple(sorted(dict((*parents, *_parents(output_paths))).items()))
    if _subject(tp) != subject or _policy(cfg) != policy:
        raise ResumeValidationError("interruption_preparation_changed")
    encoded = _encoded(payload)
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise ResumeValidationError("interruption_payload_limit")
    return PreparedInterruption(encoded.decode(), hashlib.sha256(encoded).hexdigest(),
                                str(idea_dir), str(project_root), policy, subject,
                                files, parents, outputs)


def _sync_readback(path: Path, expected: bytes):
    resume._reject_symlink_components(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_RECEIPT_BYTES:
            raise ResumeValidationError("interruption_publication_file_invalid")
        raw = bytearray()
        while len(raw) <= MAX_RECEIPT_BYTES:
            block = os.read(fd, min(8192, MAX_RECEIPT_BYTES + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        if bytes(raw) != expected or _identity(path) != (
                info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
                info.st_size, info.st_mtime_ns, info.st_ctime_ns):
            raise ResumeValidationError("interruption_publication_readback_failed")
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync_parents(path, idea_dir):
    for parent in (path.parent, *path.parent.parents):
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if parent == idea_dir:
            break


def publish_interruption(prepared: PreparedInterruption, tp, results_dir: Path, cfg: dict) -> dict:
    """Write compact effects only. Caller holds its current-attempt effect lease."""
    if (type(prepared) is not PreparedInterruption or type(prepared.payload_json) is not str
            or len(prepared.payload_json.encode()) > MAX_RECEIPT_BYTES
            or hashlib.sha256(prepared.payload_json.encode()).hexdigest() != prepared.payload_sha256):
        raise ResumeValidationError("interruption_preparation_invalid")
    idea_dir, project_root = _scope(tp, results_dir, cfg)
    require_no_unconfirmed_stop(idea_dir)
    payload = json.loads(prepared.payload_json)

    def verify():
        if (str(idea_dir) != prepared.idea_dir or str(project_root) != prepared.project_root
                or _subject(tp) != prepared.subject or _policy(cfg) != prepared.policy_sha256):
            raise ResumeValidationError("interruption_publication_scope_changed")
        _verify_bindings(prepared.files, prepared.parents)

    verify()
    for raw, identity in prepared.outputs:
        path = Path(raw)
        resume._reject_symlink_components(path)
        if _identity(path, missing=True) != identity:
            raise ResumeValidationError("interruption_publication_output_changed")
    output = idea_dir / "interruption.json"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise ResumeValidationError("interruption_payload_limit")
    resume.atomic_write(output, encoded.decode())
    _sync_readback(output, encoded)
    _sync_parents(output, idea_dir)
    receipt = accounting.record_compute_terminal(tp, idea_dir, "interrupted",
        resume._interruption_reason_code(payload["reason"]), phase=prepared.subject[3],
        return_code=payload["return_code"])
    expected = accounting._base(tp, prepared.subject[3], "terminal", "interrupted")
    expected.update(reason_code=resume._interruption_reason_code(payload["reason"]),
                    return_code=payload["return_code"], started_at_epoch=round(prepared.subject[6], 6),
                    process_pid=prepared.subject[5])
    if (not isinstance(receipt, dict) or not canonical_identity_equal(
            {key: receipt.get(key) for key in expected}, expected)):
        raise ResumeValidationError("interruption_compute_receipt_conflict")
    compute_path = idea_dir / "_compute_receipts" / prepared.subject[1] / "terminal.json"
    _sync_readback(compute_path, _encoded(receipt))
    _sync_parents(compute_path, idea_dir)
    _sync_readback(output, encoded)
    verify()
    return payload
