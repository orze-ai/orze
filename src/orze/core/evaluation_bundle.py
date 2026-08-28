"""Content-addressed, immutable local-code bundles for evaluation.

Evaluation scripts often spawn other project-local scripts after the parent
process has started.  Hashing the parent at launch does not protect those late
loads from a concurrently edited working tree.  This module copies every
declared local executable/config into one content-addressed directory and
verifies the copy before it can be executed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


BUNDLE_DIR = "_evaluation_bundle"
BUNDLE_MANIFEST = "manifest.json"
BUNDLE_SCHEMA_VERSION = 1


class EvaluationBundleError(RuntimeError):
    """Raised when an evaluation bundle cannot be proven immutable."""


@dataclass(frozen=True)
class PreparedEvaluationBundle:
    root: Path
    entrypoint: Path
    manifest_path: Path
    sha256: str

    def environment(self, project_root: Path) -> dict[str, str]:
        return {
            "ORZE_EVALUATION_BUNDLE_ROOT": str(self.root),
            "ORZE_EVALUATION_BUNDLE_MANIFEST": str(self.manifest_path),
            "ORZE_EVALUATION_BUNDLE_SHA256": self.sha256,
            "ORZE_ORIGINAL_PROJECT_ROOT": str(Path(project_root).resolve()),
        }


def get_evaluation_bundle_config(cfg: Mapping) -> Optional[dict]:
    spec = cfg.get("evaluation_bundle")
    if not isinstance(spec, dict) or spec.get("enabled") is not True:
        return None
    return spec


def _safe_relative(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationBundleError(f"{label}_must_be_nonempty_relative_path")
    path = Path(value)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise EvaluationBundleError(f"{label}_must_stay_inside_project_root")
    if any(ord(char) < 32 for char in value):
        raise EvaluationBundleError(f"{label}_contains_control_character")
    return path


def _assert_no_symlink_path(root: Path, relative: Path, label: str) -> Path:
    if root.is_symlink():
        raise EvaluationBundleError(f"{label}_root_symlink_forbidden")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise EvaluationBundleError(f"{label}_symlink_forbidden:{relative}")
    return current


def validate_evaluation_bundle_config(cfg: Mapping) -> list[str]:
    """Return fail-closed configuration errors for ``evaluation_bundle``."""
    spec = cfg.get("evaluation_bundle")
    if spec is None:
        return []
    prefix = "evaluation_bundle"
    if not isinstance(spec, dict):
        return [f"{prefix}: must be a mapping"]
    enabled = spec.get("enabled", False)
    if not isinstance(enabled, bool):
        return [f"{prefix}.enabled: must be true or false"]
    if not enabled:
        return []

    errors: list[str] = []
    files = spec.get("files")
    if (not isinstance(files, list) or not files
            or not all(isinstance(value, str) and value.strip()
                       for value in files)
            or len(files) != len(set(files))):
        errors.append(
            f"{prefix}.files: must be a non-empty list of unique paths")
        files = []
    for value in files:
        try:
            _safe_relative(value, f"{prefix}.files")
        except EvaluationBundleError as exc:
            errors.append(str(exc))

    eval_script = cfg.get("eval_script")
    if not isinstance(eval_script, str) or not eval_script:
        errors.append(f"{prefix}: requires eval_script")
    else:
        try:
            _safe_relative(eval_script, f"{prefix}.entrypoint")
        except EvaluationBundleError as exc:
            errors.append(str(exc))
        if eval_script not in files:
            errors.append(f"{prefix}.files: must include eval_script")

    pinned = cfg.get("sealed_hashes")
    if not isinstance(pinned, Mapping):
        errors.append(f"{prefix}: requires sealed_hashes")
    else:
        for value in files:
            digest = pinned.get(value)
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdefABCDEF"
                           for char in digest)):
                errors.append(
                    f"{prefix}.files: {value} must have a sealed SHA-256")
    return errors


def _manifest_for(cfg: Mapping) -> dict:
    spec = get_evaluation_bundle_config(cfg)
    if spec is None:
        raise EvaluationBundleError("evaluation_bundle_disabled")
    errors = validate_evaluation_bundle_config(cfg)
    if errors:
        raise EvaluationBundleError("; ".join(errors))
    pins = cfg["sealed_hashes"]
    files = {
        str(path): str(pins[path]).lower()
        for path in sorted(spec["files"])
    }
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "entrypoint": str(cfg["eval_script"]),
        "files": files,
    }


def _canonical_manifest(manifest: Mapping) -> bytes:
    return (json.dumps(
        manifest, sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")


def _manifest_sha256(manifest: Mapping) -> str:
    return hashlib.sha256(_canonical_manifest(manifest)).hexdigest()


def _copy_verified(source: Path, destination: Path, expected: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(str(source), flags)
    except OSError as exc:
        raise EvaluationBundleError(f"bundle_source_unreadable:{source}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise EvaluationBundleError(f"bundle_source_not_regular:{source}")
        destination_fd = os.open(
            str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    if written <= 0:
                        raise EvaluationBundleError(
                            f"bundle_destination_short_write:{destination}")
                    view = view[written:]
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
        after = os.fstat(source_fd)
    finally:
        os.close(source_fd)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise EvaluationBundleError(f"bundle_source_changed_during_copy:{source}")
    actual = digest.hexdigest()
    if actual != expected:
        raise EvaluationBundleError(
            f"bundle_source_hash_drift:{source}:expected={expected}:actual={actual}")
    destination.chmod(0o444)


def _load_manifest(path: Path) -> dict:
    if path.is_symlink():
        raise EvaluationBundleError("bundle_manifest_symlink_forbidden")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationBundleError("bundle_manifest_invalid") from exc
    if not isinstance(value, dict):
        raise EvaluationBundleError("bundle_manifest_invalid")
    return value


def verify_evaluation_bundle(
    idea_dir: Path, cfg: Mapping,
) -> PreparedEvaluationBundle:
    """Verify the exact content-addressed bundle already staged for an idea."""
    idea_dir = Path(idea_dir).resolve()
    expected_manifest = _manifest_for(cfg)
    identity = _manifest_sha256(expected_manifest)
    bundle_parent = Path(idea_dir) / BUNDLE_DIR
    bundle_root = bundle_parent / identity
    if bundle_parent.is_symlink() or bundle_root.is_symlink():
        raise EvaluationBundleError("bundle_directory_symlink_forbidden")
    manifest_path = bundle_root / BUNDLE_MANIFEST
    observed = _load_manifest(manifest_path)
    if observed != expected_manifest:
        raise EvaluationBundleError("bundle_manifest_identity_mismatch")
    if _manifest_sha256(observed) != identity:
        raise EvaluationBundleError("bundle_manifest_hash_mismatch")
    for relative_text, expected_digest in expected_manifest["files"].items():
        relative = _safe_relative(relative_text, "bundle_manifest_file")
        path = _assert_no_symlink_path(bundle_root, relative, "bundle_file")
        if not path.is_file():
            raise EvaluationBundleError(f"bundle_file_missing:{relative_text}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise EvaluationBundleError(
                f"bundle_file_hash_mismatch:{relative_text}")
    entrypoint = _assert_no_symlink_path(
        bundle_root, Path(expected_manifest["entrypoint"]), "bundle_entrypoint")
    return PreparedEvaluationBundle(
        root=bundle_root,
        entrypoint=entrypoint,
        manifest_path=manifest_path,
        sha256=identity,
    )


def stage_evaluation_bundle(
    idea_dir: Path, cfg: Mapping,
) -> PreparedEvaluationBundle:
    """Create or reuse a verified content-addressed evaluation bundle."""
    expected_manifest = _manifest_for(cfg)
    identity = _manifest_sha256(expected_manifest)
    project_root = Path(cfg.get("_project_root") or ".").resolve()
    idea_dir = Path(idea_dir).resolve()
    bundle_parent = idea_dir / BUNDLE_DIR
    if idea_dir.is_symlink() or bundle_parent.is_symlink():
        raise EvaluationBundleError("bundle_directory_symlink_forbidden")
    bundle_parent.mkdir(parents=True, exist_ok=True)
    final_root = bundle_parent / identity
    if final_root.exists():
        return verify_evaluation_bundle(idea_dir, cfg)

    temporary = bundle_parent / f".{identity}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700)
        for relative_text, expected_digest in expected_manifest["files"].items():
            relative = _safe_relative(relative_text, "evaluation_bundle.files")
            source = _assert_no_symlink_path(
                project_root, relative, "bundle_source")
            _copy_verified(source, temporary / relative, expected_digest)
        manifest_path = temporary / BUNDLE_MANIFEST
        manifest_path.write_bytes(_canonical_manifest(expected_manifest))
        manifest_path.chmod(0o444)
        for directory in sorted(
                (path for path in temporary.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts), reverse=True):
            directory.chmod(0o555)
        temporary.chmod(0o555)
        try:
            os.rename(temporary, final_root)
        except OSError:
            if not final_root.is_dir():
                raise
            temporary.chmod(0o700)
            shutil.rmtree(temporary)
    except Exception:
        if temporary.exists():
            for path in temporary.rglob("*"):
                try:
                    path.chmod(0o700 if path.is_dir() else 0o600)
                except OSError:
                    pass
            try:
                temporary.chmod(0o700)
                shutil.rmtree(temporary)
            except OSError:
                pass
        raise
    return verify_evaluation_bundle(idea_dir, cfg)
