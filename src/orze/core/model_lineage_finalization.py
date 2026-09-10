"""Read-only lineage preparation and bounded file-only publication.

The model_lineage facade owns legacy helpers and exception types. Resolve it
lazily to preserve its existing instrumentation seams without import cycles.
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _lineage():
    from orze.core import model_lineage
    return model_lineage


@dataclass(frozen=True)
class PreparedModelLineageFinalization:
    """Process-local immutable preparation, not standalone attempt authority."""

    payload_json: str
    payload_sha256: str
    idea_dir: str
    subject: tuple
    policy_sha256: str
    files: tuple
    parents: tuple


def _finalization_policy(cfg: Mapping) -> str:
    return _lineage()._canonical_hash({
        "model_lineage": cfg.get("model_lineage", {}),
        "data_boundaries": cfg.get("data_boundaries", {}),
        "data_separation": cfg.get("data_separation", {}),
        "_project_root": str(cfg.get("_project_root", ".")),
        "_orze_dir": str(cfg.get("_orze_dir", "")),
    })


def _finalization_subject(tp, idea_dir: Path) -> tuple:
    subject = (getattr(tp, "idea_id", None), str(getattr(tp, "attempt_id", "")),
               getattr(tp, "execution_identity", None))
    if (subject[0] != idea_dir.name or any(
            not isinstance(value, str) or _lineage()._TOKEN.fullmatch(value) is None
            or value in (".", "..") for value in subject[:2])
            or not isinstance(subject[2], str) or _lineage()._HEX64.fullmatch(subject[2]) is None):
        raise _lineage().ModelLineageError("model_lineage_publication_subject_invalid")
    return subject


def _finalization_paths(tp, idea_dir: Path, cfg: Mapping) -> tuple[Path, ...]:
    receipt_dir = _lineage()._receipt_dir(idea_dir, str(tp.attempt_id), create=False)
    state_root = Path(cfg.get("_orze_dir") or (
        Path(cfg.get("_project_root", ".")) / ".orze"))
    separation = cfg["data_separation"]
    if any(not isinstance(separation.get(key), str) or not separation[key]
           for key in ("train_manifest", "evaluation_manifest")):
        raise _lineage().ModelLineageError("model_lineage_manifest_path_invalid")
    return tuple(sorted({path.absolute() for path in (
        _lineage()._idea_path(idea_dir, cfg["model_lineage"]["artifact"]),
        receipt_dir / "boundary.json", receipt_dir / "start.json",
        state_root / "state" / "data_separation.json",
        Path(separation["train_manifest"]), Path(separation["evaluation_manifest"]),
    )}, key=str))


def _publication_bindings(paths: tuple[Path, ...]) -> tuple[tuple, tuple]:
    """Capture six files and bounded lexical parents, without reading bytes."""
    files, parents = [], {}
    try:
        for path in paths:
            if ".." in path.parts:
                raise _lineage().ModelLineageError("model_lineage_publication_path_invalid")
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise _lineage().ModelLineageError("model_lineage_publication_file_invalid")
            files.append((str(path), _lineage()._stat_identity(info) + (info.st_nlink,)))
            for parent in (path.parent, *path.parent.parents):
                key = str(parent)
                if key in parents:
                    continue
                if len(files) + len(parents) >= _lineage()._MAX_PUBLICATION_BINDINGS:
                    raise _lineage().ModelLineagePublicationUnsupported(
                        "model_lineage_publication_binding_limit")
                info = parent.lstat()
                if not stat.S_ISDIR(info.st_mode):
                    raise _lineage().ModelLineageError("model_lineage_publication_path_invalid")
                # A coordinator may legitimately create its effect/lock files
                # between phases. Bind directory identity, not entry timestamps.
                parents[key] = (info.st_dev, info.st_ino, info.st_mode)
        if len(files) + len(parents) > _lineage()._MAX_PUBLICATION_BINDINGS:
            raise _lineage().ModelLineagePublicationUnsupported("model_lineage_publication_binding_limit")
    except OSError as exc:
        raise _lineage().ModelLineageError("model_lineage_publication_input_unavailable") from exc
    return tuple(files), tuple(sorted(parents.items()))


def _prepared_payload(payload: dict, idea_dir: Path, subject: tuple,
                      policy: str, files=(), parents=()) -> PreparedModelLineageFinalization:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > _lineage()._MAX_FINALIZATION_PAYLOAD_BYTES:
        raise _lineage().ModelLineagePublicationUnsupported("model_lineage_publication_payload_limit")
    return PreparedModelLineageFinalization(
        encoded, _lineage()._canonical_hash(payload), str(idea_dir), subject, policy, files, parents)


def prepare_model_lineage_finalization(
    tp, idea_dir: Path, cfg: Mapping,
) -> PreparedModelLineageFinalization:
    """Read-only slow preparation for a single regular-file artifact.

    Reads an existing separation receipt; never creates directories or audits
    manifests. Hashes the model outside the caller's short effect lock. A
    directory artifact is unsupported here, because root metadata cannot prove
    its nested files stayed unchanged and a 100k-file scan is not a short lock.
    The legacy finalize wrapper retains its full directory-hashing behavior.
    """
    if not isinstance(cfg, Mapping) or _lineage().validate_model_lineage_config(cfg):
        raise _lineage().ModelLineageError("model_lineage_policy_invalid")
    idea_dir = Path(idea_dir).absolute()
    policy = _finalization_policy(cfg)
    if not cfg.get("model_lineage", {}).get("enabled", False):
        return _prepared_payload({"status": "disabled"}, idea_dir, (), policy)
    subject = _finalization_subject(tp, idea_dir)
    artifact = _lineage()._idea_path(idea_dir, cfg["model_lineage"]["artifact"])
    try:
        if stat.S_ISDIR(artifact.lstat().st_mode):
            raise _lineage().ModelLineagePublicationUnsupported(
                "model_lineage_short_publication_directory_unsupported")
    except OSError as exc:
        raise _lineage().ModelLineageError("model_lineage_artifact_missing") from exc
    paths = _lineage()._finalization_paths(tp, idea_dir, cfg)
    files, parents = _publication_bindings(paths)
    try:
        payload = _lineage()._model_lineage_finalization_payload(tp, idea_dir, cfg, readonly=True)
    except _lineage().DataSeparationError as exc:
        raise _lineage().ModelLineageError("model_lineage_data_separation_unavailable") from exc
    if (subject != _finalization_subject(tp, idea_dir) or policy != _finalization_policy(cfg)
            or (files, parents) != _publication_bindings(paths)):
        raise _lineage().ModelLineageError("model_lineage_preparation_changed")
    return _prepared_payload(payload, idea_dir, subject, policy, files, parents)


def publish_model_lineage_finalization(
    prepared: PreparedModelLineageFinalization, tp, idea_dir: Path, cfg: Mapping,
) -> dict:
    """Publish a compact envelope after current-attempt checks by the caller.

    Does not rehash model/manifest contents. Any uncertain post-write exception
    must make the caller HOLD its effect intent; no rollback/replay is claimed.
    """
    if (type(prepared) is not PreparedModelLineageFinalization
            or not isinstance(cfg, Mapping) or _lineage().validate_model_lineage_config(cfg)
            or type(prepared.payload_json) is not str
            or len(prepared.payload_json.encode("utf-8")) > _lineage()._MAX_FINALIZATION_PAYLOAD_BYTES):
        raise _lineage().ModelLineageError("model_lineage_preparation_invalid")
    idea_dir = Path(idea_dir).absolute()
    if str(idea_dir) != prepared.idea_dir or _finalization_policy(cfg) != prepared.policy_sha256:
        raise _lineage().ModelLineageError("model_lineage_publication_scope_changed")
    try:
        payload = json.loads(prepared.payload_json)
    except (ValueError, UnicodeError) as exc:
        raise _lineage().ModelLineageError("model_lineage_preparation_invalid") from exc
    if not isinstance(payload, dict) or _lineage()._canonical_hash(payload) != prepared.payload_sha256:
        raise _lineage().ModelLineageError("model_lineage_preparation_invalid")
    if not cfg.get("model_lineage", {}).get("enabled", False):
        if payload != {"status": "disabled"}:
            raise _lineage().ModelLineageError("model_lineage_preparation_invalid")
        return payload
    if set(payload) != _lineage()._LINEAGE_KEYS or payload.get("artifact_kind") != "file":
        raise _lineage().ModelLineageError("model_lineage_preparation_invalid")
    paths = _lineage()._finalization_paths(tp, idea_dir, cfg)

    def verify():
        if (prepared.subject != _finalization_subject(tp, idea_dir)
                or prepared.policy_sha256 != _finalization_policy(cfg)
                or (prepared.files, prepared.parents) != _publication_bindings(paths)):
            raise _lineage().ModelLineageError("model_lineage_publication_input_changed")

    verify()
    output = _lineage()._idea_path(idea_dir, _lineage().LINEAGE_FILE)
    _lineage()._write_envelope_once(output, payload)
    try:
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except OSError as exc:
        raise _lineage().ModelLineageError("model_lineage_publication_sync_failed") from exc
    actual, digest = _lineage()._read_envelope(output, _lineage()._LINEAGE_KEYS)
    if actual != payload or digest != prepared.payload_sha256:
        raise _lineage().ModelLineageError("model_lineage_publication_readback_failed")
    verify()
    return payload
