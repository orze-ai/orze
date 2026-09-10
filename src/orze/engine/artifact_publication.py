"""Prepare declared, independent artifact inodes outside the terminal writer.

CALLING SPEC:
    prepare_artifacts(ref, idea_dir, binding) -> PreparedArtifacts
        Copies and hashes bounded declared regular files with no SQLite/effect
        lock held. New occurrence directories are create-only. Failed staging
        can remain, but is not an accepted artifact without a committed row.
    verify_prepared_artifacts(prepared, ref, idea_dir, binding) -> tuple[dict, ...]
        Cheap identity checks only, for the current-attempt terminal writer.
        The caller registers these records in the same SQL transaction as the
        terminal. This module does not grant attempt or scientific authority.

Snapshots use new inodes, never links/renames of worker files. Read-only mode
is a cooperative-writer boundary, not a sandbox against a hostile same UID.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

from orze.core.execution_attempts import AttemptRef
from orze.engine.attempt_effect_lock import AttemptEffectBusy

_CHUNK_BYTES = 1024 * 1024


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _identity(st, directory=False):
    if directory:
        return st.st_dev, st.st_ino, st.st_mode
    return (st.st_dev, st.st_ino, st.st_mode, st.st_nlink, st.st_size,
            st.st_mtime_ns, st.st_ctime_ns)


def _open_directory(path, *, create=False):
    """Traverse without following redirects, transferring each FD once."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise AttemptEffectBusy("artifact_directory_invalid")
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                else:
                    os.fsync(fd)
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=fd)
            old, fd = fd, child
            os.close(old)
        result, fd = fd, None
        return result
    finally:
        if fd is not None:
            os.close(fd)


def _path_identities(path):
    """Capture component identity without hashing content or directory mtime."""
    path = Path(path)
    result = []
    for parent in reversed(path.parents):
        st = parent.lstat()
        if not stat.S_ISDIR(st.st_mode):
            raise AttemptEffectBusy("artifact_parent_redirected")
        result.append((str(parent), True, _identity(st, True)))
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise AttemptEffectBusy("artifact_file_invalid")
    result.append((str(path), False, _identity(st)))
    return tuple(result)


def _verify_identities(identities):
    for name, directory, expected in identities:
        st = Path(name).lstat()
        if _identity(st, directory) != expected:
            raise AttemptEffectBusy("artifact_file_changed")


def _binding(binding, ref, idea_dir):
    from orze.core.artifact_contract import validate_artifact_publication_binding
    normalized = validate_artifact_publication_binding(binding)
    folder = Path(idea_dir).absolute()
    if (not isinstance(ref, AttemptRef) or ref.phase != "training"
            or ref.task_id != folder.name or normalized["scope"] != str(folder.parent)):
        raise AttemptEffectBusy("artifact_producer_scope_invalid")
    return normalized, folder


def _snapshot_hash(destination, maximum):
    """Read back bounded snapshot bytes outside the lifecycle transaction."""
    identities = _path_identities(destination)
    parent_fd = _open_directory(destination.parent)
    fd = None
    try:
        fd = os.open(destination.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=parent_fd)
        before = os.fstat(fd)
        if _identity(before) != identities[-1][2] or before.st_size > maximum:
            raise AttemptEffectBusy("artifact_readback_invalid")
        digest, size = hashlib.sha256(), 0
        while True:
            chunk = os.read(fd, min(_CHUNK_BYTES, maximum - size + 1))
            if not chunk:
                break
            size += len(chunk)
            if size > maximum:
                raise AttemptEffectBusy("artifact_readback_size_invalid")
            digest.update(chunk)
        if size != before.st_size or _identity(os.fstat(fd)) != _identity(before):
            raise AttemptEffectBusy("artifact_readback_changed")
        _verify_identities(identities)
        return digest.hexdigest()
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            os.close(parent_fd)


@dataclass(frozen=True)
class PreparedArtifacts:
    """Immutable metadata only; no mutable record dictionaries cross the seam."""
    producer_json: str
    idea_dir: str
    binding_json: str
    records_json: str
    records_sha256: str
    identities: tuple


def _copy_one(source, destination, maximum):
    source_fd = output_fd = source_parent = output_parent = None
    try:
        source_ids = _path_identities(source)
        source_parent = _open_directory(source.parent)
        source_fd = os.open(source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                            dir_fd=source_parent)
        before = os.fstat(source_fd)
        if (_identity(before) != source_ids[-1][2]
                or before.st_size > maximum):
            raise AttemptEffectBusy("artifact_source_invalid")
        output_parent = _open_directory(destination.parent)
        output_fd = os.open(destination.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                            | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600,
                            dir_fd=output_parent)
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(source_fd, min(_CHUNK_BYTES, maximum - total + 1))
            if not block:
                break
            total += len(block)
            if total > maximum:
                raise AttemptEffectBusy("artifact_size_limit")
            digest.update(block)
            remaining = memoryview(block)
            while remaining:
                written = os.write(output_fd, remaining)
                if type(written) is not int or written <= 0 or written > len(remaining):
                    raise OSError("artifact_short_write")
                remaining = remaining[written:]
        if total != before.st_size or _identity(os.fstat(source_fd)) != _identity(before):
            raise AttemptEffectBusy("artifact_source_changed")
        _verify_identities(source_ids)
        os.fchmod(output_fd, 0o400)
        os.fsync(output_fd)
        output_stat = os.fstat(output_fd)
        closing, output_fd = output_fd, None
        os.close(closing)
        closing, source_fd = source_fd, None
        os.close(closing)
        os.fsync(output_parent)
        output_ids = _path_identities(destination)
        if (_identity(output_stat) != output_ids[-1][2]
                or output_stat.st_size != total
                or (before.st_dev, before.st_ino) ==
                   (output_stat.st_dev, output_stat.st_ino)):
            raise AttemptEffectBusy("artifact_copy_identity_invalid")
        _verify_identities(source_ids)
        if _snapshot_hash(destination, maximum) != digest.hexdigest():
            raise AttemptEffectBusy("artifact_readback_digest_mismatch")
        return digest.hexdigest(), total, source_ids + output_ids
    finally:
        # Every descriptor has one close owner; uncertain close is not retried.
        pending = [fd for fd in (output_fd, source_fd, output_parent, source_parent)
                   if fd is not None]
        failure = None
        for fd in pending:
            try:
                os.close(fd)
            except OSError as exc:
                failure = failure or exc
        if failure is not None:
            raise failure


def prepare_artifacts(ref, idea_dir, binding):
    """Perform the expensive, unaccepted staging phase outside all writers."""
    try:
        normalized, folder = _binding(binding, ref, idea_dir)
        root = Path(normalized["root"])
        records, identities = [], []
        outputs = normalized["contract"]["outputs"]
        # An explicitly empty declaration is a valid zero-artifact attempt.
        root_fd = _open_directory(root, create=True) if outputs else None
        try:
            for logical_name, output in sorted(outputs.items()):
                artifact_id = hashlib.sha256(_canonical({
                    "producer": asdict(ref), "logical_name": logical_name,
                    "scope": normalized["scope"],
                }).encode("utf-8")).hexdigest()
                # One occurrence gets one create-only staging directory. An
                # unaccepted partial copy blocks retries instead of creating
                # another potentially large orphan on every monitor tick.
                try:
                    os.mkdir(artifact_id, 0o700, dir_fd=root_fd)
                except FileExistsError as exc:
                    raise AttemptEffectBusy("artifact_staging_requires_resolution") from exc
                os.fsync(root_fd)
                destination = root / artifact_id / "content"
                digest, size, captured = _copy_one(
                    folder / output["path"], destination, output["max_bytes"])
                identities.extend(captured)
                records.append({
                    "schema": 1, "artifact_id": artifact_id, "producer": asdict(ref),
                    "spec_fingerprint": normalized["spec_fingerprint"],
                    "scope": normalized["scope"], "logical_name": logical_name,
                    "path": str(destination), "content_sha256": digest,
                    "size_bytes": size,
                })
        finally:
            if root_fd is not None:
                os.close(root_fd)
        encoded_records = _canonical(records)
        result = PreparedArtifacts(_canonical(asdict(ref)), str(folder),
                                   _canonical(normalized), encoded_records,
                                   hashlib.sha256(encoded_records.encode("utf-8")).hexdigest(),
                                   tuple(identities))
        verify_prepared_artifacts(result, ref, folder, normalized)
        return result
    except (OSError, ValueError, TypeError) as exc:
        raise AttemptEffectBusy("artifact_snapshot_unavailable") from exc


def verify_prepared_artifacts(prepared, ref, idea_dir, binding):
    """Verify bounded metadata, never reread/hash large files in the SQL lock."""
    try:
        normalized, folder = _binding(binding, ref, idea_dir)
        if (not isinstance(prepared, PreparedArtifacts)
                or prepared.producer_json != _canonical(asdict(ref))
                or prepared.idea_dir != str(folder)
                or prepared.binding_json != _canonical(normalized)
                or hashlib.sha256(prepared.records_json.encode("utf-8")).hexdigest()
                   != prepared.records_sha256):
            raise AttemptEffectBusy("artifact_prepared_scope_changed")
        _verify_identities(prepared.identities)
        return tuple(json.loads(prepared.records_json))
    except (OSError, ValueError, TypeError) as exc:
        raise AttemptEffectBusy("artifact_prepared_unavailable") from exc
