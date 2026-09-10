"""Bounded, stable trigger-file snapshots; this adapter never unlinks files.

CALLING SPEC: read_trigger(path) -> snapshot dict | None (missing only).
One file descriptor supplies bytes and before/after metadata; final pathname
identity must still match. This is not a lossless multi-message file producer
protocol or protection against hostile concurrent filesystem replacement.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from orze.engine.trigger_delivery_storage import TriggerDeliveryError

MAX_PAYLOAD_BYTES = 64 * 1024


def _identity(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink,
            metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)


def read_trigger(trigger_file):
    path = Path(trigger_file).absolute()
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_PAYLOAD_BYTES:
            raise TriggerDeliveryError("trigger_file_invalid")
        data = bytearray()
        while len(data) <= MAX_PAYLOAD_BYTES:
            part = os.read(fd, MAX_PAYLOAD_BYTES + 1 - len(data))
            if not part:
                break
            data.extend(part)
        after = os.fstat(fd)
        current = path.lstat()
        if (len(data) > MAX_PAYLOAD_BYTES or len(data) != after.st_size
                or _identity(before) != _identity(after)
                or _identity(after) != _identity(current)):
            raise TriggerDeliveryError("trigger_file_changed")
        try:
            payload = bytes(data).decode("utf-8")
        except UnicodeDecodeError:
            raise TriggerDeliveryError("trigger_file_encoding_invalid") from None
        payload_hash = hashlib.sha256(data).hexdigest()
        resolved = str(path.resolve(strict=True))
        # Legacy compatibility deliberately keeps exactly the old fingerprint.
        fingerprint = f"ino={after.st_ino}:size={after.st_size}:mtime_ns={after.st_mtime_ns}"
        key = hashlib.sha256(json.dumps(
            [resolved, *_identity(after), payload_hash], separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        return {"payload": payload, "payload_sha256": payload_hash,
                "file_path": resolved, "fingerprint": fingerprint, "source_key": key}
    finally:
        os.close(fd)
