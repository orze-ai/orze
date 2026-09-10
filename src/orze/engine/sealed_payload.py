"""Bounded, immutable Linux launch-input bytes; no execution authority.

CALLING SPEC: ``with sealed_payload(raw) as fd`` owns one anonymous memfd
until context exit. ``raw`` must be exact bytes, 1..65536 bytes. Before yield,
all bytes are written and read back, and WRITE/GROW/SHRINK/SEAL are verified.
The descriptor is CLOEXEC by default; callers explicitly pass it through
``prepare_supervised(worker_only_fds=(fd,))`` and retain the context until
preparation has forked its worker. No payload is placed in argv/environment.

``read_sealed_payload(fd, expected_sha256)`` borrows, never closes, a descriptor
and uses pread without changing its shared offset. The descriptor must remain
owned/stable during the call; concurrent caller close/reassignment is outside
this contract. The result is verified bytes, not a claim or attempt capability.
A fixed launch driver must close its inherited descriptor before importing
untrusted adapters. There is no pipe, disk-file, or unsupported-platform
fallback. Close is attempted once even after an uncertain close error; no
reused descriptor may be closed by a retry. No crash-recovery proof is added.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
import re
import stat
import sys
from typing import Iterator

try:
    import fcntl
except ImportError:  # Allow import on unsupported systems, never a fallback.
    fcntl = None

MAX_PAYLOAD_BYTES = 65536
_HEX = re.compile(r"[0-9a-f]{64}")


class SealedPayloadError(ValueError):
    """Payload transport failed closed; messages contain no payload bytes."""


def _features():
    if sys.platform != "linux" or fcntl is None:
        raise SealedPayloadError("sealed_payload_unsupported")
    try:
        create = os.memfd_create
        flags = os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
        seals = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW
                 | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        add, get = fcntl.F_ADD_SEALS, fcntl.F_GET_SEALS
    except AttributeError:
        raise SealedPayloadError("sealed_payload_unsupported") from None
    return create, flags, seals, add, get


def _metadata(fd):
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise SealedPayloadError("sealed_payload_not_regular")
    if not 1 <= info.st_size <= MAX_PAYLOAD_BYTES:
        raise SealedPayloadError("sealed_payload_size_invalid")
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _require_seals(fd, expected, get):
    actual = fcntl.fcntl(fd, get)
    if type(actual) is not int or actual & expected != expected:
        raise SealedPayloadError("sealed_payload_seals_missing")


def read_sealed_payload(fd: int, expected_sha256: str) -> bytes:
    """Return bounded verified bytes without owning fd or moving its offset."""
    if type(fd) is not int or fd < 3:
        raise SealedPayloadError("sealed_payload_descriptor_invalid")
    if type(expected_sha256) is not str or _HEX.fullmatch(expected_sha256) is None:
        raise SealedPayloadError("sealed_payload_digest_invalid")
    _, _, seals, _, get = _features()
    try:
        before = _metadata(fd)
        _require_seals(fd, seals, get)
        size, offset, chunks = before[3], 0, []
        while offset < size:
            chunk = os.pread(fd, min(16384, size - offset), offset)
            if type(chunk) is not bytes or not chunk or len(chunk) > size - offset:
                raise SealedPayloadError("sealed_payload_read_incomplete")
            chunks.append(chunk)
            offset += len(chunk)
        if os.pread(fd, 1, size) != b"":
            raise SealedPayloadError("sealed_payload_size_changed")
        _require_seals(fd, seals, get)
        if _metadata(fd) != before:
            raise SealedPayloadError("sealed_payload_descriptor_changed")
        raw = b"".join(chunks)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise SealedPayloadError("sealed_payload_digest_mismatch")
        return raw
    except (OSError, OverflowError):
        raise SealedPayloadError("sealed_payload_read_failed") from None


@contextmanager
def sealed_payload(raw: bytes) -> Iterator[int]:
    """Yield one fully sealed/read-back memfd; close its ownership exactly once."""
    if type(raw) is not bytes:
        raise SealedPayloadError("sealed_payload_bytes_invalid")
    if not 1 <= len(raw) <= MAX_PAYLOAD_BYTES:
        raise SealedPayloadError("sealed_payload_size_invalid")
    create, flags, seals, add, get = _features()
    try:
        fd = create("orze-launch-input", flags)
    except OSError:
        raise SealedPayloadError("sealed_payload_create_failed") from None
    try:
        if type(fd) is not int or fd < 3:
            raise SealedPayloadError("sealed_payload_descriptor_invalid")
        offset, view = 0, memoryview(raw)
        while offset < len(raw):
            try:
                written = os.write(fd, view[offset:])
            except OSError:
                raise SealedPayloadError("sealed_payload_write_failed") from None
            if type(written) is not int or not 1 <= written <= len(raw) - offset:
                raise SealedPayloadError("sealed_payload_write_failed")
            offset += written
        try:
            fcntl.fcntl(fd, add, seals)
            _require_seals(fd, seals, get)
        except OSError:
            raise SealedPayloadError("sealed_payload_seal_failed") from None
        if read_sealed_payload(fd, hashlib.sha256(raw).hexdigest()) != raw:
            raise SealedPayloadError("sealed_payload_readback_failed")
        yield fd
    finally:
        try:
            os.close(fd)
        except OSError:
            raise SealedPayloadError("sealed_payload_close_failed") from None
