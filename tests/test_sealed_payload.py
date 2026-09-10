"""New Linux sealed-input mechanisms, not missing-API or old-behavior reds.

Real memfds exercise kernel seals, independent offsets and the actual blocked
supervisor/worker handoff. Separately labelled fault tests substitute only
specific syscalls; they do not claim process-tree or kernel proof.
"""
import errno
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import subprocess
import sys

import pytest

from orze.engine import sealed_payload as api
from orze.engine.supervised_process import prepare_supervised


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "memfd_create"),
    reason="Linux memfd contract")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def test_real_seals_reject_write_resize_and_shared_writable_mapping():
    raw = b"bounded immutable input\x00\xff"
    with api.sealed_payload(raw) as fd:
        assert not os.get_inheritable(fd)
        expected = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW
                    | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        assert fcntl.fcntl(fd, fcntl.F_GET_SEALS) & expected == expected
        for mutate in (lambda: os.pwrite(fd, b"X", 0),
                       lambda: os.ftruncate(fd, len(raw) + 1),
                       lambda: os.ftruncate(fd, len(raw) - 1),
                       lambda: mmap.mmap(fd, len(raw), access=mmap.ACCESS_WRITE)):
            with pytest.raises(OSError) as error:
                mutate()
            assert error.value.errno == errno.EPERM
        assert api.read_sealed_payload(fd, digest(raw)) == raw
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize("size", [1, 65536])
def test_real_reader_preserves_offset_and_caller_ownership_at_both_bounds(size):
    raw = (b"\x00\xffabcd" * (size // 6 + 1))[:size]
    with api.sealed_payload(raw) as fd:
        os.lseek(fd, size // 2, os.SEEK_SET)
        assert api.read_sealed_payload(fd, digest(raw)) == raw
        assert os.lseek(fd, 0, os.SEEK_CUR) == size // 2
        with pytest.raises(api.SealedPayloadError, match="digest_mismatch"):
            api.read_sealed_payload(fd, "0" * 64)
        assert os.fstat(fd).st_size == size
        assert os.lseek(fd, 0, os.SEEK_CUR) == size // 2


def test_actual_blocked_worker_reads_after_parent_releases_its_payload_fd(tmp_path):
    raw = "sealed transport 空白\n".encode() + b"\x00\xff"
    script = """
import os, sys
from orze.engine.sealed_payload import read_sealed_payload
fd = int(sys.argv[1])
raw = read_sealed_payload(fd, sys.argv[2])
os.close(fd)
from pathlib import Path
Path('received.bin').write_bytes(raw)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    process = None
    try:
        with api.sealed_payload(raw) as fd:
            command = [sys.executable, "-c", script, str(fd), digest(raw)]
            process = prepare_supervised(
                command, identity={"attempt_ref": {
                    "task_id": "idea-payload", "phase": "posthoc",
                    "attempt_id": "payload-roundtrip", "generation": 1},
                    "scope": str(tmp_path)},
                env=environment, cwd=tmp_path, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, worker_only_fds=(fd,))
            assert process.poll() is None
            assert not (tmp_path / "received.bin").exists()
            assert process.binding["command_sha256"] == digest(
                json.dumps(command, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode())
        with pytest.raises(OSError):
            os.fstat(fd)
        process.start()
        assert process.wait(timeout=5) == 0
        assert (tmp_path / "received.bin").read_bytes() == raw
        closure = process.closure_receipt()
        assert closure["worker_returncode"] == 0
        assert closure["stop_requested"] is False
        assert closure["wait_proof"] == "ECHILD_WALL"
    finally:
        if process is not None:
            process.stop(timeout=3)


class BytesSubclass(bytes):
    pass


@pytest.mark.parametrize("raw", [None, "text", bytearray(b"x"), BytesSubclass(b"x"),
                                  b"", b"x" * 65537])
def test_invalid_bytes_reject_before_creating_any_descriptor(monkeypatch, raw):
    monkeypatch.setattr(api.os, "memfd_create", lambda *a: pytest.fail("must not create"))
    with pytest.raises(api.SealedPayloadError):
        with api.sealed_payload(raw):
            pytest.fail("invalid input must not yield")


@pytest.mark.parametrize("fd,sha", [(True, "0" * 64), (2, "0" * 64),
                                    (3, "A" * 64), (3, None)])
def test_invalid_reader_arguments_reject_without_io_or_close(monkeypatch, fd, sha):
    monkeypatch.setattr(api.os, "fstat", lambda *a: pytest.fail("must not inspect"))
    monkeypatch.setattr(api.os, "close", lambda *a: pytest.fail("reader cannot own fd"))
    with pytest.raises(api.SealedPayloadError):
        api.read_sealed_payload(fd, sha)


@pytest.mark.parametrize("kind", ["unsealed", "write_only_seal", "pipe"])
def test_reader_refuses_unsealed_or_nonregular_descriptor_without_closing(kind):
    extra = None
    if kind == "pipe":
        fd, extra = os.pipe()
    else:
        fd = os.memfd_create("test-incomplete-seals", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        os.write(fd, b"x")
        if kind == "write_only_seal":
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE)
    try:
        with pytest.raises(api.SealedPayloadError):
            api.read_sealed_payload(fd, digest(b"x"))
        os.fstat(fd)
    finally:
        os.close(fd)
        if extra is not None:
            os.close(extra)


@pytest.mark.parametrize("fault", ["zero_write", "partial_then_error", "seal", "readback"])
def test_explicit_preparation_io_fault_never_yields_and_closes_once(monkeypatch, fault):
    """Real descriptor with selected syscall faults; no process is launched."""
    real_create, real_write, real_close = os.memfd_create, os.write, os.close
    real_fcntl, real_pread = fcntl.fcntl, os.pread
    created, closes, writes = [], [], []

    def create(*args):
        fd = real_create(*args)
        created.append(fd)
        return fd

    def write(fd, body):
        if fd in created and fault in {"zero_write", "partial_then_error"}:
            writes.append(True)
            if fault == "zero_write":
                return 0
            if len(writes) == 1:
                return real_write(fd, body[:2])
            raise OSError(errno.EIO, "injected partial write")
        return real_write(fd, body)

    def seal(fd, operation, *args):
        if fd in created and fault == "seal" and operation == fcntl.F_ADD_SEALS:
            raise OSError(errno.EPERM, "injected seal failure")
        return real_fcntl(fd, operation, *args)

    def pread(fd, size, offset):
        if fd in created and fault == "readback":
            return b""
        return real_pread(fd, size, offset)

    def close(fd):
        if fd in created:
            closes.append(fd)
        return real_close(fd)

    monkeypatch.setattr(api.os, "memfd_create", create)
    monkeypatch.setattr(api.os, "write", write)
    monkeypatch.setattr(api.fcntl, "fcntl", seal)
    monkeypatch.setattr(api.os, "pread", pread)
    monkeypatch.setattr(api.os, "close", close)
    with pytest.raises(api.SealedPayloadError):
        with api.sealed_payload(b"actual bytes"):
            pytest.fail("uncertain preparation cannot reach launch")
    assert len(created) == 1 and closes == created
    if fault == "partial_then_error":
        assert len(writes) == 2
    with pytest.raises(OSError):
        os.fstat(created[0])


def test_uncertain_close_is_not_retried_on_reused_descriptor(monkeypatch):
    real_close = os.close
    owned, replacement, closes = [], [], []

    def close(fd):
        real_close(fd)
        if fd in owned:
            closes.append(fd)
            replacement.append(os.open(os.devnull, os.O_RDONLY))
            assert replacement[-1] == fd, "exercise real descriptor-number reuse"
            raise OSError(errno.EIO, "injected post-close uncertainty")

    monkeypatch.setattr(api.os, "close", close)
    try:
        with pytest.raises(api.SealedPayloadError, match="close_failed"):
            with api.sealed_payload(b"owned") as fd:
                owned.append(fd)
        assert closes == owned
        os.fstat(replacement[0])
    finally:
        for fd in replacement:
            real_close(fd)


def test_unsupported_creation_never_substitutes_another_transport(monkeypatch):
    monkeypatch.delattr(api.os, "memfd_create")
    with pytest.raises(api.SealedPayloadError, match="unsupported"):
        with api.sealed_payload(b"payload"):
            pytest.fail("unsupported host must not yield")
