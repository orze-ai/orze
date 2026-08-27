"""Host-local, process-safe leases for physical GPUs.

Orze's in-process slot manager coordinates jobs owned by one controller.  This
module closes the separate-controller gap with kernel ``flock`` leases.  Lease
descriptors are deliberately passed to GPU children: if a controller crashes
or detaches a child, the kernel keeps the GPU leased until that child exits.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


class GpuLeaseError(RuntimeError):
    """Raised when exclusive ownership of a physical GPU cannot be proven."""


_registry_lock = threading.RLock()
_registered: dict[int, "GpuLease"] = {}


def _validate_gpu(gpu: int) -> int:
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise GpuLeaseError("gpu_lease_invalid_physical_gpu")
    return gpu


def _lease_dir() -> Path:
    """Return a deterministic, private runtime directory for this uid."""
    uid = os.getuid()
    run_user = Path("/run/user") / str(uid)
    try:
        st = run_user.lstat()
        if (stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode)
                and st.st_uid == uid):
            parent = run_user
        else:
            parent = Path("/tmp")
    except OSError:
        parent = Path("/tmp")

    path = parent / f"orze-gpu-leases-{uid}"
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise GpuLeaseError("gpu_lease_directory_unavailable") from exc

    try:
        st = path.lstat()
    except OSError as exc:
        raise GpuLeaseError("gpu_lease_directory_unavailable") from exc
    if (not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode)
            or st.st_uid != uid or stat.S_IMODE(st.st_mode) != 0o700):
        raise GpuLeaseError("gpu_lease_directory_integrity_failed")
    return path


@dataclass
class GpuLease:
    gpu: int
    fd: int
    path: Path
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Do not issue LOCK_UN: inherited descriptors share the same open-file
        # description, so an explicit unlock in the parent would also drop a
        # detached child's protection.  The kernel releases the flock when
        # the final inherited descriptor closes.
        os.close(self.fd)


def _acquire_one(gpu: int) -> GpuLease:
    gpu = _validate_gpu(gpu)
    path = _lease_dir() / f"gpu-{gpu}.lock"
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise GpuLeaseError(
            f"gpu_lease_file_unavailable: physical_gpu={gpu}") from exc
    try:
        st = os.fstat(fd)
        if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid()
                or st.st_nlink != 1
                or stat.S_IMODE(st.st_mode) & 0o077):
            raise GpuLeaseError(
                f"gpu_lease_file_integrity_failed: physical_gpu={gpu}")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GpuLeaseError(
                f"gpu_lease_contended: physical_gpu={gpu}") from exc
        metadata = json.dumps({
            "physical_gpu": gpu,
            "pid": os.getpid(),
        }, sort_keys=True).encode("utf-8") + b"\n"
        os.ftruncate(fd, 0)
        os.write(fd, metadata)
        os.fsync(fd)
        return GpuLease(gpu=gpu, fd=fd, path=path)
    except Exception:
        os.close(fd)
        raise


class GpuLeaseSet:
    """An all-or-nothing lease over a controller's physical GPU scope."""

    def __init__(self, leases: Sequence[GpuLease]):
        self._leases = list(leases)
        self._closed = False

    @property
    def gpus(self) -> tuple[int, ...]:
        return tuple(lease.gpu for lease in self._leases)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with _registry_lock:
            for lease in self._leases:
                if _registered.get(lease.gpu) is lease:
                    _registered.pop(lease.gpu, None)
            for lease in reversed(self._leases):
                lease.close()


def acquire_gpu_leases(gpu_ids: Sequence[int]) -> GpuLeaseSet:
    """Acquire and register every requested GPU, or acquire none."""
    if (isinstance(gpu_ids, (str, bytes))
            or len(gpu_ids) != len(set(gpu_ids))):
        raise GpuLeaseError("gpu_lease_scope_invalid")
    ordered = sorted(_validate_gpu(gpu) for gpu in gpu_ids)
    acquired: list[GpuLease] = []
    with _registry_lock:
        overlap = [gpu for gpu in ordered if gpu in _registered]
        if overlap:
            raise GpuLeaseError(
                f"gpu_lease_already_registered: physical_gpu={overlap[0]}")
        try:
            for gpu in ordered:
                acquired.append(_acquire_one(gpu))
            for lease in acquired:
                _registered[lease.gpu] = lease
        except Exception:
            for lease in reversed(acquired):
                lease.close()
            raise
    return GpuLeaseSet(acquired)


@contextlib.contextmanager
def gpu_execution_lease(gpu: int | None) -> Iterator[tuple[int, ...]]:
    """Yield lease FDs for one GPU-visible child launch.

    A controller lease is borrowed when present.  Direct launcher calls lazily
    acquire a temporary parent lease; the child receives the descriptor via
    ``pass_fds`` and therefore keeps ownership after the parent closes it.
    """
    if gpu is None or gpu < 0:
        yield ()
        return
    gpu = _validate_gpu(gpu)
    temporary: GpuLeaseSet | None = None
    try:
        with _registry_lock:
            lease = _registered.get(gpu)
            if lease is not None:
                inherited_fd = os.dup(lease.fd)
            else:
                temporary = acquire_gpu_leases([gpu])
                lease = _registered[gpu]
                inherited_fd = os.dup(lease.fd)
    except Exception:
        if temporary is not None:
            temporary.close()
        raise
    try:
        yield (inherited_fd,)
    finally:
        os.close(inherited_fd)
        if temporary is not None:
            temporary.close()
