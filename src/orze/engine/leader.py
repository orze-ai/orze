"""Filesystem-based leader election for multi-host orze deployments.

Why this exists
---------------
When two orze daemons on different hosts share the same ``results/``
directory (common on FSx / NFS), both will run every LLM-role cycle
and burn the same API quota twice. ``idea_lake.db`` is already shared,
so experiment execution and metric harvesting can safely run on both
nodes (they claim distinct GPUs / ideas). But the LLM-backed roles
(professor, research, thinker, engineer, bug_fixer, data_analyst) are
the ones that cost real money and tokens, so they should run on a
single "leader" at a time.

Design
------
A non-blocking ``fcntl.flock(LOCK_EX|LOCK_NB)`` on
``<results>/.orze_leader.lock`` is the source of truth. The call is
POSIX-local to the host but works on FSx/NFSv4 via fcntl advisory
locks. The **holder** of the fd is the leader; losing the fd (process
death, unmount) releases the lock automatically — no heartbeat-based
reclamation is required for the common case.

A sidecar heartbeat file ``<results>/.orze_leader.heartbeat`` (JSON
with ``{host, pid, ts}``) is written every ``HEARTBEAT_INTERVAL_S``
seconds, and a human-readable ``.orze_leader.host`` file is updated
with ``<hostname>:<pid>``. Followers read the heartbeat on startup
and every cycle so they can log *which* host holds leadership.

Takeover of a stale lease is intentionally conservative: ``flock``
releases on process exit automatically, so a truly dead leader will
free the lock on its own. The ``STALE_LEASE_S`` threshold is a
defensive fallback for pathological states (e.g., NFS flake where
the lock is still held by a dead node); on stale detection the
follower forcibly ``unlink`` the lockfile and retries the
``flock``. See ``try_acquire`` for the retry semantics.

Public contract
---------------
    LeaderHandle.try_acquire(results_dir) -> LeaderHandle | None
        Returns a handle if this process is the leader; ``None`` if
        another process already holds the lock. The handle's fd is
        owned by the caller for the process lifetime.

    LeaderHandle.heartbeat()
        Refresh the heartbeat file. Safe to call on every scheduler
        tick; writes only if ``HEARTBEAT_INTERVAL_S`` have elapsed.

    LeaderHandle.release()
        Flush heartbeat, unlink sidecar, close fd. Idempotent.

    read_current_leader(results_dir) -> dict | None
        Returns the last-seen leader's heartbeat record (for follower
        logging), or ``None`` if no heartbeat file or it's unreadable.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orze")

LOCK_NAME = ".orze_leader.lock"
HEARTBEAT_NAME = ".orze_leader.heartbeat"
HOST_NAME = ".orze_leader.host"

HEARTBEAT_INTERVAL_S = 30.0
STALE_LEASE_S = 300.0  # 5 minutes


@dataclass
class LeaderHandle:
    """Holds the locked fd + sidecar paths. Treat as opaque."""
    fd: int
    lock_path: Path
    heartbeat_path: Path
    host_path: Path
    host: str
    pid: int
    _last_heartbeat: float = 0.0
    _released: bool = False

    def heartbeat(self, force: bool = False) -> None:
        if self._released:
            return
        now = time.time()
        if not force and (now - self._last_heartbeat) < HEARTBEAT_INTERVAL_S:
            return
        payload = json.dumps({
            "host": self.host, "pid": self.pid, "ts": now,
        })
        try:
            self.heartbeat_path.write_text(payload, encoding="utf-8")
            self.host_path.write_text(f"{self.host}:{self.pid}\n",
                                      encoding="utf-8")
        except OSError as exc:
            logger.warning("leader heartbeat write failed: %s", exc)
            return
        self._last_heartbeat = now

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass
        for p in (self.heartbeat_path, self.host_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


def _open_lockfile(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)


def _heartbeat_is_stale(heartbeat_path: Path) -> bool:
    """True if the heartbeat file's recorded timestamp is older than
    STALE_LEASE_S seconds (or unreadable)."""
    try:
        text = heartbeat_path.read_text(encoding="utf-8")
        data = json.loads(text)
        ts = float(data.get("ts", 0))
    except (OSError, ValueError, TypeError):
        return True
    return (time.time() - ts) > STALE_LEASE_S


def read_current_leader(results_dir: Path | str) -> Optional[dict]:
    """Return the last-seen leader's heartbeat record, or None."""
    hb = Path(results_dir) / HEARTBEAT_NAME
    try:
        return json.loads(hb.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def try_acquire(results_dir: Path | str,
                host: Optional[str] = None,
                pid: Optional[int] = None) -> Optional[LeaderHandle]:
    """Try to become leader. Returns LeaderHandle on success, None otherwise.

    If the lockfile is held but the recorded heartbeat is older than
    ``STALE_LEASE_S``, we unlink the lockfile and retry once — this
    recovers from a ghost lock after a crashed node on networked FS.
    """
    results_dir = Path(results_dir)
    lock_path = results_dir / LOCK_NAME
    heartbeat_path = results_dir / HEARTBEAT_NAME
    host_path = results_dir / HOST_NAME

    def _attempt() -> Optional[LeaderHandle]:
        fd = _open_lockfile(lock_path)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return None
        handle = LeaderHandle(
            fd=fd,
            lock_path=lock_path,
            heartbeat_path=heartbeat_path,
            host_path=host_path,
            host=host or socket.gethostname(),
            pid=pid or os.getpid(),
        )
        handle.heartbeat(force=True)
        return handle

    first = _attempt()
    if first is not None:
        return first

    # Primary is held. Only attempt takeover if the heartbeat is stale.
    if _heartbeat_is_stale(heartbeat_path):
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            return None
        second = _attempt()
        if second is not None:
            logger.warning(
                "leader: took over stale lease (last heartbeat > %ds)",
                int(STALE_LEASE_S))
            return second
    return None


# Role names that should NOT run on a follower node. These are the
# LLM-backed research/steering roles that burn API tokens. Experiment
# launching and metric harvesting continue on followers.
LLM_ROLE_NAMES = frozenset({
    "professor", "research", "research_gemini", "research_gpt",
    "research_claude", "research_anthropic", "research_local",
    "thinker", "engineer", "bug_fixer", "data_analyst",
    "documenter", "code_evolution",
})


def should_skip_role_as_follower(role_name: str) -> bool:
    """Return True if this role should be gated behind leader election."""
    return role_name in LLM_ROLE_NAMES
