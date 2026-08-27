"""Process dataclasses and low-level process management utilities.

CALLING SPEC:
    TrainingProcess (dataclass)
        Fields: idea_id (str), gpu (int), process (Popen), start_time (float),
                log_path (Path), timeout (float)
        Private: _log_fh, _last_log_size, _last_log_check, _stall_since
        Methods: close_log() — closes the log file handle if open

    EvalProcess (dataclass)
        Fields: idea_id (str), gpu (int), process (Popen), start_time (float),
                log_path (Path), timeout (float)
        Private: _log_fh
        Methods: close_log() — closes the log file handle if open

    RoleProcess (dataclass)
        Fields: role_name (str), process (Popen), start_time (float),
                log_path (Path), timeout (float), lock_dir (Path), cycle_num (int),
                ideas_pre_size (int), ideas_pre_count (int)
        Private: _log_fh
        Methods: close_log() — closes the log file handle if open

    _kill_pg(proc, sig=SIGTERM) -> None
        proc: subprocess.Popen
        sig: signal number (default SIGTERM)
        side effects: sends signal to the entire process group; falls back to proc.send_signal

    _terminate_and_reap(proc, label="", timeout=10) -> None
        proc: subprocess.Popen
        label: str — for log messages
        timeout: float — seconds to wait after SIGTERM before SIGKILL
        side effects: SIGTERM -> wait -> SIGKILL surviving process-group
                      descendants; logs warnings on force kill

    _new_process_group() -> None
        preexec_fn for subprocess.Popen; calls os.setpgrp() to create a new process group

    run_pre_script(idea_id, gpu, cfg, results_dir=None) -> bool
        idea_id: str
        gpu: int — set as CUDA_VISIBLE_DEVICES
        cfg: dict — uses 'pre_script', 'pre_args', 'pre_timeout', 'python', 'train_extra_env'
        results_dir: optional result root for framework compute receipts
        returns: True if no pre_script configured or script exited 0, False on failure/timeout
        side effects: runs blocking subprocess

    run_artifact_preflight(idea_id, results_dir, cfg) -> bool
        returns: True if disabled or resolver exited 0, False otherwise
        side effects: runs a zero-GPU resolver and writes a hash-only receipt
"""
import datetime
import hashlib
import json
import os
import signal
import secrets
import subprocess
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

logger = logging.getLogger("orze")


def capture_process_identity(pid: int) -> dict:
    """Capture stable Linux process identity fields for crash recovery."""
    stat_text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    # The comm field is parenthesized and may contain spaces. Fields after
    # the final ')' start at process-state (field 3); starttime is field 22.
    rest = stat_text.rsplit(")", 1)[1].strip().split()
    return {
        "pid": int(pid),
        "pgid": os.getpgid(int(pid)),
        "start_ticks": int(rest[19]),
    }


def process_is_running(pid: int, start_ticks: Optional[int] = None) -> bool:
    """True only for the same, non-zombie process recorded at launch."""
    try:
        identity = capture_process_identity(int(pid))
        stat_text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        state = stat_text.rsplit(")", 1)[1].strip().split()[0]
    except (OSError, ValueError, ProcessLookupError):
        return False
    if start_ticks is not None and identity["start_ticks"] != int(start_ticks):
        return False
    return state != "Z"


def process_group_members(pgid: int) -> list:
    """Return non-zombie PIDs currently belonging to a Linux process group."""
    members = []
    try:
        proc_entries = Path("/proc").iterdir()
    except OSError:
        return members
    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
            rest = stat_text.rsplit(")", 1)[1].strip().split()
            state = rest[0]
            member_pgid = int(rest[2])
        except (OSError, ValueError, IndexError):
            continue
        if member_pgid == int(pgid) and state != "Z":
            members.append(int(entry.name))
    return sorted(members)


def process_descendant_identities(root_pid: int) -> list[dict]:
    """Snapshot every non-zombie descendant, including escaped groups.

    Process groups are not an ancestry boundary: a shell can call ``setsid``
    and survive a later ``killpg`` of its role parent. Capturing stable
    ``(pid, start_ticks)`` identities before termination lets the reaper kill
    those already-existing escapees without risking a reused PID.
    """
    rows: dict[int, dict] = {}
    try:
        entries = Path("/proc").iterdir()
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
            rest = stat_text.rsplit(")", 1)[1].strip().split()
            state = rest[0]
            pid = int(entry.name)
            rows[pid] = {
                "pid": pid,
                "ppid": int(rest[1]),
                "pgid": int(rest[2]),
                "start_ticks": int(rest[19]),
                "state": state,
            }
        except (OSError, ValueError, IndexError):
            continue

    descendants = []
    frontier = {int(root_pid)}
    while frontier:
        children = [row for row in rows.values()
                    if row["ppid"] in frontier and row["state"] != "Z"]
        if not children:
            break
        descendants.extend(children)
        frontier = {row["pid"] for row in children}
    return sorted(descendants, key=lambda row: row["pid"])


def process_tree_cpu_ticks(root_pid: int) -> Optional[int]:
    """Return cumulative live-tree CPU ticks, or ``None`` if unavailable.

    Linux ``/proc`` utime/stime plus waited-child cutime/cstime provide a
    content-free activity signal. Including the root and every current
    descendant catches useful silent computation without reading commands,
    prompts, output, or environment values.
    """
    if (isinstance(root_pid, bool)
            or not isinstance(root_pid, int) or root_pid < 1):
        return None
    rows: dict[int, dict] = {}
    try:
        entries = Path("/proc").iterdir()
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
            rest = stat_text.rsplit(")", 1)[1].strip().split()
            pid = int(entry.name)
            rows[pid] = {
                "ppid": int(rest[1]),
                "ticks": sum(int(rest[index]) for index in range(11, 15)),
            }
        except (OSError, ValueError, IndexError):
            continue
    if root_pid not in rows:
        return None
    members = {root_pid}
    frontier = {root_pid}
    while frontier:
        children = {
            pid for pid, row in rows.items()
            if row["ppid"] in frontier and pid not in members
        }
        members.update(children)
        frontier = children
    return sum(rows[pid]["ticks"] for pid in members)


def progress_paths_fingerprint(paths) -> Optional[str]:
    """Hash only lstat metadata for declared role outputs.

    No file contents are read or persisted. A path creation, replacement,
    size change, or mtime change is enough to count as artifact progress.
    """
    normalized = sorted({str(Path(path)) for path in (paths or ())})
    if not normalized:
        return None
    digest = hashlib.sha256()
    for value in normalized:
        digest.update(value.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        try:
            stat = Path(value).lstat()
        except OSError:
            digest.update(b"missing\0")
            continue
        digest.update(
            f"{stat.st_mode}:{stat.st_size}:{stat.st_mtime_ns}".encode("ascii")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _signal_tracked_processes(identities: list[dict], sig: int) -> None:
    """Signal exact process identities, refusing PID-reuse collisions."""
    for identity in reversed(identities):
        pid = int(identity["pid"])
        if not process_is_running(pid, identity.get("start_ticks")):
            continue
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def terminate_recorded_process_group(pid: int, pgid: int, start_ticks: int,
                                     idea_id: str, timeout: float = 3.0) -> bool:
    """Terminate a durably identified orphan process group.

    Refuses to signal on PID reuse, command mismatch, PGID mismatch, or if the
    recorded group is this orchestrator's group. Returns only after the
    recorded process is absent or a zombie (therefore no longer executing).
    """
    pid = int(pid)
    pgid = int(pgid)
    members = process_group_members(pgid)
    if not members:
        return True
    try:
        current = capture_process_identity(pid)
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except (OSError, ValueError, ProcessLookupError):
        return not process_group_members(pgid)
    expected = idea_id.encode("utf-8")
    if (current["start_ticks"] != int(start_ticks)
            or current["pgid"] != pgid or pgid == os.getpgrp()
            or expected not in cmdline or b"--idea-id" not in cmdline):
        logger.error(
            "Refusing orphan termination for %s: recorded pid/pgid identity no longer matches",
            idea_id,
        )
        return False

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_group_members(pgid):
            return True
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_group_members(pgid):
            return True
        time.sleep(0.05)
    remaining = process_group_members(pgid)
    if remaining:
        logger.error(
            "Orphan process group %d for %s still has live members after SIGKILL: %s",
            pgid, idea_id, remaining,
        )
    return not remaining


def _kill_pg(proc: subprocess.Popen, sig=signal.SIGTERM):
    """Send a signal to the entire process group of *proc*.
    lookup fails (e.g. process already dead).
    """
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


def _terminate_and_reap(proc: subprocess.Popen, label: str = "",
                        timeout: float = 10, pgid: Optional[int] = None,
                        tracked_identities: Optional[list[dict]] = None,
                        discover_pgid: bool = True):
    """Terminate a process and every descendant in its dedicated group."""
    escaped_descendants = list(tracked_identities or [])
    proc_pid = getattr(proc, "pid", None)
    if type(proc_pid) is int and discover_pgid:
        escaped_descendants.extend(process_descendant_identities(proc_pid))
    # Retain only one record per stable process identity. A role may have
    # observed the same child over many watchdog polls.
    escaped_descendants = list({
        (int(identity["pid"]), int(identity["start_ticks"])): identity
        for identity in escaped_descendants
        if isinstance(identity, dict)
        and type(identity.get("pid")) is int
        and type(identity.get("start_ticks")) is int
    }.values())
    if pgid is None and discover_pgid and type(proc_pid) is int:
        try:
            pgid = os.getpgid(proc_pid)
        except (ProcessLookupError, PermissionError, OSError):
            pgid = None
    if pgid == os.getpgrp():
        logger.error("Refusing to terminate our own process group for %s",
                     label or "process")
        pgid = None

    # Signal captured descendants first. A child that created a separate
    # process group/session is not reached by killpg(pgid); the exact identity
    # check prevents signaling a different process after PID reuse.
    _signal_tracked_processes(escaped_descendants, signal.SIGTERM)
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    else:
        try:
            proc.terminate()
        except (ProcessLookupError, OSError):
            pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Force killing %s (PID %d)", label or "process", proc.pid)
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        else:
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error("Failed to reap %s (PID %d) after SIGKILL",
                         label or "process", proc.pid)

    # The group leader can exit on SIGTERM while a descendant ignores it.
    # Waiting only for ``proc`` then leaks that descendant indefinitely.
    group_remaining = process_group_members(pgid) if pgid is not None else []
    if pgid is not None:
        if group_remaining:
            logger.warning("Force killing %d surviving descendant(s) of %s: %s",
                           len(group_remaining), label or "process",
                           group_remaining)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            deadline = time.time() + 5
            while time.time() < deadline and process_group_members(pgid):
                time.sleep(0.05)
            group_remaining = process_group_members(pgid)
            if group_remaining:
                logger.error("Failed to reap process group %d for %s: %s",
                             pgid, label or "process", group_remaining)

    escaped_remaining = [
        identity for identity in escaped_descendants
        if process_is_running(identity["pid"], identity.get("start_ticks"))
    ]
    if escaped_remaining:
        logger.warning(
            "Force killing %d escaped descendant(s) of %s: %s",
            len(escaped_remaining), label or "process",
            [identity["pid"] for identity in escaped_remaining],
        )
        _signal_tracked_processes(escaped_remaining, signal.SIGKILL)
        deadline = time.time() + 5
        while time.time() < deadline:
            escaped_remaining = [
                identity for identity in escaped_remaining
                if process_is_running(
                    identity["pid"], identity.get("start_ticks"))
            ]
            if not escaped_remaining:
                break
            time.sleep(0.05)
        if escaped_remaining:
            logger.error(
                "Failed to reap escaped descendant(s) of %s: %s",
                label or "process",
                [identity["pid"] for identity in escaped_remaining],
            )
    return not group_remaining and not escaped_remaining and proc.poll() is not None


_OFFLINE_ENV_KEYS = (
    "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE",
)


def _truthy_env(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _drain_and_hash(stream, digest) -> None:
    """Drain a subprocess pipe at constant memory while hashing its bytes."""
    try:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(block)
    finally:
        stream.close()


def _artifact_preflight_identity(idea_id: str, results_dir: Path,
                                 cfg: dict) -> dict:
    """Return current hash-only identity for an enabled resolver contract."""
    spec = cfg.get("artifact_preflight") or {}
    results_dir = Path(results_dir)
    idea_dir = results_dir / idea_id
    policy = str(spec.get("network", "inherit")).strip().lower()
    script_text = str(spec.get("script") or "")
    project_root = Path(cfg.get("_project_root", "."))
    script_path = Path(script_text)
    if not script_path.is_absolute():
        script_path = project_root / script_path
    config_path = idea_dir / "idea_config.yaml"
    if not config_path.exists():
        config_path = Path(cfg.get("base_config", "configs/base.yaml"))
        if not config_path.is_absolute():
            config_path = project_root / config_path
    extra_env_raw = cfg.get("train_extra_env") or {}
    extra_env = extra_env_raw if isinstance(extra_env_raw, dict) else {}
    contract = {
        "script": script_text,
        "args": [str(value) for value in (spec.get("args") or [])],
        "interpreter": str(spec.get(
            "interpreter", cfg.get("python", sys.executable)) or ""),
        "network": policy,
        # Bind environment shape and the artifact network switches without
        # hashing arbitrary secret values into a durable offline oracle.
        "train_extra_env_keys": sorted(
            str(key) for key in extra_env),
        "train_extra_env_type": type(extra_env_raw).__name__,
        "offline_flags": {
            key: _truthy_env(str(extra_env.get(
                key, os.environ.get(key, ""))))
            for key in _OFFLINE_ENV_KEYS
        },
    }
    return {
        "network_policy": policy,
        "script_sha256": _sha256_file(script_path),
        "config_sha256": _sha256_file(config_path),
        "contract_sha256": hashlib.sha256(json.dumps(
            contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
    }


def verify_artifact_preflight_receipt(idea_id: str, results_dir: Path,
                                      cfg: dict) -> bool:
    """True only when a passed receipt matches the current launch contract."""
    spec = cfg.get("artifact_preflight") or {}
    if not isinstance(spec, dict) or not spec.get("enabled", False):
        return True
    try:
        receipt = json.loads((
            Path(results_dir) / idea_id / "artifact_preflight.json"
        ).read_text(encoding="utf-8"))
        expected = _artifact_preflight_identity(idea_id, results_dir, cfg)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError,
            ValueError):
        return False
    return (
        isinstance(receipt, dict)
        and receipt.get("schema_version") == 1
        and receipt.get("idea_id") == idea_id
        and receipt.get("status") == "passed"
        and receipt.get("gpu_visibility") == "hidden"
        and all(expected.get(key) is not None
                and receipt.get(key) == expected.get(key)
                for key in (
                    "network_policy", "script_sha256", "config_sha256",
                    "contract_sha256",
                ))
    )


def run_artifact_preflight(idea_id: str, results_dir: Path, cfg: dict) -> bool:
    """Run a bounded dataset/model resolver with accelerators hidden.

    The configured script owns domain-specific resolution. Orze enforces the
    execution contract: no visible accelerator, explicit metadata-network
    policy, process-group timeout, and a non-secret audit receipt.
    """
    spec = cfg.get("artifact_preflight") or {}
    if not isinstance(spec, dict) or not spec.get("enabled", False):
        return True

    started = time.time()
    results_dir = Path(results_dir)
    idea_dir = results_dir / idea_id
    idea_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = idea_dir / "artifact_preflight.json"
    policy = str(spec.get("network", "inherit")).strip().lower()
    script_text = str(spec.get("script") or "")
    project_root = Path(cfg.get("_project_root", "."))
    script_path = Path(script_text)
    if not script_path.is_absolute():
        script_path = project_root / script_path
    config_path = idea_dir / "idea_config.yaml"
    if not config_path.exists():
        config_path = Path(cfg.get("base_config", "configs/base.yaml"))
        if not config_path.is_absolute():
            config_path = project_root / config_path

    identity = _artifact_preflight_identity(idea_id, results_dir, cfg)
    receipt = {
        "schema_version": 1,
        "idea_id": idea_id,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu_visibility": "hidden",
        **identity,
    }

    def finish(status: str, **fields) -> bool:
        receipt.update(fields)
        receipt["status"] = status
        receipt["finished_at"] = datetime.datetime.now(
            datetime.timezone.utc).isoformat()
        receipt["duration_seconds"] = round(time.time() - started, 3)
        from orze.core.fs import atomic_write
        atomic_write(receipt_path, json.dumps(receipt, indent=2) + "\n")
        return status == "passed"

    if policy not in {"inherit", "required", "offline"}:
        return finish("configuration_error", reason="invalid_network_policy")
    if not script_text or not script_path.is_file():
        return finish("configuration_error", reason="script_missing")

    extra_env = cfg.get("train_extra_env") or {}
    if not isinstance(extra_env, dict):
        return finish("configuration_error", reason="train_extra_env_not_mapping")
    env = os.environ.copy()
    for key, value in extra_env.items():
        env[key] = str(value)
    offline_conflicts = [key for key in _OFFLINE_ENV_KEYS
                         if _truthy_env(env.get(key, ""))]
    if policy == "required" and offline_conflicts:
        return finish(
            "configuration_error",
            reason="network_required_but_offline_flags_set",
            conflicting_env_keys=offline_conflicts,
        )
    if policy == "offline":
        for key in _OFFLINE_ENV_KEYS:
            env[key] = "1"

    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "none",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "ORZE_ARTIFACT_PREFLIGHT": "1",
        "ORZE_ARTIFACT_NETWORK_POLICY": policy,
        "ORZE_IDEA_ID": idea_id,
        "ORZE_RESULTS_DIR": str(results_dir),
        "ORZE_IDEA_CONFIG": str(config_path),
    })

    from orze.engine.launcher import _format_args
    values = {
        "idea_id": idea_id,
        "results_dir": str(results_dir),
        "config": str(config_path),
        "project_root": str(project_root),
    }
    interpreter = spec.get("interpreter", cfg.get("python", sys.executable))
    cmd = ([str(interpreter)] if interpreter else []) + [str(script_path)]
    cmd.extend(_format_args(spec.get("args") or [], values))
    timeout = float(spec.get("timeout", 300))

    logger.info("Running zero-GPU artifact preflight for %s (%s network)",
                idea_id, policy)
    proc = None
    stdout_digest = hashlib.sha256()
    stderr_digest = hashlib.sha256()
    drain_threads = []
    try:
        proc = subprocess.Popen(
            cmd, env=env, cwd=str(project_root),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        for stream, digest in (
                (proc.stdout, stdout_digest),
                (proc.stderr, stderr_digest)):
            thread = threading.Thread(
                target=_drain_and_hash, args=(stream, digest), daemon=True)
            thread.start()
            drain_threads.append(thread)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        assert proc is not None
        _terminate_and_reap(proc, f"artifact preflight {idea_id}",
                            timeout=1, pgid=proc.pid)
        for thread in drain_threads:
            thread.join()
        return finish(
            "timed_out", timeout_seconds=timeout,
            stdout_sha256=stdout_digest.hexdigest(),
            stderr_sha256=stderr_digest.hexdigest(),
        )
    except Exception as exc:
        if proc is not None:
            _terminate_and_reap(proc, f"artifact preflight {idea_id}",
                                timeout=1, pgid=proc.pid)
        for thread in drain_threads:
            thread.join()
        return finish("execution_error", reason=type(exc).__name__)

    for thread in drain_threads:
        thread.join()

    fields = {
        "exit_code": proc.returncode,
        "stdout_sha256": stdout_digest.hexdigest(),
        "stderr_sha256": stderr_digest.hexdigest(),
    }
    if proc.returncode != 0:
        logger.warning("Artifact preflight failed for %s (exit %d)",
                       idea_id, proc.returncode)
        return finish("failed", **fields)
    logger.info("Artifact preflight passed for %s", idea_id)
    return finish("passed", **fields)


# ---------------------------------------------------------------------------
# Utilities

@dataclass
class TrainingProcess:
    """Tracks a non-blocking training subprocess."""
    idea_id: str
    gpu: int
    process: subprocess.Popen
    start_time: float
    log_path: Path
    timeout: float
    train_script: Optional[str] = None
    config_path: Optional[str] = None
    attempt_id: Optional[str] = None
    execution_identity: Optional[str] = None
    _log_fh: Any = field(default=None, repr=False)
    _last_log_size: int = field(default=0, repr=False)
    _last_log_check: float = field(default=0.0, repr=False)
    _stall_since: float = field(default=0.0, repr=False)

    def close_log(self):
        """Close the log file handle if open."""
        if self._log_fh and not self._log_fh.closed:
            try:
                self._log_fh.close()
            except Exception:
                pass


@dataclass
class EvalProcess:
    """Tracks a non-blocking eval subprocess."""
    idea_id: str
    gpu: int
    process: subprocess.Popen
    start_time: float
    log_path: Path
    timeout: float
    attempt_id: Optional[str] = None
    _log_fh: Any = field(default=None, repr=False)

    def close_log(self):
        if self._log_fh and not self._log_fh.closed:
            try:
                self._log_fh.close()
            except Exception:
                pass


@dataclass
class RoleProcess:
    """Tracks a non-blocking agent role subprocess."""
    role_name: str
    process: subprocess.Popen
    start_time: float
    log_path: Path
    timeout: float
    lock_dir: Path
    cycle_num: int
    _log_fh: Any = field(default=None, repr=False)
    ideas_pre_size: int = 0  # ideas.md size before role started
    ideas_pre_count: int = 0  # idea count before role started
    # Running tally of ideas consumed from ideas.md while this role was
    # still active. Incremented by the consumption phase whenever it
    # ingests ideas and a research-writer role is running. Used to avoid
    # the _ideas_were_modified false-negative where a role appends ideas
    # that get consumed (ideas.md wiped) before the role exits — without
    # this counter, the post-exit size/count check cannot distinguish
    # "appended then consumed" from "never appended".
    ideas_consumed_during_run: int = 0
    # ideas.md mtime snapshot taken at role launch. Used as a cross-daemon
    # fallback signal: in multi-daemon deployments (shared ideas.md,
    # separate active_roles dicts) the ideas_consumed_during_run counter
    # is only incremented by the daemon whose active_roles contains this
    # RoleProcess. If a DIFFERENT daemon performs the consumption (wipes
    # ideas.md), this daemon's counter stays 0 and the size/count check
    # also returns False (ideas_pre_size matches the post-wipe size by
    # coincidence). mtime changes on any write to the shared file, so
    # current_mtime > ideas_md_mtime_pre means ideas.md was touched
    # during the role's lifetime — enough to credit the role.
    ideas_md_mtime_pre: float = 0.0
    # Stall-detection state. Mirrors TrainingProcess: `_last_log_size`
    # tracks the last observed log bytes and `_stall_since` is the
    # epoch-time the log last stopped growing. Used by
    # check_active_roles to kill roles whose stdout has been silent for
    # longer than `role_stall_minutes` — catches claude-CLI hangs that
    # produce a 0-byte log and would otherwise burn the full wall-clock
    # timeout.
    _last_log_size: int = field(default=0, repr=False)
    _stall_since: float = field(default=0.0, repr=False)
    # Round-2 B2: optional per-role override of role_stall_minutes.
    # When None, the global ``role_stall_minutes`` (from orze.yaml) is
    # used. Roles with `<role>.stall_minutes:` set this at launch so
    # check_active_roles can enforce a per-role timer instead of the
    # global one.
    stall_minutes_override: Optional[float] = None
    # Round-2 B3: warmup tolerance — the stall timer doesn't begin
    # counting until either (a) the first stdout byte is observed, or
    # (b) ``stall_warmup_seconds`` has elapsed since process spawn,
    # whichever is sooner. This protects LLM-mode roles whose first
    # 30-90s is model init / skill composition with no stdout yet.
    stall_warmup_seconds: float = 60.0
    # Declared output paths are monitored by metadata only. Together with
    # process-tree CPU ticks and log growth, this makes the no-progress
    # watchdog distinguish a genuine stall from silent useful work.
    progress_paths: tuple[Path, ...] = field(default_factory=tuple)
    _last_cpu_ticks: Optional[int] = field(default=None, repr=False)
    _last_progress_fingerprint: Optional[str] = field(default=None, repr=False)
    _last_progress_kinds: tuple[str, ...] = field(
        default_factory=tuple, repr=False)
    # True for research-type roles whose job is to append to ideas.md.
    # False for strategy roles (professor, data_analyst, thinker,
    # engineer, code_evolution) that modify other files — skipping the
    # ideas-modified soft-failure check for those avoids spurious
    # "completed successfully but ideas file was not modified" warnings.
    writes_ideas_file: bool = True
    # Stable process identities observed while the role is alive. Keeping
    # live descendants across polls lets normal-exit and graceful-shutdown
    # paths reap children that reparented themselves with setsid().
    _root_start_ticks: Optional[int] = field(default=None, repr=False)
    _pgid: Optional[int] = field(default=None, repr=False)
    _tracked_descendants: list[dict] = field(default_factory=list, repr=False)

    def __post_init__(self):
        pid = getattr(self.process, "pid", None)
        if type(pid) is not int:
            return
        try:
            identity = capture_process_identity(pid)
        except (OSError, ValueError, ProcessLookupError):
            return
        self._root_start_ticks = identity["start_ticks"]
        self._pgid = identity["pgid"]
        refresh_role_process_descendants(self)

    def close_log(self):
        if self._log_fh and not self._log_fh.closed:
            try:
                self._log_fh.close()
            except Exception:
                pass


def refresh_role_process_descendants(rp: RoleProcess) -> int:
    """Refresh the live stable-identity set for a managed role.

    Dead identities are pruned, so process churn cannot grow this state
    indefinitely. The return value is the current number of retained live
    descendants.
    """
    pid = getattr(rp.process, "pid", None)
    if type(pid) is not int:
        return 0
    retained = {
        (int(identity["pid"]), int(identity["start_ticks"])): identity
        for identity in getattr(rp, "_tracked_descendants", [])
        if isinstance(identity, dict)
        and type(identity.get("pid")) is int
        and type(identity.get("start_ticks")) is int
        and process_is_running(identity["pid"], identity["start_ticks"])
    }
    root_start_ticks = getattr(rp, "_root_start_ticks", None)
    if (root_start_ticks is not None
            and process_is_running(pid, root_start_ticks)):
        for identity in process_descendant_identities(pid):
            retained[(identity["pid"], identity["start_ticks"])] = identity
    rp._tracked_descendants = sorted(
        retained.values(), key=lambda identity: identity["pid"])
    return len(rp._tracked_descendants)


def terminate_role_process(
    rp: RoleProcess,
    label: str,
    timeout: float = 10,
    *,
    reaper=None,
) -> bool:
    """Terminate a role using its stable root and descendant identities."""
    refresh_role_process_descendants(rp)
    pid = getattr(rp.process, "pid", None)
    root_live = (
        type(pid) is int
        and rp._root_start_ticks is not None
        and process_is_running(pid, rp._root_start_ticks)
    )
    # Never signal a numeric process group after its recorded leader has
    # exited: Linux may already have reused that ID. Exact retained descendant
    # identities remain safe to signal after reparenting.
    pgid = rp._pgid if root_live else None
    result = (reaper or _terminate_and_reap)(
        rp.process,
        label,
        timeout=timeout,
        pgid=pgid,
        tracked_identities=list(rp._tracked_descendants),
        discover_pgid=root_live,
    )
    rp._tracked_descendants.clear()
    # Legacy/mocked reapers returned None; only an explicit False means the
    # termination proof failed.
    return result is not False


def run_pre_script(idea_id: str, gpu: int, cfg: dict,
                   results_dir: Optional[Path] = None) -> bool:
    """Run pre-training script if configured. Returns True if OK to proceed."""
    import sys
    pre_script = cfg.get("pre_script")
    if not pre_script:
        return True

    python = cfg.get("python", sys.executable)
    pre_args = cfg.get("pre_args") or []
    pre_timeout = cfg.get("pre_timeout", 3600)

    from orze.engine.launcher import _format_args
    cmd = [python, pre_script]
    cmd.extend(_format_args(pre_args, {"idea_id": idea_id, "gpu": gpu}))

    env = os.environ.copy()
    for k, v in (cfg.get("train_extra_env") or {}).items():
        env[k] = str(v)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    logger.info("Running pre-script for %s on GPU %s", idea_id, gpu)
    proc = None
    handle = None
    try:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, preexec_fn=_new_process_group,
        )
        from types import SimpleNamespace
        handle = SimpleNamespace(
            idea_id=idea_id,
            gpu=gpu,
            process=proc,
            start_time=time.time(),
            attempt_id=secrets.token_hex(16),
        )
        if results_dir is not None:
            from orze.engine.accounting import record_compute_start
            record_compute_start(
                handle, Path(results_dir) / idea_id, phase="pre_script")
        _, stderr = proc.communicate(timeout=pre_timeout)
        if proc.returncode == 0:
            logger.info("Pre-script OK for %s", idea_id)
            if results_dir is not None:
                from orze.engine.accounting import record_compute_terminal
                record_compute_terminal(
                    handle, Path(results_dir) / idea_id, "completed",
                    "pre_script_completed", phase="pre_script",
                    return_code=proc.returncode)
            return True
        logger.warning("Pre-script failed for %s (exit %d): %s",
                       idea_id, proc.returncode,
                       stderr[-200:] if stderr else "")
        if results_dir is not None:
            from orze.engine.accounting import record_compute_terminal
            record_compute_terminal(
                handle, Path(results_dir) / idea_id, "failed",
                "pre_script_nonzero", phase="pre_script",
                return_code=proc.returncode)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Pre-script timed out for %s after %ds",
                       idea_id, pre_timeout)
        _terminate_and_reap(proc, f"pre-script {idea_id}")
        if results_dir is not None:
            from orze.engine.accounting import record_compute_terminal
            record_compute_terminal(
                handle, Path(results_dir) / idea_id, "interrupted",
                "pre_script_timeout", phase="pre_script",
                return_code=proc.poll())
        return False
    except Exception as e:
        if proc is not None and proc.poll() is None:
            _terminate_and_reap(proc, f"pre-script {idea_id}", timeout=3)
        if results_dir is not None and handle is not None:
            from orze.engine.accounting import record_compute_terminal
            record_compute_terminal(
                handle, Path(results_dir) / idea_id, "failed",
                "pre_script_error", phase="pre_script",
                return_code=proc.poll())
        elif results_dir is not None:
            from orze.engine.accounting import record_zero_gpu_outcome
            record_zero_gpu_outcome(
                idea_id, Path(results_dir) / idea_id, gpu, "rejected",
                "pre_script_popen_failed", phase="pre_script")
        logger.warning("Pre-script error for %s: %s", idea_id, e)
        return False


def _new_process_group():
    """preexec_fn for subprocess.Popen to create a new process group."""
    os.setpgrp()
