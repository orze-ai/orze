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
        side effects: SIGTERM -> wait -> SIGKILL if needed; logs warnings on force kill

    _new_process_group() -> None
        preexec_fn for subprocess.Popen; calls os.setpgrp() to create a new process group

    run_pre_script(idea_id, gpu, cfg) -> bool
        idea_id: str
        gpu: int — set as CUDA_VISIBLE_DEVICES
        cfg: dict — uses 'pre_script', 'pre_args', 'pre_timeout', 'python', 'train_extra_env'
        returns: True if no pre_script configured or script exited 0, False on failure/timeout
        side effects: runs blocking subprocess
"""
import os
import signal
import subprocess
import logging
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
                        timeout: float = 10):
    """SIGTERM the process group, wait, then SIGKILL if needed."""
    _kill_pg(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Force killing %s (PID %d)", label or "process", proc.pid)
        _kill_pg(proc, signal.SIGKILL)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error("Failed to reap %s (PID %d) after SIGKILL",
                         label or "process", proc.pid)


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
    stall_minutes_override: Optional[int] = None
    # Round-2 B3: warmup tolerance — the stall timer doesn't begin
    # counting until either (a) the first stdout byte is observed, or
    # (b) ``stall_warmup_seconds`` has elapsed since process spawn,
    # whichever is sooner. This protects LLM-mode roles whose first
    # 30-90s is model init / skill composition with no stdout yet.
    stall_warmup_seconds: float = 60.0
    # True for research-type roles whose job is to append to ideas.md.
    # False for strategy roles (professor, data_analyst, thinker,
    # engineer, code_evolution) that modify other files — skipping the
    # ideas-modified soft-failure check for those avoids spurious
    # "completed successfully but ideas file was not modified" warnings.
    writes_ideas_file: bool = True

    def close_log(self):
        if self._log_fh and not self._log_fh.closed:
            try:
                self._log_fh.close()
            except Exception:
                pass


def run_pre_script(idea_id: str, gpu: int, cfg: dict) -> bool:
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
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=pre_timeout,
        )
        if result.returncode == 0:
            logger.info("Pre-script OK for %s", idea_id)
            return True
        else:
            logger.warning("Pre-script failed for %s (exit %d): %s",
                           idea_id, result.returncode,
                           result.stderr[-200:] if result.stderr else "")
            return False
    except subprocess.TimeoutExpired:
        logger.warning("Pre-script timed out for %s after %ds",
                       idea_id, pre_timeout)
        return False
    except Exception as e:
        logger.warning("Pre-script error for %s: %s", idea_id, e)
        return False


def _new_process_group():
    """preexec_fn for subprocess.Popen to create a new process group."""
    os.setpgrp()
