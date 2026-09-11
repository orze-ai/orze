"""Orze lifecycle management — stop, start, restart.

Provides the core logic for `orze stop`, `orze start`, and `orze restart`.
Stop publishes a cooperative request, not proof of controller/tree closure.
Pending stop markers refuse start; restart requires a future qualified
closure consumer. No process-name, integer-PID or GPU orphan cleanup is used.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# Pattern used to find the orze orchestrator via pgrep/pkill.
# The [o] trick prevents the grep/pgrep process from matching itself.
#
# Match ONLY the detached orchestrator — `python -m orze.cli -c …orze.yaml`
# (daemon mode) and the foreground equivalent after os.execv. A looser
# pattern like `orze.*orze\.yaml` also matches the *caller* shell whose
# cmdline is literally `orze start -c orze.yaml`, producing a phantom
# "already running" error on the very first `orze start` from bash.
_ORZE_PAT = r"^[^ ]+ -m [o]rze\.cli( |$).*orze\.yaml( |$)"

def _log(prefix, msg):
    print(f"[{prefix}] {msg}", flush=True)


def _read_pid(results_dir: Path, hostname: str):
    """Read PID from .orze.pid.{hostname} or legacy .orze.pid."""
    for name in [f".orze.pid.{hostname}", ".orze.pid"]:
        pid_file = results_dir / name
        if pid_file.exists():
            try:
                return int(pid_file.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pass
    return None


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _kill_pid(pid: int, timeout: int = 10):
    """Deprecated: a bare PID cannot authorize signaling or prove closure."""
    from orze.core.control_outcome import StopOutcome
    return StopOutcome("hold", "owned_process_handle_required")


def _pgrep(pattern: str) -> list:
    """Return PIDs matching a pgrep -f pattern (excluding ourselves)."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True, text=True, timeout=5,
        )
        return [
            int(p) for p in result.stdout.strip().split()
            if p.strip() and int(p) != os.getpid()
        ]
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return []


def _cleanup_gpu_orphans(workdir: str, gpu_ids=None):
    """Deprecated: GPU/cmdline matching supplies no execution ownership."""
    from orze.core.control_outcome import StopOutcome
    return StopOutcome("hold", "owned_process_handle_required")


def _kill_children_of(parent_pid: int, timeout: int = 60):
    """Deprecated: enumerating children of an integer PID is not authority."""
    from orze.core.control_outcome import StopOutcome
    return StopOutcome("hold", "owned_process_handle_required")


# ── stop ─────────────────────────────────────────────────────────────

def do_stop(cfg: dict, timeout: int = 60):
    """Publish a cooperative stop request, never claim controller closure.

    Existing owned-handle shutdown consumes the stop sentinel. A PID file,
    process name, GPU assignment, CLI return code or shutdown sentinel is not
    proof that all former writers exited. The legacy timeout argument remains
    accepted, but this request-only API performs no process wait or scan.
    """
    from orze.core.control_outcome import StopOutcome
    from orze.core.fs import atomic_write
    import stat

    try:
        results_dir = Path(cfg["results_dir"]).absolute()
        # Refuse redirected request publication instead of overwriting another
        # scope's markers. These checks do not create a global filesystem lock.
        for path in (results_dir, *results_dir.parents):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(info.st_mode):
                raise OSError("stop_request_directory_unverifiable")
        results_dir.mkdir(parents=True, exist_ok=True)
        for name, content in (
            (".orze_disabled", "Controller stop requested; closure unconfirmed"),
            (".orze_stop_all", "kill"),
        ):
            path = results_dir / name
            try:
                info = path.lstat()
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise OSError("stop_request_marker_unverifiable")
            atomic_write(path, content)
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                before = os.fstat(fd)
                expected = content.encode("utf-8")
                if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                        or before.st_size != len(expected)):
                    raise OSError("stop_request_readback_unconfirmed")
                raw = os.read(fd, len(expected) + 1)
                after = os.fstat(fd)
                witness = lambda info: (
                    info.st_dev, info.st_ino, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)
                if (raw != expected or witness(before) != witness(after)
                        or witness(before) != witness(path.lstat())):
                    raise OSError("stop_request_readback_unconfirmed")
            finally:
                os.close(fd)
        directory = os.open(results_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except (OSError, ValueError, TypeError, KeyError):
        _log("stop", "HOLD: cooperative stop request publication unconfirmed")
        return StopOutcome("hold", "controller_stop_request_unconfirmed")

    _log("stop", "Cooperative stop requested; controller/process-tree closure unconfirmed")
    return StopOutcome("requested", "controller_stop_confirmation_required")


# ── start ────────────────────────────────────────────────────────────

def do_start(cfg: dict, foreground: bool = False, config_path: str = None,
             gpus: str = None, timeout: int = None):
    """Start orze on the local node.

    1. Verify the opt-in controller runtime contract
    2. Check not already running
    3. Refuse any stop/shutdown sentinel without clearing it
    4. Build the child command
    5. Launch orze (detached daemon or foreground via os.execv)

    Args:
        gpus: Comma-separated GPU IDs (e.g. "0,1,3"). None = auto-detect.
        timeout: Max training time per job in seconds. None = use config.

    Returns PID in daemon mode. In foreground mode, replaces the process
    via os.execv (never returns).
    """
    # This function is also a public Python entry point, so enforce identity
    # here rather than relying only on cli.py or on the eventual child. The
    # check must precede directory creation and, critically, sentinel removal.
    from orze.service.runtime_contract import require_controller_runtime_contract
    require_controller_runtime_contract(cfg.get("controller_runtime"))

    results_dir = Path(cfg["results_dir"])
    from orze.core.control_outcome import require_controller_start_allowed
    require_controller_start_allowed(results_dir)
    hostname = socket.gethostname()
    config_path = config_path or cfg.get("_config_path", "orze.yaml")
    python = sys.executable
    log_file = str(Path(cfg.get("results_dir", "orze_results")) / "orze.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # 2. Check not already running
    pid = _read_pid(results_dir, hostname)
    if pid and _is_alive(pid):
        _log("start", f"Orze is already running (PID {pid}). "
             f"Use 'orze restart' instead.")
        sys.exit(1)

    # Also check via pgrep
    running = _pgrep(_ORZE_PAT)
    if running:
        _log("start", f"Orze is already running (PID {running[0]}). "
             f"Use 'orze restart' instead.")
        sys.exit(1)

    # 3. Recheck stop state at the final launch boundary. Never clear it here.
    results_dir.mkdir(parents=True, exist_ok=True)
    require_controller_start_allowed(results_dir)

    # 4. Build command
    cmd = [python, "-m", "orze.cli", "-c", config_path]
    if gpus:
        cmd.extend(["--gpus", gpus])
    if timeout is not None:
        cmd.extend(["--timeout", str(timeout)])

    # 5. Launch
    if foreground:
        gpu_msg = f" on GPUs {gpus}" if gpus else ""
        _log("start", f"Starting orze in foreground{gpu_msg}...")
        require_controller_start_allowed(results_dir)
        os.execv(python, cmd)
        # Never returns

    with open(log_file, "a") as lf:
        require_controller_start_allowed(results_dir)
        proc = subprocess.Popen(
            cmd,
            stdout=lf, stderr=lf,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    time.sleep(3)
    if proc.poll() is not None:
        _log("start", f"ERROR: Failed to start (exit code {proc.returncode}). "
             f"Check {log_file}")
        sys.exit(1)

    gpu_msg = f" on GPUs {gpus}" if gpus else ""
    _log("start", f"Orze started{gpu_msg} (PID {proc.pid})")
    _log("start", f"Log: {log_file}")
    return proc.pid


# ── restart ──────────────────────────────────────────────────────────

def do_restart(cfg: dict, timeout: int = 60, foreground: bool = False,
               config_path: str = None, gpus: str = None):
    """Request stop; no restart without a source-qualified closure consumer."""
    from orze.core.control_outcome import StopOutcome
    from orze.service.runtime_contract import require_controller_runtime_contract

    outcome = do_stop(cfg, timeout=timeout)
    # Preserve the independent runtime refusal without invoking do_start,
    # whose old side effects used to erase the pending stop markers.
    require_controller_runtime_contract(cfg.get("controller_runtime"))
    if isinstance(outcome, StopOutcome) and outcome.status in ("requested", "hold"):
        return outcome
    # No confirmed-stop consumer exists in this slice. Even a typed label,
    # legacy True/None or an unexpected caller result cannot authorize start.
    return StopOutcome("hold", "controller_stop_confirmation_required")
