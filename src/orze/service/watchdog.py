"""Orze watchdog — checks if orze is alive and restarts if needed.

Invokable as: python -m orze.service.watchdog
Designed to be called every minute from crontab or every 5 minutes from systemd timer.
Also manages Docker containers defined in orze.yaml (auto-pull, auto-recreate).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("orze.watchdog")

SERVICE_CONFIG_PATH = Path.home() / ".orze_service.json"


def load_service_config(path=None):
    """Read ~/.orze_service.json. Returns dict or None."""
    p = Path(path) if path else SERVICE_CONFIG_PATH
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _read_pid(results_dir, hostname):
    """Read PID from .orze.pid.{hostname}. Returns int or None."""
    pid_file = Path(results_dir) / f".orze.pid.{hostname}"
    if not pid_file.exists():
        # Fall back to legacy single PID file
        pid_file = Path(results_dir) / ".orze.pid"
        if not pid_file.exists():
            return None
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def _is_pid_alive(pid):
    """Check if a process is alive via kill(0)."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _is_heartbeat_stale(results_dir, hostname, threshold):
    """Check if the most recent heartbeat for this host is stale.

    Returns (stale: bool, age_seconds: float) or (False, 0) if no heartbeat found.
    """
    results_dir = Path(results_dir)
    best_epoch = 0
    hb_dir = results_dir.parent / ".orze" / "heartbeats"
    candidates = list(hb_dir.glob(f"{hostname}_*.json")) + \
        list(results_dir.glob(f"_host_{hostname}_*.json"))
    for hb_path in candidates:
        try:
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
            epoch = hb.get("epoch", 0)
            if isinstance(epoch, (int, float)) and epoch > best_epoch:
                best_epoch = epoch
        except (json.JSONDecodeError, OSError):
            continue

    if best_epoch == 0:
        return False, 0

    age = time.time() - best_epoch
    return age > threshold, age


def _should_restart(results_dir):
    """Check sentinel files. Returns (should_skip: bool, reason: str)."""
    results_dir = Path(results_dir)

    disabled = results_dir / ".orze_disabled"
    if disabled.exists():
        return True, "disabled (.orze_disabled exists)"

    stop_all = results_dir / ".orze_stop_all"
    if stop_all.exists():
        return True, "stopped (.orze_stop_all exists)"

    shutdown = results_dir / ".orze_shutdown"
    if shutdown.exists():
        try:
            age = time.time() - shutdown.stat().st_mtime
        except OSError:
            age = 0
        if age < 120:
            return True, f"graceful shutdown {age:.0f}s ago (waiting 120s)"

    return False, ""


def _kill_stale(pid):
    """Kill a stale process. SIGTERM first, SIGKILL after 5s."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return

    for _ in range(10):
        time.sleep(0.5)
        if not _is_pid_alive(pid):
            return

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _is_orze_running():
    """Secondary check: pgrep for orze processes (any launch method).

    Uses ``orze\\.cli`` which matches ``python -m orze.cli`` regardless of
    config file name.  The PID file is the primary liveness check; this
    pgrep is only a fallback.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", r"orze\.cli"],
            capture_output=True, timeout=5,
        )
        # Filter out our own watchdog process
        pids = [p for p in result.stdout.decode().split() if p.strip() and int(p) != os.getpid()]
        return len(pids) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return False


def _launch_orze(svc_cfg):
    """Launch orze as a detached process."""
    python = svc_cfg["python"]
    config_file = svc_cfg["config_file"]
    workdir = svc_cfg.get("workdir", ".")
    log_file = svc_cfg.get("log_file", str(Path(svc_cfg.get("results_dir", "/tmp")) / "orze.log"))

    with open(log_file, "a") as lf:
        proc = subprocess.Popen(
            [python, "-m", "orze.cli", "-c", config_file],
            cwd=workdir,
            stdout=lf,
            stderr=lf,
            start_new_session=True,
        )
    return proc.pid


def _write_restart_marker(results_dir, hostname, reason, prev_pid=None):
    """Write a marker file so orchestrator can send notifications on startup."""
    results_dir = Path(results_dir)
    marker = results_dir / f".orze_watchdog_restart_{hostname}.json"
    data = {
        "hostname": hostname,
        "reason": reason,
        "prev_pid": prev_pid,
        "timestamp": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    marker.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _has_docker():
    """Check if docker CLI is available."""
    return shutil.which("docker") is not None


def _docker_run(args, timeout=60):
    """Run a docker command. Returns (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(
            ["docker"] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return -1, "", str(exc)


def _container_image_id(name):
    """Return the image ID currently used by a running/stopped container, or None."""
    rc, out, _ = _docker_run(["inspect", "--format", "{{.Image}}", name])
    return out if rc == 0 else None


def _pull_image(image):
    """Pull image. Returns (changed: bool, new_id: str | None)."""
    rc_before, id_before, _ = _docker_run(
        ["images", "--format", "{{.ID}}", "--no-trunc", image],
    )
    rc, _, stderr = _docker_run(["pull", image], timeout=300)
    if rc != 0:
        return False, None
    rc_after, id_after, _ = _docker_run(
        ["images", "--format", "{{.ID}}", "--no-trunc", image],
    )
    changed = (rc_before != 0) or (id_before != id_after)
    return changed, id_after


def _recreate_container(name, spec):
    """Stop + remove + run a container from spec dict."""
    _docker_run(["stop", name], timeout=30)
    _docker_run(["rm", name], timeout=15)

    cmd = ["run", "-d", "--name", name, "--restart=unless-stopped"]
    for port in spec.get("ports", []):
        cmd += ["-p", str(port)]
    for vol in spec.get("volumes", []):
        cmd += ["-v", str(vol)]
    for k, v in spec.get("env", {}).items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(spec["image"])
    cmd += spec.get("command", [])

    rc, out, err = _docker_run(cmd, timeout=60)
    return rc == 0, out or err


def check_containers(svc_cfg, _log=None):
    """Check Docker containers defined in config — auto-pull and recreate if image changed."""
    if _log is None:
        _log = logger.info

    if not _has_docker():
        return

    config_file = svc_cfg.get("config_file")
    if not config_file or not Path(config_file).exists():
        return

    try:
        import yaml
        cfg = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    except Exception:
        return

    containers = cfg.get("containers", {})
    if not containers:
        return

    for name, spec in containers.items():
        if not isinstance(spec, dict) or "image" not in spec:
            continue

        image = spec["image"]
        old_image_id = _container_image_id(name)

        changed, new_id = _pull_image(image)
        if changed:
            _log(f"Container '{name}': new image pulled ({image}), recreating.")
            ok, detail = _recreate_container(name, spec)
            if ok:
                _log(f"Container '{name}': recreated successfully.")
            else:
                _log(f"Container '{name}': recreate failed — {detail}")
        else:
            # Ensure container is running even if image didn't change
            rc, state, _ = _docker_run(
                ["inspect", "--format", "{{.State.Running}}", name],
            )
            if rc != 0:
                _log(f"Container '{name}': not found, creating.")
                ok, detail = _recreate_container(name, spec)
                if ok:
                    _log(f"Container '{name}': created successfully.")
                else:
                    _log(f"Container '{name}': create failed — {detail}")
            elif state != "true":
                _log(f"Container '{name}': not running, starting.")
                _docker_run(["start", name])


def check_and_restart(svc_cfg):
    """Main watchdog logic: check alive -> check stale -> check sentinels -> restart."""
    hostname = socket.gethostname()
    results_dir = svc_cfg["results_dir"]
    threshold = svc_cfg.get("stall_threshold", 1800)
    log_file = svc_cfg.get("log_file", str(Path(results_dir) / "orze_watchdog.log"))

    def _log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [watchdog] {msg}"
        logger.info(msg)
        try:
            with open(log_file, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    # Check sentinels first — don't restart if disabled/stopped
    skip, reason = _should_restart(results_dir)
    if skip:
        _log(f"Skipping restart: {reason}")
        return

    pid = _read_pid(results_dir, hostname)

    if pid and _is_pid_alive(pid):
        # Process alive — check for stalls
        stale, age = _is_heartbeat_stale(results_dir, hostname, threshold)
        if stale:
            _log(f"Orze PID {pid} alive but heartbeat stale ({age:.0f}s > {threshold}s). Killing.")
            _kill_stale(pid)
            time.sleep(2)
            _write_restart_marker(results_dir, hostname, f"stale heartbeat ({age:.0f}s)", pid)
        else:
            # All good, nothing to do
            return
    elif pid and not _is_pid_alive(pid):
        _log(f"Orze PID {pid} not alive.")
        _write_restart_marker(results_dir, hostname, "process died", pid)
    else:
        # No PID file — check if somehow running anyway
        if _is_orze_running():
            _log("No PID file but orze process found. Skipping.")
            return
        _write_restart_marker(results_dir, hostname, "no PID file found", None)

    # Double-check: no orze.cli already running (race condition guard)
    if _is_orze_running():
        _log("orze already running (pgrep). Skipping launch.")
        return

    # Re-check sentinels (may have changed during stall kill)
    skip, reason = _should_restart(results_dir)
    if skip:
        _log(f"Skipping restart after kill: {reason}")
        return

    new_pid = _launch_orze(svc_cfg)
    _log(f"Restarted orze (new PID {new_pid})")


def main():
    """Entry point for python -m orze.service.watchdog."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    svc_cfg = load_service_config()
    if not svc_cfg:
        print(f"No service config found at {SERVICE_CONFIG_PATH}", file=sys.stderr)
        print("Run 'orze service install -c orze.yaml' first.", file=sys.stderr)
        sys.exit(1)

    check_and_restart(svc_cfg)
    check_containers(svc_cfg)


if __name__ == "__main__":
    main()
