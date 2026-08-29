"""Training subprocess launcher and lifecycle monitor.

CALLING SPEC:
    launch(idea_id, gpu, results_dir, cfg) -> TrainingProcess
        idea_id: str — experiment identifier (e.g. "idea-abc123")
        gpu: int — CUDA device index (set as CUDA_VISIBLE_DEVICES)
        results_dir: Path — parent dir; logs written to results_dir/idea_id/train_output.log
        cfg: dict — orze config; requires keys 'train_script', 'ideas_file', 'base_config';
                     optional 'python', 'train_extra_args', 'train_extra_env', 'timeout'
        returns: TrainingProcess with a running Popen in its own process group
        side effects: creates results_dir/idea_id/train_output.log, spawns subprocess

    check_active(active, results_dir, cfg, failure_counts, fix_counts=None) -> list[(idea_id, gpu)]
        active: Dict[int, TrainingProcess] — gpu -> running process; MUTATED in-place (finished entries removed)
        results_dir: Path
        cfg: dict — uses 'stall_minutes', 'max_fix_attempts', executor config
        failure_counts: dict — idea_id -> int; MUTATED to track consecutive failures
        fix_counts: dict | None — idea_id -> int; MUTATED to track fix attempts
        returns: list of (idea_id, gpu) tuples for processes that finished this cycle
        side effects: kills timed-out/stalled/hung processes, writes metrics.json for failures,
                      may invoke executor LLM to auto-fix and relaunch failed ideas,
                      sends notifications on stall/timeout

    _format_args(args, template_vars) -> list[str]
        args: list | str | None — raw arguments (coerced to list)
        template_vars: dict — e.g. {"idea_id": "idea-abc", "gpu": 0}; replaces {key} in each arg
        returns: list of formatted string arguments

    _write_failure(idea_dir, reason) -> None
        idea_dir: Path — e.g. results_dir / idea_id
        reason: str — error description
        side effects: atomically writes {"status": "FAILED", "error": reason} to idea_dir/metrics.json

    _detect_zombie(tp) -> bool
        tp: TrainingProcess — the process to check
        returns: True if process is stuck (alive but no CPU/GPU activity and no log growth)
                 for 3 consecutive checks; False otherwise
        side effects: stores _zombie_cpu, _zombie_log_size, _zombie_count on tp for state tracking

    _get_checkpoint_dir(cfg) -> Path | None
        cfg: dict — orze config
        returns: value of --checkpoint-dir from train_extra_args, or None
"""
import datetime
import json
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

from orze.engine.process import (
    TrainingProcess, _new_process_group, _terminate_and_reap,
    capture_process_identity, verify_artifact_preflight_receipt,
)
from orze.engine.resume import (
    mark_resume_launched, prepare_resume_launch, write_interruption_receipt,
)
from orze.engine.execution_identity import (
    DuplicateExecutionError, compute_execution_identity,
    release_execution_identity, reserve_execution_identity,
)
from orze.core.fs import atomic_write, tail_file
from orze.core.gpu_lease import gpu_execution_lease
from orze.core.research_policy import validate_idea_against_research_policy
from orze.reporting.notifications import notify

logger = logging.getLogger("orze")

# #10: Rolling buffer for anomaly detection
_recent_completions: list = []


def _resolve_train_script(ts: str, cfg: dict) -> str:
    """Resolve a per-idea train_script override to an existing path.

    Ideas (and portfolios) sometimes set a bare basename like
    'train_orze_adapter.py' while the canonical script lives at
    'vjepa2/train_orze_adapter.py'. A strict Path(ts).exists() check from
    repo root then rejects the idea with "train_script not found", blocking
    the whole portfolio. If the literal path is missing but its basename
    matches the canonical cfg train_script (or a sweep_allowlist entry) that
    DOES exist, resolve to that real path. Returns ts unchanged when it
    already exists or no match is found.
    """
    if not ts or Path(ts).exists():
        return ts
    name = Path(ts).name
    candidates = [cfg.get("train_script", "")]
    candidates += list(cfg.get("sweep_allowlist", []) or [])
    for cand in candidates:
        if cand and Path(cand).name == name and Path(cand).exists():
            logger.info("Resolved train_script '%s' -> '%s'", ts, cand)
            return cand
    return ts


def _get_checkpoint_dir(cfg: dict) -> Optional[Path]:
    """Extract --checkpoint-dir from train_extra_args."""
    args = cfg.get("train_extra_args") or []
    for i, arg in enumerate(args):
        if str(arg) == "--checkpoint-dir" and i + 1 < len(args):
            return Path(str(args[i + 1]))
    return None


_BOUNDARY_ENV_KEYS = (
    "ORZE_FORBIDDEN_PATHS",
    "ORZE_WATCH_PATHS",
    "ORZE_ACCESS_LOG",
    "ORZE_REQUIRE_KERNEL_BOUNDARY",
    "ORZE_KERNEL_BOUNDARY_ACTIVE",
    "ORZE_TRAINING_NETWORK",
    "ORZE_BOUNDARY_ATTEST_FD",
    "ORZE_BOUNDARY_ATTEST_NONCE",
)


def _resolve_paths(paths) -> list[str]:
    """Resolve a validated list of path strings to canonical paths."""
    return [os.path.realpath(str(path)) for path in (paths or [])]


def _validated_data_boundary_policy(db_cfg) -> tuple[list[str], list[str], str]:
    """Return canonical boundary policy or reject a direct launch.

    Hard-block targets must be stable ordinary files or directories. A
    redirected or missing target cannot support a claim that it was hidden
    from training, so it fails before GPU telemetry.
    """
    if not isinstance(db_cfg, dict):
        raise LaunchIntegrityError("data_boundary_policy_not_mapping")
    for key in ("forbidden_in_training", "watch_paths"):
        raw = db_cfg.get(key, [])
        if (not isinstance(raw, list)
                or any(not isinstance(path, str) or not path.strip()
                       for path in raw)):
            raise LaunchIntegrityError(
                f"data_boundary_{key}_invalid")
        if any(":" in path or any(ord(char) < 32 for char in path)
               for path in raw):
            raise LaunchIntegrityError(
                f"data_boundary_{key}_unsafe_characters")
        if any(not Path(path).is_absolute() for path in raw):
            raise LaunchIntegrityError(
                f"data_boundary_{key}_must_be_absolute")

    forbidden = _resolve_paths(db_cfg.get("forbidden_in_training"))
    watch = _resolve_paths(db_cfg.get("watch_paths"))
    network = db_cfg.get("training_network", "inherit")
    if network not in ("inherit", "deny"):
        raise LaunchIntegrityError("data_boundary_training_network_invalid")

    canonical_forbidden = []
    for raw, resolved in zip(
            db_cfg.get("forbidden_in_training", []), forbidden):
        if os.path.abspath(raw) != resolved:
            raise LaunchIntegrityError("data_boundary_forbidden_path_redirected")
        if resolved == os.path.sep:
            raise LaunchIntegrityError("data_boundary_forbidden_root_invalid")
        try:
            mode = os.stat(resolved, follow_symlinks=False).st_mode
        except OSError as exc:
            raise LaunchIntegrityError(
                "data_boundary_forbidden_path_unavailable") from exc
        import stat
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise LaunchIntegrityError(
                "data_boundary_forbidden_path_type_unsupported")
        if resolved not in canonical_forbidden:
            canonical_forbidden.append(resolved)

    return canonical_forbidden, list(dict.fromkeys(watch)), network


def _apply_data_boundary_env(env: Dict[str, str], db_cfg: dict,
                             idea_dir: Path) -> None:
    """Populate ORZE_FORBIDDEN_PATHS/WATCH_PATHS/ACCESS_LOG env vars from
    data_boundaries config. Used both by the in-process builtins.open patch
    and by the kernel namespace isolation path.
    """
    forbidden, watch, network = _validated_data_boundary_policy(db_cfg)
    for key in _BOUNDARY_ENV_KEYS:
        env.pop(key, None)
    if forbidden:
        env["ORZE_FORBIDDEN_PATHS"] = ":".join(forbidden)
    if watch:
        env["ORZE_WATCH_PATHS"] = ":".join(watch)
    try:
        idea_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    env["ORZE_ACCESS_LOG"] = str(idea_dir / "_access_log.tsv")
    if forbidden or network == "deny":
        env["ORZE_REQUIRE_KERNEL_BOUNDARY"] = "1"
    env["ORZE_TRAINING_NETWORK"] = network


def _probe_kernel_boundary(*, deny_network: bool) -> None:
    """Prove user/mount (and optionally network) isolation without a GPU."""
    unshare = shutil.which("unshare")
    bash = shutil.which("bash")
    mount = shutil.which("mount")
    umount = shutil.which("umount")
    if None in (unshare, bash, mount, umount):
        raise LaunchIntegrityError("data_boundary_kernel_tools_unavailable")
    args = [unshare, "-U", "--map-root-user", "-m"]
    if deny_network:
        args.append("-n")
    env = os.environ.copy()
    for key in (
        "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    ):
        env[key] = ""
    try:
        with tempfile.TemporaryDirectory(
                prefix="orze-boundary-probe-") as target:
            completed = subprocess.run(
                args + [
                    bash, "-ceu",
                    f"{mount} --make-rprivate /; "
                    f"{mount} -t tmpfs -o "
                    "nosuid,nodev,noexec,size=4096,mode=000 "
                    f"tmpfs \"$1\"; {umount} \"$1\"",
                    "orze-boundary-probe", target,
                ],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LaunchIntegrityError(
            "data_boundary_kernel_probe_failed") from exc
    if completed.returncode != 0:
        raise LaunchIntegrityError("data_boundary_kernel_probe_failed")


def _build_isolated_cmd(base_cmd: list, forbidden_paths: list,
                        *, deny_network: bool = False) -> list:
    """Wrap `base_cmd` so it runs inside a private user+mount namespace with
    each forbidden path bind-mounted over by an empty dir. Any file read
    rooted at a forbidden path returns ENOENT at the kernel layer — no
    Python patches, no library-specific hooks.

    Every namespace or mount operation is mandatory. Any setup failure exits
    before the training script; there is no Python-only fallback for a hard
    boundary.
    """
    import shlex
    unshare = shutil.which("unshare")
    bash = shutil.which("bash")
    mount = shutil.which("mount")
    if None in (unshare, bash, mount):
        raise LaunchIntegrityError("data_boundary_kernel_tools_unavailable")

    q_mount = shlex.quote(mount)
    mount_lines = ["set -eu", f"{q_mount} --make-rprivate /"]
    for p in forbidden_paths:
        q_path = shlex.quote(p)
        mount_lines.append(
            f"if [ -L {q_path} ]; then exit 125; "
            f"elif [ -d {q_path} ]; then "
            f"{q_mount} -t tmpfs -o "
            f"nosuid,nodev,noexec,size=4096,mode=000 "
            f"tmpfs {q_path}; "
            f"elif [ -f {q_path} ]; then "
            f"{q_mount} --bind /dev/null {q_path}; "
            f"{q_mount} -o remount,bind,ro {q_path}; "
            f"else exit 125; fi"
        )
    mount_lines.extend([
        "export ORZE_KERNEL_BOUNDARY_ACTIVE=1",
        f"exec {shlex.join(base_cmd)}",
    ])
    args = [unshare, "-U", "--map-root-user", "-m"]
    if deny_network:
        args.append("-n")
    return args + [bash, "-c", "\n".join(mount_lines)]


def _format_args(args, template_vars: dict) -> list:
    """Safely format arguments without crashing on literal {} braces."""
    if args is None:
        args = []
    elif isinstance(args, str):
        args = [args]
    elif not isinstance(args, list):
        try:
            args = list(args)
        except TypeError:
            args = [args]
    formatted = []
    for arg in args:
        s = str(arg)
        for k, v in template_vars.items():
            s = s.replace(f"{{{k}}}", str(v))
        formatted.append(s)
    return formatted


def _tree_cpu_jiffies(root_pid: int) -> int:
    """Sum utime+stime (jiffies) across *root_pid* and all its descendants.

    A naked `accelerate launch` / `python -u` parent sleeps in epoll for
    most of the run — it's the child process that does the heavy data
    indexing, model load, and training steps. Reading the parent's
    /proc/pid/stat alone massively underestimates real work and yielded
    false positives from `_detect_zombie` during cold-cache dataset
    indexing (which legitimately shows 0 GPU activity for minutes).
    """
    total = 0
    stack = [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                parts = f.read().split()
            total += int(parts[13]) + int(parts[14])
        except (OSError, IndexError, ValueError):
            # Reading /proc for a transient pid is inherently racy: the process
            # can exit between being listed and stat'd, raising ProcessLookupError
            # (ESRCH) or FileNotFoundError (ENOENT) — both OSError. A vanished pid
            # contributes 0 jiffies; never let it propagate (it once killed run()).
            continue
        try:
            with open(f"/proc/{pid}/task/{pid}/children") as f:
                stack.extend(int(p) for p in f.read().split())
        except (FileNotFoundError, ValueError, OSError):
            pass
    return total


def _detect_zombie(tp) -> bool:
    """Check if a training process is stuck (alive but doing nothing).

    Returns True if the process has:
    - Near-zero CPU usage (parent AND all descendants summed)
    - No GPU memory usage (parent and children)
    - No log file growth
    All three must be true to avoid false positives.
    Requires 3 consecutive positive detections (~90s at 30s poll).
    """
    pid = tp.process.pid

    # 1. Check CPU usage across the process tree.
    try:
        total_cpu = _tree_cpu_jiffies(pid)

        prev = getattr(tp, '_zombie_cpu', None)
        tp._zombie_cpu = (time.time(), total_cpu)

        if prev is not None:
            dt = tp._zombie_cpu[0] - prev[0]
            dcpu = tp._zombie_cpu[1] - prev[1]
            if dt > 30 and dcpu > 10:  # any meaningful CPU in last 30s
                tp._zombie_count = 0
                return False
    except (OSError, IndexError, ValueError):
        return False  # can't check (incl. racy /proc ProcessLookupError) — assume alive

    # 2. Check GPU memory (nvidia-smi for this PID and children)
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={tp.gpu}",
             "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        nvsmi_lines = result.stdout.strip().splitlines()

        # Check parent PID
        for line in nvsmi_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and int(parts[0]) == pid:
                tp._zombie_count = 0
                return False  # process has GPU memory allocated

        # Check child process PIDs
        try:
            with open(f"/proc/{pid}/task/{pid}/children") as f:
                child_pids = [int(p) for p in f.read().split()]
            for cpid in child_pids:
                for line in nvsmi_lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2 and int(parts[0]) == cpid:
                        tp._zombie_count = 0
                        return False  # child has GPU memory
        except (FileNotFoundError, ValueError, OSError):
            pass  # can't read children, continue with other checks
    except Exception:
        pass  # nvidia-smi failed, check other signals

    # 3. Check log file growth
    if tp.log_path and tp.log_path.exists():
        try:
            current_size = tp.log_path.stat().st_size
            prev_size = getattr(tp, '_zombie_log_size', 0)
            tp._zombie_log_size = current_size
            if current_size > prev_size:
                tp._zombie_count = 0
                return False  # log is growing
        except OSError:
            pass

    # All checks failed — this process looks stuck.
    # Require 3 consecutive detections to avoid transient false positives.
    zombie_count = getattr(tp, '_zombie_count', 0) + 1
    tp._zombie_count = zombie_count
    return zombie_count >= 3  # 3 consecutive checks (~90s at 30s poll)


# --------------------------------------------------------------------- #
# F3: zombie/stuck-training watchdog                                     #
# --------------------------------------------------------------------- #

# Watchdog activates once the training shows a first-batch marker OR
# WATCHDOG_GRACE_MIN minutes pass since launch (whichever first). Then
# every poll it samples GPU util / log mtime / process-tree CPU time.
# After WATCHDOG_CONSECUTIVE consecutive samples where ALL THREE
# signals are stuck (GPU<5%, log mtime unchanged, CPU delta <1s),
# the process is killed and the idea is marked failed.
#
# Test override via env vars so unit tests don't have to wait minutes.
WATCHDOG_GRACE_MIN = int(os.environ.get("ORZE_WD_GRACE_MIN", "60"))
WATCHDOG_CONSECUTIVE = int(os.environ.get("ORZE_WD_CONSECUTIVE", "15"))
WATCHDOG_GPU_UTIL_THRESHOLD = int(os.environ.get("ORZE_WD_GPU_UTIL", "5"))
WATCHDOG_CPU_DELTA_JIFFIES = int(
    os.environ.get("ORZE_WD_CPU_DELTA_JIFFIES", "100"))  # ~1s @ HZ=100

_FIRST_BATCH_RE = __import__("re").compile(
    r"\bbatch\s+\d+\s*/\s*\d+|epoch\s+\d+|Epoch\s+\d+|step\s+\d+",
    __import__("re").IGNORECASE,
)


def _scan_first_batch_marker(log_path: Optional[Path]) -> bool:
    """True if the log file contains a batch/epoch/step marker."""
    if not log_path:
        return False
    try:
        if not log_path.exists():
            return False
    except OSError:
        return False
    try:
        # Tail the last 32KB — first-batch markers reappear regularly,
        # we don't need full-file scan and tail is bounded.
        text = tail_file(log_path, 32768)
    except Exception:
        return False
    return bool(_FIRST_BATCH_RE.search(text))


def _gpu_util_for_pid(pid: int, gpu: int) -> Optional[int]:
    """Return GPU util (0-100) for the GPU process *pid* runs on, or None
    if pid is not currently using a GPU / nvidia-smi is unavailable.

    Uses ``--query-compute-apps=pid,gpu_uuid`` to map pid → GPU, then
    ``--query-gpu=utilization.gpu,uuid`` for per-GPU util.
    """
    try:
        a = subprocess.run(
            ["nvidia-smi", f"--id={gpu}",
             "--query-compute-apps=pid,gpu_uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    gpu_uuid = None
    for line in a.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                if int(parts[0]) == pid:
                    gpu_uuid = parts[1]
                    break
            except ValueError:
                continue
    if gpu_uuid is None:
        return None
    try:
        b = subprocess.run(
            ["nvidia-smi", f"--id={gpu}",
             "--query-gpu=uuid,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    for line in b.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == gpu_uuid:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def _watchdog_check(tp) -> bool:
    """True once the training process has been stuck for
    WATCHDOG_CONSECUTIVE consecutive samples post-grace.
    Mutates ``tp`` to keep watchdog state.
    """
    now = time.time()
    elapsed_min = (now - tp.start_time) / 60.0

    # 1. First-batch detection (latched).
    if not getattr(tp, "_wd_first_batch", False):
        if _scan_first_batch_marker(tp.log_path):
            tp._wd_first_batch = True

    # 2. Activation gate.
    activated = (
        getattr(tp, "_wd_first_batch", False)
        or elapsed_min >= WATCHDOG_GRACE_MIN
    )
    if not activated:
        return False

    # 3. Sample all three signals.
    gpu_util = _gpu_util_for_pid(tp.process.pid, tp.gpu)
    try:
        log_mtime = tp.log_path.stat().st_mtime if tp.log_path else 0.0
    except OSError:
        log_mtime = 0.0
    try:
        cpu_jiffies = _tree_cpu_jiffies(tp.process.pid)
    except Exception:
        cpu_jiffies = 0

    prev = getattr(tp, "_wd_prev_sample", None)
    tp._wd_prev_sample = (now, log_mtime, cpu_jiffies, gpu_util)

    if prev is None:
        # Need a baseline sample first.
        tp._wd_bad_count = 0
        return False

    _, prev_mtime, prev_jiffies, _ = prev
    log_unchanged = log_mtime <= prev_mtime
    cpu_delta = cpu_jiffies - prev_jiffies
    cpu_idle = cpu_delta < WATCHDOG_CPU_DELTA_JIFFIES
    gpu_idle = (gpu_util is not None and gpu_util < WATCHDOG_GPU_UTIL_THRESHOLD)

    if log_unchanged and cpu_idle and gpu_idle:
        tp._wd_bad_count = getattr(tp, "_wd_bad_count", 0) + 1
    else:
        tp._wd_bad_count = 0

    return tp._wd_bad_count >= WATCHDOG_CONSECUTIVE


# --------------------------------------------------------------------- #
# F5: launch-time nested-config validator                                #
# --------------------------------------------------------------------- #

# Top-level dict values that ARE allowed in idea_config.yaml. Trainers
# explicitly accept these as nested sub-configs; everything else must be
# argparse-style scalar kwargs.
_NESTED_CONFIG_WHITELIST = {
    # Common ML training sub-configs that trainers accept as nested dicts.
    "ema",
    "augmentation",
    "augmentations",
    "data",
    "training",
    "model",
    # orze framework-managed nested blocks.
    "data_boundaries",   # orze framework-managed
    "executor_fix",      # orze framework-managed
    "report",            # orze framework-managed
    # NOTE: project/domain-specific nested keys (e.g. data_mix, per_dataset_*,
    # decoding-method blocks, augmentation pipelines) MUST be declared in the
    # project's orze.yaml `nested_config_whitelist` — they arrive here via the
    # `extra_whitelist` argument and are hot-reloaded
    # (orchestrator._HOT_RELOAD_KEYS includes "nested_config_whitelist"). The
    # engine must stay task-agnostic and never name a domain-specific key here.
}


def normalize_nested_config(
    idea_cfg: dict,
    normalization_map: Optional[dict] = None,
) -> tuple[dict, list[str]]:
    """Apply only project-declared repairs to known nested config shapes.

    Supported rules are ``flatten_prefix`` (``lora.rank`` → ``lora_rank``)
    and ``rename:<key>``. Explicit destination keys always win. Unknown rules
    are ignored so the subsequent validator still fails closed.
    """
    if not isinstance(idea_cfg, dict) or not normalization_map:
        return idea_cfg, []
    normalized = dict(idea_cfg)
    changes: list[str] = []
    for source, rule in normalization_map.items():
        value = normalized.get(source)
        if not isinstance(value, dict):
            continue
        if rule == "flatten_prefix":
            for child, child_value in value.items():
                destination = f"{source}_{child}"
                if destination not in normalized:
                    normalized[destination] = child_value
                    changes.append(f"{source}.{child} -> {destination}")
            normalized.pop(source, None)
        elif isinstance(rule, str) and rule.startswith("rename:"):
            destination = rule.split(":", 1)[1].strip()
            if not destination:
                continue
            existing = normalized.get(destination)
            if isinstance(existing, dict):
                merged = dict(value)
                merged.update(existing)
                normalized[destination] = merged
            elif destination not in normalized:
                normalized[destination] = value
            normalized.pop(source, None)
            changes.append(f"{source} -> {destination}")
    return normalized, changes


def validate_idea_config_no_nested(
    idea_cfg: dict,
    extra_whitelist: Optional[list] = None,
) -> Optional[str]:
    """F5: reject configs whose top-level values are nested dicts.

    Returns None if valid, otherwise an error message. Whitelisted keys
    (``ema``, ``augmentation``, ...) may have dict values. ``extra_whitelist``
    extends the default whitelist (configurable via cfg).
    """
    if not isinstance(idea_cfg, dict):
        return None
    allowed = set(_NESTED_CONFIG_WHITELIST)
    if extra_whitelist:
        allowed.update(str(k) for k in extra_whitelist)
    bad = []
    for k, v in idea_cfg.items():
        if isinstance(v, dict) and k not in allowed:
            bad.append(k)
    if not bad:
        return None
    return ("nested_config_not_allowed: top-level dict values for keys "
            + ", ".join(sorted(bad))
            + " — only argparse-style scalar kwargs allowed (whitelist: "
            + ", ".join(sorted(allowed)) + ")")


# --------------------------------------------------------------------- #
# F5b: launch-time method-validator enforcement (cycle-116)              #
# --------------------------------------------------------------------- #
# Reads results/_validators/*.yaml and rejects ideas that violate any
# error-severity rule. Supports the operator set documented in
# PROFESSOR_RULES.md plus the `field_any` (list of fields) and `field_sum`
# (numeric wildcard aggregate) extensions
# used by require_nontrivial_training_op_101.yaml. Operator set:
#   equals, not_equals, in, not_in, contains, not_contains,
#   exists, not_exists, gt, gte, lt, lte.
# (contains/not_contains = substring/membership test on the field value;
#  added cyc-4766 — previously absent, making all contains/not_contains
#  validator rules silent no-ops. See GOAL.md cyc-3989 flag.) Cycle-095 committed
# professor would land this directly if the engineer trigger pended
# >5 cycles; cycle-115 confirmed pending=6, so this is that landing.

def _eval_validator_rule(rule: dict, idea_cfg: dict,
                         _in_any_of: bool = False) -> Optional[str]:
    """Return None if rule passes, otherwise an error string.

    `_in_any_of` is set when evaluating a sub-clause of an any_of exemption
    list; it makes an ABSENT field fail a positive assertion instead of being
    skipped. See the comment at the value-comparison branch. Callers outside
    this module should not pass it.
    """
    if not isinstance(rule, dict):
        return None
    op = str(rule.get("operator", "")).lower()
    explanation = str(rule.get("explanation", "")).strip()

    # `any_of: [subrule, ...]` / `all_of: [subrule, ...]` combinators.
    # Added cyc-5869 (professor E12). Previously ABSENT: a combinator rule has
    # no top-level `field`, so it fell through to `if not field: return None`
    # and SILENTLY PASSED. That made every validator whose blocking logic lives
    # inside an any_of a launch-time no-op — 38 of them in the 1.7B project,
    # including no_gs_dose_above_data_ceiling_cyc829, the champion-replica
    # guards and the step/rank budget caps. First diagnosed cyc-4241; not landed
    # then because this submodule is shared. Re-checked cyc-5869: the sibling
    # project (/hot-data/fsx/erik/auto-research) has 24 validators and ZERO
    # use any_of/all_of, so this change is a no-op there.
    #
    # Semantics (matching _libs/pre_eval_gate.py, the pre-eval path that has
    # implemented this correctly since cyc-2017): a subrule returns None when it
    # PASSES and an error string when it FAILS.
    #   any_of  -> passes iff >= 1 subrule passes. This is the EXEMPTION pattern
    #              ("block X unless inference_only / eval_only / mirror-enabled").
    #   all_of  -> passes iff ALL subrules pass.
    # An empty/malformed list passes (fail-open, as everywhere else here).
    if "any_of" in rule:
        subs = rule.get("any_of") or []
        if not isinstance(subs, list) or not subs:
            return None
        errs = [_eval_validator_rule(s, idea_cfg, _in_any_of=True)
                for s in subs]
        if all(e is not None for e in errs):
            return explanation or ("none of any_of satisfied: "
                                   + "; ".join(e for e in errs if e))
        return None
    if "all_of" in rule:
        subs = rule.get("all_of") or []
        if not isinstance(subs, list):
            return None
        for s in subs:
            # Propagate: an all_of nested inside an any_of is still an
            # exemption clause ("exempt if ALL of these hold").
            e = _eval_validator_rule(s, idea_cfg, _in_any_of=_in_any_of)
            if e is not None:
                return explanation or e
        return None

    # `field_sum: training.datasets.*.samples` — sum numeric leaves selected
    # by a dotted path with `*`. This lets launch-time compute-budget guards
    # reject an oversized data mix before a trainer process spends GPU time.
    # If no numeric leaf matches, retry without a trailing key below `*` to
    # support both accepted dataset spellings:
    #   {ami: {samples: 1000}} and {ami: 1000}.
    if "field_sum" in rule:
        def _walk_sum(parts):
            current = [idea_cfg]
            for part in parts:
                next_values = []
                for value in current:
                    if not isinstance(value, dict):
                        continue
                    if part == "*":
                        next_values.extend(value.values())
                    elif part in value:
                        next_values.append(value[part])
                current = next_values
            return [
                value for value in current
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]

        parts = str(rule.get("field_sum", "")).split(".")
        values = _walk_sum(parts)
        if not values and len(parts) >= 2 and parts[-2] == "*":
            values = _walk_sum(parts[:-1])
        if not values:
            return None
        total = sum(values)
        try:
            expected = float(rule.get("value"))
        except (TypeError, ValueError):
            return None
        failed = (
            (op in ("lte", "le") and total > expected)
            or (op == "lt" and total >= expected)
            or (op in ("gte", "ge") and total < expected)
            or (op == "gt" and total <= expected)
        )
        if failed:
            return (explanation
                    or f"sum({rule.get('field_sum')})={total:g} "
                       f"violates {op} {expected:g}")
        return None

    # `field_any: [a, b, ...]` + `operator: exists` — pass iff at
    # least one of the listed fields is present (and non-null) in the
    # config. Used by require_nontrivial_training_op_101.
    if "field_any" in rule:
        fields = rule.get("field_any") or []
        if op in ("exists", "present", ""):
            present = [f for f in fields
                       if f in idea_cfg and idea_cfg.get(f) not in (None, "", [], {})]
            if not present:
                return (explanation
                        or f"none of required fields present: {fields}")
            return None
        if op in ("not_exists", "absent"):
            present = [f for f in fields
                       if f in idea_cfg and idea_cfg.get(f) not in (None, "", [], {})]
            if present:
                return (explanation
                        or f"forbidden fields present: {present}")
            return None
        return None  # unknown operator on field_any — be permissive

    field = rule.get("field")
    if not field:
        # Silent-pass tripwire (cyc-5869). A rule with no `field`, no
        # `field_any` and no any_of/all_of is one this engine cannot evaluate,
        # so it passes — which is exactly how the any_of no-op hid for 1000+
        # cycles. Fail-open is still the right default (a parser bug must never
        # starve the launcher), but it must be LOUD.
        logger.warning(
            "[VALIDATOR-UNKNOWN-RULE] rule has no field/field_any/any_of/all_of "
            "and was SILENTLY PASSED — it is enforcing nothing. keys=%s",
            sorted(rule.keys()))
        return None
    # Dot-notation traversal: "a.b.c" → idea_cfg["a"]["b"]["c"]. Without this,
    # any validator using nested-field paths (e.g. length_aware_decoding.enabled)
    # is a silent no-op — the literal key doesn't exist at top level so val=None
    # and the rule short-circuits via the absence-skip below. Confirmed root cause
    # of the cycle-253/254 decoder-kwargs bug: idea-055b25/064773/123800/ht-1/2/3
    # all bypassed block_length_aware_decoding_linear_244.yaml because of this.
    def _resolve_dotted(cfg, path):
        cur = cfg
        for part in str(path).split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None, False
        return cur, True

    if "." in str(field):
        val, present = _resolve_dotted(idea_cfg, field)
    else:
        present = field in idea_cfg
        val = idea_cfg.get(field)
    expected = rule.get("value")
    # For value-comparison operators, absence means "use champion
    # default" — rule doesn't apply. Only exists/not_exists care.
    #
    # EXCEPT inside an any_of exemption list (_in_any_of, cyc-5869). There the
    # clause means "you are EXEMPT IF <field> is <value>", and an absent field
    # cannot satisfy a POSITIVE assertion — you did not opt in. Skipping made
    # every such clause vacuously true, which kept the whole any_of passing and
    # left validators inert even after combinator support landed:
    # no_gs_dose_above_data_ceiling_cyc829 has `inference_only equals true` and
    # `training.gigaspeech_mirror equals true` as its exemptions, and ordinary
    # ideas carry neither key — so it never blocked a single over-ceiling dose.
    #
    # NEGATIVE operators (not_equals / not_in / not_contains) are the opposite
    # role: they are SCOPING GUARDS ("this seal does not apply to you"), and an
    # absent field genuinely does satisfy them — `el2n_select not_equals true`
    # is true for every idea that does no EL2N selection at all. Those stay
    # lenient. Measured cyc-5869 over the 300 most recent idea_configs: strict
    # on negatives too would newly block 151/300 (50%) including all three live
    # GigaSpeech ladder arms; positives-only blocks 54/300, entirely the
    # intended dead-axis + require-E22 families, and leaves the ladder passing.
    if op in ("equals", "not_equals", "in", "not_in",
              "contains", "not_contains",
              "gt", "gte", "lt", "lte"):
        if not present and _in_any_of and op in (
                "equals", "in", "contains", "gt", "gte", "lt", "lte"):
            return (explanation
                    or f"{field} absent — exemption clause not satisfied")
        if not present:
            return None
    if op == "equals":
        if val != expected:
            return explanation or f"{field}={val!r} != {expected!r}"
    elif op == "not_equals":
        if val == expected:
            return explanation or f"{field}={val!r} must not equal {expected!r}"
    elif op == "in":
        if val not in (expected or []):
            return explanation or f"{field}={val!r} not in {expected!r}"
    elif op == "not_in":
        if val in (expected or []):
            return explanation or f"{field}={val!r} must not be in {expected!r}"
    elif op == "contains":
        if expected not in (val or ""):
            return explanation or f"{field}={val!r} must contain {expected!r}"
    elif op == "not_contains":
        if expected in (val or ""):
            return explanation or f"{field}={val!r} must not contain {expected!r}"
    elif op == "exists":
        if not present or val in (None, "", [], {}):
            return explanation or f"{field} must be present"
    elif op == "not_exists":
        if present and val not in (None, "", [], {}):
            return explanation or f"{field} must be absent"
    elif op in ("gt", "gte", "lt", "lte"):
        try:
            v = float(val); e = float(expected)
            if op == "gt"  and not v >  e: return explanation or f"{field}={v} not > {e}"
            if op == "gte" and not v >= e: return explanation or f"{field}={v} not >= {e}"
            if op == "lt"  and not v <  e: return explanation or f"{field}={v} not < {e}"
            if op == "lte" and not v <= e: return explanation or f"{field}={v} not <= {e}"
        except (TypeError, ValueError):
            return None  # non-numeric — skip
    return None


def log_validator_rejection(
    results_dir: Path,
    idea_id: str,
    validator: str,
    rejection: str,
    idea_cfg: Optional[dict] = None,
    stage: str = "launch",
) -> None:
    """Append a rejection record to results/_validator_rejections.jsonl.

    Backedge for the research role (problem #5b in architecture audit):
    today validator rejections are one-way (logger.warning + skipped state)
    so research re-proposes the same idea family for many cycles silently.
    This file lets research read recent rejections at prompt-build time.
    Best-effort: never raises.
    """
    try:
        import datetime as _dt
        import json as _json
        keys = ("kind", "continuation_parent", "lora_path", "inference_only",
                "train_script", "approach_family", "method_spec", "dataset",
                "data_mix", "decoder_method")
        summary = {}
        if isinstance(idea_cfg, dict):
            for k in keys:
                if k in idea_cfg:
                    summary[k] = idea_cfg[k]
        rec = {
            "ts": _dt.datetime.utcnow().isoformat() + "Z",
            "idea_id": idea_id,
            "validator": validator,
            "rejection": rejection,
            "stage": stage,
            "config_summary": summary,
        }
        path = Path(results_dir) / "_validator_rejections.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(_json.dumps(rec) + "\n")
    except Exception:
        pass


def validate_idea_against_method_validators(
    idea_cfg: dict,
    validators_dir: Path,
) -> Optional[str]:
    """Return None if all error-severity validators pass, else an error string.

    Validators are .yaml files with schema {name, description, severity,
    rules: [...]}. Only severity=='error' rules block launch. WARN rules
    are not enforced here.
    """
    if not isinstance(idea_cfg, dict):
        return None
    try:
        import yaml
    except Exception:
        return None
    try:
        files = sorted(Path(validators_dir).glob("*.yaml"))
    except Exception:
        return None
    for vf in files:
        try:
            with open(vf) as f:
                spec = yaml.safe_load(f) or {}
        except Exception:
            continue
        if str(spec.get("severity", "")).lower() != "error":
            continue
        # unblock_when: self-disable validator when a named artifact exists.
        # Resolves the half-implemented spec (problem #3 in architecture audit):
        # validator yamls advertised this contract but the engine ignored it,
        # forcing manual .disabled renames once the unblocking method-spec
        # was published. Path is resolved relative to validators_dir's parent
        # (typically the results/ dir) so authors can write "results/_methods/..."
        # the same way they reference it in idea configs.
        uw = spec.get("unblock_when")
        if isinstance(uw, dict):
            ap = uw.get("artifact_exists")
            if isinstance(ap, str) and ap:
                try:
                    cands = [Path(ap)]
                    if not Path(ap).is_absolute():
                        cands.append(Path(validators_dir).parent.parent / ap)
                    if any(c.exists() for c in cands):
                        continue
                except Exception:
                    pass
        rules = spec.get("rules") or []
        if not isinstance(rules, list):
            continue
        _vname = str(spec.get("name", vf.stem))
        for rule in rules:
            # Inversion guard (infra tripwire). A block/pause/gate-family
            # validator whose rule REQUIRES a single method-specific field to
            # be PRESENT on every idea is almost always an authoring inversion
            # of `not_exists`: instead of blocking the one refuted method it
            # rejects the ENTIRE queue (every idea that does not wire that
            # field). This has repeatedly mass-blocked the queue and tripped
            # fix-escalation when an LLM author re-emits the validator with
            # `exists` instead of `not_exists`. Skip the rule and warn rather
            # than starve the launcher. `field_any` exists-rules are untouched
            # (they are an explicit "one of these must be set" contract).
            if (isinstance(rule, dict)
                    and _vname.startswith(("block_", "pause_", "gate_"))
                    and rule.get("field")
                    and str(rule.get("operator", "")).lower()
                        in ("exists", "present")):
                logger.warning(
                    "[VALIDATOR-INVERSION-GUARD] %s: skipping `%s exists` rule "
                    "(a block-family validator must use not_exists to target a "
                    "method; an exists rule mass-rejects the whole queue)",
                    _vname, rule.get("field"))
                continue
            err = _eval_validator_rule(rule, idea_cfg)
            if err:
                return f"validator[{_vname}]: {err}"
    return None


# --------------------------------------------------------------------- #
# F12: idea kind resolution + post-hoc launch                            #
# --------------------------------------------------------------------- #


def _resolve_idea_kind(idea_id: str, idea_cfg_path: Path,
                       results_dir: Path, cfg: dict) -> Optional[str]:
    """Return the idea's kind ('train' etc.), or None if unknown."""
    # 1) idea_config.yaml wins
    if idea_cfg_path.exists():
        try:
            import yaml
            with open(idea_cfg_path) as _f:
                obj = yaml.safe_load(_f) or {}
            if isinstance(obj, dict) and obj.get("kind"):
                return str(obj["kind"])
        except Exception:  # pragma: no cover
            pass
    # 2) idea_lake row fallback
    try:
        from orze.idea_lake import IdeaLake
        db_path = (cfg.get("idea_lake_db")
                   or Path(results_dir) / "idea_lake.db")
        if Path(db_path).exists():
            lake = IdeaLake(str(db_path))
            row = lake.get(idea_id)
            if row and row.get("kind"):
                return row["kind"]
    except Exception:  # pragma: no cover
        pass
    return None


def _launch_posthoc(idea_id: str, gpu: int, results_dir: Path, cfg: dict,
                    *, kind: str,
                    idea_cfg_path: Path) -> TrainingProcess:
    """Run a post-hoc idea in a subprocess and return a TrainingProcess-like
    handle so the rest of the scheduler (check_active etc.) is unchanged.
    """
    import yaml

    idea_dir = Path(results_dir) / idea_id
    idea_dir.mkdir(parents=True, exist_ok=True)
    log_path = idea_dir / "train_output.log"

    # Read the per-idea YAML so the subprocess has it as JSON on stdin.
    idea_cfg: Dict[str, object] = {}
    if idea_cfg_path.exists():
        try:
            with open(idea_cfg_path) as _f:
                idea_cfg = yaml.safe_load(_f) or {}
        except Exception:  # pragma: no cover
            idea_cfg = {}
    idea_cfg.setdefault("kind", kind)
    if not idea_cfg.get("adapter"):
        idea_cfg["adapter"] = cfg.get("posthoc_adapter") or "null"
    # Merge posthoc_defaults (project_root, solution_csv, python, etc.) —
    # idea-level cfg still wins.
    for _dk, _dv in (cfg.get("posthoc_defaults") or {}).items():
        idea_cfg.setdefault(_dk, _dv)
    artifact_db = (cfg.get("artifact_catalog_db")
                   or str(Path(results_dir) / "idea_lake_artifacts.db"))

    python = cfg.get("python", sys.executable)
    # Run via -c so we don't need a new module in the wire format.
    driver = (
        "import json, sys; "
        "from orze.engine.posthoc_runner import run_posthoc; "
        "cfg = json.loads(sys.stdin.read()); "
        f"run_posthoc('{idea_id}', cfg, "
        f"r'{idea_dir}', "
        f"artifact_catalog_db=r'{artifact_db}')"
    )
    cmd = [python, "-c", driver]

    env = os.environ.copy()
    if gpu is not None and int(gpu) >= 0:
        env = _authorized_gpu_environment(gpu, cfg, env)

    claim_path = idea_dir / "claim.json"
    claim_data = {}
    if claim_path.exists():
        try:
            claim_data = json.loads(claim_path.read_text(encoding="utf-8"))
            if not isinstance(claim_data, dict):
                raise ValueError("claim must be a mapping")
            if ("gpu" in claim_data and claim_data.get("gpu") != gpu):
                raise ValueError("claim GPU does not match launch GPU")
        except (json.JSONDecodeError, OSError, UnicodeDecodeError,
                ValueError) as exc:
            raise LaunchIntegrityError(
                "claim_receipt_invalid_or_mismatched") from exc

    if not verify_artifact_preflight_receipt(
            idea_id, results_dir, cfg):
        raise LaunchIntegrityError(
            "artifact_preflight_receipt_missing_or_stale")

    # Final sanity-check that the claimed GPU is still free at Popen
    # time (c1136). Raises GpuUnavailableError if not.
    _assert_controller_runtime_attested(cfg)
    with gpu_execution_lease(gpu, require_idle=True) as lease_fds:
        _verify_gpu_free(gpu, _launch_min_free_vram(cfg))

        log_fh = open(log_path, "a")
        log_fh.write(f"\n[posthoc_runner] kind={kind} gpu={gpu}\n")
        log_fh.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=log_fh, stderr=subprocess.STDOUT,
                env=env, preexec_fn=_new_process_group,
                pass_fds=lease_fds,
            )
        except Exception:
            log_fh.close()
            raise
    try:
        proc.stdin.write(json.dumps(idea_cfg).encode())
        proc.stdin.close()
    except Exception:  # pragma: no cover
        pass
    attempt_id = secrets.token_hex(16)
    if claim_data.get("attempt_id"):
        attempt_id = str(claim_data["attempt_id"])
    tp = TrainingProcess(
        idea_id=idea_id, gpu=gpu, process=proc,
        start_time=time.time(),
        log_path=log_path,
        timeout=float(cfg.get("posthoc_timeout", 3600)),
        attempt_id=attempt_id,
        _log_fh=log_fh,
    )
    tp.is_posthoc = True
    from orze.engine.accounting import record_compute_start
    try:
        record_compute_start(tp, idea_dir, phase="posthoc")
    except Exception:
        _terminate_and_reap(proc, idea_id, timeout=3)
        try:
            from orze.engine.accounting import record_compute_terminal
            record_compute_terminal(
                tp, idea_dir, "failed", "posthoc_launch_initialization_failed",
                phase="posthoc", return_code=proc.poll())
        except Exception:
            pass
        tp.close_log()
        raise
    return tp


class GpuUnavailableError(RuntimeError):
    """Raised when the claimed GPU lacks free VRAM at Popen time.

    Distinct from generic launch errors so phases.py can requeue the
    idea without invoking the executor-fix path — this is a resource
    issue, not a code bug.
    """


class LaunchIntegrityError(RuntimeError):
    """Raised when a launch attempts to bypass a control-plane invariant."""


class DuplicateLaunchError(LaunchIntegrityError):
    """Raised when exact-execution admission rejects a training launch."""


def find_forbidden_launch_override(value, path: str = "config",
                                   _seen: Optional[set] = None,
                                   _depth: int = 0) -> Optional[str]:
    """Return the first nested ``force_launch`` path, if present.

    The key is forbidden even when false: idea data must never carry a switch
    that changes which integrity validators the control plane executes.
    """
    if _depth > 64:
        return f"{path}.<nesting_limit_exceeded>"
    if _seen is None:
        _seen = set()
    if isinstance(value, (dict, list)):
        identity = id(value)
        if identity in _seen:
            return f"{path}.<recursive_reference>"
        _seen.add(identity)
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) == "force_launch":
                return child_path
            found = find_forbidden_launch_override(
                child, child_path, _seen, _depth + 1)
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = find_forbidden_launch_override(
                child, f"{path}[{index}]", _seen, _depth + 1)
            if found:
                return found
    if isinstance(value, (dict, list)):
        _seen.remove(id(value))
    return None


def _verify_gpu_free(gpu, min_free_mib: int) -> None:
    """Sanity-check that ``gpu`` has at least ``min_free_mib`` free VRAM
    immediately before spawning the subprocess.

    Closes c1136 / smart_dispatch class: the scheduler's 5s-cached VRAM
    view can permit a launch that the live GPU cannot host, leading to
    multiple shards landing on one GPU and OOM-crashing.

    When the check is enabled, unavailable telemetry fails closed: launching
    blind can violate reservations or turn one monitoring fault into an OOM
    retry storm. Set the threshold to zero only to disable this check explicitly.
    """
    if gpu is None or int(gpu) < 0 or min_free_mib <= 0:
        return
    try:
        from orze.engine.gpu_slots import _query_all_gpu_usage
        usage = _query_all_gpu_usage([int(gpu)])
    except Exception as exc:
        raise GpuUnavailableError(
            "GPU telemetry unavailable at launch time") from exc
    if not usage:
        raise GpuUnavailableError(
            "GPU telemetry unavailable at launch time")
    entry = usage.get(int(gpu))
    if entry is None:
        raise GpuUnavailableError(
            f"GPU {gpu} not visible to nvidia-smi at launch time"
        )
    used, total = entry
    free = total - used
    if free < min_free_mib:
        raise GpuUnavailableError(
            f"GPU {gpu} has {free} MiB free, need >= {min_free_mib} MiB "
            f"(used={used}, total={total})"
        )


def _resolve_pause_flag_path(cfg: dict, results_dir: Path) -> Path:
    """Resolve the single canonical pause-flag path.

    Precedence:
    1. ``launcher.paused_flag_path`` in orze.yaml (absolute or relative to
       ``results_dir``).
    2. ``<results_dir>/_launcher_paused.flag`` (default).

    Always returns an absolute, fully-resolved path so that cwd shifts cannot
    flip the result between cycles (root cause of c1135).
    """
    override = cfg.get("launcher", {}).get("paused_flag_path")
    results_path = Path(results_dir)
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = results_path / candidate
    else:
        candidate = results_path / "_launcher_paused.flag"
    return candidate.resolve()


def _is_launcher_paused(cfg: dict, results_dir: Path) -> bool:
    """Return True if queue consumption should be suspended this cycle.

    Sources, in order:
    - ``launcher.paused: true`` in orze.yaml (config-level kill switch)
    - presence of the canonical pause-flag file (see ``_resolve_pause_flag_path``)

    Every call emits ``[PAUSE_CHECK] path=<...> present=<bool> config_paused=<bool>``
    so operators can confirm which sentinel a daemon is watching.
    """
    launcher_cfg = cfg.get("launcher", {})
    if not isinstance(launcher_cfg, dict):
        logger.error("launcher config is not a mapping; pausing fail-closed")
        return True
    config_paused = bool(launcher_cfg.get("paused", False))
    flag_path = _resolve_pause_flag_path(cfg, results_dir)
    try:
        flag_path.lstat()
        flag_present = True
    except FileNotFoundError:
        flag_present = False
    except OSError:
        flag_present = True
    logger.info(
        "[PAUSE_CHECK] path=%s present=%s config_paused=%s",
        flag_path, flag_present, config_paused,
    )
    return config_paused or flag_present


def _assert_launch_authorized(idea_id: str, results_dir: Path,
                              cfg: dict) -> None:
    """Enforce stop/pause controls for every direct launcher caller."""
    if (not isinstance(idea_id, str) or not idea_id
            or Path(idea_id).parts != (idea_id,)
            or idea_id in (".", "..")):
        raise LaunchIntegrityError("idea_id_invalid")
    results_dir = Path(results_dir)
    for sentinel in (".orze_disabled", ".orze_stop_all", ".orze_shutdown"):
        try:
            (results_dir / sentinel).lstat()
            present = True
        except FileNotFoundError:
            present = False
        except OSError:
            present = True
        if present:
            raise LaunchIntegrityError(
                f"launch_blocked_by_sentinel:{sentinel}")
    if _is_launcher_paused(cfg, results_dir):
        raise LaunchIntegrityError("launch_blocked_by_pause_policy")


def _assert_gpu_authorized(gpu: int, cfg: dict) -> None:
    """Require a valid, managed, non-reserved physical GPU ID."""
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise LaunchIntegrityError("gpu_id_invalid")
    scheduling = cfg.get("gpu_scheduling", {})
    if not isinstance(scheduling, dict):
        raise LaunchIntegrityError("gpu_scheduling_policy_invalid")

    def validated_ids(raw, label: str) -> Optional[set]:
        if raw is None:
            return None
        if (not isinstance(raw, list)
                or any(isinstance(value, bool)
                       or not isinstance(value, int)
                       or value < 0 for value in raw)
                or len(raw) != len(set(raw))):
            raise LaunchIntegrityError(f"{label}_invalid")
        return set(raw)

    managed = cfg.get("_managed_gpu_ids")
    configured = scheduling.get("allowed_gpus")
    managed_ids = validated_ids(managed, "managed_gpu_scope")
    configured_ids = validated_ids(configured, "gpu_allowlist")
    if ((managed_ids is not None and gpu not in managed_ids)
            or (configured_ids and gpu not in configured_ids)):
        raise LaunchIntegrityError(
            f"gpu_outside_managed_scope:{gpu}")
    reserved_ids = validated_ids(
        scheduling.get("reserved_gpus") or [], "reserved_gpu_list")
    if gpu in reserved_ids:
        raise LaunchIntegrityError(f"gpu_is_reserved:{gpu}")


def _authorized_gpu_environment(
    gpu: int,
    cfg: dict,
    base_env: Optional[dict] = None,
) -> dict:
    """Return a child environment exposing exactly one authorized GPU."""
    _assert_gpu_authorized(gpu, cfg)
    env = dict(os.environ if base_env is None else base_env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _launch_min_free_vram(cfg: dict) -> int:
    scheduling = cfg.get("gpu_scheduling") or {}
    return int(cfg.get(
        "launcher_min_free_vram_mib",
        scheduling.get("min_free_vram_mib", 1000),
    ))


def _assert_controller_runtime_attested(cfg: dict) -> None:
    """Re-attest an opt-in controller pin immediately before GPU telemetry."""
    from orze.service.runtime_contract import (
        RuntimeContractError,
        require_controller_runtime_contract,
    )
    try:
        require_controller_runtime_contract(cfg.get("controller_runtime"))
    except RuntimeContractError as exc:
        raise LaunchIntegrityError(str(exc)) from exc


def launch(idea_id: str, gpu: int, results_dir: Path, cfg: dict, lake=None) -> TrainingProcess:
    """Launch a training subprocess on the given GPU.

    F12: If the idea's YAML specifies ``kind`` other than 'train' (or the
    idea_lake row has such a kind), dispatch to posthoc_runner instead of
    the training script. The 'train' path below is preserved byte-exact
    for back-compat.

    Args:
        lake: IdeaLake instance for FSM transition recording (optional)
    """
    results_dir = Path(results_dir)
    _assert_launch_authorized(idea_id, results_dir, cfg)
    from orze.core.decision_batches import validate_idea_decision_admission
    decision_error = validate_idea_decision_admission(
        results_dir, cfg, idea_id)
    if decision_error:
        raise LaunchIntegrityError(decision_error)
    _assert_gpu_authorized(gpu, cfg)
    log_path = results_dir / idea_id / "train_output.log"

    # F12: detect non-train ideas and dispatch to posthoc_runner.
    idea_cfg_path = results_dir / idea_id / "idea_config.yaml"

    # Queue revalidation (professor cycle 140): re-run method validators at
    # launch time. Validators added/strengthened after enqueue must reject
    # orphan ideas before they consume a GPU. Train-kind only; posthoc has
    # its own schema and these validators don't apply.
    if idea_cfg_path.exists():
        try:
            import yaml as _yaml
            with open(idea_cfg_path) as _qrf:
                _qr_idea_cfg = _yaml.safe_load(_qrf) or {}
            if not isinstance(_qr_idea_cfg, dict):
                raise LaunchIntegrityError("idea_config_must_be_mapping")
            forbidden_path = find_forbidden_launch_override(_qr_idea_cfg)
            if forbidden_path:
                raise LaunchIntegrityError(
                    f"forbidden_launch_override:{forbidden_path}")
            approach_family = None
            if lake is not None:
                lake_row = lake.get(idea_id)
                if isinstance(lake_row, dict):
                    approach_family = lake_row.get("approach_family")
            policy_error = validate_idea_against_research_policy(
                _qr_idea_cfg, cfg, approach_family=approach_family)
            if policy_error:
                raise LaunchIntegrityError(policy_error)
            _validators_dir = Path(results_dir) / "_validators"
            if _validators_dir.is_dir():
                _qr_err = validate_idea_against_method_validators(
                    _qr_idea_cfg, _validators_dir)
                if _qr_err:
                    _qr_mark = results_dir / idea_id / "_schema_invalid.txt"
                    try:
                        _qr_mark.parent.mkdir(parents=True, exist_ok=True)
                        _qr_mark.write_text(
                            f"queue_revalidation: {_qr_err}\n")
                    except OSError:
                        pass
                    logger.warning(
                        "QUEUE-REVALIDATION REJECTED idea=%s: %s",
                        idea_id, _qr_err)
                    raise RuntimeError(f"queue_revalidation_{_qr_err}")
        except RuntimeError:
            raise
        except Exception as _qr_e:
            raise LaunchIntegrityError(
                "idea_config_validation_failed:"
                f"{type(_qr_e).__name__}") from _qr_e

    idea_kind = _resolve_idea_kind(idea_id, idea_cfg_path, results_dir, cfg)
    if idea_kind and idea_kind != "train":
        return _launch_posthoc(idea_id, gpu, results_dir, cfg,
                               kind=idea_kind,
                               idea_cfg_path=idea_cfg_path)

    python = cfg.get("python", sys.executable)
    train_script = cfg["train_script"]

    # Per-idea train_script override: read from idea_config.yaml if present
    if idea_cfg_path.exists():
        try:
            import yaml
            with open(idea_cfg_path) as _f:
                idea_cfg = yaml.safe_load(_f) or {}
            if idea_cfg.get("train_script"):
                train_script = _resolve_train_script(
                    idea_cfg["train_script"], cfg)
                logger.info("Per-idea train_script override: %s -> %s",
                            idea_id, train_script)
        except Exception:
            pass  # fall back to global train_script

    # Data boundary guardrails. Hard path blocks and network denial require a
    # verified user/mount namespace and never degrade to Python-only hooks.
    # The builtins.open wrapper remains defense in depth and provides the
    # explicitly audit-only watch mode.
    db_cfg = cfg.get("data_boundaries", {})
    forbidden, watch, training_network = _validated_data_boundary_policy(
        db_cfg)
    kernel_boundary = bool(forbidden or training_network == "deny")
    use_wrapper = bool(kernel_boundary or watch)

    # Use per-idea config if it exists, otherwise global base config
    config_path = cfg["base_config"]
    if idea_cfg_path.exists():
        config_path = str(idea_cfg_path)

    if use_wrapper:
        base_cmd = [
            python, "-m", "orze.data_boundaries.wrap", train_script,
            "--idea-id", idea_id,
            "--results-dir", str(results_dir),
            "--ideas-md", cfg["ideas_file"],
            "--config", config_path,
        ]
    else:
        base_cmd = [
            python, train_script,
            "--idea-id", idea_id,
            "--results-dir", str(results_dir),
            "--ideas-md", cfg["ideas_file"],
            "--config", config_path,
        ]
    for arg in (cfg.get("train_extra_args") or []):
        base_cmd.append(str(arg))

    # A resume request is explicit and hash-bound. Revalidate it before the
    # final GPU availability check so corrupted or changed evidence consumes
    # no accelerator time.
    resume_context = prepare_resume_launch(idea_id, results_dir, cfg)
    if resume_context:
        base_cmd.extend(resume_context["args"])

    cmd = base_cmd

    env = os.environ.copy()
    for k, v in (cfg.get("train_extra_env") or {}).items():
        env[k] = str(v)
    env = _authorized_gpu_environment(gpu, cfg, env)
    if use_wrapper:
        _apply_data_boundary_env(env, db_cfg, results_dir / idea_id)

    claim_path = results_dir / idea_id / "claim.json"
    stored_claim = {}
    if claim_path.exists():
        try:
            stored_claim = json.loads(claim_path.read_text(encoding="utf-8"))
            if not isinstance(stored_claim, dict):
                raise ValueError("claim must be a mapping")
            if ("gpu" in stored_claim
                    and stored_claim.get("gpu") != gpu):
                raise ValueError("claim GPU does not match launch GPU")
        except (json.JSONDecodeError, OSError, UnicodeDecodeError,
                ValueError) as exc:
            raise LaunchIntegrityError(
                "claim_receipt_invalid_or_mismatched") from exc

    if lake is not None and not claim_path.exists():
        raise RuntimeError(
            f"Lifecycle-managed launch requires an existing claim for {idea_id}"
        )

    if not verify_artifact_preflight_receipt(
            idea_id, results_dir, cfg):
        raise LaunchIntegrityError(
            "artifact_preflight_receipt_missing_or_stale")
    from orze.core.data_separation import (
        DataSeparationError, ensure_data_separation,
    )
    try:
        separation_receipt = ensure_data_separation(cfg)
    except DataSeparationError as exc:
        raise LaunchIntegrityError(str(exc)) from exc
    if kernel_boundary:
        _probe_kernel_boundary(deny_network=training_network == "deny")
        # The actual namespace setup repeats every mandatory operation after
        # the zero-GPU capability probe. A race or host-policy change exits
        # before user training code rather than weakening the boundary.
        cmd = _build_isolated_cmd(
            base_cmd,
            forbidden,
            deny_network=training_network == "deny",
        )

    attempt_id = secrets.token_hex(16)
    stored_attempt = stored_claim.get("attempt_id")
    if stored_attempt:
        attempt_id = str(stored_attempt)

    try:
        execution_identity = compute_execution_identity(
            config_path=Path(config_path),
            base_config_path=Path(cfg["base_config"]),
            train_script=Path(train_script),
            python=str(python),
            train_extra_args=list(cfg.get("train_extra_args") or []),
            train_extra_env=dict(cfg.get("train_extra_env") or {}),
            data_boundaries=dict(db_cfg),
            data_separation=dict(cfg.get("data_separation") or {}),
        )
        reserve_execution_identity(
            results_dir, cfg, execution_identity, idea_id, attempt_id)
    except DuplicateExecutionError as exc:
        raise DuplicateLaunchError(str(exc)) from exc

    from orze.core.model_lineage import (
        ModelLineageError, close_model_lineage_attestation,
        prepare_model_lineage_launch, receive_model_lineage_attestation,
    )
    lineage_context = None
    try:
        lineage_context = prepare_model_lineage_launch(
            idea_id=idea_id,
            attempt_id=attempt_id,
            execution_identity=execution_identity,
            idea_dir=results_dir / idea_id,
            cfg=cfg,
            separation_receipt=separation_receipt,
        )
        if lineage_context:
            env.update(lineage_context["env"])
    except ModelLineageError as exc:
        release_execution_identity(
            results_dir, cfg, execution_identity, idea_id, attempt_id)
        raise LaunchIntegrityError(str(exc)) from exc

    # Final sanity-check that the claimed GPU is still free at Popen
    # time (c1136). Raises GpuUnavailableError if not — handled in
    # phases.py as a requeue, not a code-fix retry.
    try:
        _assert_controller_runtime_attested(cfg)
        with gpu_execution_lease(gpu, require_idle=True) as lease_fds:
            _verify_gpu_free(gpu, _launch_min_free_vram(cfg))

            # Keep the log open for the subprocess lifetime.  The GPU lease
            # descriptor is inherited alongside any lineage attestation FD.
            log_fh = open(log_path, "w", encoding="utf-8")
            pass_fds = list(lease_fds)
            if lineage_context:
                pass_fds.append(lineage_context["write_fd"])
            proc = subprocess.Popen(
                cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
                preexec_fn=_new_process_group,
                pass_fds=tuple(pass_fds),
            )
    except Exception:
        close_model_lineage_attestation(lineage_context)
        if "log_fh" in locals():
            log_fh.close()
        release_execution_identity(
            results_dir, cfg, execution_identity, idea_id, attempt_id)
        raise

    now = time.time()
    tp = TrainingProcess(
        idea_id=idea_id, gpu=gpu, process=proc,
        start_time=now, log_path=log_path,
        timeout=cfg.get("timeout", 3600),
        train_script=str(train_script), config_path=str(config_path),
        attempt_id=attempt_id, execution_identity=execution_identity,
        _log_fh=log_fh, _last_log_size=0,
        _last_log_check=now, _stall_since=0.0,
    )

    # Persist the actual trainer identity before advertising IN_PROGRESS.
    # The trainer deliberately owns a separate process group, so recording
    # only the orchestrator PID makes crash recovery unsafe (it can relaunch
    # while the original trainer still writes effects).
    try:
        receive_model_lineage_attestation(
            lineage_context, process_pid=proc.pid)
        if claim_path.exists():
            identity = capture_process_identity(proc.pid)
            claim_data = json.loads(claim_path.read_text(encoding="utf-8"))
            claim_data.update({
                "attempt_id": attempt_id,
                "trainer_pid": identity["pid"],
                "trainer_pgid": identity["pgid"],
                "trainer_start_ticks": identity["start_ticks"],
                "trainer_started_at": now,
            })
            atomic_write(claim_path, json.dumps(claim_data, indent=2))

        from orze.engine.accounting import record_compute_start
        record_compute_start(tp, results_dir / idea_id, phase="training")

        # Record FSM transition: CLAIMED → IN_PROGRESS using the process
        # whose effects recovery must mediate.
        if lake and not lake.record_state_transition(
                idea_id,
                from_state="CLAIMED",
                to_state="IN_PROGRESS",
                reason=f"training_launched on gpu {gpu}",
                host=socket.gethostname(),
                pid=proc.pid,
                sop_type=cfg.get("sop", "training"),
        ):
            raise RuntimeError(
                f"Could not persist IN_PROGRESS transition for {idea_id}"
            )
        if resume_context:
            mark_resume_launched(resume_context, claim_path)
    except Exception:
        close_model_lineage_attestation(lineage_context)
        _terminate_and_reap(proc, idea_id, timeout=3)
        try:
            from orze.engine.accounting import record_compute_terminal
            record_compute_terminal(
                tp, results_dir / idea_id, "failed",
                "training_launch_initialization_failed",
                phase="training", return_code=proc.poll())
        except Exception:
            pass
        log_fh.close()
        raise

    return tp


def _write_failure(idea_dir: Path, reason: str, lake=None, idea_id=None, cfg=None):
    """Write a failure metrics.json atomically and record FSM transition."""
    metrics = {
        "status": "FAILED",
        "error": reason,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    atomic_write(idea_dir / "metrics.json", json.dumps(metrics, indent=2))

    # Record FSM transition: current_state → FAILED (v4.5: determine actual state)
    if lake and idea_id:
        try:
            # Query the actual current state from the FSM
            current_state = lake.get_fsm_state(idea_id)
            if current_state:
                lake.record_state_transition(
                    idea_id,
                    from_state=current_state,
                    to_state="FAILED",
                    reason=reason,
                    host=socket.gethostname(),
                    pid=os.getpid(),
                    sop_type=(cfg or {}).get("sop", "training"),
                )
            else:
                # No state found, default to IN_PROGRESS for backward compatibility
                lake.record_state_transition(
                    idea_id,
                    from_state="IN_PROGRESS",
                    to_state="FAILED",
                    reason=reason,
                    host=socket.gethostname(),
                    pid=os.getpid(),
                    sop_type=(cfg or {}).get("sop", "training"),
                )
        except Exception as e:
            logger.warning("FSM transition failed (non-blocking): %s", e)


def _terminate_training(tp: TrainingProcess, results_dir: Path, cfg: dict,
                        reason: str) -> None:
    """Terminate one trainer and persist an auditable interruption receipt."""
    _terminate_and_reap(tp.process, tp.idea_id)
    tp.close_log()
    try:
        write_interruption_receipt(
            tp, results_dir, cfg, reason=reason,
            terminating_signal="SIGTERM", return_code=tp.process.poll(),
        )
    except Exception as exc:
        logger.warning(
            "Could not persist interruption receipt for %s: %s",
            tp.idea_id, type(exc).__name__,
        )


def check_active(active: Dict[int, TrainingProcess], results_dir: Path,
                 cfg: dict, failure_counts: dict,
                 fix_counts: Optional[dict] = None, lake=None) -> list:
    """Check running processes. Reap completed/timed-out/stalled/OOM.
    Returns list of (idea_id, gpu) tuples for finished ideas.

    When fix_counts is provided and max_fix_attempts > 0, failed ideas
    are sent to the executor LLM for diagnosis before recording failure.
    If the LLM applies a fix, the idea is re-launched on the same GPU.

    Args:
        lake: IdeaLake instance for FSM transition recording (optional)
    """
    from orze.engine.health import check_stalled, detect_fatal_in_log, _adaptive_stall_minutes
    from orze.engine.failure import _record_failure, _try_executor_fix, _reset_idea_for_retry
    from orze.engine.failure_analysis import classify_failure, write_failure_analysis as _write_fa_orig
    from orze.engine.accounting import record_compute_terminal

    def account_terminal(tp, outcome: str, reason_code: str,
                         return_code: Optional[int]) -> None:
        phase = "posthoc" if getattr(tp, "is_posthoc", False) else "training"
        record_compute_terminal(
            tp,
            results_dir / tp.idea_id,
            outcome,
            reason_code,
            phase=phase,
            return_code=return_code,
        )

    # Wrap write_failure_analysis to also run SOP feedback (orze-pro)
    def write_failure_analysis(idea_dir, category, error_msg):
        _write_fa_orig(idea_dir, category, error_msg)
        if cfg.get("sops", {}).get("failure_feedback", True):
            try:
                from orze.extensions import get_extension
                _sops = get_extension("sops")
                if _sops:
                    _sops.analyze_failure_feedback(idea_dir, results_dir, cfg)
            except Exception:
                pass

    finished = []
    stall_minutes = _adaptive_stall_minutes(
        results_dir, cfg.get("stall_minutes", 0))
    if fix_counts is None:
        fix_counts = {}

    for gpu in list(active.keys()):
        tp = active[gpu]
        # With multi-slot, gpu is a slot key like "0:42". Use tp.gpu for actual GPU ID.
        actual_gpu = tp.gpu if hasattr(tp, 'gpu') else gpu
        ret = tp.process.poll()
        elapsed = time.time() - tp.start_time

        # --- Still running ---
        if ret is None:
            if elapsed > tp.timeout:
                logger.warning("[TIMEOUT] %s after %.0fm — killing",
                               tp.idea_id, elapsed / 60)
                notify("stall", {"idea_id": tp.idea_id, "gpu": gpu,
                                 "reason": f"Timeout after {elapsed / 60:.0f}m"}, cfg)
                _terminate_training(tp, results_dir, cfg, "timeout")
                error_msg = "Timed out"
                if _try_executor_fix(tp.idea_id, error_msg,
                                     results_dir, cfg, fix_counts):
                    _reset_idea_for_retry(results_dir / tp.idea_id)
                    try:
                        new_tp = launch(tp.idea_id, actual_gpu, results_dir, cfg)
                        active[gpu] = new_tp
                        logger.info("[FIX-RETRY] %s relaunched on GPU %s",
                                     tp.idea_id, gpu)
                        continue
                    except Exception as e:
                        logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                      tp.idea_id, e)
                _write_failure(results_dir / tp.idea_id, error_msg, lake=lake, idea_id=tp.idea_id, cfg=cfg)
                write_failure_analysis(results_dir / tp.idea_id, classify_failure(error_msg, -1, "training"), error_msg)
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append((tp.idea_id, gpu))
                continue

            if check_stalled(tp, stall_minutes):
                logger.warning("[STALLED] %s — no log output for %dm, killing",
                               tp.idea_id, stall_minutes)
                notify("stall", {"idea_id": tp.idea_id, "gpu": gpu,
                                 "reason": f"Stalled ({stall_minutes}m no output)"}, cfg)
                _terminate_training(tp, results_dir, cfg, "stall")
                error_msg = f"Stalled (no output for {stall_minutes}m)"
                if _try_executor_fix(tp.idea_id, error_msg,
                                     results_dir, cfg, fix_counts):
                    _reset_idea_for_retry(results_dir / tp.idea_id)
                    try:
                        new_tp = launch(tp.idea_id, actual_gpu, results_dir, cfg)
                        active[gpu] = new_tp
                        logger.info("[FIX-RETRY] %s relaunched on GPU %s",
                                     tp.idea_id, gpu)
                        continue
                    except Exception as e:
                        logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                      tp.idea_id, e)
                _write_failure(results_dir / tp.idea_id, error_msg, lake=lake, idea_id=tp.idea_id, cfg=cfg)
                write_failure_analysis(results_dir / tp.idea_id, classify_failure(error_msg, -1, "training"), error_msg)
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append((tp.idea_id, gpu))
                continue

            # --- Zombie detection: process alive but not using resources ---
            if ret is None and elapsed > 120:
                is_zombie = _detect_zombie(tp)
                if is_zombie:
                    logger.warning("[ZOMBIE] %s — alive for %.0fs but no CPU/GPU activity, killing",
                                   tp.idea_id, elapsed)
                    notify("stall", {"idea_id": tp.idea_id, "gpu": gpu,
                                     "reason": "Zombie process (no CPU/GPU activity)"}, cfg)
                    _terminate_training(tp, results_dir, cfg, "zombie")
                    error_msg = "Process stuck (zombie: no CPU/GPU activity)"
                    if _try_executor_fix(tp.idea_id, error_msg,
                                         results_dir, cfg, fix_counts):
                        _reset_idea_for_retry(results_dir / tp.idea_id)
                        try:
                            new_tp = launch(tp.idea_id, actual_gpu, results_dir, cfg)
                            active[gpu] = new_tp
                            logger.info("[FIX-RETRY] %s relaunched on GPU %s",
                                         tp.idea_id, gpu)
                            continue
                        except Exception as e:
                            logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                          tp.idea_id, e)
                    _write_failure(results_dir / tp.idea_id, error_msg, lake=lake, idea_id=tp.idea_id, cfg=cfg)
                    write_failure_analysis(results_dir / tp.idea_id, classify_failure(error_msg, -1, "training"), error_msg)
                    _record_failure(failure_counts, tp.idea_id)
                    del active[gpu]
                    finished.append((tp.idea_id, gpu))
                    continue

            # --- F3: triple-signal watchdog (post-grace) ---
            if ret is None and _watchdog_check(tp):
                logger.warning(
                    "[WATCHDOG] %s — stuck (no GPU/log/CPU progress for "
                    "%d consecutive samples), killing",
                    tp.idea_id, WATCHDOG_CONSECUTIVE)
                notify("stall", {"idea_id": tp.idea_id, "gpu": gpu,
                                 "reason": "Watchdog: stuck_no_progress"}, cfg)
                _terminate_training(tp, results_dir, cfg, "watchdog")
                error_msg = "stuck_no_progress"
                _write_failure(results_dir / tp.idea_id, error_msg, lake=lake, idea_id=tp.idea_id, cfg=cfg)
                write_failure_analysis(
                    results_dir / tp.idea_id,
                    classify_failure(error_msg, -1, "training"),
                    error_msg)
                _record_failure(failure_counts, tp.idea_id)
                # Mark in idea_lake too (best-effort).
                try:
                    from orze.engine.failure import _mark_lake_failure
                    _mark_lake_failure(
                        tp.idea_id, cfg, results_dir, "stuck_no_progress")
                except Exception:
                    pass
                del active[gpu]
                finished.append((tp.idea_id, gpu))
                continue

            fatal = detect_fatal_in_log(tp)
            if fatal and tp.process.poll() is None:
                logger.warning("[FATAL-HUNG] %s — fatal error in log but "
                               "process still alive, killing:\n%s",
                               tp.idea_id, fatal[:200])
                notify("stall", {"idea_id": tp.idea_id, "gpu": gpu,
                                 "reason": f"Fatal error (hung): {fatal[:100]}"}, cfg)
                _terminate_training(tp, results_dir, cfg, "fatal_log")
                error_msg = f"Process hung after fatal error:\n{fatal[:500]}"
                if _try_executor_fix(tp.idea_id, error_msg,
                                     results_dir, cfg, fix_counts):
                    _reset_idea_for_retry(results_dir / tp.idea_id)
                    try:
                        new_tp = launch(tp.idea_id, actual_gpu, results_dir, cfg)
                        active[gpu] = new_tp
                        logger.info("[FIX-RETRY] %s relaunched on GPU %s",
                                     tp.idea_id, gpu)
                        continue
                    except Exception as e:
                        logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                      tp.idea_id, e)
                _write_failure(results_dir / tp.idea_id, error_msg, lake=lake, idea_id=tp.idea_id, cfg=cfg)
                write_failure_analysis(results_dir / tp.idea_id, classify_failure(error_msg, -1, "training"), error_msg)
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append((tp.idea_id, gpu))
                continue

            kill_file = results_dir / tp.idea_id / ".kill"
            if kill_file.exists():
                logger.info("Admin kill signal for %s — terminating", tp.idea_id)
                _terminate_training(tp, results_dir, cfg, "admin_kill")
                kill_file.unlink(missing_ok=True)
                _write_failure(results_dir / tp.idea_id, "Killed by admin", lake=lake, idea_id=tp.idea_id, cfg=cfg)
                write_failure_analysis(results_dir / tp.idea_id, "crash", "Killed by admin")
                del active[gpu]
                finished.append((tp.idea_id, gpu))

            continue

        # --- Process exited ---
        # Reap zombie to prevent accumulation
        try:
            tp.process.wait(timeout=1)
        except Exception:
            pass
        tp.close_log()
        metrics_path = results_dir / tp.idea_id / "metrics.json"

        if ret == 0 and metrics_path.exists():
            metrics_error = ""
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if not isinstance(metrics, dict):
                    metrics_error = "metrics.json must contain a JSON object"
                    metrics = {}
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                metrics_error = f"metrics.json is unreadable: {exc}"
                metrics = {}
            status = metrics.get("status")
            if status not in ("COMPLETED", "FAILED"):
                metrics_error = metrics_error or (
                    "metrics.json must declare status COMPLETED or FAILED"
                )
                status = "INVALID"
            logger.info("[%s] %s on GPU %s in %.1fm",
                        status, tp.idea_id, gpu, elapsed / 60)

            # Validate metric consistency + anomaly detection
            primary = (cfg.get("report") or {}).get("primary_metric", "")
            if primary:
                from orze.engine.guardrails import validate_avg_metric, check_identical_results
                metric_warning = validate_avg_metric(metrics, primary)
                if metric_warning:
                    logger.warning("[METRIC] %s: %s", tp.idea_id, metric_warning)
                _recent_completions.append({"idea_id": tp.idea_id, "metrics": metrics})
                if len(_recent_completions) > 20:
                    _recent_completions.pop(0)
                anomaly = check_identical_results(_recent_completions, primary)
                if anomaly:
                    logger.warning("[ANOMALY] %s", anomaly)
            if metrics_error:
                account_terminal(tp, "failed", "metrics_invalid", ret)
                invalid_path = metrics_path.with_name(
                    f"metrics.invalid.{time.time_ns()}.json")
                try:
                    os.replace(metrics_path, invalid_path)
                except OSError:
                    invalid_path = None
                reason = metrics_error
                if invalid_path is not None:
                    reason += f"; original preserved as {invalid_path.name}"
                _write_failure(
                    results_dir / tp.idea_id, reason,
                    lake=lake, idea_id=tp.idea_id, cfg=cfg,
                )
                write_failure_analysis(
                    results_dir / tp.idea_id,
                    classify_failure(reason, 0, "training"), reason,
                )
                _record_failure(failure_counts, tp.idea_id)
            elif status == "FAILED":
                error_msg = metrics.get("error", "Training script reported FAILED")
                # VRAM precheck failure is environmental (GPU busy), not a code bug.
                # Re-queue so the idea retries when VRAM is available; don't burn fix budget.
                if str(error_msg).startswith("insufficient_vram:"):
                    account_terminal(
                        tp, "requeued", "trainer_vram_precheck", ret)
                    logger.warning("[VRAM-CONTENTION] %s — %s — re-queuing",
                                   tp.idea_id, str(error_msg)[:100])
                    _reset_idea_for_retry(
                        results_dir / tp.idea_id, release_claim=True)
                    if lake is not None:
                        try:
                            if not lake.set_status(tp.idea_id, "queued"):
                                logger.error(
                                    "[VRAM-CONTENTION] audited requeue rejected "
                                    "for %s", tp.idea_id)
                        except Exception as exc:
                            logger.error(
                                "[VRAM-CONTENTION] audited requeue failed for "
                                "%s: %s", tp.idea_id, exc)
                elif _try_executor_fix(tp.idea_id, error_msg,
                                       results_dir, cfg, fix_counts,
                                       exit_code=ret if ret is not None else -1):
                    account_terminal(
                        tp, "failed", "trainer_declared_failed", ret)
                    _reset_idea_for_retry(results_dir / tp.idea_id)
                    try:
                        new_tp = launch(tp.idea_id, actual_gpu, results_dir, cfg)
                        active[gpu] = new_tp
                        logger.info("[FIX-RETRY] %s relaunched on GPU %s",
                                     tp.idea_id, gpu)
                        continue
                    except Exception as e:
                        logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                      tp.idea_id, e)
                else:
                    account_terminal(
                        tp, "failed", "trainer_declared_failed", ret)
                    write_failure_analysis(results_dir / tp.idea_id, classify_failure(error_msg, ret or -1, "training"), error_msg)
                    _record_failure(failure_counts, tp.idea_id)
            elif status == "COMPLETED":
                try:
                    from orze.core.model_lineage import finalize_model_lineage
                    finalize_model_lineage(
                        tp, results_dir / tp.idea_id, cfg)
                except Exception:
                    account_terminal(
                        tp, "failed", "model_lineage_invalid", ret)
                    invalid_path = metrics_path.with_name(
                        f"metrics.lineage_invalid.{time.time_ns()}.json")
                    try:
                        os.replace(metrics_path, invalid_path)
                    except OSError:
                        pass
                    error_msg = "model_lineage_validation_failed"
                    _write_failure(
                        results_dir / tp.idea_id, error_msg,
                        lake=lake, idea_id=tp.idea_id, cfg=cfg,
                    )
                    write_failure_analysis(
                        results_dir / tp.idea_id, "integrity", error_msg)
                    _record_failure(failure_counts, tp.idea_id)
                    status = "FAILED"
                else:
                    account_terminal(
                        tp, "completed", "trainer_completed", ret)
            if (status == "COMPLETED" and lake is not None
                    and cfg.get("eval_script")):
                training_stage = lake.get_stage_state(
                    tp.idea_id, "training")
                persisted = training_stage == "COMPLETE"
                if training_stage in ("NOT_STARTED", "PENDING", "IN_PROGRESS"):
                    persisted = lake.record_stage_transition(
                        tp.idea_id,
                        stage="training",
                        from_state=training_stage,
                        to_state="COMPLETE",
                        reason="training_completed_evaluation_pending",
                        host=socket.gethostname(),
                        pid=os.getpid(),
                    )
                if not persisted:
                    raise RuntimeError(
                        f"Could not persist training stage completion for "
                        f"{tp.idea_id}; current stage is "
                        f"{lake.get_stage_state(tp.idea_id, 'training')}"
                    )
            if (status == "COMPLETED" and lake is not None
                    and not cfg.get("eval_script")):
                # With no separate evaluation phase, successful training is
                # the terminal lifecycle event.  Persist it immediately;
                # periodic catch-up is only a crash-recovery fallback.
                current_state = lake.get_fsm_state(tp.idea_id)
                persisted = current_state == "COMPLETE"
                if current_state == "IN_PROGRESS":
                    persisted = lake.record_state_transition(
                        tp.idea_id,
                        from_state="IN_PROGRESS",
                        to_state="COMPLETE",
                        reason="training_completed",
                        host=socket.gethostname(),
                        pid=os.getpid(),
                        sop_type=(cfg or {}).get("sop", "training"),
                    )
                    # Catch-up or another host may win after the read above.
                    if not persisted:
                        persisted = lake.get_fsm_state(tp.idea_id) == "COMPLETE"
                if not persisted:
                    raise RuntimeError(
                        f"Could not persist terminal lifecycle for {tp.idea_id}; "
                        f"current state is {lake.get_fsm_state(tp.idea_id)}"
                    )
        else:
            reason = f"exit code {ret}"
            try:
                tail_str = tail_file(tp.log_path, 8192)
                lines = tail_str.strip().split("\n")
                tail = "\n".join(lines[-5:])
                reason += f"\n{tail}"
            except Exception:
                pass
            logger.warning("[FAILED] %s on GPU %s — %s", tp.idea_id, gpu, reason)
            exit_reason_code = (
                "process_exit_nonzero" if ret not in (None, 0)
                else "metrics_missing"
            )
            account_terminal(tp, "failed", exit_reason_code, ret)
            if _try_executor_fix(tp.idea_id, reason,
                                 results_dir, cfg, fix_counts,
                                 exit_code=ret if ret is not None else -1):
                _reset_idea_for_retry(results_dir / tp.idea_id)
                try:
                    new_tp = launch(tp.idea_id, actual_gpu, results_dir, cfg)
                    active[gpu] = new_tp
                    logger.info("[FIX-RETRY] %s relaunched on GPU %s",
                                 tp.idea_id, gpu)
                    continue
                except Exception as e:
                    logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                  tp.idea_id, e)
            # CRITICAL: Always record FSM transition for failed jobs, regardless of metrics.json status.
            # Training processes may write metrics.json before crashing, so we can't skip FSM
            # recording just because metrics.json exists. The FSM transition must be recorded
            # for every failure to maintain complete audit trail.
            if not metrics_path.exists():
                _write_failure(results_dir / tp.idea_id,
                               f"Process exited with code {ret}",
                               lake=lake, idea_id=tp.idea_id, cfg=cfg)
            else:
                # metrics.json exists but we still need to record the FSM transition
                # (don't overwrite metrics.json, just record FSM state change)
                if lake and tp.idea_id:
                    try:
                        current_state = lake.get_fsm_state(tp.idea_id)
                        if current_state:
                            lake.record_state_transition(
                                tp.idea_id,
                                from_state=current_state,
                                to_state="FAILED",
                                reason=reason,
                                host=socket.gethostname(),
                                pid=os.getpid(),
                                sop_type=(cfg or {}).get("sop", "training"),
                            )
                    except Exception as e:
                        logger.warning("FSM transition failed (non-blocking): %s", e)

            write_failure_analysis(results_dir / tp.idea_id, classify_failure(reason, ret or -1, "training"), reason)
            _record_failure(failure_counts, tp.idea_id)

        del active[gpu]
        finished.append((tp.idea_id, gpu))

    return finished
