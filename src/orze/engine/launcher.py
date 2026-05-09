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
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from orze.engine.process import TrainingProcess, _new_process_group, _terminate_and_reap
from orze.core.fs import atomic_write, tail_file
from orze.reporting.notifications import notify

logger = logging.getLogger("orze")

# #10: Rolling buffer for anomaly detection
_recent_completions: list = []


def _get_checkpoint_dir(cfg: dict) -> Optional[Path]:
    """Extract --checkpoint-dir from train_extra_args."""
    args = cfg.get("train_extra_args") or []
    for i, arg in enumerate(args):
        if str(arg) == "--checkpoint-dir" and i + 1 < len(args):
            return Path(str(args[i + 1]))
    return None


_UNSHARE_WARNED = False
_OVERLAY_DIR = Path("/tmp/orze_empty_overlay")


def _resolve_paths(paths) -> list:
    """Realpath-resolve a list of path strings, dropping empties."""
    out = []
    for p in paths or []:
        try:
            out.append(os.path.realpath(str(p)))
        except Exception:
            out.append(str(p))
    return [p for p in out if p]


def _apply_data_boundary_env(env: Dict[str, str], db_cfg: dict,
                             idea_dir: Path) -> None:
    """Populate ORZE_FORBIDDEN_PATHS/WATCH_PATHS/ACCESS_LOG env vars from
    data_boundaries config. Used both by the in-process builtins.open patch
    and by the kernel namespace isolation path.
    """
    forbidden = _resolve_paths(db_cfg.get("forbidden_in_training"))
    watch = _resolve_paths(db_cfg.get("watch_paths"))
    if forbidden:
        env["ORZE_FORBIDDEN_PATHS"] = ":".join(forbidden)
    if watch:
        env["ORZE_WATCH_PATHS"] = ":".join(watch)
    try:
        idea_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    env["ORZE_ACCESS_LOG"] = str(idea_dir / "_access_log.tsv")


def _has_unshare() -> bool:
    """True if `unshare` is on PATH and supports unprivileged user+mount ns."""
    import shutil
    return shutil.which("unshare") is not None


def _ensure_empty_overlay() -> Path:
    """Ensure /tmp/orze_empty_overlay exists (empty dir used as bind-mount
    source to hide forbidden paths inside the training namespace)."""
    try:
        _OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return _OVERLAY_DIR


def _build_isolated_cmd(base_cmd: list, forbidden_paths: list) -> list:
    """Wrap `base_cmd` so it runs inside a private user+mount namespace with
    each forbidden path bind-mounted over by an empty dir. Any file read
    rooted at a forbidden path returns ENOENT at the kernel layer — no
    Python patches, no library-specific hooks.

    Requires Linux `unshare` with unprivileged user namespaces. Callers
    should check _has_unshare() first and fall back to the in-process
    builtins.open patch (the 3.2.26 behavior) when unavailable.
    """
    import shlex
    overlay = _ensure_empty_overlay()

    # Build the mount script. We bind-mount the empty overlay over each
    # forbidden path. Nonexistent paths are skipped (|| true) — orze may
    # declare paths that don't exist on every host.
    mount_lines = []
    for p in forbidden_paths:
        q_overlay = shlex.quote(str(overlay))
        q_path = shlex.quote(p)
        mount_lines.append(
            f"[ -e {q_path} ] && mount --bind {q_overlay} {q_path} 2>/dev/null || true"
        )
    mount_script = "\n".join(mount_lines)

    inner = f"{mount_script}\nexec {shlex.join(base_cmd)}"
    return ["unshare", "-U", "--map-root-user", "-m", "bash", "-c", inner]


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
        except (FileNotFoundError, IndexError, ValueError):
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
    except (FileNotFoundError, IndexError, ValueError):
        return False  # can't check, assume alive

    # 2. Check GPU memory (nvidia-smi for this PID and children)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
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


def _gpu_util_for_pid(pid: int) -> Optional[int]:
    """Return GPU util (0-100) for the GPU process *pid* runs on, or None
    if pid is not currently using a GPU / nvidia-smi is unavailable.

    Uses ``--query-compute-apps=pid,gpu_uuid`` to map pid → GPU, then
    ``--query-gpu=utilization.gpu,uuid`` for per-GPU util.
    """
    try:
        a = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
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
            ["nvidia-smi", "--query-gpu=uuid,utilization.gpu",
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
    gpu_util = _gpu_util_for_pid(tp.process.pid)
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
    "ema",
    "augmentation",
    "augmentations",
    "data_boundaries",   # orze framework-managed
    "executor_fix",      # orze framework-managed
    "report",            # orze framework-managed
    # cycle-346 (professor): hard-code keys that were previously only in
    # orze.yaml's nested_config_whitelist. The cfg-driven extra_whitelist
    # path requires an orze.cli restart after orze.yaml edits; hard-coding
    # them here makes the validator robust to stale-cfg situations and
    # prevents legitimate `soup` / `data_mix` / `length_aware_decoding`
    # ideas from being silently rejected for ~weeks at a time. See
    # results/_retrospection.txt cycle-343..346 for the failure history.
    "data_mix",
    "per_dataset_max_samples",
    "per_dataset_prompts",
    "per_dataset_eval_sample_strategy",
    "per_dataset_enable_thinking",
    "soup",
    "audio_cleanup",
    "length_aware_decoding",
    "reverb_augmentation",
    "ctc_aux",
    "editor",
    # cycle-098 (professor direct-edit, deadline from cycle-095..097):
    # 16,102 queued ideas were blocked by schema-validator on these four
    # nested keys. Engineer trigger pending 13 cycles (085→098); the
    # _engineer_blocker.txt sentinel never appeared, so professor lands
    # the patch directly per the cycle-095 commitment. Semantic gates
    # (decode_constraints, lock_model_to_higgs_v3_8b, eval_tasks subset
    # rejection) still apply downstream — this only widens the SCHEMA
    # validator, not the semantic policy.
    "data",
    "training",
    "model",
    "decoding",
}


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
# PROFESSOR_RULES.md plus the `field_any` (list of fields) extension
# used by require_nontrivial_training_op_101.yaml. Cycle-095 committed
# professor would land this directly if the engineer trigger pended
# >5 cycles; cycle-115 confirmed pending=6, so this is that landing.

def _eval_validator_rule(rule: dict, idea_cfg: dict) -> Optional[str]:
    """Return None if rule passes, otherwise an error string."""
    if not isinstance(rule, dict):
        return None
    op = str(rule.get("operator", "")).lower()
    explanation = str(rule.get("explanation", "")).strip()

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
        return None
    val = idea_cfg.get(field)
    expected = rule.get("value")
    # For value-comparison operators, absence means "use champion
    # default" — rule doesn't apply. Only exists/not_exists care.
    if op in ("equals", "not_equals", "in", "not_in",
              "gt", "gte", "lt", "lte"):
        if field not in idea_cfg:
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
    elif op == "exists":
        if field not in idea_cfg or val in (None, "", [], {}):
            return explanation or f"{field} must be present"
    elif op == "not_exists":
        if field in idea_cfg and val not in (None, "", [], {}):
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
        rules = spec.get("rules") or []
        if not isinstance(rules, list):
            continue
        for rule in rules:
            err = _eval_validator_rule(rule, idea_cfg)
            if err:
                return f"validator[{spec.get('name', vf.stem)}]: {err}"
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
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    log_fh = open(log_path, "a")
    log_fh.write(f"\n[posthoc_runner] kind={kind} gpu={gpu}\n")
    log_fh.flush()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=log_fh, stderr=subprocess.STDOUT,
        env=env, preexec_fn=_new_process_group,
    )
    try:
        proc.stdin.write(json.dumps(idea_cfg).encode())
        proc.stdin.close()
    except Exception:  # pragma: no cover
        pass
    tp = TrainingProcess(
        idea_id=idea_id, gpu=gpu, process=proc,
        start_time=time.time(),
        log_path=log_path,
        timeout=float(cfg.get("posthoc_timeout", 3600)),
        _log_fh=log_fh,
    )
    tp.is_posthoc = True
    return tp


def launch(idea_id: str, gpu: int, results_dir: Path, cfg: dict) -> TrainingProcess:
    """Launch a training subprocess on the given GPU.

    F12: If the idea's YAML specifies ``kind`` other than 'train' (or the
    idea_lake row has such a kind), dispatch to posthoc_runner instead of
    the training script. The 'train' path below is preserved byte-exact
    for back-compat.
    """
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
        except Exception as _qr_e:  # pragma: no cover
            logger.warning(
                "Queue-revalidation soft-failed for idea=%s: %s",
                idea_id, _qr_e)

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
                train_script = idea_cfg["train_script"]
                logger.info("Per-idea train_script override: %s -> %s",
                            idea_id, train_script)
        except Exception:
            pass  # fall back to global train_script

    # Data boundary guardrails. Two layered defenses, activated when
    # data_boundaries is configured:
    #   1. Kernel isolation (primary): unshare -U -m bash -c "mount --bind
    #      empty_dir forbidden_path; exec python -m orze.data_boundaries.wrap
    #      train.py ...". Any file read rooted at a forbidden path returns
    #      ENOENT at the kernel layer — works regardless of which library
    #      does the I/O (pyarrow, h5py, tfrecord, C extensions, network).
    #   2. In-process audit (secondary): orze.data_boundaries.wrap activates
    #      a monkey-patched builtins.open() that appends to ORZE_ACCESS_LOG
    #      for post-hoc audit of Python-level file accesses.
    db_cfg = cfg.get("data_boundaries") or {}
    forbidden = _resolve_paths(db_cfg.get("forbidden_in_training"))
    watch = _resolve_paths(db_cfg.get("watch_paths"))
    use_wrapper = bool(forbidden or watch)

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

    # Wrap with namespace isolation if forbidden paths are configured and
    # unshare is available. Falls back to the in-process patch otherwise.
    if forbidden and _has_unshare():
        cmd = _build_isolated_cmd(base_cmd, forbidden)
    else:
        cmd = base_cmd
        if forbidden and not _has_unshare():
            global _UNSHARE_WARNED
            if not _UNSHARE_WARNED:
                logger.warning(
                    "data_boundaries.forbidden_in_training is set but `unshare` "
                    "is not available. Falling back to in-process builtins.open "
                    "patch, which does NOT catch pyarrow/h5py/C-extension reads. "
                    "Install util-linux (provides unshare) for kernel-level "
                    "isolation."
                )
                _UNSHARE_WARNED = True

    env = os.environ.copy()
    for k, v in (cfg.get("train_extra_env") or {}).items():
        env[k] = str(v)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if use_wrapper:
        _apply_data_boundary_env(env, db_cfg, results_dir / idea_id)

    # Keep file handle open for subprocess lifetime
    log_fh = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
            preexec_fn=_new_process_group,
        )
    except Exception:
        log_fh.close()
        raise

    now = time.time()
    return TrainingProcess(
        idea_id=idea_id, gpu=gpu, process=proc,
        start_time=now, log_path=log_path,
        timeout=cfg.get("timeout", 3600),
        _log_fh=log_fh, _last_log_size=0,
        _last_log_check=now, _stall_since=0.0,
    )


def _write_failure(idea_dir: Path, reason: str):
    """Write a failure metrics.json atomically."""
    metrics = {
        "status": "FAILED",
        "error": reason,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    atomic_write(idea_dir / "metrics.json", json.dumps(metrics, indent=2))


def check_active(active: Dict[int, TrainingProcess], results_dir: Path,
                 cfg: dict, failure_counts: dict,
                 fix_counts: Optional[dict] = None) -> list:
    """Check running processes. Reap completed/timed-out/stalled/OOM.
    Returns list of (idea_id, gpu) tuples for finished ideas.

    When fix_counts is provided and max_fix_attempts > 0, failed ideas
    are sent to the executor LLM for diagnosis before recording failure.
    If the LLM applies a fix, the idea is re-launched on the same GPU.
    """
    from orze.engine.health import check_stalled, detect_fatal_in_log, _adaptive_stall_minutes
    from orze.engine.failure import _record_failure, _try_executor_fix, _reset_idea_for_retry
    from orze.engine.failure_analysis import classify_failure, write_failure_analysis as _write_fa_orig

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
                _terminate_and_reap(tp.process, tp.idea_id)
                tp.close_log()
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
                _write_failure(results_dir / tp.idea_id, error_msg)
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
                _terminate_and_reap(tp.process, tp.idea_id)
                tp.close_log()
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
                _write_failure(results_dir / tp.idea_id, error_msg)
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
                    _terminate_and_reap(tp.process, tp.idea_id)
                    tp.close_log()
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
                    _write_failure(results_dir / tp.idea_id, error_msg)
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
                _terminate_and_reap(tp.process, tp.idea_id)
                # Give SIGKILL after 30s if still alive.
                try:
                    tp.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    import signal as _sig
                    from orze.engine.process import _kill_pg
                    _kill_pg(tp.process, _sig.SIGKILL)
                tp.close_log()
                error_msg = "stuck_no_progress"
                _write_failure(results_dir / tp.idea_id, error_msg)
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
                _terminate_and_reap(tp.process, tp.idea_id)
                tp.close_log()
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
                _write_failure(results_dir / tp.idea_id, error_msg)
                write_failure_analysis(results_dir / tp.idea_id, classify_failure(error_msg, -1, "training"), error_msg)
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append((tp.idea_id, gpu))
                continue

            kill_file = results_dir / tp.idea_id / ".kill"
            if kill_file.exists():
                logger.info("Admin kill signal for %s — terminating", tp.idea_id)
                _terminate_and_reap(tp.process)
                tp.close_log()
                kill_file.unlink(missing_ok=True)
                _write_failure(results_dir / tp.idea_id, "Killed by admin")
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
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                metrics = {"status": "UNKNOWN"}
            status = metrics.get("status", "COMPLETED")
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
            if status == "FAILED":
                error_msg = metrics.get("error", "Training script reported FAILED")
                if _try_executor_fix(tp.idea_id, error_msg,
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
                write_failure_analysis(results_dir / tp.idea_id, classify_failure(error_msg, ret or -1, "training"), error_msg)
                _record_failure(failure_counts, tp.idea_id)
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
            if not metrics_path.exists():
                _write_failure(results_dir / tp.idea_id,
                               f"Process exited with code {ret}")
            write_failure_analysis(results_dir / tp.idea_id, classify_failure(reason, ret or -1, "training"), reason)
            _record_failure(failure_counts, tp.idea_id)

        del active[gpu]
        finished.append((tp.idea_id, gpu))

    return finished
