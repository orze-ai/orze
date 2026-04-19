"""GPU scheduler with exclusive and VRAM-based modes.

Config (orze.yaml):
    gpu_scheduling:
      mode: exclusive            # 'exclusive' (default) = 1 job per GPU
                                 # 'vram' = pack by VRAM until full
      max_vram_pct: 85           # (vram mode) stop when GPU VRAM hits 85%
      min_free_vram_mib: 2000    # (vram mode) stop if less than 2GB free
      max_jobs_per_gpu: 20       # (vram mode) safety cap
      max_load_per_cpu: 2.0      # pause scheduling when load/cpu exceeds this
      min_free_ram_gb: 16        # pause scheduling when free RAM drops below

Exclusive mode: each GPU runs at most 1 job. Simple, predictable, safe
for multi-tenant and when you need to reserve GPUs. This is the default.

VRAM mode: packs multiple jobs per GPU based on actual VRAM usage.
Higher throughput for small jobs, but can cause contention.
"""

import logging
import os
import subprocess
import time
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("orze")

# ---------------------------------------------------------------------------
# GPU memory queries (batched + cached)
# ---------------------------------------------------------------------------

_vram_cache: Dict[int, Tuple[float, int, int]] = {}
_CACHE_TTL = 5.0  # seconds


def _query_all_gpu_usage() -> Dict[int, Tuple[int, int]]:
    """Query (used_mib, total_mib) for ALL GPUs in one nvidia-smi call."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        out = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                out[int(parts[0])] = (int(parts[1]), int(parts[2]))
        return out
    except Exception:
        return {}


def _get_gpu_usage(gpu_id: int) -> Optional[Tuple[int, int]]:
    """Get (used_mib, total_mib) with batch cache (5s TTL)."""
    now = time.time()
    cached = _vram_cache.get(gpu_id)
    if cached and (now - cached[0]) < _CACHE_TTL:
        return cached[1], cached[2]
    # Refresh all GPUs in one call
    for gid, (used, total) in _query_all_gpu_usage().items():
        _vram_cache[gid] = (now, used, total)
    entry = _vram_cache.get(gpu_id)
    return (entry[1], entry[2]) if entry else None


def _invalidate_cache(gpu_id: Optional[int] = None):
    """Invalidate cache after job launch/finish."""
    if gpu_id is not None:
        _vram_cache.pop(gpu_id, None)
    else:
        _vram_cache.clear()


# ---------------------------------------------------------------------------
# System resource checks
# ---------------------------------------------------------------------------

_system_cache: Dict[str, Tuple[float, float]] = {}
_SYSTEM_CACHE_TTL = 10.0  # seconds


def _get_load_per_cpu() -> float:
    """Return 1-min load average divided by CPU count."""
    cached = _system_cache.get("load")
    now = time.time()
    if cached and (now - cached[0]) < _SYSTEM_CACHE_TTL:
        return cached[1]
    try:
        load1 = os.getloadavg()[0]
        ncpu = os.cpu_count() or 1
        val = load1 / ncpu
    except OSError:
        val = 0.0
    _system_cache["load"] = (now, val)
    return val


def _count_user_processes() -> int:
    """Count running processes owned by current user (via /proc, no subprocess).

    This is a system-wide count — it sees jobs from ALL orze instances,
    not just the current one. Used to prevent multi-instance burst overload.
    """
    cached = _system_cache.get("nprocs")
    now = time.time()
    if cached and (now - cached[0]) < _SYSTEM_CACHE_TTL:
        return int(cached[1])
    uid = os.getuid()
    count = 0
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                stat = os.stat(f"/proc/{entry}")
                if stat.st_uid == uid:
                    count += 1
            except (FileNotFoundError, PermissionError, OSError):
                continue
    except OSError:
        count = 0
    _system_cache["nprocs"] = (now, float(count))
    return count


def _get_free_ram_gb() -> float:
    """Return free RAM in GB (Linux only, falls back to large value)."""
    cached = _system_cache.get("ram")
    now = time.time()
    if cached and (now - cached[0]) < _SYSTEM_CACHE_TTL:
        return cached[1]
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if parts[0] in ("MemAvailable:", "MemFree:", "Buffers:", "Cached:"):
                    info[parts[0]] = int(parts[1])
            avail_kb = info.get("MemAvailable:", 0)
            if not avail_kb:
                avail_kb = (info.get("MemFree:", 0) +
                            info.get("Buffers:", 0) +
                            info.get("Cached:", 0))
            val = avail_kb / (1024 * 1024)
    except Exception:
        val = 9999.0
    _system_cache["ram"] = (now, val)
    return val


# ---------------------------------------------------------------------------
# GpuSlotManager
# ---------------------------------------------------------------------------

class GpuSlotManager:
    """VRAM-based GPU job scheduler with dict-compatible interface.

    Replaces the old ``self.active: Dict[int, TrainingProcess]``.
    Schedules by actual VRAM usage — no guessing slot counts.
    """

    def __init__(self, gpu_ids: List[int],
                 mode: str = "exclusive",
                 max_vram_pct: float = 90,
                 min_free_vram_mib: int = 1000,
                 max_jobs_per_gpu: int = 20,
                 max_load_per_cpu: float = 2.0,
                 min_free_ram_gb: float = 16.0,
                 slots_per_gpu: int = 1):  # legacy compat, maps to max_jobs
        self.gpu_ids = list(gpu_ids)
        if mode not in ("exclusive", "vram"):
            logger.warning("Unknown gpu_scheduling.mode '%s' — defaulting to 'exclusive'", mode)
            mode = "exclusive"
        self.mode = mode
        self.max_vram_pct = max_vram_pct
        self.min_free_vram_mib = min_free_vram_mib
        if mode == "exclusive":
            self.max_jobs_per_gpu = 1
        else:
            self.max_jobs_per_gpu = max(max_jobs_per_gpu, slots_per_gpu)
        self.max_load_per_cpu = max_load_per_cpu
        self.min_free_ram_gb = min_free_ram_gb
        self._active: Dict[str, object] = {}
        self._gpu_jobs: Dict[int, List[str]] = {g: [] for g in self.gpu_ids}
        self._next_id: int = 0
        self._system_throttle_logged: float = 0.0

        # Legacy compat
        self.slots_per_gpu = self.max_jobs_per_gpu

    def _system_has_capacity(self) -> bool:
        """Check CPU load, process count, and RAM before scheduling.

        Three independent guards:
        1. System-wide process count (instant, sees ALL orze instances)
        2. OS load average (catches pressure from other users too)
        3. Free RAM

        The process count guard is the primary burst preventer — it counts
        all processes owned by our user across the entire system, so two
        orze instances can't independently burst past the limit.
        """
        n_cpus = os.cpu_count() or 1
        max_concurrent = int(n_cpus * self.max_load_per_cpu)

        # Guard 1: system-wide process count (sees all instances)
        n_procs = _count_user_processes()
        if n_procs >= max_concurrent:
            now = time.time()
            if now - self._system_throttle_logged > 60:
                logger.info("Throttling: %d user processes >= %d max "
                            "(%.1f × %d CPUs) — waiting for processes to finish",
                            n_procs, max_concurrent,
                            self.max_load_per_cpu, n_cpus)
                self._system_throttle_logged = now
            return False

        # Guard 2: OS load average (catches pressure from other users)
        load = _get_load_per_cpu()
        if load > self.max_load_per_cpu:
            now = time.time()
            if now - self._system_throttle_logged > 60:
                logger.info("Throttling: load/cpu=%.1f (max %.1f) — "
                            "waiting for system to cool down",
                            load, self.max_load_per_cpu)
                self._system_throttle_logged = now
            return False

        # Guard 3: RAM
        ram_gb = _get_free_ram_gb()
        if ram_gb < self.min_free_ram_gb:
            now = time.time()
            if now - self._system_throttle_logged > 60:
                logger.info("Throttling: free RAM=%.1fGB (min %.1fGB) — "
                            "waiting for memory to free up",
                            ram_gb, self.min_free_ram_gb)
                self._system_throttle_logged = now
            return False
        return True

    def _gpu_has_vram(self, gpu_id: int) -> bool:
        """Check VRAM headroom and job count cap (hard limits only)."""
        n_jobs = len(self._gpu_jobs.get(gpu_id, []))
        if n_jobs >= self.max_jobs_per_gpu:
            return False
        usage = _get_gpu_usage(gpu_id)
        if usage is None:
            if n_jobs > 0:
                logger.warning("GPU %d: VRAM query failed with %d active jobs — "
                               "refusing new work until nvidia-smi recovers", gpu_id, n_jobs)
                return False
            return True
        used, total = usage
        if total <= 0:
            return True
        return (used / total * 100 < self.max_vram_pct and
                total - used >= self.min_free_vram_mib)

    def _gpu_has_capacity(self, gpu_id: int) -> bool:
        """Check if GPU can accept another job.
        Exclusive mode: job count + system resources (no VRAM check).
        VRAM mode: VRAM + job count + system resources."""
        if self.mode == "exclusive":
            if len(self._gpu_jobs.get(gpu_id, [])) >= 1:
                return False
            return self._system_has_capacity()
        if not self._gpu_has_vram(gpu_id):
            return False
        if not self._system_has_capacity():
            return False
        return True

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def assign(self, tp, gpus: List[int]) -> str:
        """Assign process to GPU(s). Returns slot key (e.g. "0:42" or "0:42+1:43").
        Raises RuntimeError if GPU has no capacity."""
        parts = []
        for g in gpus:
            if g not in self._gpu_jobs:
                raise RuntimeError(f"GPU {g} not managed ({self.gpu_ids})")
            if not self._gpu_has_capacity(g):
                raise RuntimeError(f"GPU {g} at capacity ({self.job_count(g)} jobs, "
                                   f"max {self.max_jobs_per_gpu})")
            sid = self._next_id
            self._next_id += 1
            part = f"{g}:{sid}"
            parts.append(part)
            self._gpu_jobs[g].append(part)

        slot_key = "+".join(parts)
        self._active[slot_key] = tp
        if hasattr(tp, 'slot_key'):
            tp.slot_key = slot_key
        for g in gpus:
            _invalidate_cache(g)
        return slot_key

    def release(self, slot_key: str):
        """Release job. Returns the process."""
        tp = self._active.pop(slot_key, None)
        if tp is None:
            return None
        for part in slot_key.split("+"):
            try:
                g = int(part.split(":")[0])
                if part in self._gpu_jobs.get(g, []):
                    self._gpu_jobs[g].remove(part)
                _invalidate_cache(g)
            except (ValueError, IndexError):
                pass
        return tp

    def free_gpu_ids(self, exclude: Optional[Set[int]] = None) -> List[int]:
        """GPUs with VRAM capacity, sorted most-free-memory first."""
        exc = exclude or set()
        candidates = []
        for g in self.gpu_ids:
            if g in exc or not self._gpu_has_capacity(g):
                continue
            usage = _get_gpu_usage(g)
            free_mib = (usage[1] - usage[0]) if usage else 99999
            candidates.append((free_mib, g))
        candidates.sort(key=lambda x: -x[0])
        return [g for _, g in candidates]

    def gpu_ids_in_use(self) -> Set[int]:
        return {g for g in self.gpu_ids if self._gpu_jobs.get(g)}

    def job_count(self, gpu_id: int) -> int:
        return len(self._gpu_jobs.get(gpu_id, []))

    @property
    def total_free_slots(self) -> int:
        return sum(1 for g in self.gpu_ids if self._gpu_has_capacity(g))

    @property
    def total_used_slots(self) -> int:
        return len(self._active)

    # ------------------------------------------------------------------
    # Dict-compatible interface
    # ------------------------------------------------------------------

    def keys(self):    return self._active.keys()
    def values(self):  return self._active.values()
    def items(self):   return self._active.items()
    def __len__(self): return len(self._active)
    def __bool__(self): return bool(self._active)
    def __iter__(self): return iter(self._active)

    def __contains__(self, key):
        if isinstance(key, int):
            return bool(self._gpu_jobs.get(key))
        return key in self._active

    def __getitem__(self, key):
        if isinstance(key, int):
            jobs = self._gpu_jobs.get(key, [])
            if jobs:
                return self._active[jobs[0]]
            raise KeyError(key)
        return self._active[key]

    def __setitem__(self, key, tp):
        if isinstance(key, int):
            self.assign(tp, [key])
        else:
            self._active[key] = tp
            for part in str(key).split("+"):
                try:
                    g = int(part.split(":")[0])
                    if g in self._gpu_jobs and part not in self._gpu_jobs[g]:
                        self._gpu_jobs[g].append(part)
                except (ValueError, IndexError):
                    pass

    def __delitem__(self, key):
        if isinstance(key, int):
            jobs = self._gpu_jobs.get(key, [])
            if jobs:
                self.release(jobs[0])
                return
            raise KeyError(key)
        self.release(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


# ---------------------------------------------------------------------------
# F13: cross-host opportunistic scheduler
# ---------------------------------------------------------------------------


_CHEAP_KINDS = {"posthoc_eval", "tta_sweep", "bundle_combine", "agg_search"}


def _parse_nvidia_smi_stdout(stdout: str) -> Dict[int, Dict[str, int]]:
    """Parse `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total`
    CSV output into {gpu_id: {util, free_mib, total_mib}}.
    """
    out: Dict[int, Dict[str, int]] = {}
    for line in stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            util = int(parts[1])
            used = int(parts[2])
            total = int(parts[3])
        except ValueError:
            continue
        out[idx] = {
            "util": util,
            "free_mib": max(total - used, 0),
            "total_mib": total,
        }
    return out


def _run_ssh(host: str, cmd: List[str], *, timeout: int = 10) -> str:
    """Run a remote command via ssh. Returns stdout (empty on failure)."""
    full = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=accept-new",
        host, "--",
    ] + cmd
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout
    except Exception as e:  # pragma: no cover
        logger.debug("ssh to %s failed: %s", host, e)
    return ""


def poll_fleet(
    hosts: List[str],
    *,
    smi_cmd: Optional[List[str]] = None,
) -> Dict[str, Dict[int, Dict[str, int]]]:
    """Return ``{host: {gpu_id: {util, free_mib, total_mib}}}`` for every
    reachable host. Local host(s) use a direct nvidia-smi call; others ssh.

    Missing / unreachable hosts are simply omitted.
    """
    smi_cmd = smi_cmd or [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    out: Dict[str, Dict[int, Dict[str, int]]] = {}
    local_names = {"localhost", "127.0.0.1", os.uname().nodename}
    for host in hosts:
        if host in local_names:
            try:
                r = subprocess.run(smi_cmd, capture_output=True,
                                    text=True, timeout=5)
                if r.returncode == 0:
                    out[host] = _parse_nvidia_smi_stdout(r.stdout)
            except Exception as e:  # pragma: no cover
                logger.debug("local nvidia-smi failed: %s", e)
        else:
            stdout = _run_ssh(host, smi_cmd)
            if stdout:
                out[host] = _parse_nvidia_smi_stdout(stdout)
    return out


def pick_least_loaded(
    fleet: Dict[str, Dict[int, Dict[str, int]]],
    *,
    min_free_mib: int = 1024,
) -> Optional[Tuple[str, int]]:
    """Return (host, gpu_id) with the lowest utilization among those with
    at least ``min_free_mib`` free VRAM, or None if none qualify.

    Ties on utilization break on larger free_mib.
    """
    best = None  # (util, -free_mib, host, gpu)
    for host, gpus in fleet.items():
        for gpu_id, stats in gpus.items():
            if stats["free_mib"] < min_free_mib:
                continue
            key = (stats["util"], -stats["free_mib"], host, gpu_id)
            if best is None or key < best:
                best = key
    if best is None:
        return None
    return best[2], best[3]


def wrap_remote_cmd(host: str, cmd: List[str]) -> List[str]:
    """Build the argv to execute ``cmd`` on ``host`` via ssh. For the local
    host the command is returned unchanged. Use this in the scheduler
    when dispatching to a remote node selected by pick_least_loaded().
    """
    local_names = {"localhost", "127.0.0.1", os.uname().nodename}
    if host in local_names:
        return list(cmd)
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
        host, "--",
    ] + [_shlex_quote(a) for a in cmd]


def _shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(str(s))


def is_fleet_eligible_kind(kind: Optional[str]) -> bool:
    """Only cheap inference-only kinds are dispatched across hosts."""
    return kind in _CHEAP_KINDS
