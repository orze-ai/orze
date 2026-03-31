"""VRAM-based GPU scheduler for multi-job-per-GPU and multi-GPU-per-job.

Schedules by actual GPU memory — no fixed slot counts needed. Queries
nvidia-smi (batched, cached) to check free VRAM before each assignment.

Config (orze.yaml):
    gpu_scheduling:
      max_vram_pct: 85           # stop assigning when GPU VRAM hits 85%
      min_free_vram_mib: 2000    # stop if less than 2GB free
      max_jobs_per_gpu: 20       # safety cap (raise for many tiny jobs)

Backward compatible: without gpu_scheduling config, max_jobs_per_gpu=1
gives identical behavior to old Dict[int, TrainingProcess].
"""

import logging
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
# GpuSlotManager
# ---------------------------------------------------------------------------

class GpuSlotManager:
    """VRAM-based GPU job scheduler with dict-compatible interface.

    Replaces the old ``self.active: Dict[int, TrainingProcess]``.
    Schedules by actual VRAM usage — no guessing slot counts.
    """

    def __init__(self, gpu_ids: List[int],
                 max_vram_pct: float = 90,
                 min_free_vram_mib: int = 1000,
                 max_jobs_per_gpu: int = 20,
                 slots_per_gpu: int = 1):  # legacy compat, maps to max_jobs
        self.gpu_ids = list(gpu_ids)
        self.max_vram_pct = max_vram_pct
        self.min_free_vram_mib = min_free_vram_mib
        self.max_jobs_per_gpu = max(max_jobs_per_gpu, slots_per_gpu)
        self._active: Dict[str, object] = {}
        self._gpu_jobs: Dict[int, List[str]] = {g: [] for g in self.gpu_ids}
        self._next_id: int = 0

        # Legacy compat
        self.slots_per_gpu = self.max_jobs_per_gpu

    def _gpu_has_capacity(self, gpu_id: int) -> bool:
        """Check VRAM headroom AND job count cap."""
        n_jobs = len(self._gpu_jobs.get(gpu_id, []))
        if n_jobs >= self.max_jobs_per_gpu:
            return False
        usage = _get_gpu_usage(gpu_id)
        if usage is None:
            # nvidia-smi failed — only allow new jobs if GPU is completely idle.
            # This prevents unbounded spawning when the driver is degraded.
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
