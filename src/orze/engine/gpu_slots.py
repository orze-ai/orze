"""GPU slot manager for multi-job-per-GPU and multi-GPU-per-job scheduling.

VRAM-aware: checks actual GPU memory before assigning new jobs.
Supports both slot-based limits and memory-based limits.

Config (orze.yaml):
    gpu_scheduling:
      slots_per_gpu: 20          # max concurrent jobs per GPU
      max_vram_pct: 85           # don't assign if GPU VRAM exceeds this %
      min_free_vram_mib: 2000    # don't assign if free VRAM below this

Backward compatible: slots_per_gpu=1 + no VRAM config behaves identically
to the old Dict[int, TrainingProcess].
"""

import logging
import time
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("orze")


def _query_gpu_usage(gpu_id: int) -> Optional[Tuple[int, int]]:
    """Query (used_mib, total_mib) for a GPU. Returns None on failure."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=5,
        )
        parts = result.stdout.strip().split(",")
        return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        return None


class GpuSlotManager:
    """Manages GPU slot allocation for training processes.

    VRAM-aware scheduling: before assigning a slot, checks that the GPU
    has enough free memory. This prevents OOM when many jobs share a GPU.

    Implements dict-like interface for backward compatibility with code
    that did ``self.active: Dict[int, TrainingProcess]``.
    """

    def __init__(self, gpu_ids: List[int], slots_per_gpu: int = 1,
                 max_vram_pct: float = 90, min_free_vram_mib: int = 1000):
        self.gpu_ids = list(gpu_ids)
        self.slots_per_gpu = max(1, slots_per_gpu)
        self.max_vram_pct = max_vram_pct
        self.min_free_vram_mib = min_free_vram_mib
        self._slots: Dict[int, List] = {
            g: [None] * self.slots_per_gpu for g in self.gpu_ids
        }
        self._active: Dict[str, object] = {}
        self._vram_cache: Dict[int, Tuple[float, bool]] = {}
        self._vram_cache_ttl = 10.0  # seconds

    # ------------------------------------------------------------------
    # VRAM checks
    # ------------------------------------------------------------------

    def _gpu_has_capacity(self, gpu_id: int) -> bool:
        """Check if GPU has enough free VRAM (cached, 10s TTL)."""
        now = time.time()
        cached = self._vram_cache.get(gpu_id)
        if cached and (now - cached[0]) < self._vram_cache_ttl:
            return cached[1]

        usage = _query_gpu_usage(gpu_id)
        if usage is None:
            result = True  # can't query — allow (fail later if OOM)
        else:
            used, total = usage
            pct = used / total * 100 if total > 0 else 0
            free = total - used
            result = pct < self.max_vram_pct and free >= self.min_free_vram_mib

        self._vram_cache[gpu_id] = (now, result)
        return result

    def _invalidate_cache(self, gpu_id: int):
        """Invalidate VRAM cache for a GPU after job launch/finish."""
        self._vram_cache.pop(gpu_id, None)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def assign(self, tp, gpus: List[int]) -> str:
        """Assign a process to slot(s) on the given GPU(s).

        Returns the slot key (e.g. "0:3" or "0:1+1:2" for multi-GPU).
        Raises RuntimeError if no free slot on a requested GPU.
        """
        parts = []
        for g in gpus:
            if g not in self._slots:
                raise RuntimeError(f"GPU {g} not in managed list {self.gpu_ids}")
            slots = self._slots[g]
            assigned = False
            for i, s in enumerate(slots):
                if s is None:
                    slots[i] = tp
                    parts.append(f"{g}:{i}")
                    assigned = True
                    break
            if not assigned:
                # Rollback any already-assigned slots
                for p in parts:
                    gs, si = p.split(":")
                    self._slots[int(gs)][int(si)] = None
                raise RuntimeError(f"No free slot on GPU {g}")

        slot_key = "+".join(parts)
        self._active[slot_key] = tp

        if hasattr(tp, 'slot_key'):
            tp.slot_key = slot_key

        # Invalidate VRAM cache for affected GPUs
        for g in gpus:
            self._invalidate_cache(g)

        return slot_key

    def release(self, slot_key: str):
        """Release slot(s) and return the process."""
        tp = self._active.pop(slot_key, None)
        if tp is None:
            return None

        for part in slot_key.split("+"):
            try:
                gpu_str, slot_str = part.split(":")
                g, i = int(gpu_str), int(slot_str)
                if g in self._slots and i < len(self._slots[g]):
                    self._slots[g][i] = None
                    self._invalidate_cache(g)
            except (ValueError, IndexError):
                logger.warning("Invalid slot key part: %s", part)

        return tp

    def free_gpu_ids(self, exclude: Optional[Set[int]] = None) -> List[int]:
        """GPUs with free slot(s) AND VRAM capacity, sorted least-loaded first."""
        exc = exclude or set()
        candidates = []
        for g in self.gpu_ids:
            if g in exc:
                continue
            free_count = sum(1 for s in self._slots[g] if s is None)
            if free_count == 0:
                continue
            if not self._gpu_has_capacity(g):
                continue
            candidates.append((free_count, g))
        candidates.sort(key=lambda x: -x[0])  # most free slots first
        return [g for _, g in candidates]

    def gpu_ids_in_use(self) -> Set[int]:
        """GPUs with ANY slot occupied."""
        return {g for g in self.gpu_ids
                if any(s is not None for s in self._slots[g])}

    def fully_busy_gpu_ids(self) -> Set[int]:
        """GPUs with ALL slots occupied."""
        return {g for g in self.gpu_ids
                if all(s is not None for s in self._slots[g])}

    def slot_usage(self, gpu_id: int) -> Tuple[int, int]:
        """Return (used, total) slots for a GPU."""
        slots = self._slots.get(gpu_id, [])
        used = sum(1 for s in slots if s is not None)
        return used, len(slots)

    @property
    def total_free_slots(self) -> int:
        return sum(1 for slots in self._slots.values() for s in slots if s is None)

    @property
    def total_used_slots(self) -> int:
        return len(self._active)

    # ------------------------------------------------------------------
    # Dict-compatible interface (backward compat)
    # ------------------------------------------------------------------
    # The main loop does: active[gpu] = tp, del active[gpu], for k in active, etc.
    # With multi-slot, int keys auto-map to slot assignment/release.
    # String slot keys (e.g. "0:3") work directly.

    def keys(self):
        return self._active.keys()

    def values(self):
        return self._active.values()

    def items(self):
        return self._active.items()

    def __len__(self):
        return len(self._active)

    def __contains__(self, key):
        if isinstance(key, int):
            # Check if any slot on this GPU is occupied
            return any(s is not None for s in self._slots.get(key, []))
        return key in self._active

    def __getitem__(self, key):
        if isinstance(key, int):
            # Return first active process on this GPU
            for sk, tp in self._active.items():
                if sk.startswith(f"{key}:"):
                    return tp
            raise KeyError(key)
        return self._active[key]

    def __setitem__(self, key, tp):
        if isinstance(key, int):
            self.assign(tp, [key])
        else:
            # Direct slot key assignment — update both _active and _slots
            self._active[key] = tp
            for part in key.split("+"):
                try:
                    gs, si = part.split(":")
                    g, i = int(gs), int(si)
                    if g in self._slots and i < len(self._slots[g]):
                        self._slots[g][i] = tp
                except (ValueError, IndexError):
                    pass

    def __delitem__(self, key):
        if isinstance(key, int):
            # Delete first matching slot on this GPU (backward compat)
            for sk in list(self._active.keys()):
                if sk.startswith(f"{key}:"):
                    self.release(sk)
                    return
            raise KeyError(key)
        else:
            self.release(key)

    def __iter__(self):
        return iter(self._active)

    def __bool__(self):
        return bool(self._active)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
