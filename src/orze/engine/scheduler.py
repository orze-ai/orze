"""Idea scheduling, claiming, orphan cleanup, and status counting.

CALLING SPEC:
    get_unclaimed(ideas, results_dir, skipped=None) -> list[str]
        ideas: Dict[str, dict] — parsed ideas (id -> metadata with 'priority')
        results_dir: Path — parent dir for experiment results
        skipped: set | None — idea IDs to exclude
        returns: idea IDs that have no results_dir/idea_id directory,
                 sorted by priority (critical > high > medium > low) then ID

    claim(idea_id, results_dir, gpu, lake=None) -> bool
        idea_id: str
        results_dir: Path
        gpu: int — recorded in claim.json
        lake: IdeaLake | None — if provided, sets status to 'running'
        returns: True if mkdir succeeded (we got the lock), False if already claimed
        side effects: creates results_dir/idea_id/ directory, writes claim.json

    cleanup_orphans(results_dir, hours, lake=None) -> int
        results_dir: Path
        hours: float — max age of stale claims (0 disables cleanup)
        lake: IdeaLake | None — if provided, resets cleaned ideas to 'queued'
        returns: number of stale local claims safely archived
        side effects: preserves result directories and archives claims/partial
                      metrics only after stable process identity proves them dead

    _count_statuses(ideas, results_dir) -> dict
        ideas: Dict[str, dict]
        results_dir: Path
        returns: {"QUEUED": n, "IN_PROGRESS": n, "COMPLETED": n, "FAILED": n, ...}

    run_cleanup(results_dir, cfg) -> None
        results_dir: Path
        cfg: dict — uses 'cleanup' (patterns, script, timeout), 'gc' (enabled,
                     checkpoints_dir, keep_top, keep_recent, min_free_gb), 'report', 'ideas_file'
        side effects: deletes checkpoint dirs via GC, deletes files matching glob patterns
                      in results dirs, runs custom cleanup script
"""
import os
import re
import subprocess
import time
import socket
import datetime
import json
import logging
import secrets
import sys
from typing import Dict, List, Optional
from pathlib import Path
from orze.core.fs import _fs_lock, _fs_unlock, atomic_write
from orze.engine.process import capture_process_identity, process_is_running

logger = logging.getLogger("orze")
PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

def get_unclaimed(ideas: Dict[str, dict], results_dir: Path,
                  skipped: Optional[set] = None,
                  lake=None) -> List[str]:
    """Return idea IDs with no results dir, sorted by priority then ID.
    Excludes ideas in the skipped set and ideas with missing strategy files.
    When lake is provided, permanently marks unfixable ideas as 'skipped'
    so they don't clog the queue forever."""
    unclaimed = []
    for idea_id in ideas:
        idea_dir = results_dir / idea_id
        resume_requested = (idea_dir / "resume_request.json").is_file()
        if skipped and idea_id in skipped and not resume_requested:
            continue
        # Stale-dir recovery (cycle-2175 engineer P3):
        # portfolio expansion or a prior partial claim can create the
        # results dir (with idea_config.yaml) without claim.json or
        # metrics.json.  These dirs block dispatch because the old
        # `not exists()` gate treated any existing dir as "claimed".
        # Now: dirs without claim.json *and* without metrics.json are
        # stale — treat them as unclaimed so the next dispatch cycle
        # can pick them up.  claim() has a matching fix.
        stale_dir = (
            idea_dir.exists()
            and not (idea_dir / "claim.json").exists()
            and not (idea_dir / "metrics.json").exists()
        )
        if stale_dir:
            logger.warning(
                "Treating %s as unclaimed: stale dir (no claim, no metrics)",
                idea_id)
        if not idea_dir.exists() or stale_dir:
            # Validate strategy file exists before counting as queued
            idea_config = ideas[idea_id].get("config", {})
            strategy_name = idea_config.get("strategy")
            if strategy_name:
                strategy_path = Path("strategies") / f"{strategy_name}.py"
                if not strategy_path.exists():
                    logger.warning(
                        "Skipping %s: strategy file %s does not exist",
                        idea_id, strategy_path)
                    if lake:
                        try:
                            lake.set_status(idea_id, "skipped")
                        except Exception:
                            pass
                    continue
            unclaimed.append(idea_id)

    no_medal_comps = set()
    no_medal_file = results_dir / "_no_medal_competitions.txt"
    if no_medal_file.exists():
        try:
            no_medal_comps = {
                line.strip() for line in no_medal_file.read_text().splitlines()
                if line.strip()
            }
        except Exception:
            pass

    def sort_key(idea_id):
        pri = PRIORITY_ORDER.get(ideas[idea_id]["priority"], 2)
        cfg = ideas[idea_id].get("config") or {}
        strategy = str(cfg.get("strategy") or "").lower()
        is_inference_only = (
            bool(cfg.get("inference_only"))
            or strategy == "eval_only"
            or strategy.endswith("_eval"))
        inference_boost = 0 if is_inference_only else 1
        comp_id = cfg.get("competition_id") or ""
        if not comp_id and isinstance(cfg.get("data"), dict):
            comp_id = cfg["data"].get("competition_id", "")
        medal_need = 0 if comp_id in no_medal_comps else 1
        portfolio_boost = 0 if idea_id.startswith("idea-pf-") else 1
        # Stable-hash tiebreaker replaces alphabetic idea_id sort to avoid
        # starving ideas whose IDs sort late (per project_pf_idea_alphabetic_starvation
        # memory: numeric-prefix IDs always won the previous tiebreaker, so any
        # alphabetic-prefix idea like idea-bb*/idea-cc* was structurally last).
        import hashlib
        hash_tiebreak = hashlib.md5(idea_id.encode()).hexdigest()
        return (pri, portfolio_boost, medal_need, inference_boost, hash_tiebreak)

    unclaimed.sort(key=sort_key)

    return unclaimed


# ---------------------------------------------------------------------------
# Claiming (atomic mkdir)
# ---------------------------------------------------------------------------

def get_critical_force_pack_eligible(ideas: Dict[str, dict],
                                     results_dir: Path) -> List[str]:
    """Unclaimed critical idea IDs that declare min_free_vram_mib_for_eval (force-pack eligible)."""
    eligible = []
    for idea_id, meta in ideas.items():
        if meta.get("priority") != "critical":
            continue
        if (results_dir / idea_id).exists():
            continue
        if (meta.get("config") or {}).get("min_free_vram_mib_for_eval") is not None:
            eligible.append(idea_id)
    return eligible


def claim(idea_id: str, results_dir: Path, gpu: int,
          lake=None) -> bool:
    """Atomically claim an idea via mkdir. Returns True if we got it.
    If lake is provided, also updates the DB status to 'running'.

    Multi-host safety: if the directory already exists with a claim.json
    owned by another host, refuse to claim (even if the idea was reset
    to 'queued' in the DB by a stale reconciliation).
    """
    idea_dir = results_dir / idea_id
    try:
        idea_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # Dir exists — check if another host owns it
        claim_path = idea_dir / "claim.json"
        if claim_path.exists():
            try:
                existing = json.loads(claim_path.read_text(encoding="utf-8"))
                if existing.get("claimed_by") != socket.gethostname():
                    return False  # another host owns this, don't steal
            except (json.JSONDecodeError, OSError):
                pass
            # Dir has a claim.json (ours or corrupt).  Other host check
            # above already handled the cross-host case.  If the claim is
            # ours (or unreadable), refuse to double-claim.
            return False
        # Stale-dir recovery (cycle-2175 engineer P3):
        # Dir exists but no claim.json and no metrics.json — portfolio
        # expansion or a prior partial claim created the dir.  Fall
        # through to write claim.json and claim the slot.
        if (idea_dir / "metrics.json").exists():
            return False  # completed/failed idea, not claimable

    claim_info = {
        "attempt_id": secrets.token_hex(16),
        "claimed_by": socket.gethostname(),
        "claimed_at": datetime.datetime.now().isoformat(),
        "pid": os.getpid(),
        "gpu": gpu,
    }
    try:
        claim_info["owner_start_ticks"] = capture_process_identity(
            os.getpid())["start_ticks"]
    except (OSError, ValueError, ProcessLookupError):
        # Recovery still handles legacy claims conservatively; a normal Linux
        # launch should always make this identity available.
        pass
    atomic_write(idea_dir / "claim.json", json.dumps(claim_info, indent=2))

    if lake:
        try:
            # Record FSM transition: QUEUED → CLAIMED
            persisted = lake.record_state_transition(
                idea_id,
                from_state="QUEUED",
                to_state="CLAIMED",
                reason=f"claimed by {socket.gethostname()} on gpu {gpu}",
                host=socket.gethostname(),
                pid=os.getpid(),
            )
            if not persisted:
                raise RuntimeError("FSM claim was rejected")
        except Exception as exc:
            # The filesystem lock and the audited queue must agree. Leaving
            # claim.json behind after a rejected DB claim strands the idea and
            # makes a later scheduler believe another worker owns it.
            try:
                (idea_dir / "claim.json").unlink(missing_ok=True)
            except OSError:
                pass
            logger.warning("Claim rollback for %s: %s", idea_id, exc)
            return False

    return True


# ---------------------------------------------------------------------------
# GPU management
# ---------------------------------------------------------------------------

def cleanup_orphans(results_dir: Path, hours: float,
                    lake=None) -> int:
    """Archive stale local claims without deleting experiment evidence.

    Age alone never proves an orphan: another host may own a quiet job, or a
    local PID may have been reused. A claim is recoverable only when hostname,
    PID, and process start ticks identify a dead local owner and any recorded
    trainer is also dead. Partial/invalid metrics are archived before requeue;
    valid terminal metrics remain authoritative.
    """
    if hours <= 0:
        return 0

    cleaned = 0
    cutoff = time.time() - hours * 3600

    for d in results_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        claim_path = d / "claim.json"
        metrics_path = d / "metrics.json"

        if not claim_path.exists():
            continue

        try:
            last_activity = claim_path.stat().st_mtime
            log_path = d / "train_output.log"
            if log_path.exists():
                last_activity = max(last_activity,
                                    log_path.stat().st_mtime)
            if last_activity >= cutoff:
                continue

            idea_id = d.name
            age_hours = (time.time() - last_activity) / 3600
            try:
                claim_data = json.loads(
                    claim_path.read_text(encoding="utf-8"))
                if not isinstance(claim_data, dict):
                    raise ValueError("claim must be a mapping")
                if claim_data.get("claimed_by") != socket.gethostname():
                    continue
                owner_pid = int(claim_data["pid"])
                owner_ticks = int(claim_data["owner_start_ticks"])
            except (json.JSONDecodeError, OSError, UnicodeDecodeError,
                    KeyError, TypeError, ValueError):
                logger.warning(
                    "Keeping stale claim for %s: ownership identity is "
                    "incomplete or invalid", idea_id)
                continue
            if process_is_running(owner_pid, owner_ticks):
                continue

            trainer_pid = claim_data.get("trainer_pid")
            trainer_ticks = claim_data.get("trainer_start_ticks")
            if trainer_pid is not None or trainer_ticks is not None:
                try:
                    if process_is_running(int(trainer_pid), int(trainer_ticks)):
                        continue
                except (TypeError, ValueError):
                    logger.warning(
                        "Keeping stale claim for %s: trainer identity is "
                        "incomplete or invalid", idea_id)
                    continue

            metrics_status = None
            if metrics_path.exists():
                try:
                    metrics = json.loads(
                        metrics_path.read_text(encoding="utf-8"))
                    if isinstance(metrics, dict):
                        metrics_status = metrics.get("status")
                except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                    pass

            terminal_state = {
                "COMPLETED": "COMPLETE",
                "FAILED": "FAILED",
            }.get(metrics_status)
            if lake is not None:
                try:
                    if terminal_state is not None:
                        persisted = lake.reconcile_terminal_state(
                            idea_id,
                            terminal_state,
                            "reconcile_stale_claim_terminal_metrics",
                        )
                    else:
                        persisted = lake.set_status(idea_id, "queued")
                except Exception as exc:
                    logger.warning(
                        "Failed to persist orphan recovery for %s: %s",
                        idea_id, exc)
                    continue
                if not persisted:
                    logger.warning(
                        "Keeping stale claim for %s: lifecycle recovery was "
                        "rejected", idea_id)
                    continue

            suffix = time.time_ns()
            logger.info(
                "Archiving stale local claim: %s (last activity %.1fh ago)",
                idea_id, age_hours)
            archived_metrics = None
            if metrics_path.exists() and terminal_state is None:
                archived_metrics = d / f"metrics.orphan.{suffix}.json"
                os.replace(metrics_path, archived_metrics)
            archived_claim = d / f"claim.orphan.{suffix}.json"
            os.replace(claim_path, archived_claim)
            atomic_write(
                d / f"recovery.orphan.{suffix}.json",
                json.dumps({
                    "schema_version": 1,
                    "idea_id": idea_id,
                    "outcome": ("terminal_preserved" if terminal_state
                                else "requeued"),
                    "terminal_state": terminal_state,
                    "claim_archive": archived_claim.name,
                    "metrics_archive": (
                        archived_metrics.name if archived_metrics else None),
                    "recovered_at": datetime.datetime.now(
                        datetime.timezone.utc).isoformat(),
                }, indent=2, sort_keys=True) + "\n",
            )
            cleaned += 1
        except Exception as e:
            logger.warning("Failed to clean orphan %s: %s", d.name, e)

    return cleaned


# ---------------------------------------------------------------------------
# Status counting
# ---------------------------------------------------------------------------

def _count_statuses(ideas: Dict[str, dict], results_dir: Path,
                    lake=None) -> dict:
    """Count idea statuses without full report generation.

    When a lake (IdeaLake) is provided, its audited FSM is authoritative for
    every lifecycle state. Mixing filesystem guesses for active work with DB
    totals for terminal work can double-count ideas and advertise states that
    contradict the transition ledger.
    """
    if lake is not None:
        return lake.get_lifecycle_counts()

    counts = {}

    # Count ideas from the current queue (queued / in-progress)
    for idea_id in ideas:
        idea_dir = results_dir / idea_id
        if not idea_dir.exists():
            counts["QUEUED"] = counts.get("QUEUED", 0) + 1
        elif (idea_dir / "metrics.json").exists():
            try:
                m = json.loads((idea_dir / "metrics.json").read_text(encoding="utf-8"))
                st = m.get("status", "UNKNOWN")
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                st = "FAILED"
            counts[st] = counts.get(st, 0) + 1
        else:
            counts["IN_PROGRESS"] = counts.get("IN_PROGRESS", 0) + 1

    return counts


# ---------------------------------------------------------------------------
# Garbage collection / cleanup
# ---------------------------------------------------------------------------

_PROTECTED_EVIDENCE_NAMES = {
    "artifact_preflight.json",
    "interruption.json",
    "progress.json",
    "resume_request.json",
}


def _is_protected_evidence(idea_dir: Path, path: Path) -> bool:
    """True for lifecycle/accounting evidence cleanup must never erase."""
    try:
        relative = path.relative_to(idea_dir)
    except ValueError:
        return True
    if relative.parts and relative.parts[0] == "_compute_receipts":
        return True
    name = path.name
    return (
        name in _PROTECTED_EVIDENCE_NAMES
        or (name.startswith("claim") and name.endswith(".json"))
        or (name.startswith("metrics") and name.endswith(".json"))
        or (name.startswith("recovery") and name.endswith(".json"))
    )


def run_cleanup(results_dir: Path, cfg: dict):
    """Run periodic cleanup: GC checkpoints, delete file patterns, run script."""
    cleanup_cfg = cfg.get("cleanup") or {}

    # GC: delete checkpoint dirs for non-top experiments
    gc_cfg = cfg.get("gc") or {}
    if gc_cfg.get("enabled") and gc_cfg.get("checkpoints_dir"):
        try:
            from orze.agents.orze_gc import run_gc
            report_cfg = cfg.get("report") or {}
            lake_path = Path(cfg.get("idea_lake_db") or Path(cfg.get("results_dir", "orze_results")) / "idea_lake.db")
            stats = run_gc(
                results_dir=results_dir,
                checkpoints_dir=Path(gc_cfg["checkpoints_dir"]),
                primary_metric=report_cfg.get("primary_metric", ""),
                sort_order=report_cfg.get("sort", "descending"),
                lake_db_path=lake_path if lake_path.exists() else None,
                keep_top=gc_cfg.get("keep_top", 50),
                keep_recent=gc_cfg.get("keep_recent", 20),
                min_free_gb=gc_cfg.get("min_free_gb", 0),
            )
            cs = stats.get("checkpoints", {})
            if cs.get("deleted", 0) > 0:
                logger.info("GC: deleted %d checkpoint dirs, kept %d",
                            cs["deleted"], cs["kept"])
        except Exception as e:
            logger.warning("GC failed: %s", e)

    # Built-in: delete files matching glob patterns in results dirs
    patterns = cleanup_cfg.get("patterns") or []
    if patterns:
        deleted = 0
        for d in results_dir.iterdir():
            if not d.is_dir() or not d.name.startswith("idea-"):
                continue
            for pattern in patterns:
                for f in d.glob(pattern):
                    try:
                        if f.is_file() and not _is_protected_evidence(d, f):
                            f.unlink()
                            deleted += 1
                    except Exception:
                        pass
        if deleted:
            logger.info("Cleanup: deleted %d files matching %s",
                        deleted, patterns)

    # Custom cleanup script
    script = cleanup_cfg.get("script")
    if script:
        python = cfg.get("python", sys.executable)
        timeout = cleanup_cfg.get("timeout", 300)
        try:
            result = subprocess.run(
                [python, script], capture_output=True, text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                logger.info("Cleanup script completed")
            else:
                logger.warning("Cleanup script failed (exit %d)",
                               result.returncode)
        except Exception as e:
            logger.warning("Cleanup script error: %s", e)


# ---------------------------------------------------------------------------
# Process checking (with health monitoring)
