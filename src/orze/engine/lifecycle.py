"""Startup, shutdown, and PID management for Orze.

CALLING SPEC:
    startup_checks(results_dir, cfg, hostname, instance_uuid) -> HealthMonitor
    reconcile_stale_running(cfg) -> None
    print_startup_summary(cfg) -> None
    write_pid_file(results_dir) -> Path
    remove_pid_file(pid_file_path, results_dir) -> None
    write_shutdown_heartbeat(results_dir, hostname, instance_uuid, active) -> None
    graceful_shutdown(results_dir, cfg, active, active_evals, active_roles,
                      iteration, state_dict, lake, hostname, kill_all=False) -> None
    atexit_cleanup(active, active_evals, active_roles) -> None
"""

import datetime
import json
import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from orze import __version__
from orze.engine.process import (
    _kill_pg, process_is_running, process_group_members,
    reconcile_orphaned_role_receipts, terminate_recorded_process_group,
    terminate_role_process,
)
from orze.engine.health import fs_startup_check, cleanup_stale_locks, HealthMonitor
from orze.engine.resume import write_interruption_receipt
from orze.engine.upgrade_cleanup import check_and_clean as upgrade_check_and_clean
from orze.reporting.state import save_state
from orze.reporting.notifications import notify
from orze.core.fs import _fs_unlock, atomic_write

logger = logging.getLogger("orze")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def startup_checks(results_dir: Path, cfg: dict,
                   hostname: str, instance_uuid: str) -> HealthMonitor:
    """Run pre-flight checks before entering main loop.

    Returns a HealthMonitor instance for per-iteration health checks.
    """
    logger.info("=== Startup self-checks ===")
    logger.info("orze v%s | host=%s | instance=%s | pid=%d",
                __version__, hostname, instance_uuid, os.getpid())

    # 1. Verify shared filesystem is mounted and writable
    if not fs_startup_check(results_dir):
        raise SystemExit(
            f"FATAL: Shared filesystem at {results_dir} is not "
            f"writable. Check mount status.")
    logger.info("Filesystem check OK: %s", results_dir)

    # 1.5 Normalize the "results/" contract: RULES files, SOP prompts and
    # agent-authored scripts address experiment state as ./results/ (the
    # historical default), but results_dir is configurable. When the two
    # diverge, agents split-brain their notes across both directories and
    # drop trigger files where no consumer looks. Keep a project-root
    # "results" symlink pointing at the real results_dir so both paths
    # resolve to the same directory.
    try:
        link = Path("results")
        target = results_dir.resolve()
        if link.resolve() != target:
            if link.is_symlink():
                link.unlink()
                link.symlink_to(target)
                logger.info("Re-pointed results symlink -> %s", target)
            elif not link.exists():
                link.symlink_to(target)
                logger.info("Linked results -> %s", target)
            else:
                logger.warning(
                    "Both ./results/ and results_dir=%s exist — SOP-authored "
                    "files in results/ will not be seen by orze. Merge "
                    "results/ into %s and replace it with a symlink.",
                    results_dir, results_dir)
    except OSError as e:
        logger.warning("results symlink normalization failed: %s", e)

    # 2. Reconcile nonce-bound managed roles before stale lock cleanup.
    role_recovery = reconcile_orphaned_role_receipts(
        Path(cfg.get("_orze_dir", ".orze")), hostname)
    if role_recovery["errors"]:
        raise SystemExit(
            "FATAL: managed role recovery failed: "
            + ",".join(role_recovery["errors"])
        )
    if role_recovery["recovered"]:
        logger.warning("Recovered orphaned managed role(s): %s",
                       ", ".join(role_recovery["recovered"]))
    if role_recovery["remote"]:
        logger.info("Leaving remote managed role receipt(s) untouched: %s",
                    ", ".join(role_recovery["remote"]))

    # 2.5 Clean up legacy stale locks from our own hostname
    cleanup_stale_locks(results_dir, hostname)

    # 3. Scrub stale state files if orze (or orze-pro) was upgraded since
    # the last boot. Must run before the FSM / roles start so they don't
    # observe pre-upgrade one-shot triggers or a pause sentinel that the
    # old binary wrote. Silent no-op on first boot.
    upgrade_check_and_clean(results_dir)

    # 4. Initialize per-iteration health monitor
    health_monitor = HealthMonitor(results_dir)

    # 5. Detect watchdog restart marker and notify
    marker = results_dir / f".orze_watchdog_restart_{hostname}.json"
    if marker.exists():
        try:
            mdata = json.loads(marker.read_text(encoding="utf-8"))
            reason = mdata.get("reason", "unknown")
            prev_pid = mdata.get("prev_pid")
            logger.info("Watchdog restart detected: %s (prev PID %s)",
                        reason, prev_pid)
            notify("watchdog_restart", {
                "host": hostname,
                "reason": reason,
                "prev_pid": prev_pid,
                "timestamp": mdata.get("iso", ""),
            }, cfg)
            marker.unlink()
        except Exception as e:
            logger.warning("Failed to process watchdog restart marker: %s", e)

    # 6. Reconcile stale "running" ideas from prior unclean shutdown
    reconcile_stale_running(cfg)
    # 6b. F7: any 'running' rows still missing their training process
    # are orphans → mark failed (covers crashes that escaped 6).
    try:
        reconcile_running_dead_pids(cfg)
    except Exception as e:
        logger.warning("F7 startup reconcile_running_dead_pids: %s", e)

    logger.info("=== Startup checks passed ===")

    print_startup_summary(cfg)

    return health_monitor


def reconcile_stale_running(cfg: dict) -> None:
    """Reset ideas stuck in 'running' from a prior unclean shutdown.

    Only resets ideas that were claimed by THIS host (checked via
    claim.json). Ideas claimed by other hosts are left as 'running'
    since the other machine may still be training them.
    """
    import socket as _socket
    hostname = _socket.gethostname()
    results_dir = Path(cfg.get("results_dir", "orze_results"))
    lake_path = Path(cfg.get("idea_lake_db") or results_dir / "idea_lake.db")
    if not lake_path.exists():
        return
    lake = None
    try:
        from orze.idea_lake import IdeaLake
        lake = IdeaLake(str(lake_path))
        cur = lake.conn.execute(
            "SELECT idea_id FROM ideas WHERE status = 'running'")
        all_running = [row[0] for row in cur.fetchall()]

        recoveries = []
        others = []
        live_legacy = _running_idea_pids()
        evaluation_required = bool(cfg.get("eval_script"))

        def _metrics_target(idea_id):
            metrics_path = results_dir / idea_id / "metrics.json"
            if not metrics_path.exists():
                return "QUEUED"
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                return None
            target = {
                "COMPLETED": (
                    "EVALUATION_PENDING"
                    if evaluation_required else "COMPLETE"
                ),
                "FAILED": "FAILED",
            }.get(metrics.get("status"))
            # This function is only consulted after the recorded owner and
            # trainer are proven dead.  A partial metrics file can therefore
            # no longer become terminal; retrying it would also overwrite the
            # interrupted run's evidence.  Preserve it as a failed attempt.
            return target or "FAILED"

        for idea_id in all_running:
            claim_path = results_dir / idea_id / "claim.json"
            if claim_path.exists():
                try:
                    claim = json.loads(claim_path.read_text(encoding="utf-8"))
                    if claim.get("claimed_by") != hostname:
                        others.append(idea_id)
                        continue
                    claim_pid = int(claim.get("pid") or 0)
                    owner_start = claim.get("owner_start_ticks")
                    pid_alive = claim_pid > 0 and process_is_running(
                        claim_pid,
                        int(owner_start) if owner_start is not None else None,
                    )
                    if pid_alive:
                        others.append(idea_id)
                        continue

                    trainer_pid = claim.get("trainer_pid")
                    trainer_pgid = claim.get("trainer_pgid")
                    trainer_start = claim.get("trainer_start_ticks")
                    termination_attempted = False
                    terminated = True
                    if all(v is not None for v in (
                            trainer_pid, trainer_pgid, trainer_start)):
                        termination_attempted = process_is_running(
                            int(trainer_pid), int(trainer_start))
                        terminated = terminate_recorded_process_group(
                            int(trainer_pid), int(trainer_pgid),
                            int(trainer_start), idea_id,
                        )
                    elif idea_id in live_legacy:
                        # Legacy claims lack enough stable process identity to
                        # terminate safely. Keep the lock rather than risk PID
                        # reuse or duplicate execution.
                        logger.error(
                            "Cannot recover %s: live legacy trainer has no durable identity",
                            idea_id,
                        )
                        others.append(idea_id)
                        continue

                    if not terminated:
                        logger.error(
                            "Cannot recover %s: orphan trainer could not be proven terminated",
                            idea_id,
                        )
                        others.append(idea_id)
                        continue

                    target = _metrics_target(idea_id)
                    if target is None:
                        logger.error(
                            "Cannot recover %s: metrics exist but are invalid/non-terminal",
                            idea_id,
                        )
                        others.append(idea_id)
                        continue
                    # New execution-identity admission retains ownership until
                    # this immutable terminal receipt exists.  Only emit it
                    # after the process group has been proven empty above.
                    if claim.get("attempt_id"):
                        try:
                            from orze.engine.accounting import (
                                record_recovered_compute_terminal,
                            )
                            record_recovered_compute_terminal(
                                results_dir / idea_id,
                                claim,
                                outcome="interrupted",
                                reason_code="startup_recovery",
                            )
                        except Exception as exc:
                            logger.error(
                                "Cannot recover %s: compute ledger closure "
                                "failed: %s", idea_id, type(exc).__name__)
                            others.append(idea_id)
                            continue
                    metrics_status = {
                        "COMPLETE": "COMPLETED",
                        "EVALUATION_PENDING": "COMPLETED",
                        "FAILED": "FAILED",
                    }.get(target)
                    atomic_write(
                        results_dir / idea_id / "recovery.json",
                        json.dumps({
                            "idea_id": idea_id,
                            "owner_pid": claim_pid,
                            "trainer_pid": trainer_pid,
                            "trainer_pgid": trainer_pgid,
                            "trainer_start_ticks": trainer_start,
                            "termination_attempted": termination_attempted,
                            "trainer_proven_stopped": terminated,
                            "metrics_status_after_stop": metrics_status,
                            "target_state": target,
                            "recovered_at_epoch": time.time(),
                            "recovered_at": datetime.datetime.now().isoformat(),
                        }, indent=2),
                    )
                    recovered_claim = claim_path.with_name(
                        f"claim.recovered.{int(time.time())}.json")
                    os.replace(claim_path, recovered_claim)
                    recoveries.append((idea_id, target))
                except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                    logger.error(
                        "Cannot safely recover claim for %s: %s", idea_id, exc)
                    others.append(idea_id)
            else:
                recovery_path = results_dir / idea_id / "recovery.json"
                if recovery_path.exists():
                    try:
                        recovery = json.loads(
                            recovery_path.read_text(encoding="utf-8"))
                        recovery_pgid = recovery.get("trainer_pgid")
                        target = recovery.get("target_state")
                        if (recovery.get("trainer_proven_stopped") is not True
                                or target not in (
                                    "QUEUED", "EVALUATION_PENDING", "COMPLETE",
                                    "FAILED",
                                )
                                or (recovery_pgid is not None
                                    and process_group_members(int(recovery_pgid)))):
                            raise ValueError("recovery WAL does not prove a stopped trainer")
                        recoveries.append((idea_id, target))
                        continue
                    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                        logger.error(
                            "Cannot consume recovery WAL for %s: %s", idea_id, exc)
                        others.append(idea_id)
                        continue
                if idea_id in live_legacy:
                    logger.error(
                        "Cannot recover %s without claim: a matching trainer is live",
                        idea_id,
                    )
                    others.append(idea_id)
                    continue
                target = _metrics_target(idea_id)
                if target is None:
                    logger.error(
                        "Cannot recover %s without claim: metrics are invalid/non-terminal",
                        idea_id,
                    )
                    others.append(idea_id)
                    continue
                recoveries.append((idea_id, target))

        if recoveries:
            reconciled = []
            for idea_id, target_state in recoveries:
                current_state = lake.get_fsm_state(idea_id)
                persisted = False
                if target_state in ("COMPLETE", "FAILED"):
                    reason = (
                        "reconcile_startup_recover_completed_output"
                        if target_state == "COMPLETE"
                        else "reconcile_startup_recover_failed_output"
                    )
                    persisted = lake.reconcile_terminal_state(
                        idea_id, target_state, reason)
                elif target_state == "EVALUATION_PENDING":
                    if current_state == "CLAIMED":
                        persisted = lake.record_state_transition(
                            idea_id,
                            "CLAIMED",
                            "IN_PROGRESS",
                            reason="reconcile_training_process_started",
                            host=hostname,
                            pid=os.getpid(),
                            sop_type="training",
                        )
                        current_state = lake.get_fsm_state(idea_id)
                    else:
                        persisted = current_state == "IN_PROGRESS"
                    if persisted:
                        training_stage = lake.get_stage_state(
                            idea_id, "training")
                        persisted = training_stage == "COMPLETE" or (
                            training_stage in (
                                "NOT_STARTED", "PENDING", "IN_PROGRESS",
                            ) and lake.record_stage_transition(
                                idea_id,
                                stage="training",
                                from_state=training_stage,
                                to_state="COMPLETE",
                                reason=(
                                    "reconcile_training_completed_"
                                    "evaluation_pending"
                                ),
                                host=hostname,
                                pid=os.getpid(),
                            )
                        )
                    if persisted:
                        evaluation_stage = lake.get_stage_state(
                            idea_id, "evaluation")
                        if evaluation_stage in ("NOT_STARTED", "IN_PROGRESS"):
                            persisted = lake.record_stage_transition(
                                idea_id,
                                stage="evaluation",
                                from_state=evaluation_stage,
                                to_state="PENDING",
                                reason=(
                                    "reconcile_evaluation_pending"
                                    if evaluation_stage == "NOT_STARTED" else
                                    "reconcile_interrupted_evaluation_pending"
                                ),
                                host=hostname,
                                pid=os.getpid(),
                            )
                        else:
                            persisted = evaluation_stage == "PENDING"
                elif current_state in ("CLAIMED", "IN_PROGRESS"):
                    persisted = lake.record_state_transition(
                        idea_id,
                        current_state,
                        "QUEUED",
                        reason="startup_recover_orphan_terminated",
                        host=hostname,
                        pid=os.getpid(),
                        sop_type="training",
                    )
                elif current_state == "QUEUED":
                    persisted = lake.set_status(idea_id, "queued")
                if persisted:
                    reconciled.append((idea_id, target_state))
                else:
                    logger.error(
                        "Rejected startup lifecycle recovery for %s: %s -> %s",
                        idea_id, current_state, target_state,
                    )
            if reconciled:
                logger.info(
                    "Reconciled %d stale local ideas: %s",
                    len(reconciled), ", ".join(
                    f"{idea_id}->{target.lower()}"
                    for idea_id, target in reconciled[:10]),
                )
        if others:
            logger.info("Kept %d 'running' ideas owned by other hosts: %s",
                        len(others), ", ".join(others[:10]))
    except Exception as e:
        logger.warning("Failed to reconcile stale ideas: %s", e)
    finally:
        if lake is not None:
            lake.close()


def _running_idea_pids() -> set:
    """Return the set of idea_ids that have a live python training
    subprocess on this host (cmdline contains '--idea-id <id>').

    Uses psutil if available; falls back to /proc scanning. Returns an
    empty set if neither is usable (caller treats as 'unknown' and
    skips reconcile to avoid false orphan-marks).
    """
    found = set()
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None  # noqa: N806

    if psutil is not None:
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = p.info.get("cmdline") or []
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            for i, tok in enumerate(cmd):
                if tok == "--idea-id" and i + 1 < len(cmd):
                    found.add(cmd[i + 1])
                    break
        return found

    # /proc fallback (Linux only).
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as f:
                    raw = f.read().split(b"\x00")
                cmd = [x.decode("utf-8", errors="replace") for x in raw if x]
            except (OSError, IOError):
                continue
            for i, tok in enumerate(cmd):
                if tok == "--idea-id" and i + 1 < len(cmd):
                    found.add(cmd[i + 1])
                    break
    except OSError:
        return set()
    return found


def reconcile_running_dead_pids(cfg: dict) -> int:
    """F7: For every status='running' idea, verify a python training
    process exists on this host with ``--idea-id <id>`` in its cmdline.
    If the process is gone but metrics.json shows COMPLETED, mark as
    'completed'. Otherwise mark 'failed' with reason 'orphaned_pid'.

    Multi-host safety: only acts on rows whose claim.json says they
    belong to THIS host. Rows owned by another host are left alone.

    Returns the number of rows reconciled.
    """
    import json as _json
    import socket as _socket
    hostname = _socket.gethostname()
    results_dir = Path(cfg.get("results_dir", "orze_results"))
    lake_path = Path(cfg.get("idea_lake_db") or results_dir / "idea_lake.db")
    if not lake_path.exists():
        return 0

    alive_ideas = _running_idea_pids()
    # Grace period: skip reconcile for rows whose claim.json or
    # idea-dir activity is younger than this. Covers the launch
    # race window (claim.json written → subprocess spawn) and the
    # post-completion metrics flush window. Cycle-092 cross-domain row
    # 65: standard distributed-task-queue grace is 120s.
    grace_seconds = 180
    now_ts = time.time()
    evaluation_required = bool(cfg.get("eval_script"))

    n_completed = 0
    n_orphaned = 0
    n_requeued = 0
    n_skipped_grace = 0
    n_warned = 0
    lake = None
    try:
        from orze.idea_lake import IdeaLake
        lake = IdeaLake(str(lake_path))
        rows = lake.conn.execute(
            "SELECT idea_id, eval_metrics FROM ideas "
            "WHERE status = 'running'").fetchall()
        for idea_id, em_raw in rows:
            idea_dir = results_dir / idea_id
            claim_path = idea_dir / "claim.json"

            def completed_on_disk() -> bool:
                metrics_path = idea_dir / "metrics.json"
                if not metrics_path.is_file():
                    return False
                try:
                    metrics = _json.loads(
                        metrics_path.read_text(encoding="utf-8"))
                except (_json.JSONDecodeError, OSError, UnicodeDecodeError):
                    return False
                return (isinstance(metrics, dict)
                        and metrics.get("status") == "COMPLETED")

            def reconcile_completed_training() -> bool:
                if not evaluation_required:
                    return lake.set_status(idea_id, "completed")
                current = lake.get_fsm_state(idea_id)
                if current == "CLAIMED":
                    if not lake.record_state_transition(
                            idea_id, "CLAIMED", "IN_PROGRESS",
                            reason="reconcile_training_process_started",
                            host=hostname, pid=os.getpid(),
                            sop_type="training"):
                        return False
                    current = "IN_PROGRESS"
                if current != "IN_PROGRESS":
                    return False
                stage = lake.get_stage_state(idea_id, "training")
                training_ok = stage == "COMPLETE"
                if stage in ("NOT_STARTED", "PENDING", "IN_PROGRESS"):
                    training_ok = lake.record_stage_transition(
                        idea_id,
                        stage="training",
                        from_state=stage,
                        to_state="COMPLETE",
                        reason=(
                            "reconcile_training_completed_"
                            "evaluation_pending"
                        ),
                        host=hostname,
                        pid=os.getpid(),
                    )
                if not training_ok:
                    return False
                evaluation_stage = lake.get_stage_state(
                    idea_id, "evaluation")
                if evaluation_stage in ("NOT_STARTED", "IN_PROGRESS"):
                    return lake.record_stage_transition(
                        idea_id,
                        stage="evaluation",
                        from_state=evaluation_stage,
                        to_state="PENDING",
                        reason=(
                            "reconcile_evaluation_pending"
                            if evaluation_stage == "NOT_STARTED" else
                            "reconcile_interrupted_evaluation_pending"
                        ),
                        host=hostname,
                        pid=os.getpid(),
                    )
                return evaluation_stage == "PENDING"

            # ---- Ownership check (multi-host safety) ----
            if claim_path.exists():
                try:
                    claim = _json.loads(claim_path.read_text(encoding="utf-8"))
                    if claim.get("claimed_by") != hostname:
                        continue
                except (_json.JSONDecodeError, OSError):
                    # Ownership cannot be proved. Never mutate another host's
                    # possible work from a corrupt claim.
                    continue
            else:
                # Completion may flush immediately before claim cleanup. It is
                # the only terminal filesystem evidence accepted here.
                if completed_on_disk():
                    if reconcile_completed_training():
                        n_completed += 1
                elif lake.set_status(idea_id, "queued"):
                    n_requeued += 1
                continue

            # ---- Live process? ----
            if idea_id in alive_ideas:
                continue

            # ---- Grace window: ----
            # If the idea_dir was modified within `grace_seconds`, the
            # subprocess may be in the launch race window or the
            # completion-flush window. Skip this iteration.
            try:
                latest = claim_path.stat().st_mtime
                # Also consider real trainer progress and terminal metrics.
                # A submission alone is not lifecycle evidence.
                for fname in ("metrics.json", "train_output.log"):
                    fp = idea_dir / fname
                    if fp.exists():
                        latest = max(latest, fp.stat().st_mtime)
                if (now_ts - latest) < grace_seconds:
                    n_skipped_grace += 1
                    continue
            except OSError:
                pass

            # ---- Completed-on-disk check ----
            if completed_on_disk():
                if reconcile_completed_training():
                    n_completed += 1
                continue

            try:
                em = _json.loads(em_raw) if em_raw else {}
                if not isinstance(em, dict):
                    em = {}
            except (ValueError, TypeError):
                em = {}
            # Require 2 consecutive liveness misses before orphan-marking
            # (Phi-Accrual anti-flap). Persist the first miss immediately;
            # the previous implementation forgot to commit warning-only
            # cycles, so an orphan could remain "running" forever.
            miss_count = em.get("liveness_misses", 0) + 1
            em["liveness_misses"] = miss_count
            if miss_count >= 2:
                em["failure_reason"] = "orphaned_pid"
            lake.conn.execute(
                "UPDATE ideas SET eval_metrics = ? WHERE idea_id = ?",
                (_json.dumps(em), idea_id))
            lake.conn.commit()
            if miss_count >= 2:
                if lake.set_status(idea_id, "failed"):
                    n_orphaned += 1
            else:
                n_warned += 1
        if n_completed:
            target = (
                "evaluation pending" if evaluation_required else "completed")
            logger.info(
                "Reconciled %d 'running' rows (training completed on disk) "
                "-> %s", n_completed, target)
        if n_requeued:
            logger.info(
                "Reconciled %d 'running' rows (no claim.json) -> queued",
                n_requeued)
        if n_skipped_grace:
            logger.debug(
                "Reconcile: %d 'running' rows skipped (within %ds grace)",
                n_skipped_grace, grace_seconds)
        if n_warned:
            logger.info(
                "Reconcile: %d 'running' rows — 1st liveness miss (need 2 to orphan)",
                n_warned)
        if n_orphaned:
            logger.info(
                "Reconciled %d orphaned 'running' rows (dead PID, 2+ misses) -> failed",
                n_orphaned)
    except Exception as e:
        logger.warning("Failed to reconcile dead-PID rows: %s", e)
    finally:
        if lake is not None:
            lake.close()
    return n_completed + n_orphaned


def print_startup_summary(cfg: dict) -> None:
    """Print a human-readable table of what's configured."""
    W = 60
    line = "=" * W

    # Detect .env
    env_path = None
    config_path = cfg.get("_config_path")
    if config_path:
        candidate = Path(config_path).resolve().parent / ".env"
        if candidate.is_file():
            env_path = str(candidate)
    if not env_path and (Path.cwd() / ".env").is_file():
        env_path = str(Path.cwd() / ".env")

    # Evaluation
    eval_script = cfg.get("eval_script")
    eval_on = bool(eval_script and Path(eval_script).exists())

    # Research roles
    roles = cfg.get("roles") or {}
    research_names = [
        rname for rname, rcfg in roles.items()
        if isinstance(rcfg, dict) and rcfg.get("mode") in ("research", "claude")
    ]

    # Notifications
    ncfg = cfg.get("notifications", {})
    notif_on = ncfg.get("enabled", False)
    notif_channels = [
        ch.get("type", "?") for ch in ncfg.get("channels", [])
        if isinstance(ch, dict)
    ] if notif_on else []

    # Cleanup
    cleanup_cfg = cfg.get("cleanup", {})
    cleanup_on = bool(cleanup_cfg.get("script"))
    cleanup_interval = cleanup_cfg.get("interval", 100)

    # API key check for auto-research
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_any_llm_key = has_anthropic or has_gemini

    lines = [
        "",
        line,
        f"  orze v{__version__} -- Startup Summary",
        line,
        f"  REQUIRED:",
        f"    train_script : {cfg.get('train_script', '?')}",
        f"    ideas_file   : {cfg.get('ideas_file', '?')}",
        f"    results_dir  : {cfg.get('results_dir', '?')}",
        "",
        f"  OPTIONAL FEATURES:",
        f"    evaluation   : {'ON  (' + str(eval_script) + ')' if eval_on else 'OFF'}",
        f"    research     : {'ON  (' + ', '.join(research_names) + ')' if research_names else 'OFF'}",
        f"    notifications: {'ON  (' + ', '.join(notif_channels) + ')' if notif_on and notif_channels else 'OFF'}",
        f"    auto-cleanup : {'ON  (every ' + str(cleanup_interval) + ' ideas)' if cleanup_on else 'OFF'}",
        f"    .env file    : {'loaded (' + env_path + ')' if env_path else 'not found'}",
        line,
        "",
    ]

    from orze.extensions import has_pro
    if has_pro():
        if research_names and not has_any_llm_key:
            lines.append(
                "  \033[33m⚠ WARNING: No ANTHROPIC_API_KEY or GEMINI_API_KEY found.\033[0m"
            )
            lines.append(
                "  \033[33m  Auto-research is configured but will not work without an API key.\033[0m"
            )
            lines.append(
                "  \033[33m  Add ANTHROPIC_API_KEY or GEMINI_API_KEY to your .env file.\033[0m"
            )
            lines.append("")
            logger.warning(
                "Auto-research is configured but no ANTHROPIC_API_KEY or "
                "GEMINI_API_KEY found — research agent will not be able to "
                "generate ideas. Add a key to .env to enable auto-research."
            )
    else:
        lines.append(
            "  \033[2m💡 Upgrade to orze-pro for AI-powered idea generation,"
            " auto-fix, and code evolution → orze.ai/pro\033[0m"
        )
        lines.append("")

    for l in lines:
        print(l)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def graceful_shutdown(results_dir: Path, cfg: dict,
                      active: dict, active_evals: dict, active_roles: dict,
                      iteration: int, state_dict: dict, lake,
                      hostname: str, instance_uuid: str,
                      kill_all: bool = False, *, managed: bool = False,
                      pid_file_path: Path | None = None) -> None:
    """Terminate roles, detach or kill training/eval, save state, clean up.

    Args:
        results_dir: Path to results directory.
        cfg: Config dict.
        active: Dict of gpu -> TrainingProcess.
        active_evals: Dict of gpu -> EvalProcess.
        active_roles: Dict of role_name -> RoleProcess.
        iteration: Current iteration number.
        state_dict: Pre-built state dict for persistence.
        lake: IdeaLake instance (or None).
        hostname: This node's hostname.
        instance_uuid: This node's instance UUID.
        kill_all: If True, kill training and eval processes too (not just
                  detach them). Used by `orze --stop` to fully stop everything.
    """
    logger.info("Shutting down gracefully (kill_all=%s)...", kill_all)
    training_count = len(active)
    eval_count = len(active_evals)

    def close_interrupted_evaluation(ep) -> None:
        """Close compute/stage evidence after a controlled evaluator stop."""
        try:
            from orze.engine.accounting import record_compute_terminal
            record_compute_terminal(
                ep, results_dir / ep.idea_id, "interrupted",
                "evaluation_controller_shutdown", phase="evaluation",
                return_code=ep.process.poll(),
            )
        except Exception as exc:
            logger.warning(
                "Could not persist evaluation interruption receipt for %s: %s",
                ep.idea_id, type(exc).__name__,
            )
        if lake is not None:
            try:
                stage = lake.get_stage_state(ep.idea_id, "evaluation")
                if stage == "IN_PROGRESS" and not lake.record_stage_transition(
                    ep.idea_id,
                    stage="evaluation",
                    from_state="IN_PROGRESS",
                    to_state="PENDING",
                    reason="evaluation_controller_shutdown_retry",
                    host=hostname,
                    pid=os.getpid(),
                ):
                    raise RuntimeError("evaluation_stage_retry_rejected")
            except Exception as exc:
                logger.warning(
                    "Could not reset evaluation stage for %s: %s",
                    ep.idea_id, type(exc).__name__,
                )

    # 0. Write "shutting_down" heartbeat so other nodes know our state
    if not managed:
        try:
            write_shutdown_heartbeat(
                results_dir, hostname, instance_uuid, active)
        except Exception:
            pass

    if kill_all:
        # Kill ALL child processes: training, eval, and roles
        all_procs = []
        for gpu, tp in active.items():
            logger.info("Killing training %s on GPU %s (PID %d)",
                        tp.idea_id, gpu, tp.process.pid)
            _kill_pg(tp.process, signal.SIGTERM)
            all_procs.append(("training", tp))
        for gpu, ep in active_evals.items():
            logger.info("Killing eval %s on GPU %s (PID %d)",
                        ep.idea_id, gpu, ep.process.pid)
            _kill_pg(ep.process, signal.SIGTERM)
            all_procs.append(("eval", ep))
        for role_name, rp in active_roles.items():
            logger.info("Killing role '%s' (PID %d)",
                        role_name, rp.process.pid)
            reaped = terminate_role_process(rp, f"role {role_name}")
            rp.close_log()
            if reaped:
                _fs_unlock(rp.lock_dir)

        # Wait up to 10s then SIGKILL
        deadline = time.time() + 10
        for label, proc in all_procs:
            remaining = max(1, deadline - time.time())
            try:
                proc.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing %s (PID %d)",
                               label, proc.process.pid)
                _kill_pg(proc.process, signal.SIGKILL)
                try:
                    proc.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            proc.close_log()
            if label == "training":
                try:
                    write_interruption_receipt(
                        proc, results_dir, cfg, reason="orze_stop",
                        terminating_signal="SIGTERM",
                        return_code=proc.process.poll(),
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not persist interruption receipt for %s: %s",
                        proc.idea_id, type(exc).__name__,
                    )
            else:
                close_interrupted_evaluation(proc)
            if hasattr(proc, 'lock_dir') and proc.lock_dir:
                _fs_unlock(proc.lock_dir)
    else:
        # Training has a durable process/claim recovery path and may safely
        # detach. Evaluators do not: detaching left an unowned allocation that
        # could never write its terminal receipt. Interrupt evals cleanly and
        # return their stage to PENDING so the next controller can retry.
        for gpu, tp in active.items():
            logger.info("Detaching training %s on GPU %s (PID %d) "
                        "-- will finish in background",
                        tp.idea_id, gpu, tp.process.pid)
            tp.close_log()
        for gpu, ep in active_evals.items():
            logger.info("Interrupting eval %s on GPU %s (PID %d) "
                        "-- next controller will retry",
                        ep.idea_id, gpu, ep.process.pid)
            _kill_pg(ep.process, signal.SIGTERM)
            try:
                ep.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _kill_pg(ep.process, signal.SIGKILL)
                try:
                    ep.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            ep.close_log()
            close_interrupted_evaluation(ep)
        for role_name, rp in active_roles.items():
            logger.info("Terminating role '%s' (PID %d)...",
                        role_name, rp.process.pid)
            reaped = terminate_role_process(rp, f"role {role_name}")
            rp.close_log()
            if reaped:
                _fs_unlock(rp.lock_dir)

    # 3. Write shutdown sentinel (tells the watchdog not to restart us)
    if not managed:
        sentinel = results_dir / ".orze_shutdown"
        try:
            sentinel.write_text(
                f"pid={os.getpid()} iteration={iteration} "
                f"time={datetime.datetime.now().isoformat()}\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    # 4. Save state for restart recovery
    if not managed:
        save_state(results_dir, state_dict)

    # 5. Notify (best effort)
    if not managed:
        try:
            notify("shutdown", {
                "host": hostname,
                "message": (f"Graceful shutdown after iteration "
                            f"{iteration}"),
            }, cfg)
        except Exception:
            pass

    # 6. Close IdeaLake (flushes WAL on shared filesystems)
    if lake:
        try:
            lake.close()
        except Exception:
            pass

    # 7. Clean up PID file
    if managed:
        try:
            if pid_file_path is not None:
                pid_file_path.unlink(missing_ok=True)
        except OSError:
            pass
    else:
        remove_pid_file(results_dir / f".orze.pid.{hostname}", results_dir)

    logger.info(
        "Shutdown complete%s after iteration %d. %d training and %d eval "
        "process(es) handled.",
        " (managed; campaign state unchanged)" if managed else "; state saved",
        iteration, training_count if not kill_all else 0,
        eval_count if not kill_all else 0)

    # Detached children must no longer be visible to atexit_cleanup, whose
    # last-resort contract is to kill every process still tracked here.
    active_roles.clear()
    active.clear()
    active_evals.clear()


def atexit_cleanup(active: dict, active_evals: dict,
                   active_roles: dict) -> None:
    """Last-resort cleanup of tracked groups and exact role descendants."""
    for gpu, tp in list(active.items()):
        _kill_pg(tp.process, signal.SIGKILL)
        tp.close_log()
    for gpu, ep in list(active_evals.items()):
        _kill_pg(ep.process, signal.SIGKILL)
        ep.close_log()
    for role_name, rp in list(active_roles.items()):
        terminate_role_process(rp, f"atexit role {role_name}", timeout=2)
        rp.close_log()


def write_shutdown_heartbeat(results_dir: Path, hostname: str,
                             instance_uuid: str, active: dict) -> None:
    """Write a final heartbeat marking this node as shutting_down."""
    pid = os.getpid()
    heartbeat = {
        "host": hostname,
        "pid": pid,
        "timestamp": datetime.datetime.now().isoformat(),
        "epoch": time.time(),
        "status": "shutting_down",
        "active": [
            {
                "idea_id": tp.idea_id,
                "gpu": tp.gpu,
                "elapsed_min": round((time.time() - tp.start_time) / 60, 1),
                "detached": True,
            }
            for tp in active.values()
        ],
        "free_gpus": [],
        "orze_version": __version__,
        "instance_uuid": instance_uuid,
    }
    atomic_write(results_dir / f"_host_{hostname}_{pid}.json",
                 json.dumps(heartbeat, indent=2))


# ---------------------------------------------------------------------------
# PID file management
# ---------------------------------------------------------------------------

def write_pid_file(results_dir: Path) -> Path:
    """Write host-specific PID file for clean stop via --stop or kill.

    Returns the host-specific PID file path (caller should store it for
    later removal).
    """
    hostname = socket.gethostname()
    pid_file = results_dir / f".orze.pid.{hostname}"
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    # Legacy single PID file (for backward compat)
    legacy = results_dir / ".orze.pid"
    legacy.write_text(str(os.getpid()), encoding="utf-8")
    return pid_file


def remove_pid_file(pid_file_path, results_dir: Path) -> None:
    """Remove PID files on exit."""
    for f in [pid_file_path, results_dir / ".orze.pid"]:
        try:
            if f and f.exists():
                f.unlink()
        except Exception:
            pass
