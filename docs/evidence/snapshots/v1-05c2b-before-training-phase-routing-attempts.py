"""Training adapter identities, including an explicit pre-native import.

Legacy import consumes an already observed execution's claim/start receipts
inside its completion transaction. It is not a pre-Popen intent or permission
to start another process. Native history never permits tokenless enrollment.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import stat

from orze.core.execution_attempts import (
    AttemptAuthorityError, AttemptRef, StaleAttempt, create_attempt,
    current_attempt, finish_attempt, mark_running, require_current,
)
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.execution_authority import (
    canonical_identity_equal, execution_transaction, lifecycle_fence,
)


def _read(path, limit=65536):
    """Read one stable regular receipt without creating or repairing it."""
    path = Path(path)
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise AttemptEffectBusy("training_identity_redirected")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    identity = lambda st: (st.st_dev, st.st_ino, st.st_mode, st.st_nlink,
                           st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise AttemptEffectBusy("training_identity_invalid")
        raw = os.read(fd, limit + 1)
        if (len(raw) > limit or identity(os.fstat(fd)) != identity(before)
                or identity(path.lstat()) != identity(before)):
            raise AttemptEffectBusy("training_identity_changed")
    finally:
        os.close(fd)
    try:
        def pairs(items):
            obj = {}
            for key, value in items:
                if key in obj:
                    raise ValueError("duplicate key")
                obj[key] = value
            return obj
        value = json.loads(raw, object_pairs_hook=pairs)
        if type(value) is not dict:
            raise ValueError("object required")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AttemptEffectBusy("training_identity_json_invalid") from exc
    return value, hashlib.sha256(raw).hexdigest()


def _claim(tp, idea_dir, lake):
    claim, digest = _read(idea_dir / "claim.json", 8192)
    if claim.get("attempt_id") != tp.attempt_id:
        raise StaleAttempt("training_claim_replaced")
    if type(claim.get("gpu")) is not type(tp.gpu) or claim.get("gpu") != tp.gpu:
        raise AttemptEffectBusy("training_claim_resource_mismatch")
    if "lifecycle_db" in claim:
        databases = lake.conn.execute("PRAGMA database_list").fetchall()
        paths = [row[2] for row in databases if row[1] == "main"]
        if (len(paths) != 1 or not paths[0]
                or type(claim["lifecycle_db"]) is not str
                or str(Path(paths[0]).absolute()) != claim["lifecycle_db"]):
            raise AttemptEffectBusy("training_claim_catalog_mismatch")
    return claim, digest


def require_catalog(lake, idea_dir, cfg, *, handle=None):
    """Missing in-memory wiring cannot erase persisted native routing."""
    from orze.engine.execution_catalog import declared_catalog
    from orze.engine.claim_authority import read_claim
    declared = declared_catalog(idea_dir)
    claim = read_claim(Path(idea_dir) / "claim.json")
    bound = (claim or {}).get("lifecycle_db")
    if lake is None:
        if (declared is not None or bound is not None
                or (handle is not None and getattr(handle, "attempt_ref", None) is not None)):
            raise AttemptEffectBusy("training_native_catalog_required")
        from orze.reporting.evidence import report_lifecycle_db_path
        from orze.core.evaluation_retry_state import open_existing_lake
        path = report_lifecycle_db_path(Path(idea_dir).parent, cfg)
        if path.exists() or path.is_symlink():
            existing = None
            try:
                existing = open_existing_lake(path)
                from orze.core.replication_requests import request_for_task
                if request_for_task(existing.conn, Path(idea_dir).name) is not None:
                    raise AttemptEffectBusy("training_replication_catalog_required")
                if current_attempt(existing.conn, Path(idea_dir).name, "training") is not None:
                    raise AttemptEffectBusy("training_native_catalog_required")
            except AttemptEffectBusy:
                raise
            except Exception as exc:
                raise AttemptEffectBusy("training_catalog_unverifiable") from exc
            finally:
                if existing is not None:
                    existing.close()
        return
    paths = [row[2] for row in lake.conn.execute("PRAGMA database_list") if row[1] == "main"]
    if len(paths) != 1 or not paths[0]:
        raise AttemptEffectBusy("training_persistent_catalog_required")
    actual = str(Path(paths[0]).absolute())
    if any(value is not None and value != actual for value in (bound, declared)):
        raise AttemptEffectBusy("training_catalog_scope_mismatch")


def _legacy_start(tp, idea_dir):
    start, digest = _read(idea_dir / "_compute_receipts" / tp.attempt_id / "start.json")
    expected = {
        "schema_version": 1, "event": "start", "outcome": "started",
        "idea_id": tp.idea_id, "attempt_id": tp.attempt_id, "phase": "training",
        "physical_gpu": tp.gpu, "process_pid": tp.process.pid,
    }
    if any(type(start.get(key)) is not type(value) or start.get(key) != value
           for key, value in expected.items()):
        raise AttemptEffectBusy("training_legacy_start_identity_mismatch")
    # The old watchdog may adjust an in-memory clock; persisted start remains
    # authoritative for identity, while PID alone is never an ownership grant.
    if (getattr(tp, "execution_identity", None) is not None
            and start.get("execution_identity_sha256") != tp.execution_identity):
        raise AttemptEffectBusy("training_legacy_execution_identity_mismatch")
    return digest


def current(lake, tp, idea_dir):
    """Return a current row/legacy candidate; stale or closed returns False."""
    try:
        row = current_attempt(lake.conn, tp.idea_id, "training")
        ref = getattr(tp, "attempt_ref", None)
        if ref is not None:
            if (not isinstance(ref, AttemptRef) or ref.phase != "training"
                    or ref.task_id != tp.idea_id or ref.attempt_id != tp.attempt_id):
                raise AttemptEffectBusy("training_attempt_identity_invalid")
            if row is None:
                raise AttemptEffectBusy("training_attempt_record_missing")
            if (row["attempt_id"], row["generation"]) != (ref.attempt_id, ref.generation):
                return False
        elif row is not None:
            if row["binding"].get("origin") != "legacy_import":
                raise AttemptEffectBusy("training_native_token_required")
            if row["attempt_id"] == tp.attempt_id:
                if row["state"] in ("TERMINAL", "NOT_STARTED"):
                    return False
                raise AttemptEffectBusy("training_legacy_import_unclosed")
            if row["state"] not in ("TERMINAL", "NOT_STARTED"):
                raise AttemptEffectBusy("training_previous_attempt_unclosed")
        if row is not None and ref is not None and row["state"] in ("TERMINAL", "NOT_STARTED"):
            return False
        claim, claim_sha = _claim(tp, idea_dir, lake)
        if ref is None:
            start_sha = _legacy_start(tp, idea_dir)
            terminal = idea_dir / "_compute_receipts" / tp.attempt_id / "terminal.json"
            if terminal.exists() or terminal.is_symlink():
                # A pre-native terminal cannot be newly delivered as an
                # imported completion, regardless of its domain outcome.
                return False
            fence = lifecycle_fence(lake, tp.idea_id, "training")
            if fence["global_state"] != "IN_PROGRESS" or fence["phase_state"] != "IN_PROGRESS":
                return False
            return {"legacy": True, "claim_sha256": claim_sha,
                    "start_sha256": start_sha, "lifecycle": fence}
        row = require_current(lake.conn, ref)
        if (row["binding"].get("origin") == "native_training"
                and (type(row["binding"].get("process_pid")) is not type(tp.process.pid)
                     or row["binding"].get("process_pid") != tp.process.pid)):
            raise AttemptEffectBusy("training_process_identity_mismatch")
        if not canonical_identity_equal(
                lifecycle_fence(lake, tp.idea_id, "training"), row["binding"].get("lifecycle")):
            return False
        if row["binding"].get("origin") == "native_training":
            from orze.engine.training_supervision import bound_binding
            bound_binding(tp, row, idea_dir)
        return row
    except StaleAttempt:
        return False
    except OSError as exc:
        raise AttemptEffectBusy("training_authority_unreadable") from exc


def import_for_completion(tx, tp, candidate):
    """Import only inside the terminal transaction, never advertise RUNNING."""
    if not candidate.get("legacy"):
        return tp.attempt_ref
    ref = create_attempt(tx.conn, tp.idea_id, "training", tp.attempt_id, {
        "origin": "legacy_import", "claim_sha256": candidate["claim_sha256"],
        "start_sha256": candidate["start_sha256"], "lifecycle": candidate["lifecycle"],
    })
    mark_running(tx.conn, ref)
    return ref


def _launch_state(lake, idea_id):
    row = lake.conn.execute(
        "SELECT i.status,s.current_state,h.id,h.to_state FROM main.ideas i "
        "JOIN main.idea_state s ON s.idea_id=i.idea_id COLLATE BINARY "
        "JOIN main.idea_transitions h ON h.idea_id=i.idea_id COLLATE BINARY "
        "AND h.id=(SELECT MAX(id) FROM main.idea_transitions "
        "WHERE idea_id=i.idea_id COLLATE BINARY) WHERE i.idea_id=? COLLATE BINARY",
        (idea_id,),
    ).fetchall()
    if (len(row) != 1 or tuple(row[0][:2]) != ("running", "CLAIMED")
            or type(row[0][2]) is not int or row[0][2] <= 0 or row[0][3] != "CLAIMED"):
        raise AttemptEffectBusy("training_launch_lifecycle_not_claimed")
    return {"legacy_status": row[0][0], "global_state": row[0][1],
            "global_transition_id": row[0][2]}


def begin(lake, tp, idea_dir, cfg=None):
    """Persist a native intent before Popen, without inventing a started stage."""
    from orze.core.artifact_contract import artifact_publication_binding
    artifact_binding = artifact_publication_binding(
        cfg or {}, idea_dir, getattr(tp, "execution_identity", None))
    with execution_transaction(lake, idea_dir) as tx:
        from orze.engine.execution_catalog import bind_catalog
        bind_catalog(lake, idea_dir, tx.lease)
        _, claim_sha = _claim(tp, idea_dir, lake)
        launch_state = _launch_state(lake, tp.idea_id)
        from orze.engine.replication import replication_authorization
        replication = replication_authorization(
            lake, tp.idea_id, idea_dir, cfg or {}, getattr(tp, "execution_identity", None),
            claim_id=tp.attempt_id)
        captured_replication = getattr(tp, "replication_authorization", None)
        if ((replication is not None or captured_replication is not None)
                and not canonical_identity_equal(replication, captured_replication)):
            raise AttemptEffectBusy("training_replication_authorization_changed")
        binding = {
            "origin": "native_training", "claim_sha256": claim_sha,
            "launch_lifecycle": launch_state,
        }
        from orze.engine.training_supervision import PROTOCOL
        binding["process_supervision_protocol"] = PROTOCOL
        if artifact_binding is not None:
            binding["artifact_publication"] = artifact_binding
        if replication is not None:
            binding["replication"] = replication
        ref = create_attempt(tx.conn, tp.idea_id, "training", tp.attempt_id, binding)
        if not canonical_identity_equal(_launch_state(lake, tp.idea_id), launch_state):
            raise AttemptAuthorityError("training_launch_lifecycle_changed")
        if replication is not None and not canonical_identity_equal(
                replication_authorization(lake, tp.idea_id, idea_dir, cfg or {},
                                          tp.execution_identity, claim_id=tp.attempt_id), replication):
            raise AttemptEffectBusy("training_replication_authorization_changed")
        tx.watch_attempt(ref)
    return ref


def started(lake, tp, idea_dir, process_identity, *, resume_context=None,
            record_start=None):
    """Publish trainer identity and IN_PROGRESS in the observed attempt txn."""
    from orze.engine import launcher
    with execution_transaction(lake, idea_dir) as tx:
        row = require_current(tx.conn, tp.attempt_ref, states=("LAUNCHING",))
        from orze.engine.training_supervision import PROTOCOL, ready_binding
        if row["binding"].get("process_supervision_protocol") != PROTOCOL:
            raise AttemptEffectBusy("training_supervision_unbound")
        supervision = ready_binding(tp, idea_dir)
        claim, claim_sha = _claim(tp, idea_dir, lake)
        if (claim_sha != row["binding"].get("claim_sha256")
                or not canonical_identity_equal(_launch_state(lake, tp.idea_id),
                                                row["binding"].get("launch_lifecycle"))):
            raise StaleAttempt("training_launch_authority_changed")
        if (process_identity.get("pid") != tp.process.pid
                or type(process_identity.get("pid")) is not int
                or type(process_identity.get("start_ticks")) is not int):
            raise AttemptAuthorityError("training_process_identity_mismatch")
        if process_identity["start_ticks"] != supervision["worker"]["start_ticks"]:
            raise AttemptAuthorityError("training_process_identity_mismatch")
        from orze.engine.native_evaluation import _verify_compute
        if record_start is not None:
            record_start(tp, idea_dir, phase="training")
        start, _ = _read(idea_dir / "_compute_receipts" / tp.attempt_id / "start.json")
        _verify_compute(idea_dir, start, process=tp, phase="training",
                        event="start", outcome="started")
        claim.update({
            "trainer_pid": process_identity["pid"],
            "trainer_pgid": process_identity["pgid"],
            "trainer_start_ticks": process_identity["start_ticks"],
            "trainer_started_at": tp.start_time,
        })
        try:
            launcher.atomic_write(idea_dir / "claim.json", json.dumps(claim, indent=2))
            actual, _ = _claim(tp, idea_dir, lake)
            if not canonical_identity_equal(actual, claim):
                raise AttemptAuthorityError("training_started_claim_readback_failed")
            if not lake._record_state_transition_in_tx(
                    tp.idea_id, "CLAIMED", "IN_PROGRESS",
                    reason=f"training_launched on gpu {tp.gpu}", host=socket.gethostname(),
                    pid=tp.process.pid, sop_type="training"):
                raise AttemptAuthorityError("training_started_lifecycle_rejected")
            binding = dict(row["binding"])
            binding.update({"process_pid": tp.process.pid, "supervision": supervision,
                            "lifecycle": lifecycle_fence(lake, tp.idea_id, "training")})
            mark_running(tx.conn, tp.attempt_ref, binding=binding)
            if resume_context:
                launcher.mark_resume_launched(resume_context, idea_dir / "claim.json", effect_lease=tx.lease)
            tx.watch_attempt(tp.attempt_ref)
        except BaseException as exc:
            # The durable LAUNCHING intent and retained guard mediate a
            # partially changed claim. A stopped child does not undo files.
            raise AttemptEffectInDoubt("training_start_publication_unconfirmed") from exc


def failed_launch(lake, tp, idea_dir, ret, *, not_started=False):
    """Close a known unstarted/stopped launch without inventing task completion."""
    from orze.engine.accounting import record_compute_terminal
    from orze.engine.native_evaluation import _verify_compute
    from orze.engine.training_supervision import require_closed, bind_launch_cleanup
    if not_started and (getattr(tp, "process", None) is not None
                        or getattr(tp, "_termination_unconfirmed", False) is True):
        raise AttemptEffectBusy("training_created_process_cannot_be_not_started")
    ref = tp.attempt_ref
    closure = None
    if not not_started:
        row = require_current(lake.conn, ref, states=("LAUNCHING", "RUNNING"))
        closure = require_closed(tp, row, idea_dir, ret, allow_launch_cleanup=True)
    with execution_transaction(lake, idea_dir) as tx:
        row = require_current(tx.conn, ref, states=("LAUNCHING", "RUNNING"))
        _claim(tp, idea_dir, lake)
        if not not_started and type(ret) is not int:
            raise AttemptEffectBusy("training_failed_launch_exit_unconfirmed")
        if closure is not None and not canonical_identity_equal(closure, require_closed(
                tp, row, idea_dir, ret, allow_launch_cleanup=True)):
            raise AttemptEffectBusy("training_process_tree_receipt_changed")
        plan = {
            "operation": "training_failed_launch", "return_code": ret,
            "not_started": not_started,
        }
        if closure is not None:
            plan["process_tree"] = closure
        digest = tx.prepare(ref, plan)
        if row["state"] == "LAUNCHING" and not not_started:
            mark_running(tx.conn, ref, binding=bind_launch_cleanup(row, closure))
        if not not_started:
            payload = record_compute_terminal(
                tp, idea_dir, "failed", "training_launch_initialization_failed",
                phase="training", return_code=ret)
            _verify_compute(idea_dir, payload, process=tp, phase="training",
                            event="terminal", outcome="failed",
                            reason_code="training_launch_initialization_failed",
                            return_code=ret, require_start=row["state"] == "RUNNING")
        terminal = {"outcome": "not_started" if not_started else "failed",
                    "return_code": ret, "effect_receipt_sha256": digest}
        if closure is not None:
            terminal["process_tree"] = closure
        if finish_attempt(tx.conn, ref, terminal, not_started=not_started) != "committed":
            raise AttemptAuthorityError("training_failed_launch_not_new")
