"""Explicit native replication admission and launch-time authorization.

CALLING SPEC:
    request_replication(source_task_id, results_dir, cfg, lake, *, request_id,
                        reason='explicit_replication') -> status/request_id/task_id
        Reads and pins a confirmed B1 source, hashes launch inputs outside the
        writer, then atomically creates one request and one new QUEUED task.
    replication_authorization(lake, idea_id, idea_dir, cfg, execution_identity,
                              *, claim_id=None) -> dict | None
        Bounded read-only revalidation, including inside native begin's owned
        transaction. No request mapping means ordinary dedup remains enabled.

The first adapter supports confirmed native training with an artifact contract.
No config salt, new seed, OS launch, source append, or observation is invented.
One request authorizes a fixed task; its ordinary bounded failure retries remain
subject to generic attempt/FSM authority. Historical reads alone grant nothing.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import sqlite3
import sys
import uuid

from orze.core.artifact_contract import (
    artifact_publication_binding, validate_artifact_publication_binding,
)
from orze.core.execution_attempts import AttemptRef, current_attempt
from orze.core.integrity import hash_config
from orze.core.replication_requests import (
    ReplicationError, canonical, digest, get_request, insert_request,
    request_for_task, seal_record, token,
)
from orze.core.research_artifacts import artifacts_for_attempt
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.claim_authority import _lake_path, read_claim
from orze.engine.completion_events import CompletionEvent, require_completion
from orze.engine.execution_authority import execution_transaction, lifecycle_fence
from orze.engine.execution_catalog import declared_catalog


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _path(value):
    path = Path(os.path.abspath(value))
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ReplicationError("replication_path_redirected")
    return path


def _scope(lake, results_dir, cfg):
    from orze.reporting.evidence import report_lifecycle_db_path
    if lake is None or type(cfg) is not dict:
        raise ReplicationError("replication_native_catalog_required")
    results = _path(results_dir)
    database = _lake_path(lake)
    if not database:
        raise ReplicationError("replication_native_catalog_required")
    _path(database)
    if (cfg.get("results_dir") is not None and _path(cfg["results_dir"]) != results
            or "idea_lake_db" in cfg
            and str(_path(report_lifecycle_db_path(results, cfg))) != database):
        raise ReplicationError("replication_catalog_scope_mismatch")
    return results, database


def _config(raw):
    from orze.reporting.catalog import _display_config
    parsed, available = _display_config(raw)
    if not available:
        raise ReplicationError("replication_config_unavailable")
    return parsed


def _task(lake, task_id):
    rows = lake.conn.execute(
        "SELECT idea_id, CASE WHEN typeof(config)='text' AND "
        "length(CAST(config AS BLOB))<=65536 THEN config ELSE NULL END, "
        "kind,priority,category FROM main.ideas WHERE idea_id=? COLLATE BINARY LIMIT 2",
        (task_id,),
    ).fetchall()
    if (len(rows) != 1 or rows[0][0] != task_id or type(rows[0][1]) is not str
            or rows[0][2] != "train" or type(rows[0][3]) is not str or len(rows[0][3]) > 128
            or rows[0][4] is not None and (type(rows[0][4]) is not str or len(rows[0][4]) > 128)):
        raise ReplicationError("replication_task_unavailable")
    return {"config": rows[0][1], "kind": rows[0][2],
            "priority": rows[0][3], "category": rows[0][4]}


def _file_config(folder):
    from orze.engine.observation_publication import _read
    raw = _read(folder / "idea_config.yaml", 65536)
    return raw, _config(raw.decode("utf-8"))


def _source(lake, task_id, results, cfg):
    """Only bounded receipts/config and metadata; never artifact content hash."""
    from orze.engine.training_attempts import _read
    row = current_attempt(lake.conn, task_id, "training")
    if row is None:
        raise ReplicationError("replication_confirmed_native_source_required")
    ref = AttemptRef(task_id, "training", row["attempt_id"], row["generation"])
    folder = results / task_id
    row = require_completion(CompletionEvent(task_id, None, ref), lake, results, phase="training")
    if (row["state"] != "TERMINAL" or row["terminal"].get("outcome") != "completed"
            or row["binding"].get("origin") != "native_training"
            or lifecycle_fence(lake, task_id, "training")["phase_state"] != "COMPLETE"):
        raise ReplicationError("replication_confirmed_native_source_required")
    publication = validate_artifact_publication_binding(row["binding"].get("artifact_publication"))
    if publication["scope"] != str(results):
        raise ReplicationError("replication_source_scope_mismatch")
    start, _ = _read(folder / "_compute_receipts" / ref.attempt_id / "start.json")
    expected = {"idea_id": task_id, "attempt_id": ref.attempt_id, "phase": "training",
                "event": "start", "outcome": "started", "process_pid": row["binding"].get("process_pid")}
    if canonical({k: start.get(k) for k in expected}) != canonical(expected):
        raise ReplicationError("replication_source_start_mismatch")
    execution = start.get("execution_identity_sha256")
    requested = artifact_publication_binding(cfg, folder, execution)
    if canonical(requested) != canonical(publication):
        raise ReplicationError("replication_source_contract_changed")
    records = artifacts_for_attempt(lake.conn, ref)
    if (sorted(row["terminal"].get("artifact_ids", [])) != sorted(r["artifact_id"] for r in records)
            or set(publication["contract"]["outputs"]) != {r["logical_name"] for r in records}
            or any(r["producer"] != asdict(ref) or r["scope"] != str(results)
                   or r["spec_fingerprint"] != publication["spec_fingerprint"] for r in records)):
        raise ReplicationError("replication_source_artifacts_mismatch")
    task = _task(lake, task_id)
    raw, config = _file_config(folder)
    if canonical(config) != canonical(_config(task["config"])):
        raise ReplicationError("replication_source_config_changed")
    pin = {
        "source_ref": asdict(ref), "source_row_sha256": digest(row),
        "source_terminal_sha256": digest(row["terminal"]),
        "source_config_sha256": _sha(task["config"].encode("utf-8")),
        "source_file_sha256": _sha(raw),
        "artifact_records_sha256": digest({"records": records}),
        "execution_identity": execution, "spec_fingerprint": publication["spec_fingerprint"],
        "artifact_binding_sha256": digest(publication),
    }
    return pin, task, config


def _execution_inputs(folder, cfg, config):
    from orze.engine.launcher import _resolve_train_script
    from orze.engine.execution_identity import compute_execution_identity
    from orze.engine.artifact_publication import _path_identities, _verify_identities
    script = cfg["train_script"]
    if config.get("train_script"):
        script = _resolve_train_script(config["train_script"], cfg)
    paths = [folder / "idea_config.yaml", Path(cfg["base_config"]), Path(script)]
    identities = tuple(entry for path in paths for entry in _path_identities(_path(path)))
    execution = compute_execution_identity(
        config_path=paths[0], base_config_path=paths[1], train_script=paths[2],
        python=str(cfg.get("python", sys.executable)),
        train_extra_args=list(cfg.get("train_extra_args") or []),
        train_extra_env=dict(cfg.get("train_extra_env") or {}),
        data_boundaries=dict(cfg.get("data_boundaries") or {}),
        data_separation=dict(cfg.get("data_separation") or {}),
    )
    _verify_identities(identities)
    return execution, identities


def _matches(pin, record):
    if canonical(pin) != canonical({key: record[key] for key in pin}):
        raise ReplicationError("replication_source_changed")


def _target(lake, record):
    target = _task(lake, record["task_id"])
    if _sha(target["config"].encode("utf-8")) != record["source_config_sha256"]:
        raise ReplicationError("replication_target_config_changed")
    return target


def _insert_target(lake, task_id, source_id, task, reason):
    from orze.core.proposal_admission import _new_rows
    if lake.conn.execute(
            "SELECT 1 FROM temp.sqlite_master WHERE type IN ('table','view') AND name COLLATE NOCASE "
            "IN ('ideas','idea_state','idea_stage_state','idea_transitions','idea_stage_transitions') LIMIT 1",
    ).fetchone():
        raise ReplicationError("replication_lifecycle_namespace_invalid")
    for table in ("ideas", "idea_state", "idea_stage_state", "idea_transitions", "idea_stage_transitions"):
        if lake.conn.execute(f"SELECT 1 FROM main.{table} WHERE idea_id=? COLLATE BINARY LIMIT 1", (task_id,)).fetchone():
            raise ReplicationError("replication_target_identity_exists")
    if current_attempt(lake.conn, task_id, "training") is not None:
        raise ReplicationError("replication_target_identity_exists")
    prepared = {
        "idea_id": task_id, "id_num": None, "title": f"Replication of {source_id}",
        "priority": task["priority"], "category": task["category"], "parent": source_id,
        "hypothesis": reason, "config": task["config"],
        "config_hash": hash_config(_config(task["config"])),
        "config_source_sha256": _sha(task["config"].encode("utf-8")),
        "raw_markdown": "", "config_summary": None, "eval_metrics": None,
        "status": "queued", "training_time": None, "created_at": None,
        "approach_family": "other", "kind": "train",
    }
    # This private primitive inserts only; it does not call ordinary proposal
    # dedup or own a transaction. No public skip-dedup flag is introduced.
    _new_rows(lake, prepared)


def _queued_snapshot(lake, task_id):
    values = {}
    for table in ("ideas", "idea_state", "idea_transitions"):
        rows = lake.conn.execute(f"SELECT * FROM main.{table} WHERE idea_id=? COLLATE BINARY", (task_id,)).fetchall()
        values[table] = [dict(row) for row in rows]
    return canonical(values)


def request_replication(source_task_id, results_dir, cfg, lake, *, request_id,
                        reason="explicit_replication"):
    """Create exactly one persistent queued task per explicit request key."""
    token(source_task_id)
    token(request_id)
    if type(reason) is not str or not reason.strip() or len(reason.encode("utf-8")) > 1024:
        raise ReplicationError("replication_reason_invalid")
    if lake is not None and lake.conn.in_transaction:
        raise ReplicationError("replication_caller_transaction_active")
    try:
        results, database = _scope(lake, results_dir, cfg)
        pin, task, config = _source(lake, source_task_id, results, cfg)
        execution, identities = _execution_inputs(results / source_task_id, cfg, config)
        if execution != pin["execution_identity"]:
            raise ReplicationError("replication_execution_inputs_changed")
        from orze.engine.artifact_publication import _verify_identities
        with execution_transaction(lake, results / source_task_id) as tx:
            _verify_identities(identities)
            current_pin, task, _ = _source(lake, source_task_id, results, cfg)
            _matches(current_pin, pin)
            existing = get_request(tx.conn, request_id)
            if existing is not None:
                if (existing["source_ref"]["task_id"] != source_task_id
                        or existing["scope"] != str(results) or existing["database"] != database
                        or existing["reason"] != reason):
                    raise ReplicationError("replication_request_conflict")
                _matches(pin, existing)
                _target(lake, existing)
                record, status = existing, "already_requested"
            else:
                record = seal_record({
                    "schema": 1, "request_id": request_id,
                    "task_id": "idea-rep-" + uuid.uuid4().hex,
                    "scope": str(results), "database": database, **pin,
                    "reason": reason, "created_at": lake._transition_time(tx.conn),
                })
                _insert_target(lake, record["task_id"], source_task_id, task, reason)
                target_snapshot = _queued_snapshot(lake, record["task_id"])
                insert_request(tx.conn, record)
                if _queued_snapshot(lake, record["task_id"]) != target_snapshot:
                    raise ReplicationError("replication_target_readback_changed")
                status = "created"
            _matches(_source(lake, source_task_id, results, cfg)[0], record)
            if canonical(get_request(tx.conn, request_id)) != canonical(record):
                raise ReplicationError("replication_request_readback_changed")
            _verify_identities(identities)
            tx.watch_dependency(AttemptRef(**record["source_ref"]))
        return {"status": status, "request_id": request_id, "task_id": record["task_id"]}
    except (ReplicationError, AttemptEffectInDoubt):
        raise
    except Exception as exc:
        raise ReplicationError("replication_request_rejected:" + str(exc)) from exc


def replication_authorization(lake, idea_id, idea_dir, cfg, execution_identity, *, claim_id=None):
    """Read-only launch gate; captured metadata must be rechecked in native begin."""
    token(idea_id)
    if lake is None:
        raise ReplicationError("replication_native_catalog_required")
    try:
        record = request_for_task(lake.conn, idea_id)
        if record is None:
            return None
        folder = _path(idea_dir)
        results, database = _scope(lake, folder.parent, cfg)
        if folder.name != idea_id or str(results) != record["scope"] or database != record["database"]:
            raise ReplicationError("replication_launch_scope_mismatch")
        pin, _, _ = _source(lake, record["source_ref"]["task_id"], results, cfg)
        _matches(pin, record)
        task = _target(lake, record)
        _, file_config = _file_config(folder)
        if canonical(file_config) != canonical(_config(task["config"])):
            raise ReplicationError("replication_target_file_changed")
        publication = artifact_publication_binding(cfg, folder, execution_identity)
        if (execution_identity != record["execution_identity"] or publication is None
                or digest(publication) != record["artifact_binding_sha256"]):
            raise ReplicationError("replication_launch_spec_changed")
        claim_data = read_claim(folder / "claim.json")
        declared = declared_catalog(folder)
        if (type(claim_id) is not str or not claim_data or claim_data.get("attempt_id") != claim_id
                or claim_data.get("lifecycle_db") != database
                or declared is not None and declared != database):
            raise ReplicationError("replication_launch_claim_mismatch")
        rows = lake.conn.execute(
            "SELECT i.status,s.current_state,t.to_state FROM main.ideas i "
            "JOIN main.idea_state s ON s.idea_id=i.idea_id COLLATE BINARY "
            "JOIN main.idea_transitions t ON t.idea_id=i.idea_id COLLATE BINARY "
            "AND t.id=(SELECT MAX(id) FROM main.idea_transitions WHERE idea_id=i.idea_id COLLATE BINARY) "
            "WHERE i.idea_id=? COLLATE BINARY LIMIT 2", (idea_id,),
        ).fetchall()
        if len(rows) != 1 or tuple(rows[0]) != ("running", "CLAIMED", "CLAIMED"):
            raise ReplicationError("replication_launch_not_claimed")
        return {key: record[key] for key in (
            "schema", "request_id", "task_id", "source_ref", "source_terminal_sha256",
            "scope", "spec_fingerprint", "execution_identity", "request_sha256")}
    except (ReplicationError, AttemptEffectInDoubt):
        raise
    except Exception as exc:
        raise ReplicationError("replication_authorization_rejected:" + str(exc)) from exc
