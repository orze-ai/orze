"""Explicit, same-Lake CPU occurrence admission and metadata-only launch gates.

Requests copy raw task configuration without a new seed, purpose or command.
The source must still be the current confirmed completed action. Verification
reads bounded SQL/receipts, not artifact contents or Domain callbacks; it grants
neither cross-task writes nor GO. Actual dispatch separately captures sources,
reserves its own wall envelope, claims and prepares a supervised action.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import uuid

from orze.core.artifact_contract import validate_artifact_publication_binding
from orze.core.cpu_action_contract import action_fingerprint, artifact_binding
from orze.core.cpu_execution import cpu_execution
from orze.core.execution_attempts import AttemptRef, current_attempt
from orze.core.integrity import hash_config
from orze.core.replication_requests import (
    ReplicationError, canonical, digest, get_request, insert_request,
    request_for_task, seal_record, token, validate_record,
)
from orze.engine.artifact_publication import _verify_identities
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.cpu_action_sources import _route, _effect
from orze.engine.cpu_policy_evidence import _read_sets
from orze.engine.execution_authority import execution_transaction
from orze.engine.replication import _config, _scope, _queued_snapshot


def _sha(raw):
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _task(lake, task_id):
    token(task_id)
    rows = lake.conn.execute(
        "SELECT CASE WHEN typeof(config)='text' AND length(CAST(config AS BLOB))<=65536 "
        "THEN config ELSE NULL END,kind,priority,category FROM main.ideas "
        "WHERE idea_id=? COLLATE BINARY LIMIT 2", (task_id,),
    ).fetchall()
    if (len(rows) != 1 or type(rows[0][0]) is not str or rows[0][1] != "native_cpu_action"
            or type(rows[0][2]) is not str or len(rows[0][2]) > 128
            or rows[0][3] is not None and (type(rows[0][3]) is not str or len(rows[0][3]) > 128)):
        raise ReplicationError("cpu_replication_task_unavailable")
    parsed = _config(rows[0][0])
    if parsed.get("kind", "native_cpu_action") != "native_cpu_action":
        raise ReplicationError("cpu_replication_config_kind_changed")
    return {"config": rows[0][0], "kind": rows[0][1], "priority": rows[0][2], "category": rows[0][3]}


def _source(lake, results, database, task_id):
    task = _task(lake, task_id)
    current = current_attempt(lake.conn, task_id, "action")
    if current is None:
        raise ReplicationError("cpu_replication_confirmed_source_required")
    ref = AttemptRef(task_id, "action", current["attempt_id"], current["generation"])
    row, artifacts, observations = _read_sets(lake, str(results), database, ref)
    binding = row["binding"]
    source = binding.get("source")
    if (row["terminal"]["outcome"] != "completed" or binding.get("origin") != "native_cpu_action"
            or binding.get("kind") != "native_cpu_action" or binding.get("resource") != "cpu"
            or binding.get("scope") != str(results / task_id)
            or type(source) is not dict or source.get("database") != database
            or source.get("config_sha256") != _sha(task["config"])):
        raise ReplicationError("cpu_replication_confirmed_source_required")
    publication = validate_artifact_publication_binding(binding.get("artifact_publication"))
    if (publication["scope"] != str(results)
            or publication["spec_fingerprint"] != binding.get("action_sha256")
            or set(publication["contract"]["outputs"]) != {r["logical_name"] for r in artifacts}):
        raise ReplicationError("cpu_replication_source_artifacts_changed")
    domain = binding.get("domain_run")
    configured = _config(task["config"])
    if domain is None:
        if action_fingerprint(configured.get("action")) != binding["action_sha256"]:
            raise ReplicationError("cpu_replication_source_action_changed")
    elif (type(domain) is not dict or domain.get("request_sha256") != _sha(task["config"])
          or domain.get("action_sha256") != binding["action_sha256"]):
        raise ReplicationError("cpu_replication_source_domain_changed")
    identities = _effect(ref, results / task_id, row)
    again = _read_sets(lake, str(results), database, ref)
    if canonical({"value": [row, artifacts, observations]}) != canonical({"value": list(again)}):
        raise ReplicationError("cpu_replication_source_read_changed")
    _verify_identities(identities)
    pin = {"adapter": "native_cpu_action", "source_ref": asdict(ref),
        "source_row_sha256": digest(row), "source_terminal_sha256": digest(row["terminal"]),
        "source_config_sha256": _sha(task["config"]),
        "artifact_records_sha256": digest({"records": artifacts}),
        "observation_records_sha256": digest({"records": observations}),
        "action_sha256": binding["action_sha256"], "domain_run_sha256": digest({"domain_run": domain}),
        "spec_fingerprint": publication["spec_fingerprint"], "artifact_binding_sha256": digest(publication)}
    return pin, task, publication, domain


def _match(pin, record):
    if canonical(pin) != canonical({key: record.get(key) for key in pin}):
        raise ReplicationError("cpu_replication_source_changed")


def _expected_ref(expected, actual):
    if expected is not None:
        if (type(expected) is not dict or set(expected) != set(actual)
                or asdict(AttemptRef(**expected)) != expected or canonical(expected) != canonical(actual)):
            raise ReplicationError("cpu_replication_expected_source_changed")


def _target(lake, record):
    task = _task(lake, record["task_id"])
    if _sha(task["config"]) != record["source_config_sha256"]:
        raise ReplicationError("cpu_replication_target_changed")
    return task


def verify_record(lake, results_dir, record):
    """Read-only current source/request/target fence, including in owned TXs."""
    try:
        record = validate_record(record)
        if record["schema"] != 2 or record.get("adapter") != "native_cpu_action":
            raise ReplicationError("cpu_replication_adapter_mismatch")
        scope, database, identities = _route(lake, results_dir)
        if scope != record["scope"] or database != record["database"]:
            raise ReplicationError("cpu_replication_scope_changed")
        if canonical(get_request(lake.conn, record["request_id"])) != canonical(record):
            raise ReplicationError("cpu_replication_request_changed")
        _target(lake, record)
        _match(_source(lake, Path(scope), database, record["source_ref"]["task_id"])[0], record)
        if _route(lake, results_dir) != (scope, database, identities):
            raise ReplicationError("cpu_replication_route_changed")
    except (ReplicationError, AttemptEffectInDoubt):
        raise
    except Exception as exc:
        raise ReplicationError("cpu_replication_verification_rejected") from exc


def _insert_target(lake, record, source):
    from orze.core.proposal_admission import _new_rows
    task_id = record["task_id"]
    if lake.conn.execute("SELECT 1 FROM temp.sqlite_master WHERE type IN ('table','view') "
            "AND name COLLATE NOCASE IN ('ideas','idea_state','idea_stage_state',"
            "'idea_transitions','idea_stage_transitions') LIMIT 1").fetchone():
        raise ReplicationError("cpu_replication_namespace_invalid")
    for table in ("ideas", "idea_state", "idea_stage_state", "idea_transitions", "idea_stage_transitions"):
        if lake.conn.execute(f"SELECT 1 FROM main.{table} WHERE idea_id=? COLLATE BINARY LIMIT 1",
                             (task_id,)).fetchone():
            raise ReplicationError("cpu_replication_target_exists")
    if lake.conn.execute("SELECT 1 FROM main.execution_attempts WHERE task_id=? COLLATE BINARY LIMIT 1",
                         (task_id,)).fetchone():
        raise ReplicationError("cpu_replication_target_exists")
    _new_rows(lake, {"idea_id": task_id, "id_num": None,
        "title": "Replication of " + record["source_ref"]["task_id"],
        "priority": source["priority"], "category": source["category"],
        "parent": record["source_ref"]["task_id"], "hypothesis": record["reason"],
        "config": source["config"], "config_hash": hash_config(_config(source["config"])),
        "config_source_sha256": _sha(source["config"]), "raw_markdown": "", "config_summary": None,
        "eval_metrics": None, "status": "queued", "training_time": None, "created_at": None,
        "approach_family": "other", "kind": "native_cpu_action"})


def request_replication(source_task_id, results_dir, cfg, lake, *, request_id,
                        reason="explicit_replication", expected_source_ref=None):
    """Atomically admit one unchanged CPU task; exact-key replay never resets it."""
    token(source_task_id)
    token(request_id)
    if type(reason) is not str or not reason.strip() or len(reason.encode()) > 1024:
        raise ReplicationError("replication_reason_invalid")
    if lake is not None and lake.conn.in_transaction:
        raise ReplicationError("replication_caller_transaction_active")
    try:
        if cpu_execution(cfg) is None:
            raise ReplicationError("cpu_replication_execution_required")
        results, database = _scope(lake, results_dir, cfg)
        route = _route(lake, results)
        pin, _, _, _ = _source(lake, results, database, source_task_id)
        _expected_ref(expected_source_ref, pin["source_ref"])
        with execution_transaction(lake, results / source_task_id) as tx:
            current_pin, task, _, _ = _source(lake, results, database, source_task_id)
            _match(current_pin, pin)
            _expected_ref(expected_source_ref, current_pin["source_ref"])
            record = get_request(tx.conn, request_id)
            if record is None:
                record = seal_record({"schema": 2, "request_id": request_id,
                    "task_id": "idea-rep-" + uuid.uuid4().hex, "scope": str(results), "database": database,
                    **pin, "reason": reason, "created_at": lake._transition_time(tx.conn)})
                _insert_target(lake, record, task)
                target = _queued_snapshot(lake, record["task_id"])
                insert_request(tx.conn, record)
                if _queued_snapshot(lake, record["task_id"]) != target:
                    raise ReplicationError("cpu_replication_target_readback_changed")
                status = "created"
            else:
                if record["reason"] != reason:
                    raise ReplicationError("cpu_replication_request_conflict")
                _match(pin, record)
                status = "already_requested"
            verify_record(lake, results, record)
            tx.watch_cpu_replication(record)
            if _route(lake, results) != route:
                raise ReplicationError("cpu_replication_route_changed")
        # A transaction-local readback cannot acknowledge a rolled-back commit.
        verify_record(lake, results, record)
        if _route(lake, results) != route:
            raise ReplicationError("cpu_replication_route_changed")
        return {"status": status, "request_id": request_id, "task_id": record["task_id"]}
    except (ReplicationError, AttemptEffectInDoubt):
        raise
    except Exception as exc:
        raise ReplicationError("cpu_replication_request_rejected") from exc


def authorization(lake, idea_id, results_dir, cfg, *, action, domain_run=None):
    """Pre-claim read gate; native claim, budget and strong owner remain required."""
    try:
        record = request_for_task(lake.conn, token(idea_id))
        if record is None:
            return None
        if cpu_execution(cfg) is None:
            raise ReplicationError("cpu_replication_execution_required")
        results, _ = _scope(lake, results_dir, cfg)
        verify_record(lake, results, record)
        from orze.core.research_interfaces import domain_run_metadata
        metadata = None if domain_run is None else domain_run_metadata(domain_run)
        if (action_fingerprint(action) != record["action_sha256"]
                or digest({"domain_run": metadata}) != record["domain_run_sha256"]
                or digest(artifact_binding(cfg, results / idea_id, action)) != record["artifact_binding_sha256"]):
            raise ReplicationError("cpu_replication_action_or_domain_changed")
        return json.loads(canonical(record))
    except (ReplicationError, AttemptEffectInDoubt):
        raise
    except Exception as exc:
        raise ReplicationError("cpu_replication_authorization_rejected") from exc
