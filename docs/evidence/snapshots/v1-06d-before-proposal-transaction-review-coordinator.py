"""Atomic ordinary CPU proposals, not replica or execution permissions.

propose captures selected current sources outside the writer, then composes
normal task admission and one historical request in the supplied main Lake.
It never claims, debits a budget, prepares a Domain, or starts a process.
Exact request replay retains its original outcome; ordinary config dedup is
unchanged. Commit acknowledgment fences immutable task/creation/request rows,
not a QUEUED state that a legitimate consumer may already have advanced.

Domain identity is origin/replay audit, not a private future execution grant.
Later execution uses that invocation's normal Domain and source admission.
recorded_proposals is bounded historical metadata, not live source authority.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from orze.core import cpu_action_budget as budget
from orze.core import cpu_proposal_requests as store
from orze.core import research_interfaces as interfaces
from orze.core.control_outcome import require_controller_start_allowed
from orze.core.cpu_execution import cpu_execution, execution_fingerprint
from orze.core.proposal_admission import _SOURCE_FIELDS, admit_proposal_in_tx
from orze.engine import cpu_action_sources as sources
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.replication import _scope


class ProposalHOLD(AttemptEffectInDoubt):
    """No acknowledged proposal outcome is available; never fall back to insert."""


def _fail(reason):
    raise ProposalHOLD("cpu_proposal_" + reason)


def _sha(raw):
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _namespace(conn):
    if conn.execute("SELECT 1 FROM temp.sqlite_master WHERE name COLLATE NOCASE IN "
            "('ideas','idea_state','idea_stage_state','idea_transitions',"
            "'idea_stage_transitions','cpu_proposal_requests','cpu_action_scopes') LIMIT 1").fetchone():
        _fail("temporary_catalog")


def _domain(cfg):
    if cpu_execution(cfg) is None:
        _fail("explicit_cpu_required")
    declaration = interfaces.domain_declaration(cfg)
    if declaration is None:
        _fail("selected_domain_required")
    entry = interfaces._entry(interfaces._DOMAINS, declaration["kind"])
    return {"declaration": declaration, "implementation_id": entry[0]}, entry


def _budget_stop(conn, scope, database, declaration, binding):
    # Admission may precede the first scheduler tick; never initialize a budget
    # merely to propose. An existing same-scope budget remains authoritative.
    if not conn.execute("SELECT 1 FROM main.sqlite_master WHERE name=? COLLATE NOCASE",
                        ("cpu_action_scopes",)).fetchone():
        return
    budget._schema(conn)
    rows = conn.execute("SELECT CASE WHEN typeof(binding_json)='text' AND "
        "length(CAST(binding_json AS BLOB))<=16384 THEN binding_json END "
        "FROM main.cpu_action_scopes WHERE scope=? LIMIT 2", (scope,)).fetchall()
    if not rows:
        return
    if len(rows) != 1 or type(rows[0][0]) is not str:
        _fail("budget_unavailable")
    captured = budget._scope(json.loads(rows[0][0]))
    if (captured["results_dir"] != scope or captured["database"] != database
            or store.canonical(captured["declaration"]) != store.canonical(declaration)
            or captured["database_identity"] != binding["database_identity"]
            or captured["directory_identity"] != binding["scope_identity"]):
        _fail("budget_scope_changed")
    if budget._scope_row(conn, captured) is not None:
        _fail("scope_stopped")


def _task_evidence(conn, task_id, transition_id=None):
    # Bound before fetching actual source text; no mutable status/result mirrors.
    sizes = "+".join("coalesce(length(CAST(" + key + " AS BLOB)),0)" for key in _SOURCE_FIELDS)
    rows = conn.execute("SELECT " + ",".join(_SOURCE_FIELDS) + " FROM main.ideas "
        "WHERE idea_id=? COLLATE BINARY AND (" + sizes + ")<=65536 LIMIT 2", (task_id,)).fetchall()
    if len(rows) != 1 or any(value is not None and type(value) is not str for value in rows[0]):
        _fail("admission_source_unavailable")
    evidence = {"task_id": task_id, "source_sha256": store.digest(dict(zip(_SOURCE_FIELDS, rows[0]))),
                "transition_id": transition_id, "transition_sha256": None}
    if transition_id is not None:
        rows = conn.execute("SELECT id,idea_id,from_state,to_state,reason,host,pid,sop_type,ts "
                            "FROM main.idea_transitions WHERE id=? LIMIT 2", (transition_id,)).fetchall()
        if (len(rows) != 1 or rows[0][1:5] != (task_id, "UNKNOWN", "QUEUED", "proposal_admitted")
                or rows[0][7] != "action"):
            _fail("creation_transition_changed")
        fields = ("id", "idea_id", "from_state", "to_state", "reason", "host", "pid", "sop_type", "ts")
        evidence["transition_sha256"] = store.digest(dict(zip(fields, rows[0])))
    return evidence


def _verify_record(conn, record):
    actual = store.get_request(conn, record["scope"], record["request_id"])
    if store.canonical(actual) != store.canonical(record):
        _fail("request_not_confirmed")
    evidence = record["admission_evidence"]
    if store.canonical(_task_evidence(conn, evidence["task_id"], evidence["transition_id"])) != store.canonical(evidence):
        _fail("admission_changed")


@contextmanager
def _readonly(database):
    conn = sqlite3.connect(Path(database).as_uri() + "?mode=ro", uri=True, timeout=1)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        yield conn
    finally:
        conn.close()


def _fresh_sources(conn, binding):
    # The actual B handle remains held and verifies file identities separately.
    # This reader fences real committed metadata on a fresh connection, not a
    # caller's potentially stale result proxy. It acquires no source write lock.
    for item in binding["inputs"]:
        record, row, _, _ = sources._metadata(SimpleNamespace(conn=conn), binding["scope"],
            binding["database"], item["artifact"]["artifact_id"])
        if (store.canonical(record) != store.canonical(item["artifact"])
                or store.digest(row) != item["source_sha256"]
                or row["terminal"].get("effect_receipt_sha256") != item["effect_sha256"]):
            _fail("committed_source_changed")


def propose(lake, results_dir, cfg, decision, *, expected_sources):
    """Return one durable normal-admission outcome, without executing a task."""
    conn = lake.conn
    if conn.in_transaction:
        _fail("caller_transaction")
    owns = False
    timeout = None
    try:
        decision = interfaces.validate_proposal_decision(decision)
        domain, entry = _domain(cfg)
        fingerprint = execution_fingerprint(cfg)
        _scope(lake, results_dir, cfg)
        route = sources._route(lake, results_dir)
        scope, database, _ = route
        if type(expected_sources) not in (list, tuple):
            _fail("captured_selection_required")
        expected = json.loads(store.canonical({"records": list(expected_sources)}))["records"]
        cap = sources.capture_sources(lake, results_dir, decision["domain_request"]["input_artifact_ids"])
        if store.canonical({"records": sources.records(cap)}) != store.canonical({"records": expected}):
            _fail("selected_sources_changed")
        binding = sources.snapshot(cap)
        raw = store.canonical({"kind": "native_cpu_action", "domain_request": decision["domain_request"]})
        prepared = lake.prepare_proposal(decision["task_id"], decision["domain_request"]["purpose"],
            raw, "", status="queued", kind="native_cpu_action", hypothesis=decision["reason"])
        pin = {"schema": 1, "request_id": decision["request_id"], "task_id": decision["task_id"],
               "scope": scope, "database": database, "decision": decision,
               "config_sha256": _sha(raw), "domain": domain, "source_snapshot": binding}

        def check_context():
            _namespace(conn)
            require_controller_start_allowed(Path(scope))
            current_domain, current_entry = _domain(cfg)
            if (lake.conn is not conn or sources._route(lake, results_dir) != route
                    or execution_fingerprint(cfg) != fingerprint or current_entry is not entry
                    or store.canonical(current_domain) != store.canonical(domain)):
                _fail("context_changed")
            sources.require_sources(lake, results_dir, cap, metadata_only=True)
            _budget_stop(conn, scope, database, cpu_execution(cfg), binding)

        check_context()
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        conn.execute("PRAGMA busy_timeout=1000")
        conn.execute("BEGIN IMMEDIATE")
        owns = True
        check_context()
        previous = store.get_request(conn, scope, decision["request_id"])
        if previous is not None:
            if store.canonical({key: previous[key] for key in pin}) != store.canonical(pin):
                _fail("request_identity_conflict")
            record = previous
        else:
            admitted = admit_proposal_in_tx(lake, prepared)
            outcome = {"request_id": decision["request_id"], "task_id": decision["task_id"],
                "status": admitted["status"], "reason": admitted["reason"],
                "existing_id": admitted.get("existing_id")}
            evidence = _task_evidence(conn, outcome["existing_id"] or decision["task_id"],
                                      admitted.get("transition_id"))
            record = store.seal_record({**pin, "outcome": outcome, "admission_evidence": evidence,
                "created_at": datetime.now(timezone.utc).isoformat()})
            store.insert_request(conn, record)
        _verify_record(conn, record)
        check_context()
        conn.commit()
        if conn.in_transaction:
            _fail("commit_not_confirmed")
        owns = False
        # Whole request and captured rows, not a global table fingerprint: peers
        # may add independent requests or legitimately claim this queued task.
        with _readonly(database) as fresh:
            _verify_record(fresh, record)
            _fresh_sources(fresh, binding)
            _budget_stop(fresh, scope, database, cpu_execution(cfg), binding)
        check_context()
        return json.loads(store.canonical(record["outcome"]))
    except ProposalHOLD:
        raise
    except Exception as exc:
        raise ProposalHOLD("cpu_proposal_unconfirmed") from exc
    finally:
        try:
            if owns and conn.in_transaction:
                conn.rollback()
        finally:
            if timeout is not None:
                conn.execute("PRAGMA busy_timeout=" + str(int(timeout)))


def recorded_proposals(lake, results_dir, *, limit=32):
    """Historical compact outcomes; no schema creation or current-source grant."""
    try:
        if lake.conn.in_transaction:
            _fail("caller_transaction")
        _namespace(lake.conn)
        route = sources._route(lake, results_dir)
        with _readonly(route[1]) as conn:
            view = store.records_view(conn, route[0], limit=limit)
            if any(store.get_request(conn, route[0], row["request_id"])["database"] != route[1]
                   for row in view["requests"]):
                _fail("historical_database_changed")
        if sources._route(lake, results_dir) != route:
            _fail("route_changed")
        return json.loads(store.canonical({"results": [row["outcome"] for row in view["requests"]],
                                          "more_available": view["more_available"]}))
    except ProposalHOLD:
        raise
    except Exception as exc:
        raise ProposalHOLD("cpu_proposal_view_unavailable") from exc
