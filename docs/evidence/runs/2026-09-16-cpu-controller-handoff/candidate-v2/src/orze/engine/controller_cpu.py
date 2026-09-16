"""CPU budget closure for the explicit registered stop profile.

This is read-only evidence, not launch or recovery permission. A native member
can be settled before its separate CPU budget transaction commits, so the
session must verify both obligations before publishing its drain ACK.
"""
from __future__ import annotations

import hashlib
import json

from orze.engine.controller_control import ControllerHOLD, _route
from orze.engine.supervisor_worker import canonical


def budget_drain_proof(conn, results_dir, declaration, controller_id):
    """Audit the complete v1 scope in the caller's existing read transaction.

The stop-only profile starts in a fresh ownership namespace. Every reservation
must therefore match one of this instance's native action members. In
particular a RESERVED row without an attempt cannot disappear from the proof.
This whole-scope digest is intentionally not a handoff-history format.
"""
    from orze.core import cpu_action_budget as budget, execution_attempts as attempts
    from orze.engine import controller_members as members
    try:
        budget._schema(conn)
        members._schema(conn)
        stored = conn.execute("SELECT CASE WHEN length(CAST(binding_json AS BLOB))<=16384 "
            "THEN binding_json END FROM main.cpu_action_scopes WHERE scope=?",
            (str(results_dir),)).fetchone()
        if stored is None:
            raise ControllerHOLD("controller_cpu_budget_missing")
        scope = budget._scope(budget._decode(stored[0]))
        db, identity = _route(conn)
        directory, dir_identity = budget._path(results_dir, directory=True)
        if (scope['database'] != str(db) or scope['database_identity'] != list(identity)
                or scope['results_dir'] != directory or scope['directory_identity'] != dir_identity
                or canonical(scope['declaration']) != canonical(declaration)
                or budget._key(scope) in budget._HELD):
            raise ControllerHOLD('controller_cpu_budget_binding_changed')
        budget._scope_row(conn, scope)
        _, active = budget._totals(conn, scope)
        if active:
            raise ControllerHOLD('controller_cpu_budget_unsettled')
        rows = conn.execute("SELECT CASE WHEN length(CAST(payload_json AS BLOB))<=65536 "
            "THEN payload_json END FROM main.controller_members WHERE controller_id=? "
            "ORDER BY member_id COLLATE BINARY LIMIT ?",
            (controller_id, members.MAX_MEMBERS + 1)).fetchall()
        if len(rows) > members.MAX_MEMBERS:
            raise ControllerHOLD('controller_cpu_member_limit')
        refs = {}
        for (raw,) in rows:
            member = json.loads(raw)
            ref = attempts.AttemptRef(**member['identity']['attempt_ref'])
            encoded = budget._ref(ref)
            if (member['kind'] != 'action' or member['controller_id'] != controller_id
                    or member['action_state'] != 'SETTLED' or encoded in refs):
                raise ControllerHOLD('controller_cpu_member_unsettled')
            refs[encoded] = ref
        reservations = conn.execute('SELECT ' + budget._RESERVATION_COLUMNS +
            ' FROM main.cpu_action_reservations WHERE scope=? '
            'ORDER BY reservation_id COLLATE BINARY LIMIT ?',
            (directory, members.MAX_MEMBERS + 1)).fetchall()
        if len(reservations) != len(refs):
            raise ControllerHOLD('controller_cpu_reservation_inventory_changed')
        digest = hashlib.sha256()
        for stored in reservations:
            permit = budget._permit(budget._decode(stored[4]))
            row = budget._reservation_row(stored, permit)
            ref = refs.pop(row[4], None)
            if ref is None or row[5] != 'SETTLED':
                raise ControllerHOLD('controller_cpu_reservation_unsettled')
            terminal = budget._bound_current(conn, permit, ref, states=('TERMINAL', 'NOT_STARTED'))
            if row[6] != budget._terminal(conn, permit, ref, terminal['terminal']):
                raise ControllerHOLD('controller_cpu_settlement_changed')
            digest.update(canonical(list(stored)))
        if refs:
            raise ControllerHOLD('controller_cpu_reservation_missing')
        return {'schema': 1, 'policy_sha256': scope['policy_sha256'],
            'reservation_count': len(reservations), 'active_reservations': 0,
            'reservations_sha256': digest.hexdigest()}
    except ControllerHOLD:
        raise
    except Exception as exc:
        raise ControllerHOLD('controller_cpu_budget_unconfirmed') from exc


def _history_scope(conn, results_dir, declaration):
    from orze.core import cpu_action_budget as budget
    budget._schema(conn)
    row = conn.execute("SELECT CASE WHEN length(CAST(binding_json AS BLOB))<=16384 "
        "THEN binding_json END FROM main.cpu_action_scopes WHERE scope=?",
        (str(results_dir),)).fetchone()
    if row is None:
        raise ControllerHOLD('controller_cpu_budget_missing')
    scope = budget._scope(budget._decode(row[0]))
    db, identity = _route(conn)
    directory, dir_identity = budget._path(results_dir, directory=True)
    if (scope['database'] != str(db) or scope['database_identity'] != list(identity)
            or scope['results_dir'] != directory or scope['directory_identity'] != dir_identity
            or canonical(scope['declaration']) != canonical(declaration)
            or budget._key(scope) in budget._HELD):
        raise ControllerHOLD('controller_cpu_budget_binding_changed')
    budget._scope_row(conn, scope)
    budget._recovery_available(conn, scope)
    return scope


def scope_readiness(conn, results_dir, declaration):
    """Stable scope identity in STARTED, independent of later legitimate work."""
    from orze.core import cpu_action_budget as budget
    try:
        scope = _history_scope(conn, results_dir, declaration)
        return {'resource': 'cpu', 'cpu_scope_sha256':
            hashlib.sha256(budget._json(scope).encode()).hexdigest(),
            'stage': 'cpu_budget_bound_before_first_action'}
    except ControllerHOLD:
        raise
    except Exception as exc:
        raise ControllerHOLD('controller_cpu_readiness_unconfirmed') from exc


def budget_history_proof(conn, results_dir, declaration, controller_id, *, closed_inventory=True):
    """Version 2: prove this controller's exact reservations, retaining history.

    Closed inventory checks all registered generations and all reservations.
    Historical-only reads allow later work but still recheck every captured row
    and its original attempt/effect. The per-controller digest never includes a
    later generation's reservations, and never grants old execution authority.
    """
    from orze.core import cpu_action_budget as budget, execution_attempts as attempts
    from orze.engine import controller_members as members
    try:
        scope = _history_scope(conn, results_dir, declaration)
        members._schema(conn)
        reservations = conn.execute('SELECT ' + budget._RESERVATION_COLUMNS +
            ' FROM main.cpu_action_reservations WHERE scope=? '
            'ORDER BY reservation_id COLLATE BINARY LIMIT ?',
            (scope['results_dir'], members.MAX_MEMBERS + 1)).fetchall()
        if len(reservations) > members.MAX_MEMBERS:
            raise ControllerHOLD('controller_cpu_history_limit')
        _, active = budget._totals(conn, scope)
        if closed_inventory and active:
            raise ControllerHOLD('controller_cpu_budget_unsettled')
        sql = ("SELECT m.controller_id,CASE WHEN length(CAST(m.payload_json AS BLOB))<=65536 "
               "THEN m.payload_json END FROM main.controller_members m ")
        if closed_inventory:
            sql += ('JOIN main.controller_instances i ON i.controller_id=m.controller_id '
                    'WHERE i.scope=? ORDER BY m.member_id COLLATE BINARY LIMIT ?')
            key = scope['results_dir']
        else:
            sql += 'WHERE m.controller_id=? ORDER BY m.member_id COLLATE BINARY LIMIT ?'
            key = controller_id
        rows = conn.execute(sql, (key, members.MAX_MEMBERS + 1)).fetchall()
        if len(rows) > members.MAX_MEMBERS:
            raise ControllerHOLD('controller_cpu_history_limit')
        refs, own = {}, set()
        for owner, raw in rows:
            member = json.loads(raw)
            ref = attempts.AttemptRef(**member['identity']['attempt_ref'])
            encoded = budget._ref(ref)
            if (member['kind'] != 'action' or member['controller_id'] != owner
                    or member['action_state'] != 'SETTLED' or member['hold_reason'] is not None
                    or encoded in refs or canonical(member).decode() != raw):
                raise ControllerHOLD('controller_cpu_member_unsettled')
            refs[encoded] = ref
            if owner == controller_id:
                own.add(encoded)
        if closed_inventory and len(reservations) != len(refs):
            raise ControllerHOLD('controller_cpu_reservation_inventory_changed')
        digest = hashlib.sha256()
        for stored in reservations:
            permit = budget._permit(budget._decode(stored[4]))
            row = budget._reservation_row(stored, permit)
            ref = refs.pop(row[4], None)
            if ref is None:
                if closed_inventory or row[4] in own:
                    raise ControllerHOLD('controller_cpu_reservation_unregistered')
                continue
            if row[5] != 'SETTLED' or not attempts._schema(conn):
                raise ControllerHOLD('controller_cpu_reservation_unsettled')
            terminal = attempts._row(conn.execute(attempts._SELECT +
                ' WHERE attempt_id=? COLLATE BINARY', (ref.attempt_id,)).fetchone())
            if (terminal is None or row[6] != budget._terminal_record(
                    permit, ref, terminal, terminal['terminal'])):
                raise ControllerHOLD('controller_cpu_settlement_changed')
            if row[4] in own:
                digest.update(canonical(list(stored)))
        if refs:
            raise ControllerHOLD('controller_cpu_reservation_missing')
        return {'schema': 2, 'controller_id': controller_id,
            'policy_sha256': scope['policy_sha256'], 'reservation_count': len(own),
            'active_reservations': 0, 'reservations_sha256': digest.hexdigest()}
    except ControllerHOLD:
        raise
    except Exception as exc:
        raise ControllerHOLD('controller_cpu_history_unconfirmed') from exc


def check_handoff_resources(session):
    """Read actual bound CPU resources before PREPARED and again at COMMIT."""
    from orze.core import cpu_action_budget as budget
    from orze.core.cpu_execution import cpu_execution
    from orze.engine.controller_handoff import _validate_history
    session._validate_runtime()
    engine = session.orze
    if (session._lake_conn.in_transaction or engine._cpu_closed
            or session._pid_file is not None or session._gpu is not None
            or engine._gpu_leases is not None):
        raise ControllerHOLD('controller_cpu_handoff_resources_unconfirmed')
    session._lake_conn.execute('SELECT 1')
    with session.ctx._connection() as conn:
        conn.execute('BEGIN')
        session._cpu_proof(conn)
        if budget._scope_row(conn, engine._cpu_scope) is not None:
            raise ControllerHOLD('controller_cpu_handoff_policy_stopped')
        admission = session._admission
        _validate_history(admission._route, admission.source_controller_id, admission.generation - 1,
            expected=admission._payload['history_sha256'], connection=conn)
        return scope_readiness(conn, engine.results_dir, cpu_execution(engine.cfg))
