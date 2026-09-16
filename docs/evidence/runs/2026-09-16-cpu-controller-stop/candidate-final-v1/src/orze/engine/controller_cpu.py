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
