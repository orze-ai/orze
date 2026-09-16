"""Independent stdlib audit of CLOSED targeted-v4 fixtures; no product imports.

No PID probes, executions, repairs, SQLite writes or reclassification of test
counts as research evidence. Live exit/order and transient fault assertions
remain supported by the actual passing product tests, not this static audit.
"""
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'cpu-controller-handoff'
FIXTURES = Path('/tmp/ch-t4')
CASES = {
    'test_cpu_handoff_replays_once_0': (3, 2, 2, 2, 'bounded'),
    'test_cpu_handoff_replays_once_1': (3, 2, 2, 2, 'continued'),
    'test_cpu_handoff_unknown_commi0': (2, 1, 0, 1, 'orphan_at_commit'),
    'test_cpu_handoff_unknown_commi1': (2, 1, 0, 0, 'lost_started_reply'),
    'test_cpu_handoff_unknown_commi2': (2, 1, 0, 0, 'policy_stop_at_commit'),
    'test_cpu_handoff_history_reche0': (1, 0, 1, 1, 'writer_settlement'),
    'test_cpu_handoff_history_reche1': (1, 0, 1, 1, 'effect_receipt'),
    'test_historical_cpu_terminal_d0': (0, 0, 2, 2, 'no_execution_generations'),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(values):
    value = hashlib.sha256()
    for row in values:
        value.update(canonical(row))
    return value.hexdigest()


def ref_from(row):
    return {key: row[key] for key in ('task_id', 'phase', 'attempt_id', 'generation')}


def audit_case(name, expected):
    root = FIXTURES / name
    db = root / 'lake.db'
    original_db_sha = sha(db.read_bytes())
    conn = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA query_only=ON')
    conn.execute('BEGIN')
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        def rows(table, order=''):
            if table not in tables:
                return []
            return [dict(row) for row in conn.execute('SELECT * FROM ' + table + order)]

        registrations = rows('controller_instances', ' ORDER BY generation')
        sessions = rows('controller_sessions')
        grants = rows('controller_handoffs')
        members = rows('controller_members', ' ORDER BY member_id COLLATE BINARY')
        reservations = rows('cpu_action_reservations', ' ORDER BY reservation_id COLLATE BINARY')
        attempts = rows('execution_attempts')
        assert (len(registrations), len(grants), len(attempts), len(reservations)) == expected[:4]
        assert len(sessions) == len(registrations)
        case = expected[4]
        stored_scope = rows('cpu_action_scopes')
        assert len(stored_scope) == 1
        scope_raw = stored_scope[0]['binding_json']
        scope = json.loads(scope_raw)
        assert canonical(scope).decode() == scope_raw
        assert scope['policy_sha256'] == sha(canonical({k: v for k, v in scope.items()
                                                       if k != 'policy_sha256'}))
        assert scope['results_dir'] == str(root / 'results') and scope['database'] == str(db)
        attempt_map = {}
        outcomes, corrupt_receipts = [], []
        for saved in attempts:
            value = {k: v for k, v in saved.items() if k not in ('binding_json', 'terminal_json')}
            value['binding'] = json.loads(saved['binding_json'])
            value['terminal'] = json.loads(saved['terminal_json'])
            assert set(value) == {'attempt_id', 'task_id', 'phase', 'generation', 'state',
                                  'binding', 'terminal', 'hold_reason'}
            assert value['state'] in {'TERMINAL', 'NOT_STARTED'} and value['hold_reason'] is None
            ref = ref_from(value)
            terminal = value['terminal']
            folder = root / 'results' / value['task_id'] / '_execution_effects' / value['attempt_id']
            prepared_raw = (folder / 'prepared.json').read_bytes()
            if case == 'effect_receipt':
                original = (root / 'original-effect.json').read_bytes()
                assert prepared_raw == original + b' '
                assert sha(prepared_raw) != terminal['effect_receipt_sha256']
                corrupt_receipts.append(str((folder / 'prepared.json').relative_to(root)))
                prepared_raw = original  # Compare the separately preserved original; never repair.
            assert sha(prepared_raw) == terminal['effect_receipt_sha256']
            prepared = json.loads(prepared_raw)
            assert {k: prepared[k] for k in ref} == ref
            committed = json.loads((folder / 'committed.json').read_bytes())
            assert committed == {'schema_version': 1, 'event': 'effect_committed', **ref,
                                 'prepared_sha256': sha(prepared_raw)}
            if value['state'] == 'TERMINAL':
                tree = terminal['process_tree']
                assert tree['event'] == 'TREE_CLOSED' and tree['wait_proof'] == 'ECHILD_WALL'
                assert tree['worker_returncode'] == terminal['return_code']
                assert tree['binding'] == value['binding']['supervision']
                assert tree['binding']['identity']['attempt_ref'] == ref
                assert tree['reaped_children'] >= 1
            else:
                assert case == 'no_execution_generations'
            attempt_map[value['attempt_id']] = value
            outcomes.append(terminal['outcome'])
        charged = 0
        for saved in reservations:
            permit = json.loads(saved['permit_json'])
            assert canonical(permit).decode() == saved['permit_json']
            assert permit['budget_scope'] == scope
            assert permit['reservation_id'] == saved['reservation_id']
            assert permit['task_id'] == saved['task_id'] and permit['slot'] == saved['slot']
            assert saved['scope'] == scope['results_dir']
            assert int(permit['reserved_nanoseconds']) == permit['wall_limit_seconds'] * 1_000_000_000
            charged += int(permit['reserved_nanoseconds'])
            if saved['ref_json'] is None:
                assert case == 'orphan_at_commit' and saved['state'] == 'RESERVED'
                assert saved['terminal_sha256'] is None
                continue
            ref = json.loads(saved['ref_json'])
            attempt = attempt_map[ref['attempt_id']]
            assert ref == ref_from(attempt) and saved['state'] == 'SETTLED'
            assert attempt['binding']['reservation_id'] == permit['reservation_id']
            assert saved['terminal_sha256'] == sha(canonical(attempt['terminal']))
        members_by_owner = {}
        for saved in members:
            member = json.loads(saved['payload_json'])
            assert canonical(member).decode() == saved['payload_json']
            assert member['controller_id'] == saved['controller_id']
            assert member['member_id'] == saved['member_id'] and member['kind'] == 'action'
            assert member['action_state'] == 'SETTLED' and member['os_state'] == 'CLOSED'
            assert member['hold_reason'] is None
            attempt = attempt_map[member['identity']['attempt_ref']['attempt_id']]
            assert ref_from(attempt) == member['identity']['attempt_ref']
            assert member['terminal_sha256'] == sha(canonical(attempt))
            assert member['closure'] == attempt['terminal']['process_tree']
            members_by_owner.setdefault(saved['controller_id'], []).append((saved, member))
        by_id = {row['controller_id']: row for row in registrations}
        session_by_id = {row['controller_id']: row for row in sessions}
        ack_count = 0
        for saved in sessions:
            owner = saved['controller_id']
            binding = json.loads(saved['binding_json'])
            assert binding['profile'] == {'version': 2, 'profile': 'local_cpu_handoff_v1'}
            assert binding['identity'] == json.loads(by_id[owner]['identity_json'])
            assert binding['physical_gpus'] == []
            if saved['ack_json'] is None:
                assert case in {'orphan_at_commit', 'policy_stop_at_commit'}
                assert by_id[owner]['generation'] == 1 and by_id[owner]['phase'] == 'HOLD'
                continue
            ack_count += 1
            ack = json.loads(saved['ack_json'])
            request = json.loads(saved['request_json'])
            assert ack['binding_sha256'] == sha(saved['binding_json'].encode())
            assert ack['request_sha256'] == sha(saved['request_json'].encode())
            assert ack['request_id'] == request['request_id'] == by_id[owner]['request_id']
            assert ack['controller_id'] == owner and by_id[owner]['phase'] == 'QUIESCING'
            own = members_by_owner.get(owner, [])
            assert ack['members'] == {'schema': 1, 'controller_id': owner, 'member_count': len(own),
                'members_sha256': digest([[row['member_id'], sha(row['payload_json'].encode())]
                                         for row, _ in own])}
            refs = {canonical(value['identity']['attempt_ref']).decode() for _, value in own}
            budget_rows = [row for row in reservations if row['ref_json'] in refs]
            columns = ['reservation_id', 'scope', 'task_id', 'slot', 'permit_json',
                       'ref_json', 'state', 'terminal_sha256']
            assert ack['resources'] == {'lake': 'closed', 'gpu_scope': [], 'gpu_leases': 'not_acquired',
                'pid_file': 'not_created', 'leadership': 'persistent_registration_retained',
                'request_pump': 'joined', 'cpu_budget': {'schema': 2, 'controller_id': owner,
                    'policy_sha256': scope['policy_sha256'], 'reservation_count': len(own),
                    'active_reservations': 0, 'reservations_sha256':
                    digest([[row[key] for key in columns] for row in budget_rows])}}
            assert len(budget_rows) == len(own)
        for saved in grants:
            payload, ready, started = [json.loads(saved[key]) for key in ('payload_json', 'ready_json', 'started_json')]
            old, new = payload['source_controller_id'], payload['target_controller_id']
            previous, successor = session_by_id[old], session_by_id[new]
            for field in ('binding_json', 'request_json', 'ack_json'):
                assert payload['source_' + field] == previous[field]
            assert payload['source_identity_json'] == by_id[old]['identity_json']
            assert payload['source_generation'] + 1 == payload['target_generation']
            assert started == {'schema': 1, 'event': 'STARTED', 'grant_id': saved['grant_id'],
                'controller_id': new, 'generation': by_id[new]['generation'], 'process': ready['process'],
                'binding_sha256': sha(successor['binding_json'].encode()), 'resource': 'cpu',
                'cpu_scope_sha256': sha(scope_raw.encode()), 'stage': 'cpu_budget_bound_before_first_action'}
            history_rows = [row for row in registrations if row['generation'] <= payload['source_generation']]
            assert payload['history_sha256'] == digest([[row['controller_id'], sha(row['identity_json'].encode()),
                *[sha(session_by_id[row['controller_id']][field].encode()) for field in
                  ('binding_json', 'request_json', 'ack_json')]] for row in history_rows])
        if case in {'bounded', 'continued'}:
            assert [row['generation'] for row in registrations] == [0, 1, 2]
            assert all(row['state'] == 'STARTED' for row in grants)
            assert sorted(outcomes) == ['completed', 'interrupted'] and charged == 120_000_000_000
            assert scope['declaration']['version'] == (2 if case == 'continued' else 1)
            assert len(list(root.glob('start-*.json'))) == len(list(root.glob('iteration-*.json'))) == 3
        elif case in {'orphan_at_commit', 'policy_stop_at_commit'}:
            assert grants[0]['state'] == 'PREPARED'
            assert len(list(root.glob('iteration-*.json'))) == 1
            assert charged == (1_000_000_000 if case == 'orphan_at_commit' else 0)
            if case == 'policy_stop_at_commit':
                assert json.loads(stored_scope[0]['stop_json'])['reason'] == 'fixture_policy_stop'
        elif case == 'lost_started_reply':
            assert grants[0]['state'] == 'STARTED' and charged == 0
        elif case == 'no_execution_generations':
            assert sorted(row['generation'] for row in attempts) == [1, 2]
            assert outcomes == ['not_started', 'not_started'] and charged == 4_000_000_000
        else:
            assert outcomes == ['completed'] and charged == 60_000_000_000
        assert not list((root / 'results').glob('.orze.pid*'))
        return {'case': name, 'kind': case, 'controllers': len(registrations), 'ack_count': ack_count,
            'grants': [r['state'] for r in grants], 'attempt_outcomes': outcomes,
            'charged_nanoseconds': charged, 'captured_corrupt_receipts': corrupt_receipts,
            'database_sha256': original_db_sha}
    finally:
        conn.close()
        assert sha(db.read_bytes()) == original_db_sha


def main():
    run = json.loads((RAW / 'targeted-v4/run.json').read_bytes())
    assert run['frozen'] and run['exit_code'] == 0
    reports = [audit_case(name, expected) for name, expected in CASES.items()]
    output = {'schema': 1, 'status': 'passed', 'kind': 'independent_process_mechanical_check',
        'script_sha256': sha(Path(__file__).read_bytes()),
        'limits': 'Static evidence only; live exit/order and transient faults are verified by product tests. '
                  'The receipt-corruption case remains corrupt. No research-quality experiment.',
        'targeted_run_sha256': sha((RAW / 'targeted-v4/run.json').read_bytes()), 'cases': reports}
    with (RAW / 'product-verification-v1.json').open('x') as stream:
        stream.write(json.dumps(output, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'product_cases': 7, 'native_attempts': 6,
                      'separate_no_execution_attempts': 2}))


if __name__ == '__main__':
    main()
