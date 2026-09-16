"""Stdlib-only static audit of closed service-host product fixtures.

No product imports, PID lookups, process signals, database writes or replay.
Transient liveness/order are supported by passing product tests, not this audit.
"""
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'service-host'
FIXTURES = Path('/tmp/sh-t7')
# births, registrations, grants, attempts; one additional independently owned B.
CASES = {
    'test_watchdog_handoff_keeps_ho0': (2, 2, ['STARTED'], 2),
    'test_host_refuses_changed_inpu0': (1, 1, [], 0),
    'test_host_refuses_changed_inpu1': (1, 1, [], 0),
    'test_host_refuses_changed_inpu2': (1, 1, [], 0),
    'test_initial_child_cannot_exec0': (1, 0, [], 0),
    'test_unknown_handoff_replays_o0': (2, 2, ['STARTED'], 0),
    'test_hosted_watchdog_without_a0': (0, 0, [], 0),
    'test_host_stop_before_initial_0': (1, 0, [], 0),
    'test_host_stop_before_successo0': (2, 1, ['SPAWNING'], 0),
    'test_two_live_projects_keep_se0': (2, 2, ['STARTED'], 1),
    'test_two_live_projects_keep_se0/project-b': (1, 1, [], 1),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(rows):
    return sha(b''.join(canonical(row) for row in rows))


def read(path):
    return json.loads(path.read_bytes())


def audit_case(name, expected):
    root = FIXTURES / name
    births = read(root / 'births.json')
    assert len(births) == expected[0]
    state = root / 'service.json.host.lock'
    if births:
        owner, boot = read(state / 'owner.json'), read(state / 'boot.json')
        assert owner['service_config_file'] == str(root / 'service.json')
        assert boot['service_sha256'] == owner['service_sha256']
        assert owner['nonce'] == read(state / 'lock.json')['owner_nonce']
        # One refusal deliberately appends a newline after startup.
        raw = (root / 'service.json').read_bytes()
        if name == 'test_host_refuses_changed_inpu1':
            assert raw.endswith(b'\n')
            raw = raw[:-1]
        assert owner['service_sha256'] == sha(raw)
        assert all(b['parent'] == owner['process']['pid'] for b in births)
        assert not births[0]['old_exited_before_birth']
        assert all(b['old_exited_before_birth'] for b in births[1:])
    else:
        assert not state.exists()
    db = root / 'lake.db'
    if expected[1] == 0:
        assert not db.exists() and not (state / 'ready.json').exists()
        assert not (root / 'must-not-run').exists()
        return {'case': name, 'births': len(births), 'controllers': 0, 'attempts': 0}
    original_sha = sha(db.read_bytes())
    conn = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA query_only=ON')
    conn.execute('BEGIN')
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        def rows(table, order=''):
            return [] if table not in tables else [dict(r) for r in conn.execute('SELECT * FROM ' + table + order)]
        controllers = rows('controller_instances', ' ORDER BY generation')
        sessions = {r['controller_id']: r for r in rows('controller_sessions')}
        members = rows('controller_members', ' ORDER BY member_id COLLATE BINARY')
        grants = rows('controller_handoffs')
        reservations = rows('cpu_action_reservations', ' ORDER BY reservation_id COLLATE BINARY')
        attempts = rows('execution_attempts')
        assert len(controllers) == len(sessions) == expected[1]
        assert [g['state'] for g in grants] == expected[2]
        assert len(attempts) == len(reservations) == len(members) == expected[3]
        assert [r['generation'] for r in controllers] == list(range(len(controllers)))
        scope_raw = rows('cpu_action_scopes')[0]['binding_json']
        scope = json.loads(scope_raw)
        assert scope['policy_sha256'] == sha(canonical({k: v for k, v in scope.items() if k != 'policy_sha256'}))
        assert scope['results_dir'] == str(root / 'results') and scope['database'] == str(db)
        by_id = {r['controller_id']: r for r in controllers}
        attempt_map, outcomes = {}, []
        for saved in attempts:
            value = {k: v for k, v in saved.items() if k not in ('binding_json', 'terminal_json')}
            value.update(binding=json.loads(saved['binding_json']), terminal=json.loads(saved['terminal_json']))
            ref = {k: value[k] for k in ('task_id', 'phase', 'attempt_id', 'generation')}
            terminal = value['terminal']
            assert value['state'] == 'TERMINAL' and value['hold_reason'] is None
            tree = terminal['process_tree']
            assert tree['event'] == 'TREE_CLOSED' and tree['wait_proof'] == 'ECHILD_WALL'
            assert tree['binding'] == value['binding']['supervision']
            assert tree['binding']['identity']['attempt_ref'] == ref and tree['reaped_children'] >= 1
            effect = root / 'results' / ref['task_id'] / '_execution_effects' / ref['attempt_id']
            prepared = (effect / 'prepared.json').read_bytes()
            assert sha(prepared) == terminal['effect_receipt_sha256']
            assert {k: json.loads(prepared)[k] for k in ref} == ref
            assert read(effect / 'committed.json') == {'schema_version': 1, 'event': 'effect_committed', **ref, 'prepared_sha256': sha(prepared)}
            attempt_map[ref['attempt_id']] = value
            outcomes.append(terminal['outcome'])
        charged = 0
        for saved in reservations:
            permit, ref = json.loads(saved['permit_json']), json.loads(saved['ref_json'])
            value = attempt_map[ref['attempt_id']]
            assert permit['budget_scope'] == scope and saved['state'] == 'SETTLED'
            assert value['binding']['reservation_id'] == saved['reservation_id'] == permit['reservation_id']
            assert saved['terminal_sha256'] == sha(canonical(value['terminal']))
            charged += int(permit['reserved_nanoseconds'])
        assert charged == expected[3] * 60_000_000_000
        member_owners = {}
        for saved in members:
            member = json.loads(saved['payload_json'])
            value = attempt_map[member['identity']['attempt_ref']['attempt_id']]
            assert member['controller_id'] == saved['controller_id']
            assert member['action_state'] == 'SETTLED' and member['os_state'] == 'CLOSED'
            assert member['closure'] == value['terminal']['process_tree']
            assert member['terminal_sha256'] == sha(canonical(value))
            member_owners.setdefault(saved['controller_id'], []).append((saved, member))
        for index, controller in enumerate(controllers):
            cid = controller['controller_id']
            identity = json.loads(controller['identity_json'])
            assert identity['process'] == births[index]['process']
            session = sessions[cid]
            binding, ack, request = [json.loads(session[k]) for k in ('binding_json', 'ack_json', 'request_json')]
            assert binding['identity'] == identity and binding['physical_gpus'] == []
            assert ack['binding_sha256'] == sha(session['binding_json'].encode())
            assert ack['request_sha256'] == sha(session['request_json'].encode())
            assert ack['request_id'] == request['request_id'] == controller['request_id']
            own = member_owners.get(cid, [])
            assert ack['members'] == {'schema': 1, 'controller_id': cid, 'member_count': len(own),
                'members_sha256': digest([[r['member_id'], sha(r['payload_json'].encode())] for r, _ in own])}
            refs = {canonical(m['identity']['attempt_ref']).decode() for _, m in own}
            budget_rows = [r for r in reservations if r['ref_json'] in refs]
            columns = ('reservation_id', 'scope', 'task_id', 'slot', 'permit_json', 'ref_json', 'state', 'terminal_sha256')
            assert ack['resources']['cpu_budget'] == {'schema': 2, 'controller_id': cid,
                'policy_sha256': scope['policy_sha256'], 'reservation_count': len(own),
                'active_reservations': 0, 'reservations_sha256': digest([[r[k] for k in columns] for r in budget_rows])}
        assert read(state / 'ready.json')['controller_id'] == controllers[0]['controller_id']
        for grant in grants:
            payload = json.loads(grant['payload_json'])
            old, new = payload['source_controller_id'], payload['target_controller_id']
            assert payload['issuer'] == owner['process']
            for key in ('binding_json', 'request_json', 'ack_json'):
                assert payload['source_' + key] == sessions[old][key]
            assert payload['source_identity_json'] == by_id[old]['identity_json']
            if grant['state'] == 'STARTED':
                ready, started = json.loads(grant['ready_json']), json.loads(grant['started_json'])
                assert started == {'schema': 1, 'event': 'STARTED', 'grant_id': grant['grant_id'],
                    'controller_id': new, 'generation': by_id[new]['generation'], 'process': ready['process'],
                    'binding_sha256': sha(sessions[new]['binding_json'].encode()), 'resource': 'cpu',
                    'cpu_scope_sha256': sha(scope_raw.encode()), 'stage': 'cpu_budget_bound_before_first_action'}
                assert by_id[new]['predecessor'] == old
            else:
                assert new not in sessions and grant['ready_json'] is None and grant['started_json'] is None
        if expected[2] == ['SPAWNING']:
            assert not (state / 'closed.json').exists()
        else:
            closed = read(state / 'closed.json')
            final = controllers[-1]
            assert closed['kind'] == 'stopped' and closed['controller_id'] == final['controller_id']
            assert closed['ack_sha256'] == sha(sessions[final['controller_id']]['ack_json'].encode())
            assert closed['observed_process'] == births[-1]['process']
        if name == 'test_watchdog_handoff_keeps_ho0':
            assert sorted(outcomes) == ['completed', 'interrupted'] and (root / 'done').read_text() == 'done'
        if name == 'test_unknown_handoff_replays_o0':
            assert grants[0]['request_id'] == 'lost-reply' and (root / 'lost-started-reply-observed').exists()
        return {'case': name, 'births': len(births), 'controllers': len(controllers), 'grants': expected[2],
                'attempts': len(attempts), 'outcomes': sorted(outcomes), 'charged_nanoseconds': charged,
                'database_sha256': original_sha}
    finally:
        conn.close()
        assert sha(db.read_bytes()) == original_sha


def main():
    run = read(RAW / 'targeted-v7/run.json')
    assert run['frozen'] and run['exit_code'] == 0 and run['before'] == run['after']
    cases = [audit_case(name, expected) for name, expected in CASES.items()]
    report = {'schema': 1, 'status': 'passed', 'kind': 'independent_process_mechanical_check',
        'script_sha256': sha(Path(__file__).read_bytes()), 'cases': cases,
        'targeted_run_sha256': sha((RAW / 'targeted-v7/run.json').read_bytes()),
        'limits': 'Closed fixture bytes only. Kernel exit/order and transient refusals rely on product tests. '
                  'No real systemd, cgroup deployment, license, model, GPU or research-quality acceptance.'}
    with (RAW / 'product-verification-v1.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'passed', 'scopes': len(cases), 'native_attempts': sum(c['attempts'] for c in cases)}))


if __name__ == '__main__':
    main()
