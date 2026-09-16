"""Independent stdlib read-only audit of actual cold-host CPU fixture records.

Does not import product code, query current process IDs, signal processes,
access managers, or claim to repeat a past kernel observation.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(connection, table):
    if connection.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone() is None:
        return []
    return [dict(row) for row in connection.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]


def main():
    base, output = map(Path, sys.argv[1:])
    report = []
    prefixes = ('test_closed_host_recovers_', 'test_competing_prepared_', 'test_prepared_recovery_',
                'test_closed_record_', 'test_recovery_preparation_')
    for root in sorted(base.glob('test_*')):
        if root.is_symlink() or not root.name.startswith(prefixes) or not (root / 'births.json').is_file():
            continue
        births = read(root / 'births.json')
        with sqlite3.connect((root / 'lake.db').as_uri() + '?mode=ro', uri=True) as connection:
            connection.row_factory = sqlite3.Row
            db = {name: rows(connection, name) for name in ('controller_instances', 'controller_sessions',
                'controller_scope_heads', 'controller_handoffs', 'execution_attempts', 'cpu_action_reservations',
                'cpu_action_scopes', 'cpu_action_decisions')}
        instances = sorted(db['controller_instances'], key=lambda row: row['generation'])
        sessions = {row['controller_id']: row for row in db['controller_sessions']}
        assert [row['generation'] for row in instances] == list(range(len(instances)))
        assert len(instances) == len(births) and len(db['controller_scope_heads']) == 1
        head = db['controller_scope_heads'][0]
        assert head['current_id'] == instances[-1]['controller_id'] and head['pending_grant'] is None
        assert all(row['hold_reason'] is None for row in instances)
        birth_by_process = {(row['process']['pid'], row['process']['start_ticks']): row for row in births}
        hosts = {}
        for folder in root.glob('*.host.lock'):
            owner = read(folder / 'owner.json')
            svc = Path(owner['service_config_file'])
            # Negative source-change fixtures intentionally retain altered files.
            if not root.name.startswith('test_prepared_recovery_'):
                assert owner['service_sha256'] == sha(svc)
            hosts[owner['process']['pid']] = folder
        for index, instance in enumerate(instances):
            identity = json.loads(instance['identity_json'])
            process = identity['process']
            birth = birth_by_process[(process['pid'], process['start_ticks'])]
            assert birth['parent'] in hosts
            assert instance['predecessor'] == (None if index == 0 else instances[index-1]['controller_id'])
            if index:
                assert birth['old_exited_before_birth'] is True
        complete = root.name.startswith(('test_closed_host_recovers_', 'test_competing_prepared_'))
        attempts, reservations = db['execution_attempts'], db['cpu_action_reservations']
        assert len(attempts) == len(reservations)
        for attempt in attempts:
            terminal = json.loads(attempt['terminal_json'])
            assert attempt['state'] == 'TERMINAL' and terminal['return_code'] == 0
            assert terminal['outcome'] == 'completed' and terminal['process_tree']['event'] == 'TREE_CLOSED'
            assert terminal['process_tree']['wait_proof'] == 'ECHILD_WALL'
        if complete:
            assert all(row['state'] == 'SETTLED' for row in reservations)
            assert len(db['cpu_action_scopes']) == 1
            scope = json.loads(db['cpu_action_scopes'][0]['binding_json'])
            assert scope['declaration']['slots'] == 1
            assert len(db['controller_handoffs']) == len(instances) - 1
            for grant in db['controller_handoffs']:
                assert grant['state'] == 'STARTED' and grant['hold_reason'] is None
                payload = json.loads(grant['payload_json'])
                source, target = instances[payload['source_generation']], instances[payload['target_generation']]
                assert source['controller_id'] == payload['source_controller_id']
                assert target['controller_id'] == payload['target_controller_id']
                assert payload['source_ack_json'] == sessions[source['controller_id']]['ack_json']
            for instance in instances:
                session = sessions[instance['controller_id']]
                ack = json.loads(session['ack_json'])
                assert ack['resources']['lake'] == 'closed'
            for folder in hosts.values():
                if (folder / 'ready.json').exists():
                    ready, closed = read(folder / 'ready.json'), read(folder / 'closed.json')
                    assert ready['controller_id'] == closed['controller_id']
                    assert closed['kind'] == 'stopped'
            if root.name.startswith('test_closed_host_recovers_'):
                assert len(instances) == 3
                declaration = scope['declaration']
                if declaration['wall_budget_seconds'] is None:
                    assert declaration['version'] == 2 and len(attempts) == 3
                else:
                    assert declaration['wall_budget_seconds'] == 120 and len(attempts) == 2
                    assert json.loads(db['cpu_action_scopes'][0]['stop_json'])['reason'] == 'wall_envelope_exhausted'
            else:
                assert len(instances) == 2 and len(attempts) == 2
                assert len(list(root.glob('recovery-ready-*'))) == 2
                assert len(hosts) == 3 and sum((folder / 'ready.json').exists() for folder in hosts.values()) == 2
        else:
            assert len(instances) == 1 and not db['controller_handoffs']
            if root.name.startswith('test_recovery_preparation_'):
                assert (root / 'collision.json').read_bytes() == b'operator-owned configuration\n'
            if root.name.startswith('test_closed_record_'):
                assert (root / 'service.json.host.lock/closed.simulated.json').exists()
                assert not (root / 'live-refused.json').exists()
        report.append({'root': str(root), 'positive': complete, 'controllers': len(instances),
                       'native_actions': len(attempts), 'births_sha256': sha(root / 'births.json'),
                       'database_sha256': sha(root / 'lake.db')})
    summary = {'scopes': len(report), 'positive_scopes': sum(row['positive'] for row in report),
               'controllers': sum(row['controllers'] for row in report),
               'native_actions': sum(row['native_actions'] for row in report)}
    assert summary == {'scopes': 11, 'positive_scopes': 3, 'controllers': 16, 'native_actions': 13}, summary
    with output.open('x') as stream:
        stream.write(json.dumps({'summary': summary, 'runs': report, 'script_sha256': sha(Path(__file__))},
                               sort_keys=True, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
