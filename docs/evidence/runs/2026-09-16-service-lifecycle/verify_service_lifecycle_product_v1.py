"""Stdlib-only static audit; kernel exit/order remain product-test observations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'service-lifecycle'
FIXTURES = Path('/tmp/sl-t4')
HELPER = ROOT / 'orze/docs/evidence/runs/2026-09-16-service-host/verify_service_host_product_v1.py'
spec = importlib.util.spec_from_file_location('closed_cpu_records', HELPER)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
h.FIXTURES = FIXTURES
CASES = {
    'test_natural_completion_remain0': (1, 1, [], 0),
    'test_natural_completion_remain1': (1, 1, [], 1),
    'test_initial_completion_before0': (1, 1, [], 0),
    'test_successor_completion_befo0': (2, 2, ['STARTED'], 1),
    'test_successor_completion_befo1': (2, 2, ['STARTED'], 1),
    'test_closed_host_uninstall_pre0': (1, 1, [], 1),
    'test_closed_host_status_report0': (1, 1, [], 1),
}


def readonly_rows(root, query):
    path = root / 'lake.db'
    before = h.sha(path.read_bytes())
    conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        conn.execute('PRAGMA query_only=ON')
        return conn.execute(query).fetchall()
    finally:
        conn.close()
        assert h.sha(path.read_bytes()) == before


def audit():
    run = h.read(RAW / 'targeted-v4/run.json')
    assert run['frozen'] and run['exit_code'] == 0 and run['before'] == run['after']
    positives = [h.audit_case(name, expected) for name, expected in CASES.items()]
    for name in CASES:
        root = FIXTURES / name
        sessions = readonly_rows(root, 'SELECT controller_id,request_json FROM controller_sessions')
        final = h.read(root / 'service.json.host.lock/closed.json')['controller_id']
        assert json.loads(dict(sessions)[final])['reason'] == 'normal_exit'
        if name.startswith('test_successor'):
            assert (root / 'started-after-exit').exists()
            assert readonly_rows(root, 'SELECT request_id FROM controller_handoffs') == [('completed-successor',)]
        if name == 'test_successor_completion_befo1':
            assert (root / 'lost-completed-started').exists()
        if name == 'test_initial_completion_before0':
            assert (root / 'observed-after-exit').exists()
    negative = []
    for index, mutation in enumerate(('ack_missing', 'ack_hash', 'orphan_budget', 'foreign_process')):
        name = 'test_natural_exit_needs_origin' + str(index)
        root = FIXTURES / name
        births = h.read(root / 'births.json')
        assert len(births) == 1 and not (root / 'service.json.host.lock/closed.json').exists()
        request, ack = readonly_rows(root, 'SELECT request_json,ack_json FROM controller_sessions')[0]
        if mutation == 'ack_missing':
            assert ack is None
        elif mutation == 'ack_hash':
            assert json.loads(ack)['request_sha256'] != h.sha(request.encode())
        elif mutation == 'orphan_budget':
            assert readonly_rows(root, 'SELECT state,ref_json FROM cpu_action_reservations') == [('RESERVED', None)]
            assert readonly_rows(root, 'SELECT member_id FROM controller_members') == []
        else:
            identity = json.loads(readonly_rows(root, 'SELECT identity_json FROM controller_instances')[0][0])
            assert identity['process'] != births[0]['process']
        negative.append({'case': name, 'mutation': mutation, 'births': 1, 'closed_record': False})
    mutations = ('active_main', 'active_watchdog', 'active_timer', 'live_pid', 'remaining_cgroup',
                 'timer_target', 'install_also', 'ack_missing', 'closed_hash', 'after_first_disable')
    management = [('test_closed_host_uninstall_pre0', 'uninstall'), ('test_closed_host_status_report0', 'status')]
    management.extend(('test_closed_uninstall_rechecks' + str(i), m) for i, m in enumerate(mutations))
    for name, mutation in management:
        root = FIXTURES / name
        manager = h.read(root / 'manager-observation.json')
        calls = manager['calls']
        svc = h.read(root / 'service.json')
        key = hashlib.sha256(svc['service_config_file'].encode()).hexdigest()[:16]
        # Discover the selected names from the saved effective contract.
        effective = manager['effective']
        main = next(n for n in effective if n.endswith('.service') and not n.endswith('-watchdog.service'))
        timer = next(n for n in effective if n.endswith('.timer'))
        watchdog = next(n for n in effective if n.endswith('-watchdog.service'))
        assert key in main and key in timer and key in watchdog
        expected = ([['disable', n] for n in (main, timer, watchdog)] + [['daemon-reload']]
                    if mutation == 'uninstall' else [['disable', main]] if mutation == 'after_first_disable' else [])
        assert calls == expected and not any('--now' in call for call in calls)
        assert (root / 'units/another-project.service').read_text() == 'independent project'
        assert len(h.read(root / 'births.json')) == 1 and (root / 'done').read_text() == 'done'
        assert all((root / 'units' / n).exists() == (mutation != 'uninstall') for n in effective)
        assert readonly_rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',)]
        if mutation == 'ack_missing':
            assert readonly_rows(root, 'SELECT ack_json FROM controller_sessions') == [(None,)]
        elif mutation == 'closed_hash':
            saved = h.read(root / 'service.json.host.lock/closed.json')
            ack = readonly_rows(root, 'SELECT ack_json FROM controller_sessions')[0][0]
            assert saved['ack_sha256'] != h.sha(ack.encode())
        if mutation not in ('uninstall', 'status'):
            negative.append({'case': name, 'mutation': mutation, 'calls': calls, 'births': 1})
    return positives, negative


def main():
    positives, negative = audit()
    report = {'schema': 1, 'status': 'passed', 'kind': 'independent_process_mechanical_check',
              'script_sha256': h.sha(Path(__file__).read_bytes()), 'helper_sha256': h.sha(HELPER.read_bytes()),
              'targeted_run_sha256': h.sha((RAW / 'targeted-v4/run.json').read_bytes()),
              'cases': positives, 'negative_cases': negative,
              'limits': 'Seven complete positive inventories and fourteen preserved negative states. '
                        'No product imports, PID probes, signals, database writes or experiment replay. '
                        'Manager properties and operations are substitutes; no deployed systemd, GPU, '
                        'model, license, cold restart or research-quality acceptance.'}
    with (RAW / 'product-verification-v1.json').open('x') as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'positive_scopes': len(positives), 'negative_scopes': len(negative),
                      'positive_native_attempts': sum(c['attempts'] for c in positives)}))


if __name__ == '__main__':
    main()
