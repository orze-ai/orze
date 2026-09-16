"""Independent stdlib audit of retained campaign process/cost evidence.

No product imports, model access, signals or modifications of captured runs.
The product's separate verifier owns scientific quality/usage interpretation.
"""
import hashlib
import json
from pathlib import Path
import sys


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode())


def read(path):
    return json.loads(path.read_bytes())


def main():
    base, output = map(Path, sys.argv[1:])
    rows, actions = [], 0
    for kind in ('valid', 'invalid'):
        root = base / ('campaign-' + kind + '0')
        paths = [root / folder / 'run.json' for folder in
                 ('run', 'wrong-runtime', 'provider-failed', 'late-failed', 'timed-out')]
        pointers = [json.loads(line) for line in (root / 'batch/runs.jsonl').read_bytes().splitlines()]
        assert len(pointers) == 4
        for index, pointer in enumerate(pointers):
            path = root / 'batch' / pointer['path']
            if index == 1:
                assert not path.exists()
                path = path.with_name('run.saved.json')
            assert sha(path.read_bytes()) == pointer['sha256']
            paths.append(path)
        for path in paths:
            record = read(path)
            closure, binding = record['supervision']['closure'], record['supervision']['binding']
            assert closure['event'] == 'TREE_CLOSED' and closure['wait_proof'] == 'ECHILD_WALL'
            assert closure['binding'] == binding and closure['worker_returncode'] == record['exit_code']
            assert binding['command_sha256'] == digest(record['command'])
            assert binding['identity'] == {'comparison_run': record['run_id'],
                                           'request_sha256': digest(record['request'])}
            assert record['wall_seconds'] == record['finished_monotonic'] - record['started_monotonic']
            directory = Path(record['output'])
            for name, pinned in record['files'].items():
                raw = (directory / name).read_bytes()
                assert len(raw) == pinned['bytes'] and sha(raw) == pinned['sha256']
            captures = [directory / name for name in ('capture.json', 'project/partial-capture.json')
                        if name in record['files']]
            count = 0
            if captures:
                capture = read(captures[0])
                for call in capture['calls']:
                    proof = call['controller_supervision']
                    assert proof['closure']['event'] == 'TREE_CLOSED'
                    assert proof['closure']['binding'] == proof['binding']
                    assert proof['binding']['command_sha256'] == digest(call['command'])
                attempts = capture['database']['execution_attempts']
                reservations = capture['database']['cpu_action_reservations']
                assert len(attempts) == len(reservations)
                assert all(r['state'] == 'SETTLED' for r in reservations)
                for attempt in attempts:
                    assert attempt['state'] == 'TERMINAL'
                    terminal = json.loads(attempt['terminal_json'])
                    assert terminal['outcome'] == 'completed' and terminal['return_code'] == 0
                count = len(attempts)
                actions += count
            rows.append({'path': str(path), 'sha256': sha(path.read_bytes()), 'actions': count,
                         'exit_code': record['exit_code'], 'error': record['error'],
                         'stop_requested': closure['stop_requested']})
        intact, missing = (read(root / ('batch-' + label + '.json'))
                           for label in ('intact', 'missing-original'))
        for report in (intact, missing):
            assert report['counts']['planned_runs'] == 4 and len(report['pairs']) == 2
            assert report['all_pairs_qualified'] is False
        assert missing['runs'][1]['status'] == 'unknown' and 'metrics' not in missing['runs'][1]
        for row in intact['runs']:
            assert row['metrics']['native_actions'] == 5 and row['metrics']['provider_calls'] == 2
            assert row['quality']['valid'] == (kind == 'valid')
            assert row['budget_checks']['provider_cost_usd'] == 'unknown'
            assert row['budget_checks']['gpu_seconds'] == 'unknown'
        late = read(root / 'late-failure.stdout')
        assert late['status'] == 'failed' and late['metrics']['native_actions'] == 2
        assert late['metrics']['reserved_seconds'] == 4 and late['metrics']['provider_calls'] is None
    assert len(rows) == 18 and actions == 50
    report = {'schema': 1, 'outer_runs': len(rows), 'native_actions': actions, 'runs': rows,
              'scope': 'Mechanical evidence audit; no real model or research improvement claim',
              'script_sha256': sha(Path(__file__).read_bytes())}
    with output.open('x') as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'outer_runs': len(rows), 'native_actions': actions, 'output': str(output)}))


if __name__ == '__main__':
    main()
