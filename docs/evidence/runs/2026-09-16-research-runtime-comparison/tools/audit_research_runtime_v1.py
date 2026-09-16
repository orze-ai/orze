"""Reuse the independent scientific oracle; recompute software-version pairs."""
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'research-runtime-comparison'
SOURCE = ROOT / 'orze/docs/evidence/checks/2026-09-12-efficiency-formal-review.py'
spec = importlib.util.spec_from_file_location('independent_oracle', SOURCE)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def main():
    folder = BASE / 'formal-03'
    plan, summary = [check.read(folder / name) for name in ('plan.json', 'summary.json')]
    assert len(plan['schedule']) == summary['executed_cli'] == 48
    expected = [(rep, domain, variant, arm) for rep in range(6) for domain in ('sorting', 'compression')
                for variant in ('default', 'counterfactual') for arm in (('A', 'B') if rep % 2 == 0 else ('B', 'A'))]
    assert [(s['repetition'], s['domain'], s['variant'], s['arm']) for s in plan['schedule']] == expected
    verified = []
    for row in plan['schedule']:
        root = folder / row['id']
        record = check.read(root / 'run.json')
        runtime = check.read(root / 'runtime.json')
        assert runtime['before'] == runtime['after'] == plan['runtimes'][row['arm']]
        manifest = {'python': plan['runtimes'][row['arm']]['python'],
                    'datasets': {d + '_' + v: check.digest(data) for (d, v), data in check.DATA.items()}}
        # The historical checker calls policy A the full-coverage CommonPolicy.
        result = check.check_run(root, record, {**row, 'arm': 'A'}, manifest)
        selected = record['metrics']['selection']
        request = json.loads(next(i['config'] for i in record['database']['ideas'] if i['idea_id'] == selected['task_id']))['domain_request']
        assert request['input_artifact_ids'] == [] and request['payload']['operation'] == 'measure'
        action = {'version': 1, 'adapter': 'command', 'purpose': request['purpose'],
                  'inputs': {**request['payload'], 'dataset': check.DATA[row['domain'], row['variant']], 'source_bindings': []},
                  'command': [manifest['python'], str(BASE / 'application/examples/acceptance' / (row['domain'] + '.py'))],
                  'timeout_seconds': request['timeout_seconds'], 'outputs': request['outputs']}
        assert result['selection']['signature'] == [('result', check.digest({"specification_schema": "orze.native_cpu_action.v1", "action": action}))]
        action['command'][0] = '<same-verified-python-binary>'
        result['selection']['signature'] = [('result', check.digest({"specification_schema": "orze.native_cpu_action.v1", "action": action}))]
        assert result['metrics']['native_actions'] == (5 if row['variant'] == 'default' else 3)
        verified.append({**row, **result, 'run_sha256': check.sha((root / 'run.json').read_bytes())})
    assert plan['runtimes']['A']['python_binary_sha256'] == plan['runtimes']['B']['python_binary_sha256']
    assert plan['runtimes']['A']['dependencies'] == plan['runtimes']['B']['dependencies']
    groups = {}
    for domain in ('sorting', 'compression'):
        for variant in ('default', 'counterfactual'):
            pairs = []
            for repetition in range(6):
                arms = {r['arm']: r for r in verified if (r['domain'], r['variant'], r['repetition']) == (domain, variant, repetition)}
                assert arms['A']['selection'] == arms['B']['selection'] and arms['A']['statuses'] == arms['B']['statuses']
                pairs.append({k: {'A': arms['A']['metrics'][k], 'B': arms['B']['metrics'][k],
                                  'B_minus_A': arms['B']['metrics'][k] - arms['A']['metrics'][k]} for k in check.METRICS})
            medians = {k: {part: statistics.median(p[k][part] for p in pairs) for part in ('A', 'B', 'B_minus_A')} for k in check.METRICS}
            saved = summary['groups'][domain + '_' + variant]
            assert medians == saved['medians'] and saved['quality_passed_pairs'] == saved['planned_pairs'] == 6
            groups[domain + '_' + variant] = medians
    report = {'runs': len(verified), 'pairs': 24, 'native_actions': sum(r['metrics']['native_actions'] for r in verified),
              'all_quality_coverage_and_cost_equal': True, 'groups': groups, 'runs_verified': verified,
              'script_sha256': check.sha(Path(__file__).read_bytes()), 'independent_oracle_sha256': check.sha(SOURCE.read_bytes()),
              'scope': 'Actual SQLite, artifact bytes, sorting/compression scientific oracles, exact permit/closure linkage and all nine timing/cost metrics. Both arms use the full-coverage CommonPolicy; not a model reasoning or memory experiment.'}
    assert report['native_actions'] == 192
    with (BASE / 'independent-v1.json').open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
    print(json.dumps({'runs': 48, 'pairs': 24, 'native_actions': 192, 'all_quality_coverage_and_cost_equal': True}))


if __name__ == '__main__':
    main()
