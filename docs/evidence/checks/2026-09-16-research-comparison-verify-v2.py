"""Independent historical arithmetic and raw-evidence integrity, stdlib only.

Does not import the new reducer, legacy adapter, qualifier or production code.
Does not execute any model, CPU worker, archived script or existing service.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import sys
import tarfile


CORE = Path(__file__).resolve().parents[3]
RUN = CORE / 'docs/evidence/runs/2026-09-16-research-comparison'
OLD = CORE / 'docs/evidence/runs/2026-09-12-research-efficiency'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen(name, expected):
    root = RUN / name
    record = read(root / 'run.json')
    require(record['frozen'] is True and record['exit_code'] == expected, name + ': completion')
    require(record['before'] == record['after'], name + ': inputs')
    for relative, entry in record['files'].items():
        raw = (root / relative).read_bytes()
        require(len(raw) == entry['bytes'] and sha(raw) == entry['sha256'], name + ': output ' + relative)
    return {'name': name, 'exit_code': expected, 'run_sha256': sha((root / 'run.json').read_bytes())}


def main():
    checks = [frozen(name, exit_code) for name, exit_code in (
        ('baseline', 2), ('targeted-v1', 1), ('targeted-v2', 0), ('targeted-v3', 0),
        ('replay-v1', 0), ('replay-v2', 0))]
    report = read(RUN / 'reanalysis-v2/report.json')
    plan = read(RUN / 'reanalysis-v2/protocol.json')
    require(report['new_research_evidence'] is False, 'must remain historical')
    require(report['counts'] == {'planned_runs': 48, 'provided_runs': 48, 'missing_runs': 0}, 'denominator')
    for filename, expected in report['provenance']['historical_inputs'].items():
        require(sha((OLD / filename).read_bytes()) == expected, 'historical bytes ' + filename)
    for filename, expected in report['provenance']['verifier_files'].items():
        require(sha((CORE / filename).read_bytes()) == expected, 'verifier bytes ' + filename)
    digest = sha(json.dumps(plan, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                           allow_nan=False).encode())
    require(digest == report['protocol_sha256'], 'plan binding')
    originals = read(OLD / 'summary.json')
    records = {}
    observations = {arm: {'valid': 0, 'invalid': 0, 'unknown': 0} for arm in ('A', 'B')}
    total_actions = 0
    with tarfile.open(OLD / 'formal-01.tar.gz', 'r:gz') as archive:
        for pointer in originals['records']:
            path = PurePosixPath(pointer['path'])
            member = 'formal-01/' + path.parent.name + '/run.json'
            stream = archive.extractfile(member)
            require(stream is not None, 'raw record member')
            with stream:
                raw = stream.read()
            require(sha(raw) == pointer['sha256'] and len(raw) == pointer['bytes'], 'raw record hash')
            value = json.loads(raw)
            db, trace = value['database'], value['trace']
            obs = [json.loads(row['record_json']) for row in db['research_observations']]
            terminals = [json.loads(row['terminal_json']) for row in db['execution_attempts']]
            first = next(item['monotonic'] for item in trace if any(
                observation['validation']['status'] == 'valid'
                for result in item['snapshot']['recorded_evidence']['results']
                for observation in result['observation_records']))
            metrics = {
                'native_actions': len(terminals),
                'reserved_seconds': sum(int(json.loads(row['permit_json'])['reserved_nanoseconds'])
                                        for row in db['cpu_action_reservations']) / 1e9,
                'analysis_actions': sum(row['values']['operation'] == 'analyze' for row in obs),
                'cli_wall_seconds': value['finished_monotonic'] - value['started_monotonic'],
                'first_valid_consumed_seconds': first - value['started_monotonic'],
                'confirmed_selection_seconds': trace[-1]['monotonic'] - value['started_monotonic'],
                'worker_cpu_seconds': sum(row['values']['worker_cpu_seconds'] for row in obs),
                'worker_wall_seconds': sum(row['values']['worker_wall_seconds'] for row in obs),
                'native_elapsed_seconds': sum(row['elapsed_wall_seconds'] for row in terminals),
            }
            task_id = pointer['domain'] + '_' + pointer['variant']
            run_id = task_id + '-' + str(pointer['repetition']).zfill(4) + '-' + pointer['arm']
            normalized, = [r for r in report['runs'] if r['run_id'] == run_id]
            for key, value in metrics.items():
                require(normalized['metrics'][key] == value, run_id + ': metric ' + key)
            for key in ('provider_calls', 'provider_tokens', 'provider_cost_usd', 'gpu_seconds'):
                require(normalized['metrics'][key] is None, 'unmeasured cost must stay unknown')
            for row in obs:
                observations[pointer['arm']][row['validation']['status']] += 1
            records[(task_id, pointer['repetition'], pointer['arm'])] = metrics
            total_actions += metrics['native_actions']
    for task in plan['tasks']:
        task_id = task['id']
        group = report['groups'][task_id]
        for metric in next(iter(records.values())):
            a = [records[(task_id, repetition, 'A')][metric] for repetition in range(6)]
            b = [records[(task_id, repetition, 'B')][metric] for repetition in range(6)]
            require(group['median_by_arm']['A'][metric] == statistics.median(a), 'A median')
            require(group['median_by_arm']['B'][metric] == statistics.median(b), 'B median')
            require(group['median_paired_differences'][metric] == statistics.median(
                right - left for left, right in zip(a, b)), 'paired difference')
        for arm in ('A', 'B'):
            unknown = group['cost_totals'][arm]['provider_cost_usd']
            require(unknown == {'complete': False, 'known_runs': 0, 'known_sum': 0, 'missing_runs': 6},
                    'empty known subtotal cannot be a complete free run')
    require(total_actions == 180, 'historical native attempt count')
    require(observations == {'A': {'valid': 72, 'invalid': 12, 'unknown': 12},
                             'B': {'valid': 60, 'invalid': 12, 'unknown': 12}}, 'coverage tradeoff')
    output = {'schema': 1, 'passed': True, 'validation_runs': checks,
              'historical_pairs': 24, 'historical_runs': 48, 'historical_native_attempts': total_actions,
              'observations_by_arm': observations, 'independent_metric_columns': 9,
              'report_sha256': sha((RUN / 'reanalysis-v2/report.json').read_bytes()),
              'new_cpu_executions': 0, 'new_provider_calls': 0,
              'scope': 'Independent arithmetic and byte integrity; does not replace original semantic review or new real research',
              'full_regressions_included': False}
    target = Path(sys.argv[1])
    with target.open('x') as stream:
        json.dump(output, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(output, sort_keys=True))


if __name__ == '__main__':
    main()
