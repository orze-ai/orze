"""Fixed software-version experiment; reuse the unchanged CPU application."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent / 'research-runtime-comparison'
APP = ROOT / 'application'
METRICS = ('cli_wall_seconds', 'first_valid_consumed_seconds', 'confirmed_selection_seconds',
           'native_actions', 'reserved_seconds', 'analysis_actions', 'worker_cpu_seconds',
           'worker_wall_seconds', 'native_elapsed_seconds')


def read(path):
    return json.loads(path.read_bytes())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def environment():
    return {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}


def runtime(arm):
    import orze
    package = Path(orze.__file__).parent
    assert package.is_relative_to(ROOT / arm / 'runtime')
    deps = {d.metadata['Name']: d.version for d in metadata.distributions()}
    assert 'orze-pro' not in {name.lower() for name in deps}
    wheel = ROOT / arm / 'wheels/orze-4.6.2-py3-none-any.whl'
    files = {str(p.relative_to(package)): digest(p.read_bytes()) for p in package.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')}
    with zipfile.ZipFile(wheel) as archive:
        saved = {name[5:]: digest(archive.read(name)) for name in archive.namelist()
                 if name.startswith('orze/') and not name.endswith('/')}
    assert files == saved
    source = read(ROOT / arm / 'source.json')
    assert {name[len('src/orze/'):]: value for name, value in source['files'].items()
            if name.startswith('src/orze/')} == files
    app = read(ROOT / 'application.json')
    assert all(digest((APP / name).read_bytes()) == value for name, value in app['files'].items())
    return {'arm': arm, 'commit': source['commit'], 'python': sys.executable,
            'python_binary_sha256': digest(Path(sys.executable).resolve().read_bytes()),
            'core_import': str(package), 'wheel_sha256': digest(wheel.read_bytes()),
            'files': files, 'dependencies': deps, 'application': app,
            'runner_sha256': digest(Path(__file__).read_bytes())}


def one(arm, output, domain, variant):
    before = runtime(arm)
    sys.path.insert(0, str(APP))
    from examples.research_efficiency.run import run_one
    # Both runtime arms use the same full-coverage CommonPolicy (policy arm A).
    record, pointer = run_one(output, domain, variant, 'A', deadline=time.monotonic() + 75)
    after = runtime(arm)
    save(output / 'runtime.json', {'before': before, 'after': after, 'unchanged': before == after})
    return 0 if before == after and record['quality_passed'] else 1


def selection(record, arm):
    """Permit only the known venv interpreter path difference in action hashes."""
    sys.path.insert(0, str(APP))
    from examples.acceptance.common import digest as canonical_digest
    metrics = record['metrics']
    selected = metrics['selection']
    idea = next(row for row in record['database']['ideas'] if row['idea_id'] == selected['task_id'])
    request = json.loads(idea['config'])['domain_request']
    assert request['input_artifact_ids'] == [] and request['payload']['operation'] == 'measure'
    action = {'version': 1, 'adapter': 'command', 'purpose': request['purpose'],
              'inputs': {**request['payload'], 'dataset': record['config']['action_domain']['config']['dataset'], 'source_bindings': []},
              'command': [str(ROOT / arm / 'runtime/bin/python'), str(APP / 'examples/acceptance' / (record['domain'] + '.py'))],
              'timeout_seconds': request['timeout_seconds'], 'outputs': request['outputs']}
    assert selected['action_signature'] == [['result', canonical_digest(action)]]
    action['command'][0] = '<same-verified-python-binary>'
    return {**selected, 'action_signature': [['result', canonical_digest(action)]]}


def summarize(folder):
    plan = read(folder / 'plan.json')
    records, failures = {}, []
    for spec in plan['schedule']:
        path = folder / spec['id'] / 'run.json'
        if not path.exists():
            failures.append({**spec, 'reason': 'missing'})
            continue
        record = read(path)
        pin = read(folder / spec['id'] / 'runtime.json')
        if not record['quality_passed'] or not pin['unchanged'] or pin['before'] != plan['runtimes'][spec['arm']]:
            failures.append({**spec, 'reason': 'quality_or_runtime', 'detail': record.get('quality_error')})
        records[spec['id']] = record
    pairs = []
    for repetition in range(6):
        for domain in ('sorting', 'compression'):
            for variant in ('default', 'counterfactual'):
                specs = [s for s in plan['schedule'] if (s['repetition'], s['domain'], s['variant']) == (repetition, domain, variant)]
                arms = {s['arm']: records.get(s['id']) for s in specs}
                quality = not any(f['id'] in [s['id'] for s in specs] for f in failures)
                if quality:
                    quality = selection(arms['A'], 'A') == selection(arms['B'], 'B')
                    quality &= arms['A']['metrics']['observation_statuses'] == arms['B']['metrics']['observation_statuses']
                    expected = 5 if variant == 'default' else 3
                    quality &= all(arms[a]['metrics']['native_actions'] == expected for a in ('A', 'B'))
                pair = {'repetition': repetition, 'domain': domain, 'variant': variant, 'quality_passed': bool(quality)}
                if quality:
                    pair['metrics'] = {key: {'A': arms['A']['metrics'][key], 'B': arms['B']['metrics'][key],
                        'B_minus_A': arms['B']['metrics'][key] - arms['A']['metrics'][key]} for key in METRICS}
                pairs.append(pair)
    groups = {}
    for domain in ('sorting', 'compression'):
        for variant in ('default', 'counterfactual'):
            matching = [p for p in pairs if (p['domain'], p['variant']) == (domain, variant)]
            group = {'planned_pairs': 6, 'quality_passed_pairs': sum(p['quality_passed'] for p in matching)}
            if all(p['quality_passed'] for p in matching):
                group['medians'] = {key: {part: statistics.median(p['metrics'][key][part] for p in matching)
                                         for part in ('A', 'B', 'B_minus_A')} for key in METRICS}
                group['median_cli_ratio_B_over_A'] = statistics.median(p['metrics']['cli_wall_seconds']['B'] / p['metrics']['cli_wall_seconds']['A'] for p in matching)
            groups[domain + '_' + variant] = group
    return {'planned_cli': 48, 'executed_cli': len(records), 'failures': failures, 'pairs': pairs, 'groups': groups,
            'all_quality_passed': len(records) == 48 and not failures and all(p['quality_passed'] for p in pairs),
            'native_actions': sum(len(r['database']['execution_attempts']) for r in records.values()),
            'scope': 'Core software-version effect on fixed public CPU research loops, same CommonPolicy and evidence coverage. No Pro/model/memory/GPU or general research-quality inference.'}


def campaign():
    folder = ROOT / 'formal-01'
    folder.mkdir()
    runtimes = {}
    for arm in ('A', 'B'):
        result = subprocess.run([str(ROOT / arm / 'runtime/bin/python'), '-I', '-B', str(Path(__file__)), 'describe', arm],
                                env=environment(), capture_output=True, text=True, check=True)
        runtimes[arm] = json.loads(result.stdout)
    assert runtimes['A']['dependencies'] == runtimes['B']['dependencies']
    assert runtimes['A']['python_binary_sha256'] == runtimes['B']['python_binary_sha256']
    schedule = []
    for repetition in range(6):
        for domain in ('sorting', 'compression'):
            for variant in ('default', 'counterfactual'):
                for arm in ('A', 'B') if repetition % 2 == 0 else ('B', 'A'):
                    schedule.append({'id': f'{len(schedule):02d}-{domain}-{variant}-{arm}', 'repetition': repetition,
                                     'domain': domain, 'variant': variant, 'arm': arm})
    plan = {'runtimes': runtimes, 'schedule': schedule, 'repetitions': 6, 'policy': 'CommonPolicy in both arms',
            'quality': 'same valid independently confirmed selection, observation status counts and 5/default or 3/counterfactual actions; only verified interpreter-path normalization in action hash',
            'primary': 'paired complete CLI wall time; first valid and confirmed selection times secondary',
            'interpretation': 'per-task paired distributions; no pooled general research speed claim; failures stay in denominator',
            'budget': {'cpu_slots_per_cli': 1, 'reserved_seconds_per_cli': 10, 'action_timeout_seconds': 2,
                       'cli_timeout_seconds': 30, 'outer_campaign_seconds': 900},
            'created_unix': time.time(), 'runner_sha256': digest(Path(__file__).read_bytes())}
    save(folder / 'plan.json', plan)
    deadline = time.monotonic() + 900
    for spec in schedule:
        if deadline - time.monotonic() < 90:
            break
        target = folder / spec['id']
        command = [str(ROOT / spec['arm'] / 'runtime/bin/python'), '-I', '-B', str(Path(__file__)),
                   'one', spec['arm'], str(target), spec['domain'], spec['variant']]
        with (folder / (spec['id'] + '.stdout')).open('xb') as out, (folder / (spec['id'] + '.stderr')).open('xb') as err:
            result = subprocess.run(command, env=environment(), stdout=out, stderr=err, timeout=90)
        save(folder / (spec['id'] + '.command.json'), {'command': command, 'exit_code': result.returncode})
        record = read(target / 'run.json') if (target / 'run.json').exists() else {}
        print(json.dumps({'id': spec['id'], 'exit_code': result.returncode, 'wall': record.get('wall_seconds'),
                          'quality': record.get('quality_passed')}), flush=True)
        if not record.get('controller_closure') or record.get('cleanup_error'):
            break
    summary = summarize(folder)
    save(folder / 'summary.json', summary)
    print(json.dumps({'all_quality_passed': summary['all_quality_passed'], 'runs': summary['executed_cli'], 'groups': summary['groups']}), flush=True)
    return 0 if summary['all_quality_passed'] else 1


if __name__ == '__main__':
    if sys.argv[1] == 'describe':
        print(json.dumps(runtime(sys.argv[2]), sort_keys=True))
    elif sys.argv[1] == 'one':
        raise SystemExit(one(sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5]))
    elif sys.argv[1] == 'campaign':
        raise SystemExit(campaign())
