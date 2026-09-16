"""Actual Pro HTTP/CPU workload under a complete, runtime-bound collector."""
import copy
import hashlib
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from examples.holdout import scheduling as domain
from examples.research_comparison import campaign
from examples.research_comparison.protocol import digest, schedule

def _pro_sources_available():
    # Core can include the intentionally non-runnable Pro stub. Discover the
    # actual agent source without importing its production license gate.
    spec = importlib.util.find_spec('orze_pro')
    return spec is not None and any(
        (Path(root) / 'agents' / 'research.py').is_file()
        for root in spec.submodule_search_locations or ())


pytestmark = pytest.mark.skipif(not _pro_sources_available(),
                               reason='campaign integration requires real optional Pro sources')

@pytest.fixture(scope='module', params=[False, True], ids=['valid', 'invalid'])
def finished_campaign(tmp_path_factory, request):
    root = tmp_path_factory.mktemp('campaign-invalid' if request.param else 'campaign-valid')
    repository = Path(__file__).resolve().parents[1]
    instance = json.loads((repository / 'examples/holdout/instance.json').read_bytes())
    candidate = domain.produce(instance, 'challenger')
    if request.param:
        candidate['schedule'].append(dict(candidate['schedule'][0]))
    proposal = {'title': 'Evaluate an explicit scheduling candidate',
        'hypothesis': 'The independent evaluator decides feasibility from the candidate bytes.',
        'config': {'kind': 'native_cpu_action', 'domain_request': domain.make_request('produce', candidate='challenger',
                                                                              artifact_utf8=json.dumps(candidate))}}
    trace = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == '/v1/chat/completions'
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            trace.append(payload)
            model_calls = sum(row['model'] == payload['model'] for row in trace)
            if (payload['model'] == 'offline-failure'
                    or payload['model'] == 'offline-late-failure' and model_calls > 1):
                self.send_error(503, 'owned offline failure')
                return
            response = copy.deepcopy(proposal)
            response['title'] += ' ' + str(len(trace))
            if payload['model'] == 'offline-two-candidates':
                name = 'challenger' if model_calls % 2 else 'baseline'
                explicit = domain.produce(instance, name)
                if request.param:
                    explicit['schedule'].append(dict(explicit['schedule'][0]))
                response['config']['domain_request'] = domain.make_request('produce', candidate=name,
                                                                         artifact_utf8=json.dumps(explicit))
                prompt = '\n'.join(message['content'] for message in payload['messages'])
                response['parent'] = ('idea-evaluate-0001-0000' if
                    'Eligible scored parent IDs' in prompt and 'idea-evaluate-0001-0000' in prompt else 'none')
            raw = json.dumps({'id': 'offline-campaign', 'model': 'offline-revision',
                'choices': [{'message': {'content': json.dumps([response])}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    site = root / 'fixture'
    site.mkdir()
    (site / 'sitecustomize.py').write_text('''
from unittest.mock import patch
patch('orze_pro._gate.require_license', lambda: None).start()
import urllib.request
from urllib.parse import urlparse
original = urllib.request.urlopen
def owned_http(request, *args, **kwargs):
    url = request.full_url if hasattr(request, 'full_url') else request
    assert urlparse(url).hostname == '127.0.0.1', 'fixture forbids external provider access'
    return original(request, *args, **kwargs)
urllib.request.urlopen = owned_http
''')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', LLM_API_KEY='offline-fixture-key')
    env['PYTHONPATH'] = str(site) + os.pathsep + str(repository) + os.pathsep + env.get('PYTHONPATH', '')
    described = subprocess.run([sys.executable, '-c',
        'import json;from examples.research_comparison.campaign import describe;print(json.dumps(describe()))'],
        env=env, cwd=repository, capture_output=True, text=True, timeout=30)
    assert described.returncode == 0, described.stderr
    runtime = json.loads(described.stdout)
    tools = {'workload': 'scheduling-v1', 'rounds': 1, 'num_ideas': 1, 'evaluation_protocol': domain.PROTOCOLS[0]}
    inputs = {'model': {'backend': 'custom', 'model': 'offline-revision',
                        'endpoint': 'http://127.0.0.1:' + str(server.server_address[1]) + '/v1'},
              'tools': tools, 'environment': runtime['environment'], 'treatment': {}, 'data': instance,
              'evaluator': {'source_sha256': hashlib.sha256(Path(domain.__file__).read_bytes()).hexdigest(),
                            'protocol': domain.PROTOCOLS[0]},
              'instructions': 'Produce an explicit scheduling candidate for this instance: ' + json.dumps(instance),
              'initial_history': [], 'initial_memory': None}
    task = {'id': 'scheduling-target', 'domain': 'scheduling', 'role': 'target', 'seeds': [17],
        'inputs': {k: digest(inputs[k]) for k in ('data', 'evaluator', 'instructions', 'initial_history', 'initial_memory')},
        'quality': {'direction': 'maximize', 'max_regression': 0, 'minimum_valid_observations': 2},
        'limits': {'provider_calls': 4, 'provider_tokens': 100000, 'provider_cost_usd': 0,
                   'reserved_seconds': 20, 'gpu_seconds': 0, 'cli_wall_seconds': 60}}
    control = copy.deepcopy(task)
    control.update(id='scheduling-control', role='negative_control')
    arm = {'artifact_sha256': runtime['artifact_sha256'], 'treatment_sha256': digest(inputs['treatment'])}
    plan = {'schema': 1, 'comparison_id': 'offline-campaign-wiring', 'mode': 'prospective',
            'repetitions': 1, 'ordering': 'task_and_repetition', 'arms': {'A': arm, 'B': dict(arm)},
            'shared': {k: digest(inputs[k]) for k in ('model', 'tools', 'environment')},
            'verifier_sha256': runtime['verifier_sha256'], 'tasks': [task, control]}
    try:
        record = campaign.execute(plan, schedule(plan)[0]['run_id'], inputs, root / 'run', runtime=runtime, env=env)
        assert record['exit_code'] == 0, (record, (root / 'run/stderr.log').read_text(),
                                           (root / 'run/worker.json').read_text())
        script = '''import json,sys
from pathlib import Path
from examples.research_comparison.campaign import verify
r=json.loads(Path(sys.argv[1]).read_bytes());p=r['request']['protocol']
t=next(t for t in p['tasks'] if t['id']==r['task_id'])
print(json.dumps(verify(r,t,r['arm'],plan=p),sort_keys=True))'''
        audited = subprocess.run([sys.executable, '-c', script, str(root / 'run/run.json')], env=env,
                                 cwd=repository, capture_output=True, text=True, timeout=30)
        (root / 'verify.stdout').write_text(audited.stdout)
        (root / 'verify.stderr').write_text(audited.stderr)
        assert audited.returncode == 0, audited.stderr
        yield root, plan, inputs, runtime, env, record, json.loads(audited.stdout), trace, request.param
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        (root / 'http-trace.json').write_text(json.dumps(trace, sort_keys=True, indent=2))


def test_complete_run_binds_actual_quality_usage_and_outer_clock(finished_campaign):
    root, plan, inputs, runtime, env, record, measured, trace, invalid = finished_campaign
    capture = json.loads((root / 'run/capture.json').read_bytes())
    assert len(trace) == 1 and trace[0]['model'] == inputs['model']['model']
    assert measured['quality']['valid'] == measured['quality']['confirmed'] == (not invalid)
    assert measured['quality']['score'] == (None if invalid else 30)
    metrics = measured['metrics']
    assert metrics['native_actions'] == 3 and metrics['reserved_seconds'] == 6
    assert metrics['provider_calls'] == 1 and metrics['provider_tokens'] == 5
    assert metrics['provider_cost_usd'] is None and metrics['gpu_seconds'] is None
    assert metrics['cli_wall_seconds'] == record['wall_seconds']
    assert metrics['cli_wall_seconds'] > sum(c['wall_seconds'] for c in capture['calls'])
    assert metrics['first_valid_consumed_seconds'] is None
    if invalid:
        assert metrics['confirmed_selection_seconds'] is None
    else:
        assert 0 < metrics['confirmed_selection_seconds'] <= metrics['cli_wall_seconds']
    assert record['supervision']['closure']['event'] == 'TREE_CLOSED'


def test_existing_or_changed_input_does_not_start_another_run(finished_campaign):
    root, plan, inputs, runtime, env, record, measured, trace, invalid = finished_campaign
    before = len(trace)
    with pytest.raises(FileExistsError):
        campaign.execute(plan, record['run_id'], inputs, root / 'run', runtime=runtime, env=env)
    changed = copy.deepcopy(inputs)
    changed['instructions'] += '\nchanged condition'
    with pytest.raises(ValueError, match='frozen protocol'):
        campaign.execute(plan, record['run_id'], changed, root / 'wrong-input', runtime=runtime, env=env)
    assert not (root / 'wrong-input').exists() and len(trace) == before


def test_child_observes_its_actual_import_root_before_work(finished_campaign):
    root, plan, inputs, runtime, env, record, measured, trace, invalid = finished_campaign
    changed = copy.deepcopy(runtime)
    changed['runtime'][0]['root'] = str(root / 'not-the-imported-package')
    before = len(trace)
    refused = campaign.execute(plan, record['run_id'], inputs, root / 'wrong-runtime', runtime=changed, env=env)
    assert refused['exit_code'] == 1 and len(trace) == before
    assert refused['supervision']['closure']['event'] == 'TREE_CLOSED'
    assert not (root / 'wrong-runtime/project').exists()
    worker = json.loads((root / 'wrong-runtime/worker.json').read_bytes())
    assert worker['status'] == 'failed' and worker['error'] == 'ValueError'


def test_model_command_must_match_the_observed_process_binding(finished_campaign):
    root, plan, inputs, runtime, env, record, measured, trace, invalid = finished_campaign
    captured = json.loads((root / 'run/capture.json').read_bytes())
    call = next(c for c in captured['calls'] if c['label'] == 'research-0001')
    call['command'][call['command'].index('--model') + 1] = 'another-model'
    path = root / 'changed-model-capture.json'
    path.write_text(json.dumps(captured))
    script = '''import json,sys
from pathlib import Path
from examples.research_comparison.scheduling_campaign import verify
r=json.loads(Path(sys.argv[1]).read_bytes());p=r['request']['protocol']
t=next(t for t in p['tasks'] if t['id']==r['task_id'])
c=json.loads(Path(sys.argv[2]).read_bytes())
try:
 verify(c,t,request=r['request'])
except ValueError:
 raise SystemExit(0)
raise AssertionError('changed model command was accepted as the frozen model')'''
    check = subprocess.run([sys.executable, '-c', script, str(root / 'run/run.json'), str(path)], env=env,
                           cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    (root / 'changed-model.stdout').write_text(check.stdout)
    (root / 'changed-model.stderr').write_text(check.stderr)
    assert check.returncode == 0, check.stderr


def test_failed_provider_run_retains_closed_costs_and_unknown_usage(finished_campaign):
    root, plan, inputs, runtime, env, record, measured, trace, invalid = finished_campaign
    failed_inputs, failed_plan = copy.deepcopy(inputs), copy.deepcopy(plan)
    failed_inputs['model']['model'] = 'offline-failure'
    failed_plan['shared']['model'] = digest(failed_inputs['model'])
    before = len(trace)
    failed = campaign.execute(failed_plan, record['run_id'], failed_inputs, root / 'provider-failed',
                              runtime=runtime, env=env)
    assert failed['exit_code'] == 1 and len(trace) > before
    assert failed['supervision']['closure']['event'] == 'TREE_CLOSED'
    assert 'project/partial-capture.json' in failed['files'] and 'project/usage.jsonl' in failed['files']
    script = '''import json,sys
from pathlib import Path
from examples.research_comparison.campaign import verify
r=json.loads(Path(sys.argv[1]).read_bytes());p=r['request']['protocol']
t=next(t for t in p['tasks'] if t['id']==r['task_id'])
m=verify(r,t,r['arm'],plan=p)
assert m['status']=='failed' and not m['quality']['valid'] and m['quality']['score'] is None
assert m['metrics']['native_actions']==0 and m['metrics']['reserved_seconds']==0
assert m['metrics']['provider_calls'] is None and m['metrics']['provider_tokens'] is None
assert m['metrics']['provider_cost_usd'] is None and m['metrics']['cli_wall_seconds']==r['wall_seconds']
print(json.dumps(m,sort_keys=True))'''
    checked = subprocess.run([sys.executable, '-c', script, str(root / 'provider-failed/run.json')], env=env,
                             cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    (root / 'failed-verification.stdout').write_text(checked.stdout)
    (root / 'failed-verification.stderr').write_text(checked.stderr)
    assert checked.returncode == 0, checked.stderr


def test_fixed_batch_runs_both_arms_two_rounds_and_keeps_missing_costs(finished_campaign):
    from examples.research_comparison import batch
    root, plan, inputs, runtime, env, record, measured, trace, invalid = finished_campaign
    plan, inputs = copy.deepcopy(plan), copy.deepcopy(inputs)
    inputs['tools']['rounds'] = 2
    inputs['model']['model'] = 'offline-two-candidates'
    plan['shared']['tools'] = digest(inputs['tools'])
    plan['shared']['model'] = digest(inputs['model'])
    spec = {'schema': 1, 'protocol': plan,
            'shared': {k: inputs[k] for k in ('model', 'tools', 'environment')},
            'tasks': {task['id']: {k: inputs[k] for k in batch.TASK_INPUTS} for task in plan['tasks']},
            'arms': {arm: {'runtime': runtime, 'treatment': {}} for arm in ('A', 'B')}}
    spec_path = root / 'batch-input.json'
    spec_path.write_text(json.dumps(spec))
    spec_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    directory = root / 'batch'
    before = len(trace)
    cli = subprocess.run([sys.executable, '-m', 'examples.research_comparison', 'execute-campaign',
        '--specification', str(spec_path), '--specification-sha256', spec_sha, '--output-dir', str(directory)],
        env=env, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=180)
    (root / 'batch-execution.stdout').write_text(cli.stdout)
    (root / 'batch-execution.stderr').write_text(cli.stderr)
    assert cli.returncode == 0, cli.stderr
    receipt = json.loads(cli.stdout)
    assert len(trace) - before == 8
    index = [json.loads(line) for line in (directory / 'runs.jsonl').read_text().splitlines()]
    assert [p['run_id'] for p in index] == [s['run_id'] for s in schedule(plan)]
    for pointer in index:
        run = json.loads((directory / pointer['path']).read_bytes())
        assert run['exit_code'] == 0 and run['supervision']['closure']['event'] == 'TREE_CLOSED'
    for label, mutate in [('intact', False), ('missing-original', True)]:
        if mutate:
            original = directory / index[1]['path']
            original.rename(original.with_name('run.saved.json'))
        report_path = root / ('batch-' + label + '.json')
        checked = subprocess.run([sys.executable, '-m', 'examples.research_comparison', 'audit-campaign',
            '--campaign-dir', str(directory), '--specification-sha256', receipt['specification_sha256'],
            '--output', str(report_path)], env=env, cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=60)
        (root / ('batch-' + label + '.stderr')).write_text(checked.stderr)
        assert checked.returncode == 0, checked.stderr
        report = json.loads(report_path.read_bytes())
        assert report['counts'] == {'planned_runs': 4, 'provided_runs': 4, 'missing_runs': 0}
        assert len(report['pairs']) == 2 and not report['all_pairs_qualified']
        for i, row in enumerate(report['runs']):
            if mutate and i == 1:
                assert row['status'] == 'unknown' and 'metrics' not in row
                continue
            assert row['status'] == ('failed' if invalid else 'completed'), row
            assert row['quality']['valid'] == (not invalid)
            assert row['quality']['score'] == (None if invalid else 30)
            assert row['metrics']['native_actions'] == 5
            assert row['metrics']['provider_calls'] == 2 and row['metrics']['provider_tokens'] == 10
            assert row['budget_checks']['provider_cost_usd'] == 'unknown'
            assert row['budget_checks']['gpu_seconds'] == 'unknown'
    with pytest.raises(FileExistsError):
        batch.execute(spec, directory, env=env)
    changed = copy.deepcopy(spec)
    changed['tasks'][plan['tasks'][1]['id']]['instructions'] += ' changed'
    with pytest.raises(ValueError, match='frozen protocol'):
        batch.execute(changed, root / 'bad-batch', env=env)
    assert not (root / 'bad-batch').exists() and len(trace) - before == 8


def test_later_provider_failure_keeps_earlier_native_costs(finished_campaign):
    root, plan, inputs, runtime, env, record, measured, trace, invalid = finished_campaign
    plan, inputs = copy.deepcopy(plan), copy.deepcopy(inputs)
    inputs['tools']['rounds'] = 2
    inputs['model']['model'] = 'offline-late-failure'
    plan['shared'].update(model=digest(inputs['model']), tools=digest(inputs['tools']))
    failed = campaign.execute(plan, record['run_id'], inputs, root / 'late-failed', runtime=runtime, env=env)
    assert failed['exit_code'] == 1
    script = '''import json,sys
from pathlib import Path
from examples.research_comparison.campaign import verify
r=json.loads(Path(sys.argv[1]).read_bytes());p=r['request']['protocol']
t=next(t for t in p['tasks'] if t['id']==r['task_id'])
m=verify(r,t,r['arm'],plan=p)
assert m['status']=='failed' and not m['quality']['confirmed']
assert m['metrics']['native_actions']==2 and m['metrics']['reserved_seconds']==4
assert m['metrics']['provider_calls'] is None and m['metrics']['provider_tokens'] is None
print(json.dumps(m,sort_keys=True))'''
    checked = subprocess.run([sys.executable, '-c', script, str(root / 'late-failed/run.json')], env=env,
                             cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    (root / 'late-failure.stdout').write_text(checked.stdout)
    (root / 'late-failure.stderr').write_text(checked.stderr)
    assert checked.returncode == 0, checked.stderr


def test_outer_deadline_closes_its_own_child_and_retains_unknown_result(finished_campaign):
    root, plan, inputs, runtime, env, record, measured, trace, invalid = finished_campaign
    plan = copy.deepcopy(plan)
    plan['tasks'][0]['limits']['cli_wall_seconds'] = .001
    before = len(trace)
    timed = campaign.execute(plan, record['run_id'], inputs, root / 'timed-out', runtime=runtime, env=env)
    assert timed['error'] in ('TimeoutError', 'TimeoutExpired') and len(trace) == before
    assert timed['supervision']['closure']['event'] == 'TREE_CLOSED'
    assert timed['supervision']['closure']['stop_requested'] is True
    assert 'request.json' in timed['files']
    with pytest.raises(ValueError):
        campaign.verify(timed, plan['tasks'][0], record['arm'], plan=plan)


def test_confirmation_receipt_and_clock_boundaries_use_real_capture(finished_campaign):
    """Re-audit real closed products, including consistent but misplaced clocks."""
    root, plan, inputs, runtime, env, record, measured, trace, invalid = finished_campaign
    capture = json.loads((root / 'run/capture.json').read_bytes())
    worker = json.loads((root / 'run/worker.json').read_bytes())
    folder = root / 'timing-boundaries'
    folder.mkdir()
    for name, data in [('record', record), ('capture', capture), ('worker', worker)]:
        (folder / (name + '.json')).write_text(json.dumps(data))
    script = '''import copy,json,sys
from pathlib import Path
from examples.research_comparison.scheduling_campaign import verify
r=json.loads(Path(sys.argv[1]).read_bytes());p=r['request']['protocol']
t=next(t for t in p['tasks'] if t['id']==r['task_id'])
c=json.loads(Path(sys.argv[2]).read_bytes());w=json.loads(Path(sys.argv[3]).read_bytes())
window={'outer_started':r['started_monotonic'],'worker_started':w['started_monotonic'],
        'worker_finished':w['finished_monotonic'],'outer_finished':r['finished_monotonic']}
folder=Path(sys.argv[1]).parent
results={}
def check(name,capture,expected,clock=window):
 (folder/(name+'.json')).write_text(json.dumps({'capture':capture,'window':clock},sort_keys=True))
 try:
  result=verify(capture,t,request=r['request'],clock_window=clock)
 except ValueError as exc:
  assert expected=='reject',(name,str(exc))
  results[name]={'status':'rejected','reason':str(exc)}
 else:
  assert expected!='reject',name+' unexpectedly accepted'
  timing=result['metrics']['confirmed_selection_seconds']
  assert (timing is None) if expected=='unknown' else (timing is not None and 0<timing<=r['wall_seconds'])
  results[name]={'status':expected,'metrics':result['metrics']}
valid=bool(c['campaign'].get('confirmation'))
check('original',c,'measured' if valid else 'unknown')
missing=copy.deepcopy(c);missing['campaign'].pop('confirmation',None)
check('missing-receipt',missing,'unknown')
check('missing-outer-clock',c,'unknown',None)
for label,delta in [('before-worker',-r['wall_seconds']-100),('after-worker',r['wall_seconds']+100)]:
 changed=copy.deepcopy(c)
 for call in changed['calls']:
  call['started_monotonic']+=delta;call['finished_monotonic']+=delta
  call['wall_seconds']=call['finished_monotonic']-call['started_monotonic']
 check(label,changed,'reject')
for name,value in [('before-confirmation',c['calls'][-1]['started_monotonic']),
                   ('after-worker-receipt',window['worker_finished']+1),('boolean-clock',True)]:
 changed=copy.deepcopy(c)
 changed['campaign']['confirmation']={'schema':1,'event':'selection_confirmed',
     'selected_ref':c['campaign']['scope']['selected_ref'],
     'confirmation_ref':c['campaign']['scope']['confirmation_ref'],'observed_monotonic':value}
 check(name,changed,'reject')
changed=copy.deepcopy(c)
changed['campaign']['confirmation']={'schema':1,'event':'selection_confirmed',
     'selected_ref':c['campaign']['scope']['selected_ref'],
     'confirmation_ref':c['campaign']['scope']['selected_ref'],
     'observed_monotonic':window['worker_finished']}
check('self-confirmation-receipt',changed,'reject')
if valid:
 changed=copy.deepcopy(c);changed['campaign']['confirmation']['confirmation_ref']['generation']=False
 check('boolean-reference',changed,'reject')
else:
 changed=copy.deepcopy(c)
 changed['campaign']['confirmation']={'schema':1,'event':'selection_confirmed',
     'selected_ref':c['campaign']['scope']['selected_ref'],
     'confirmation_ref':c['campaign']['scope']['confirmation_ref'],
     'observed_monotonic':window['worker_finished']}
 check('invalid-quality-receipt',changed,'reject')
(folder/'results.json').write_text(json.dumps(results,sort_keys=True,indent=2))
print(json.dumps({'cases':len(results),'valid':valid}))'''
    (folder / 'verify.py').write_text(script)
    checked = subprocess.run([sys.executable, str(folder / 'verify.py'), str(folder / 'record.json'),
        str(folder / 'capture.json'), str(folder / 'worker.json')], env=env,
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    (folder / 'stdout.log').write_text(checked.stdout)
    (folder / 'stderr.log').write_text(checked.stderr)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)['cases'] == 10
    if invalid:
        assert 'confirmation' not in capture['campaign']
    else:
        moment = capture['campaign']['confirmation']['observed_monotonic']
        assert capture['calls'][-1]['finished_monotonic'] <= moment <= worker['finished_monotonic']
        assert measured['metrics']['confirmed_selection_seconds'] == moment - record['started_monotonic']
