"""Actual Pro HTTP/CPU workload under a complete, runtime-bound collector."""
import copy
import hashlib
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
            raw = json.dumps({'id': 'offline-campaign', 'model': 'offline-revision',
                'choices': [{'message': {'content': json.dumps([proposal])}, 'finish_reason': 'stop'}],
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
    assert metrics['confirmed_selection_seconds'] is None
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
