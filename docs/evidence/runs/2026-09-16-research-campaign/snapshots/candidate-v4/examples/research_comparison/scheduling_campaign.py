"""A bounded acceptance workload: Pro proposals, native CPU evaluation, repeat.

Candidate scores always come from the independent public scheduling evaluator.
The workload seed controls evaluation order, not provider sampling. Model use
requires ordinary Pro licensing and caller authorization; no test bypass lives
in this module.
"""
import json
import os
from pathlib import Path
import random
import secrets
import sqlite3
import sys
import time

import yaml

from .campaign import _copy, _sha, _write
from .protocol import digest, keys, read_json
from .scheduling import audit_scheduling, audit_scheduling_cost, evaluator_identity, verify_scheduling
from examples.holdout import scheduling as domain

TABLES = ('ideas', 'idea_state', 'idea_transitions', 'idea_stage_state', 'idea_stage_transitions',
          'execution_attempts', 'research_artifacts', 'research_observations', 'cpu_action_reservations',
          'cpu_action_decisions', 'cpu_action_scopes', 'cpu_proposal_requests', 'replication_requests')
REF_KEYS = ('task_id', 'phase', 'attempt_id', 'generation')


def _database(root):
    path = root / 'lake.db'
    if not path.exists():
        return {name: [] for name in TABLES}
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {name: [dict(r) for r in conn.execute('SELECT * FROM ' + name + ' ORDER BY rowid')]
                if name in tables else [] for name in TABLES}


def _snapshot(run, label):
    value = {'label': str(len(run['snapshots'])) + ':' + label, 'database': _database(Path(run['root']))}
    run['snapshots'].append(value)
    return value


def _invoke(run, label, command, env=None):
    from orze.engine.supervised_process import prepare_supervised, SupervisionUncertain
    root = Path(run['root'])
    before = _snapshot(run, label + ':before')
    started = time.monotonic()
    process, binding, closure, code, error = None, None, None, None, None
    with (root / (label + '.stdout')).open('xb') as stdout, (root / (label + '.stderr')).open('xb') as stderr:
        try:
            try:
                process = prepare_supervised(command, identity={'scope': str(root), 'campaign_step': label},
                    cwd=str(root), env=dict(os.environ) if env is None else env, stdout=stdout, stderr=stderr)
            except SupervisionUncertain as exc:
                process = exc.process
                raise
            process.start()
            code = process.wait(timeout=60)
            binding, closure = process.binding, process.closure_receipt()
        except Exception as exc:
            error = type(exc).__name__
        finally:
            if process is not None and process.poll() is None:
                process.stop(timeout=10)
    finished = time.monotonic()
    after = _snapshot(run, label + ':after')
    run['calls'].append({'label': label, 'command': command, 'cwd': str(root), 'exit_code': code,
        'started_monotonic': started, 'finished_monotonic': finished, 'wall_seconds': finished - started,
        'controller_supervision': {'binding': binding, 'closure': closure},
        'before_snapshot': before['label'], 'after_snapshot': after['label'], 'error': error})
    if error is not None or code != 0:
        raise RuntimeError('campaign step did not complete: ' + label)


def _native(run, label, args=None):
    root = Path(run['root'])
    _invoke(run, label, [sys.executable, '-m', 'examples.holdout',
                       *(args or ['-c', str(root / 'orze.yaml'), '--once'])])


def _admit(run, label, task, request):
    from orze.idea_lake import IdeaLake
    root = Path(run['root'])
    before = _snapshot(run, label + ':before')
    raw = json.dumps({'kind': 'native_cpu_action', 'domain_request': request}, sort_keys=True, separators=(',', ':'))
    lake = IdeaLake(str(root / 'lake.db'))
    try:
        outcome = lake.insert(task, task, raw, '', status='queued', kind='native_cpu_action', if_absent=True)
    finally:
        lake.close()
    after = _snapshot(run, label + ':after')
    run['admissions'].append({'label': label, 'task_id': task, 'domain_request': request,
        'raw_config': raw, 'outcome': outcome, 'before_snapshot': before['label'], 'after_snapshot': after['label']})
    if outcome['status'] != 'inserted':
        raise ValueError('unexpected preexisting campaign action')


def _finish(run):
    root = Path(run['root'])
    run['database'] = _database(root)
    artifacts = [read_json(r['record_json']) for r in run['database']['research_artifacts']]
    contents = {}
    for artifact in artifacts:
        path = Path(artifact['path'])
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('campaign artifact escapes its project')
        raw = path.read_bytes()
        if _sha(raw) != artifact['content_sha256']:
            raise ValueError('campaign artifact changed')
        contents[artifact['artifact_id']] = raw.decode('utf-8')
    run.update(artifacts=artifacts, artifact_contents=contents,
               observations=[read_json(r['record_json']) for r in run['database']['research_observations']],
               result_envelopes={a['producer']['task_id']: read_json(contents[a['artifact_id']])
                                 for a in artifacts if a['logical_name'] == 'evaluation'})


def _configuration(request, root):
    inputs = request['inputs']
    tools, model = inputs['tools'], inputs['model']
    keys(tools, ('workload', 'rounds', 'num_ideas', 'evaluation_protocol'), 'scheduling tools')
    if (tools['workload'] != 'scheduling-v1' or type(tools['rounds']) is not int
            or not 1 <= tools['rounds'] <= 16 or type(tools['num_ideas']) is not int
            or not 1 <= tools['num_ideas'] <= 16 or tools['evaluation_protocol'] not in domain.PROTOCOLS):
        raise ValueError('invalid acceptance workload controls')
    keys(model, ('backend', 'model', 'endpoint'), 'model settings')
    if (model['backend'] not in ('custom', 'ollama', 'openai', 'gemini', 'anthropic', 'deepseek')
            or type(model['model']) is not str or not model['model'] or type(model['endpoint']) is not str):
        raise ValueError('explicit model configuration required')
    if evaluator_identity(tools['evaluation_protocol']) != digest(inputs['evaluator']):
        raise ValueError('evaluator differs from frozen inputs')
    if type(inputs['instructions']) is not str or not inputs['instructions'].strip():
        raise ValueError('explicit task instructions required')
    task = next(t for t in request['protocol']['tasks'] if t['id'] == request['slot']['task_id'])
    budget = task['limits']['reserved_seconds']
    if type(budget) not in (int, float) or budget <= 0:
        raise ValueError('explicit CPU reservation limit required')
    cfg = {'execution': {'version': 1, 'resource': 'cpu', 'slots': 1, 'wall_budget_seconds': budget},
           'results_dir': str(root / 'results'), 'idea_lake_db': str(root / 'lake.db'),
           'ideas_file': str(root / 'ideas.md'), 'min_disk_gb': 0,
           'action_domain': {'version': 1, 'kind': 'schedule_holdout', 'config': {'instance': domain.validate_instance(inputs['data'])}},
           'action_policy': {'version': 1, 'kind': 'queue', 'idle': 'wait', 'wait_seconds': .05, 'config': {}},
           '_project_root': str(root), 'nested_config_whitelist': ['domain_request'],
           'report': {'primary_metric': 'score', 'sort': 'descending',
                      'columns': [{'key': 'score', 'source': 'assessment.json:score'}]}}
    treatment = inputs['treatment']
    if type(treatment) is not dict or set(treatment) & set(cfg):
        raise ValueError('treatment overrides shared task or resource settings')
    cfg.update(treatment)
    return cfg


def _research_command(request, root, cycle):
    model, tools = request['inputs']['model'], request['inputs']['tools']
    return [request['runtime']['python'], '-m', 'orze_pro.agents.research', '-c', str(root / 'orze.yaml'),
        '--backend', model['backend'], '--model', model['model'], '--endpoint', model['endpoint'],
        '--cycle', str(cycle), '--num-ideas', str(tools['num_ideas']), '--rules-file', str(root / 'instructions.md'),
        '--rules-sha256', _sha(request['inputs']['instructions'].encode('utf-8')), '--lake-db', str(root / 'lake.db')]


def _cost_scope(run, request):
    expected = {}
    for snapshot in run['snapshots']:
        for row in snapshot['database']['execution_attempts']:
            ref = {k: row[k] for k in REF_KEYS}
            expected[digest(ref)] = ref
    return {'instance': request['inputs']['data'], 'protocol': request['inputs']['tools']['evaluation_protocol'],
            'expected_attempt_refs': list(expected.values()),
            'worker_command': [request['runtime']['python'], str(Path(domain.__file__).resolve())]}


def run(request, output):
    from orze.core.research_result import make_native_result_ref, read_native_result
    from orze_pro.agents.usage_report import audit_usage, native_request
    root = Path(output) / 'project'
    root.mkdir()
    cfg = _configuration(request, root)
    inputs, tools, model = request['inputs'], request['inputs']['tools'], request['inputs']['model']
    config = root / 'orze.yaml'
    config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    (root / 'ideas.md').write_text('# Ideas\n')
    rules = root / 'instructions.md'
    rules.write_text(inputs['instructions'], encoding='utf-8')
    rule_sha = _sha(rules.read_bytes())
    results = root / 'agent_results'
    results.mkdir()
    run = {'root': str(root), 'cfg': cfg, 'calls': [], 'admissions': [], 'snapshots': [],
           'campaign': {'request_sha256': digest(request), 'research': [], 'evaluation_order': []}}
    _snapshot(run, 'initial')
    requests, evaluations = [], []
    rng = random.Random(request['slot']['seed'])
    try:
        _native(run, 'initialize')
        for cycle in range(1, tools['rounds'] + 1):
            nonce = secrets.token_hex(32)
            ref = make_native_result_ref(attempt_id=f'campaign-research-{cycle:04d}', role_name='research',
                project_root=root, results_dir=root / 'results', ideas_file=root / 'ideas.md',
                result_path=results / f'research-{cycle:04d}.json', process_nonce=nonce)
            env = dict(os.environ, ORZE_RESEARCH_RESULT_CONTEXT=json.dumps(ref), ORZE_ROLE_PROCESS_NONCE=nonce,
                       ORZE_LLM_USAGE_LOG=str(root / 'usage.jsonl'))
            command = _research_command(request, root, cycle)
            research = {'cycle': cycle, 'ref': ref, 'manifests': [], 'outcome': None}
            run['campaign']['research'].append(research)
            try:
                _invoke(run, f'research-{cycle:04d}', command, env)
            finally:
                for path in sorted(results.glob(ref['attempt_id'] + '.prompt*.json')):
                    suffix = path.name[len(ref['attempt_id']):]
                    number = 1 if suffix == '.prompt.json' else int(suffix[len('.prompt-'):-len('.json')])
                    manifest = read_json(path.read_bytes())
                    expected = native_request(ref, process_nonce=nonce, manifest_record=manifest,
                                              request_number=number, request_id=f'cycle-{cycle:04d}-request-{number:04d}')
                    requests.append(expected)
                    research['manifests'].append({'number': number, 'record': manifest, 'request': expected})
            outcome = research['outcome'] = read_native_result(ref, process_nonce=nonce)
            accepted = list(outcome['accepted_ids'])
            for index in range(len(accepted)):
                _native(run, f'produce-{cycle:04d}-{index:04d}')
            rng.shuffle(accepted)
            for index, task_id in enumerate(accepted):
                artifacts = [read_json(r['record_json']) for r in _database(root)['research_artifacts']]
                candidates = [a for a in artifacts if a['producer']['task_id'] == task_id and a['logical_name'] == 'candidate']
                if len(candidates) != 1:
                    raise ValueError('accepted proposal did not produce one candidate')
                source = candidates[0]
                task_id = f'idea-evaluate-{cycle:04d}-{index:04d}'
                _admit(run, task_id, task_id, domain.make_request('evaluate', protocol=tools['evaluation_protocol'],
                                                               source_id=source['artifact_id']))
                _native(run, task_id)
                quality = domain.evaluate(inputs['data'], Path(source['path']).read_bytes(), tools['evaluation_protocol'])
                run['campaign']['evaluation_order'].append({'task_id': task_id, 'source_id': source['artifact_id']})
                evaluations.append((task_id, quality))
        if not evaluations:
            raise ValueError('campaign produced no evaluable candidates')
        # Recompute from candidate bytes; provider scores never select the winner.
        valid = [item for item in evaluations if item[1]['status'] == 'valid']
        selected = max(valid, key=lambda item: item[1]['scheduled_value'])[0] if valid else evaluations[0][0]
        _native(run, 'confirmation-admit', ['replicate', selected, '-c', str(config),
            '--request-id', 'campaign-confirmation', '--reason', 'independent evaluation of the selected candidate'])
        _native(run, 'confirmation-execute')
        _snapshot(run, 'final')
        _finish(run)
        attempts = run['database']['execution_attempts']
        to_ref = lambda row: {k: row[k] for k in REF_KEYS}
        expected = {}
        for snapshot in run['snapshots']:
            for row in snapshot['database']['execution_attempts']:
                ref = to_ref(row)
                expected[digest(ref)] = ref
        replica = read_json(run['database']['replication_requests'][0]['record_json'])
        scope = {'instance': inputs['data'], 'protocol': tools['evaluation_protocol'],
                 'expected_attempt_refs': list(expected.values()),
                 'selected_ref': to_ref(next(r for r in attempts if r['task_id'] == selected)),
                 'confirmation_ref': to_ref(next(r for r in attempts if r['task_id'] == replica['task_id'])),
                 'worker_command': [sys.executable, str(Path(domain.__file__).resolve())]}
        journal = root / 'usage.jsonl'
        usage_scope = {'journals': [{'path': str(journal), 'sha256': _sha(journal.read_bytes())}], 'requests': requests}
        # Audit now, but retain originals; verification must recompute both reports.
        run['campaign'].update(scope=scope, usage_scope=usage_scope,
                               usage_audit=audit_usage(**usage_scope), task_audit=audit_scheduling(run, **scope))
        return run
    finally:
        # Preserve partial state even if a proposal, transport, worker or audit fails.
        try:
            _snapshot(run, 'collector-final')
            _finish(run)
            run['campaign']['cost_scope'] = _cost_scope(run, request)
            journal = root / 'usage.jsonl'
            run['campaign']['usage_scope'] = {'journals':
                [{'path': str(journal), 'sha256': _sha(journal.read_bytes())}] if journal.is_file() else [],
                'requests': requests}
            _write(root / 'partial-capture.json', run)
        except Exception:
            pass


def verify(capture, task, *, request, complete=True):
    from orze_pro.agents.prompt_manifest import _validate_manifest
    from orze_pro.agents.usage_report import audit_usage
    capture = _copy(capture)
    root = Path(capture['root'])
    cfg = _configuration(request, root)
    campaign = capture['campaign']
    if capture['cfg'] != cfg or campaign['request_sha256'] != digest(request):
        raise ValueError('workload configuration differs from frozen campaign')
    research = campaign['research']
    rounds = request['inputs']['tools']['rounds']
    if (len(research) > rounds or complete and len(research) != rounds
            or [r['cycle'] for r in research] != list(range(1, len(research) + 1))):
        raise ValueError('missing or reordered research round')
    calls = {call['label']: call for call in capture['calls']}
    if len(calls) != len(capture['calls']):
        raise ValueError('duplicate campaign invocation')
    for label, call in calls.items():
        binding = call['controller_supervision']['binding']
        if (binding['identity'] != {'scope': str(root), 'campaign_step': label}
                or binding['command_sha256'] != digest(call['command']) or call['cwd'] != str(root)):
            raise ValueError('campaign command differs from the observed process binding')
    expected_requests = []
    for row in research:
        ref = row['ref']
        if (ref['attempt_id'] != f"campaign-research-{row['cycle']:04d}" or ref['role_name'] != 'research'
                or ref['project_root'] != str(root)):
            raise ValueError('research invocation belongs to another campaign')
        label = f"research-{row['cycle']:04d}"
        if label not in calls or calls[label]['command'] != _research_command(request, root, row['cycle']):
            raise ValueError('research command differs from the frozen model or inputs')
        numbers = [entry['number'] for entry in row['manifests']]
        if complete and not numbers or sorted(numbers) != list(range(1, len(numbers) + 1)):
            raise ValueError('incomplete native prompt inventory')
        for entry in row['manifests']:
            saved = entry['record']
            if saved['identity'] != {k: v for k, v in ref.items() if k not in ('schema', 'result_path')}:
                raise ValueError('native manifest identity mismatch')
            manifest = _validate_manifest(saved['manifest'])
            expected = {'id': f"cycle-{row['cycle']:04d}-request-{entry['number']:04d}",
                'binding': {'kind': 'native_research_request', 'request_number': entry['number'],
                    'prompt_manifest_sha256': digest(manifest), 'native_ref_sha256': digest(ref),
                    'role_attempt_id': ref['attempt_id'], 'role_name': 'research'}}
            if entry['request'] != expected:
                raise ValueError('provider request differs from native prompt inventory')
            expected_requests.append(expected)
    usage_scope = campaign['usage_scope']
    if (usage_scope['requests'] != expected_requests or len(usage_scope['journals']) > 1
            or complete and len(usage_scope['journals']) != 1
            or any(row['path'] != str(root / 'usage.jsonl') for row in usage_scope['journals'])):
        raise ValueError('provider usage scope differs from the executed workload')
    scope = campaign['scope' if complete else 'cost_scope']
    expected_worker = [request['runtime']['python'],
                       str(Path(request['runtime']['harness_root']) / 'examples/holdout/scheduling.py')]
    if scope['worker_command'] != expected_worker:
        raise ValueError('native evaluator command differs from the frozen harness')
    refs = {}
    for snapshot in capture['snapshots']:
        for row in snapshot['database']['execution_attempts']:
            ref = {k: row[k] for k in REF_KEYS}
            refs[digest(ref)] = ref
    if scope['expected_attempt_refs'] != list(refs.values()):
        raise ValueError('native attempt coverage differs from captured history')
    if complete:
        measured = verify_scheduling(capture, task, scope=scope)
    else:
        if (task['inputs']['data'] != digest(scope['instance'])
                or task['inputs']['evaluator'] != evaluator_identity(scope['protocol'])):
            raise ValueError('failed workload has another task identity')
        measured = audit_scheduling_cost(capture, **scope)['measurement']
    usage = audit_usage(**usage_scope)
    measured['metrics'].update(usage['comparison_metrics'])
    return measured
