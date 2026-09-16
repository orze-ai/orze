"""Execute an explicitly planned, owned comparison workload and retain its inputs.

This is an experiment collector, not an execution permission or a sandbox.
The workload and collector are trusted code included in the verifier identity.
Credentials are inherited by the child and are never copied into the record.
"""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

from .protocol import digest, keys, read_json, schedule

SETTINGS = {'ORZE_OPENAI_STREAM': '0', 'ORZE_LLM_FALLBACK': '', 'ORZE_CLAUDE_FALLBACK': ''}
INPUT_NAMES = ('model', 'tools', 'environment', 'treatment', 'data', 'evaluator',
               'instructions', 'initial_history', 'initial_memory')
MAX_BYTES = 64 * 1024 * 1024


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _copy(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError('campaign input exceeds byte limit')
    return read_json(raw)


def _write(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b'\n'
    if len(raw) > MAX_BYTES:
        raise ValueError('campaign output exceeds byte limit')
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _tree(root):
    rows = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and not {'__pycache__', 'node_modules'} & set(path.relative_to(root).parts):
            if not path.resolve().is_relative_to(root):
                raise ValueError('runtime source escapes package root')
            rows[str(path.relative_to(root))] = _sha(path.read_bytes())
    if not rows:
        raise ValueError('empty runtime package')
    return {'sha256': digest(rows), 'file_count': len(rows)}


def describe():
    """Observe importable artifacts and reproducibility inputs without importing Pro."""
    runtime = []
    for name in ('orze', 'orze_pro'):
        spec = importlib.util.find_spec(name)
        roots = [] if spec is None else list(spec.submodule_search_locations or ())
        if len(roots) != 1:
            raise ValueError('comparison requires unambiguous Core and Pro packages')
        root = Path(roots[0]).resolve(strict=True)
        runtime.append({'name': name, 'root': str(root), **_tree(root)})
    artifact = [{k: row[k] for k in ('name', 'sha256', 'file_count')} for row in runtime]
    harness = Path(__file__).resolve().parent
    files = [*harness.glob('*.py'), *harness.parent.joinpath('holdout').glob('*.py')]
    sources = {str(p.relative_to(harness.parent)): _sha(p.read_bytes()) for p in sorted(files)}
    environment = {'python': sys.version, 'implementation': platform.python_implementation(),
                   'platform': platform.platform(), 'executable_sha256': _sha(Path(sys.executable).read_bytes()),
                   'distributions': sorted((d.metadata['Name'], d.version)
                                           for d in importlib.metadata.distributions() if d.metadata['Name']),
                   'settings': dict(SETTINGS),
                   'startup_modules': {name: _sha(Path(sys.modules[name].__file__).read_bytes())
                                       for name in ('sitecustomize', 'usercustomize')
                                       if name in sys.modules and getattr(sys.modules[name], '__file__', None)}}
    return _copy({'python': sys.executable, 'runtime': runtime, 'artifact_sha256': digest(artifact),
                  'harness_root': str(harness.parents[1]), 'environment': environment,
                  'verifier_sha256': digest({'harness': sources, 'dependencies': artifact})})


def _bind(plan, run_id, inputs, runtime):
    planned = {slot['run_id']: slot for slot in schedule(plan)}
    if run_id not in planned:
        raise ValueError('unplanned campaign run')
    slot = planned[run_id]
    task = next(t for t in plan['tasks'] if t['id'] == slot['task_id'])
    keys(inputs, INPUT_NAMES, 'campaign inputs')
    if (any(digest(inputs[k]) != value for k, value in plan['shared'].items())
            or any(digest(inputs[k]) != value for k, value in task['inputs'].items())
            or digest(inputs['treatment']) != plan['arms'][slot['arm']]['treatment_sha256']
            or runtime['artifact_sha256'] != plan['arms'][slot['arm']]['artifact_sha256']
            or digest(runtime['environment']) != plan['shared']['environment']):
        raise ValueError('campaign inputs do not match the frozen protocol')
    if inputs['initial_history'] != [] or inputs['initial_memory'] is not None:
        raise ValueError('this collector requires explicit fresh history and memory')
    limit = task['limits']['cli_wall_seconds']
    if type(limit) not in (int, float) or not 0 < limit <= 86400:
        raise ValueError('execution requires a finite outer acceptance timeout')
    return {'schema': 1, 'protocol': plan, 'protocol_sha256': digest(plan), 'slot': slot,
            'inputs': inputs, 'runtime': runtime, 'timeout_seconds': limit}


def execute(plan, run_id, inputs, output, *, runtime=None, env=None):
    """Run one frozen slot. Existing or interrupted output directories never replay.

Provider/account/resource permission and spending enforcement are external
prerequisites. Protocol limits are checked by the comparison reducer; this
collector enforces the outer wall limit on its own supervised process tree.
"""
    from orze.engine.supervised_process import prepare_supervised, SupervisionUncertain
    plan, inputs = _copy(plan), _copy(inputs)
    runtime = _copy(describe() if runtime is None else runtime)
    request = _bind(plan, run_id, inputs, runtime)
    # The collector/verifier environment is fixed independently of the chosen
    # execution arm. A different arm may run another compatible artifact.
    if describe()['verifier_sha256'] != plan['verifier_sha256']:
        raise ValueError('collector differs from the frozen verifier')
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _write(output / 'request.json', request)
    child_env = dict(os.environ if env is None else env)
    child_env.update(SETTINGS)
    child_env.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    harness_root = str(Path(__file__).resolve().parents[2])
    child_env['PYTHONPATH'] = harness_root + os.pathsep + child_env.get('PYTHONPATH', '')
    for name in ('ORZE_RESEARCH_RESULT_CONTEXT', 'ORZE_ROLE_PROCESS_NONCE', 'ORZE_LLM_USAGE_LOG',
                 'ORZE_LLM_TOKEN_ENVELOPE'):
        child_env.pop(name, None)
    task = next(t for t in plan['tasks'] if t['id'] == request['slot']['task_id'])
    tokens = task['limits']['provider_tokens']
    if type(tokens) is int and tokens > 0:
        child_env['ORZE_LLM_TOKEN_ENVELOPE'] = str(tokens)
    command = [runtime['python'], '-m', 'examples.research_comparison.campaign_worker',
               '--request', str(output / 'request.json'), '--request-sha256', digest(request),
               '--output', str(output)]
    process, binding, closure, exit_code, error = None, None, None, None, None
    with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
        try:
            try:
                process = prepare_supervised(command, identity={'comparison_run': run_id,
                    'request_sha256': digest(request)}, cwd=str(output), env=child_env, stdout=stdout, stderr=stderr)
            except SupervisionUncertain as exc:
                process = exc.process
                raise
            binding = process.binding
            process.start()
            remaining = request['timeout_seconds'] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('campaign outer limit')
            exit_code = process.wait(timeout=remaining)
            closure = process.closure_receipt()
        except Exception as exc:
            error = type(exc).__name__
        finally:
            if process is not None:
                try:
                    if process.poll() is None:
                        process.stop(timeout=10)
                    exit_code = process.poll()
                    closure = process.closure_receipt()
                    binding = process.binding
                except Exception as exc:
                    error = error or type(exc).__name__
    finished = time.monotonic()
    files = {}
    for name in ('request.json', 'worker.json', 'capture.json', 'stdout.log', 'stderr.log',
                 'project/partial-capture.json', 'project/usage.jsonl'):
        path = output / name
        if path.is_file() and not path.is_symlink():
            files[name] = {'bytes': path.stat().st_size, 'sha256': _sha(path.read_bytes())}
    record = {**request['slot'], 'schema': 1, 'protocol_sha256': digest(plan), 'request': request,
              'command': command, 'output': str(output), 'started_monotonic': started,
              'finished_monotonic': finished, 'wall_seconds': finished - started,
              'exit_code': exit_code, 'error': error, 'supervision': {'binding': binding, 'closure': closure},
              'files': files}
    _write(output / 'run.json', record)
    return record


def verify(record, task, arm, *, plan):
    """Re-read pinned raw files and recompute task/usage measurements for a slot."""
    from .scheduling import _closed, read_capture
    from .scheduling_campaign import verify as verify_workload
    record, plan = _copy(record), _copy(plan)
    if describe()['verifier_sha256'] != plan['verifier_sha256']:
        raise ValueError('audit differs from the frozen verifier')
    request = record['request']
    expected = _bind(plan, record['run_id'], request['inputs'], request['runtime'])
    expected_task = next(t for t in plan['tasks'] if t['id'] == record['task_id'])
    if (request != expected or digest(task) != digest(expected_task) or record['arm'] != arm
            or record['protocol_sha256'] != digest(plan)
            or any(record[k] != v for k, v in request['slot'].items())):
        raise ValueError('campaign record identity mismatch')
    folder = Path(record['output'])
    if record['error'] is not None or record['exit_code'] not in (0, 1):
        raise ValueError('campaign did not close with an auditable result')
    raw = {}
    for name in ('request.json', 'worker.json'):
        raw[name] = read_capture(folder / name, record['files'][name]['sha256'])
    if raw['request.json'] != request:
        raise ValueError('campaign did not close with an auditable result')
    _closed(record['supervision']['closure'], record['supervision']['binding'], record['exit_code'])
    binding = record['supervision']['binding']
    command = [request['runtime']['python'], '-m', 'examples.research_comparison.campaign_worker',
               '--request', str(folder / 'request.json'), '--request-sha256', digest(request), '--output', str(folder)]
    if (record['command'] != command or binding['command_sha256'] != digest(command)
            or binding['identity'] != {'comparison_run': record['run_id'], 'request_sha256': digest(request)}):
        raise ValueError('campaign command differs from the owned process')
    worker = raw['worker.json']
    complete = record['exit_code'] == 0
    capture_name = 'capture.json' if complete else 'project/partial-capture.json'
    capture_sha = worker['capture_sha256' if complete else 'partial_capture_sha256']
    if (worker['request_sha256'] != digest(request) or worker['before'] != request['runtime']
            or worker['after'] != worker['before'] or worker['status'] != ('completed' if complete else 'failed')
            or capture_sha != record['files'].get(capture_name, {}).get('sha256') or capture_sha is None):
        raise ValueError('workload runtime or inputs changed')
    capture = read_capture(folder / capture_name, capture_sha)
    start, finish = record['started_monotonic'], record['finished_monotonic']
    if (type(start) not in (int, float) or type(finish) not in (int, float)
            or not 0 <= start <= worker['started_monotonic'] <= worker['finished_monotonic'] <= finish
            or abs(finish - start - record['wall_seconds']) > 1e-8):
        raise ValueError('campaign clocks are inconsistent')
    result = verify_workload(capture, task, request=request, complete=complete)
    result['metrics']['cli_wall_seconds'] = record['wall_seconds']
    return result
