"""Execute a fixed paired schedule and audit its complete planned denominator.

No resume or automatic retry: an interrupted directory remains evidence. The
caller supplies account permission and independent spending limits. This
bounded scheduling collector supports at most 256 slots per batch.
"""
import json
import os
from pathlib import Path

from . import campaign
from .protocol import digest, keys, read_json, schedule
from .report import compare
from .scheduling import read_capture

TASK_INPUTS = ('data', 'evaluator', 'instructions', 'initial_history', 'initial_memory')


def validate(spec):
    spec = campaign._copy(spec)
    keys(spec, ('schema', 'protocol', 'shared', 'tasks', 'arms'), 'campaign specification')
    if type(spec['schema']) is not int or spec['schema'] != 1:
        raise ValueError('unsupported campaign specification')
    plan = spec['protocol']
    slots = schedule(plan)
    if plan['mode'] != 'prospective' or len(slots) > 256:
        raise ValueError('execution requires a prospective schedule of at most 256 slots')
    keys(spec['shared'], ('model', 'tools', 'environment'), 'shared execution inputs')
    keys(spec['tasks'], (t['id'] for t in plan['tasks']), 'task execution inputs')
    keys(spec['arms'], ('A', 'B'), 'execution arms')
    for value in spec['tasks'].values():
        keys(value, TASK_INPUTS, 'task execution inputs')
    for value in spec['arms'].values():
        keys(value, ('runtime', 'treatment'), 'arm execution inputs')
    # Check every task/arm before the first subprocess, using the declared seed
    # for the first repetition. Later repetitions differ only by slot identity.
    for slot in slots:
        if slot['repetition'] == 0:
            inputs, runtime = _inputs(spec, slot)
            request = campaign._bind(plan, slot['run_id'], inputs, runtime)
            from .scheduling_campaign import _configuration
            _configuration(request, Path('/campaign-preflight'))
    if campaign.describe()['verifier_sha256'] != plan['verifier_sha256']:
        raise ValueError('batch differs from the frozen verifier')
    return spec, slots


def _inputs(spec, slot):
    arm = spec['arms'][slot['arm']]
    return ({**spec['shared'], **spec['tasks'][slot['task_id']], 'treatment': arm['treatment']}, arm['runtime'])


def execute(spec, output, *, env=None):
    """Persist the complete specification, then execute AB/BA in declared order."""
    spec, slots = validate(spec)
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    campaign._write(output / 'specification.json', spec)
    specification_sha = campaign._sha((output / 'specification.json').read_bytes())
    with (output / 'runs.jsonl').open('xb') as journal:
        for index, slot in enumerate(slots):
            inputs, runtime = _inputs(spec, slot)
            child_env = dict(os.environ if env is None else env)
            roots = [str(Path(row['root']).parent) for row in runtime['runtime']]
            child_env['PYTHONPATH'] = os.pathsep.join(roots + [child_env.get('PYTHONPATH', '')])
            directory = output / 'runs' / f'{index:06d}'
            try:
                record = campaign.execute(spec['protocol'], slot['run_id'], inputs, directory,
                                          runtime=runtime, env=child_env)
            except Exception as exc:
                # A preparation/persistence failure may have left partial work.
                # Stop the batch; never hide the missing slot or replay it.
                campaign._write(output / 'interruption.json', {'run_id': slot['run_id'],
                    'error': type(exc).__name__, 'directory': str(directory)})
                break
            pointer = {'run_id': slot['run_id'], 'path': f'runs/{index:06d}/run.json',
                       'sha256': campaign._sha((directory / 'run.json').read_bytes())}
            journal.write(json.dumps(pointer, sort_keys=True).encode() + b'\n')
            journal.flush()
            os.fsync(journal.fileno())
            if (record['supervision']['closure'] or {}).get('event') != 'TREE_CLOSED':
                break  # An uncertain process tree prevents further launches.
    return {'output': str(output), 'specification_sha256': specification_sha,
            'planned_runs': len(slots)}


def audit(output, specification_sha256):
    """Re-read pinned originals; failures/missing slots cannot shrink the plan."""
    output = Path(output).absolute()
    spec, slots = validate(read_capture(output / 'specification.json', specification_sha256))
    journal = output / 'runs.jsonl'
    if journal.is_symlink() or journal.stat().st_size > campaign.MAX_BYTES:
        raise ValueError('invalid campaign index')
    pointers = []
    with journal.open('rb') as stream:
        for raw in stream:
            if len(pointers) >= len(slots):
                raise ValueError('extra campaign run')
            # An incomplete final journal write retains that planned run as
            # missing; no completed record may be inferred from a partial row.
            if not raw.endswith(b'\n'):
                break
            pointer = read_json(raw)
            keys(pointer, ('run_id', 'path', 'sha256'), 'campaign run pointer')
            index = len(pointers)
            if (pointer['run_id'] != slots[index]['run_id']
                    or pointer['path'] != f'runs/{index:06d}/run.json'):
                raise ValueError('campaign index differs from the frozen order')
            pointers.append(pointer)
    records = [{**slots[index], 'protocol_sha256': digest(spec['protocol']), 'pointer': pointer}
               for index, pointer in enumerate(pointers)]

    def verify(row, task, arm):
        pointer = row['pointer']
        record = read_capture(output / pointer['path'], pointer['sha256'])
        inputs, runtime = _inputs(spec, row)
        if (record['request']['inputs'] != inputs or record['request']['runtime'] != runtime
                or any(record[k] != row[k] for k in slots[0])):
            raise ValueError('run differs from its scheduled specification')
        return campaign.verify(record, task, arm, plan=spec['protocol'])

    report = compare(spec['protocol'], records, verify=verify)
    report['specification_sha256'] = specification_sha256
    report['scope'] = ('Audit of a fixed scheduling campaign; provider billing, GPU usage, '
                       'first evidence consumption and real research gains remain unestablished')
    return report
