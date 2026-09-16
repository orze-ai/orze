"""One fixed real-model memory pilot; credentials remain in process environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/hot-data/fsx/workspace/erik/orze-p1-continuation-20260914')
CORE = ROOT / 'orze'
PRO = ROOT / 'pro'
OUTPUT = ROOT / 'real-memory-pilot-20260916-v2'
sys.path[:0] = [str(CORE), str(CORE / 'src'), str(PRO / 'src')]
from dotenv import dotenv_values
from examples.research_comparison import campaign, batch
from examples.research_comparison.protocol import digest
from examples.holdout import scheduling

MODELS = {
    'sonnet5': {'backend': 'anthropic', 'model': 'claude-sonnet-5', 'endpoint': 'https://api.anthropic.com/v1'},
    'deepseek_flash': {'backend': 'deepseek', 'model': 'deepseek-flash', 'endpoint': 'https://api.deepseek.com/v1'},
}
RESERVES = {'connectivity_probe': 1, 'sonnet5': 64, 'deepseek_flash': 8}


def write(path, value):
    campaign._write(path, value)


def credentials():
    values = dotenv_values('/hot-data/fsx/workspace/erik/auto-research-1.7b/.env')
    for key in ('ORZE_PRO_KEY', 'ANTHROPIC_API_KEY', 'DEEPSEEK_API_KEY'):
        if not values.get(key):
            raise ValueError('missing authorized credential: ' + key)
        os.environ[key] = values[key]
    os.environ.update(campaign.SETTINGS)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONPATH'] = os.pathsep.join((str(CORE / 'src'), str(PRO / 'src'), str(CORE)))
    for key in ('ORZE_RESEARCH_RESULT_CONTEXT', 'ORZE_ROLE_PROCESS_NONCE', 'ORZE_LLM_USAGE_LOG', 'ORZE_LLM_TOKEN_ENVELOPE'):
        os.environ.pop(key, None)


def instructions(instance):
    shape = scheduling.make_request('produce', candidate='baseline', artifact_utf8='REPLACE_WITH_JSON_STRING')
    return '''You are solving a finite scheduling research task, not training a neural network.
Maximize the sum of values of scheduled jobs while respecting all mandatory jobs, release/deadline windows,
job durations, prerequisite inclusion and finish-before-start dependencies, and capacity at each integer tick.
Use the actual evaluation history to revise hypotheses and avoid repeating an already invalid schedule.
Each round requests two distinct candidate schedules. Propose your own complete schedules, not built-in generators.
If all useful distinct candidates are exhausted, returning an empty batch is allowed and remains in the experiment.
Both arms have the same ordinary history access. Use only the interfaces actually offered in the prompt.
Persistent notes, when offered, are optional: do not manufacture evidence or write merely to satisfy this experiment.

Submit the usual JSON array of ideas with title, hypothesis, parent, and config. Use parent "none" for an independent
candidate, or an eligible ID from the supplied evidence when following up. The config must have exactly two keys:
"kind": "native_cpu_action" and "domain_request": the object below. Keep the fixed envelope fields unchanged.
Replace payload.artifact_utf8 with a JSON-encoded STRING containing your complete candidate:
{"instance_id": "the actual instance ID", "schedule": [{"job_id": "an actual job ID", "start": 0}, ...]}.
Use integer starts. Omit unselected optional jobs. Always include artifact_utf8; the "baseline" label is only
an envelope field and must not select a canned solution. Do not compute or invent an authoritative score.
The independent evaluator will compute feasibility and value from your exact candidate bytes.

DOMAIN_REQUEST_SHAPE:
''' + json.dumps(shape, sort_keys=True) + '\n\nINSTANCE:\n' + json.dumps(instance, sort_keys=True) + '\n'


def prepare():
    OUTPUT.mkdir(mode=0o700)
    assert sum(RESERVES.values()) < 5000
    write(OUTPUT / 'authorization-and-budget.json', {
        'authorization': '允许所有可调用的模型做实验，费用上限5000美元',
        'total_cap_usd': 5000, 'reserved_upper_bounds_usd': RESERVES, 'prior_v1_reserved_upper_bound_usd': 73,
        'reservation_is_not_actual_spend': True,
        'enforcement': 'Each fixed batch has 8 projects x 4 research processes x a 100000-token pre-transport envelope; retries share that process envelope. No model fallback, server tools or unbounded resumption. Exclusive output directories prevent repeated execution.',
        'bound_arithmetic': {'sonnet5': '3.2M tokens x $10/M maximum token rate = $32, reserve $64',
                             'deepseek_flash': '3.2M tokens x $1.2/M peak maximum token rate = $3.84, reserve $8'},
        'prices': {'sonnet5': {'input_per_million': 2, 'output_per_million': 10, 'cache_read_per_million': .2,
                             'source': 'https://platform.claude.com/docs/en/about-claude/pricing'},
                   'deepseek_flash': {'peak_input_per_million': .3, 'peak_output_per_million': 1.2,
                                      'offpeak_input_per_million': .15, 'offpeak_output_per_million': .6,
                                      'source': 'https://api-docs.deepseek.com/quick_start/pricing/'}},
        'initial_source_heads': {name: __import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip() for name,path in [('core', CORE),('pro',PRO)]},
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'isolated CPU scheduling evaluation with authorized remote model inference; no service deployment or host GPU work',
    })
    runtime = campaign.describe()
    write(OUTPUT / 'runtime.json', runtime)
    cases = json.loads((CORE / 'docs/plans/2026-09-16-memory-pilot-inputs.json').read_bytes())
    for label, model in MODELS.items():
        tools = {'workload': 'scheduling-v1', 'rounds': 4, 'num_ideas': 2,
                 'evaluation_protocol': 'schedule-feasibility-v1', 'research_timeout_seconds': 360}
        treatments = {}
        for arm in ('A', 'B'):
            treatments[arm] = {'research_evidence': {'version': 2, 'page_size': 16, 'proposal_page_size': 16,
                                                    'max_snapshot_bytes': 32768, 'max_requests': 3}}
            if arm == 'B':
                treatments[arm]['research_memory'] = {'version': 2, 'max_updates': 1, 'resume_prepared': False}
        shared = {'model': model, 'tools': tools, 'environment': runtime['environment']}
        task_inputs, tasks = {}, []
        for name, instance in cases.items():
            task_id = 'memory-' + name.replace('_', '-')
            values = {'data': instance, 'evaluator': {'source_sha256': hashlib.sha256(Path(scheduling.__file__).read_bytes()).hexdigest(),
                      'protocol': tools['evaluation_protocol']}, 'instructions': instructions(instance),
                      'initial_history': [], 'initial_memory': {'schema': 1, 'entries': []}}
            task_inputs[task_id] = values
            tasks.append({'id': task_id, 'domain': 'scheduling', 'role': name, 'seeds': [17, 41],
                          'inputs': {key: digest(value) for key, value in values.items()},
                          'quality': {'direction': 'maximize', 'max_regression': 0, 'minimum_valid_observations': 2},
                          'limits': {'provider_calls': 24, 'provider_tokens': 400000, 'provider_cost_usd': RESERVES[label]/8,
                                     'reserved_seconds': 40, 'gpu_seconds': 0, 'cli_wall_seconds': 1800}})
        # The existing collector uses the task token limit per research process.
        # Use 100000 there, and retain the conservative 4 x process bound above.
        for task in tasks:
            task['limits']['provider_tokens'] = 100000
        plan = {'schema': 1, 'comparison_id': 'real-memory-' + label.replace('_', '-'), 'mode': 'prospective',
                'repetitions': 2, 'ordering': 'repetition',
                'arms': {arm: {'artifact_sha256': runtime['artifact_sha256'], 'treatment_sha256': digest(treatments[arm])} for arm in treatments},
                'shared': {key: digest(value) for key, value in shared.items()},
                'verifier_sha256': runtime['verifier_sha256'], 'tasks': tasks}
        spec = {'schema': 1, 'protocol': plan, 'shared': shared, 'tasks': task_inputs,
                'arms': {arm: {'runtime': runtime, 'treatment': treatments[arm]} for arm in treatments}}
        batch.validate(spec)
        write(OUTPUT / (label + '-specification.json'), spec)
    print(json.dumps({'prepared': str(OUTPUT), 'reserved_upper_bound_usd': sum(RESERVES.values()), 'spent_usd': 0}))


def probe():
    from orze_pro.agents.research_llm import call_anthropic
    directory = OUTPUT / 'connectivity-probe'
    directory.mkdir()
    os.environ['ORZE_LLM_USAGE_LOG'] = str(directory / 'usage.jsonl')
    os.environ['ORZE_LLM_TOKEN_ENVELOPE'] = '2000'
    result = {}
    started = time.monotonic()
    answer = call_anthropic('Reply with the single word READY.', os.environ['ANTHROPIC_API_KEY'],
                           model='claude-sonnet-5', max_tokens=32, result_out=result)
    record = {'elapsed_seconds': time.monotonic()-started, 'answer': answer, 'result': result}
    write(directory / 'result.json', record)
    print(json.dumps(record))


def execute(label):
    budget = json.loads((OUTPUT / 'authorization-and-budget.json').read_bytes())
    assert sum(budget['reserved_upper_bounds_usd'].values()) + budget['prior_v1_reserved_upper_bound_usd'] <= budget['total_cap_usd']
    spec = json.loads((OUTPUT / (label + '-specification.json')).read_bytes())
    receipt = batch.execute(spec, OUTPUT / label, env=dict(os.environ))
    write(OUTPUT / (label + '-receipt.json'), receipt)
    report = batch.audit(OUTPUT / label, receipt['specification_sha256'])
    write(OUTPUT / (label + '-report.json'), report)
    print(json.dumps({'model': label, 'receipt': receipt, 'counts': report['counts']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'probe', 'run'))
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    credentials()
    if args.action == 'prepare': prepare()
    elif args.action == 'probe': probe()
    else: execute(args.model)
