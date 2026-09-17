"""Fixed paired real-model diagnostic of available memory actions.

Both arms use real providers, normal licensing and actual native CPU evaluation.
A requires a counterexample note; B keeps ordinary optional memory. Neither changes generated output.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import yaml
import real_memory_pilot_v2 as pilot
from examples.research_comparison import campaign
from examples.research_comparison.scheduling_campaign import (
    _configuration, _native, _admit, _initialize_memory, _snapshot,
    _database, _finish, _consume_evaluation, _retained_duplicate_owner,
)
from examples.holdout import scheduling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('repetition', type=int, choices=range(2))
    parser.add_argument('arm', choices=['B'])
    args = parser.parse_args()
    started = time.monotonic()
    pilot.credentials()
    root = pilot.ROOT / 'model-capacity-20260917' / f'pair-{args.repetition}-{args.arm}'
    root.mkdir(mode=0o700)
    plan = json.loads((pilot.ROOT/'model_capacity_plan_20260917.json').read_bytes())
    budget = {'authorized_total_usd': 5000, 'reserved_for_slot_usd': 4,
              'whole_batch_reserved_usd': 8, 'cumulative_reserved_usd': 339,
              'bound': plan['budget_bound'], 'scope': plan['scope']}
    assert budget['cumulative_reserved_usd'] <= budget['authorized_total_usd']
    task_plan = plan['tasks'][args.repetition]
    prompt_sources = {name: hashlib.sha256((pilot.PRO/'src/orze_pro/agents'/name).read_bytes()).hexdigest()
                      for name in ('memory_consumer.py','memory_updates.py','research.py')}
    assert prompt_sources == plan['code_sha256']
    pilot.write(root/'authorization-and-budget.json', budget)
    spec = json.loads((pilot.ROOT/'real-memory-pilot-20260916-admission/haiku45-specification.json').read_bytes())
    runtime = campaign.describe()
    inputs = {**spec['shared'], **spec['tasks']['memory-target'], 'treatment': spec['arms']['B']['treatment']}
    inputs['data'] = task_plan['data']
    inputs['model'] = {**inputs['model'], 'model':plan['model']}
    inputs['environment'] = runtime['environment']
    inputs['tools'] = {**inputs['tools'], 'rounds': 2}
    inputs['initial_history'] = {'valid': task_plan['seed_valid'], 'invalid': task_plan['seed_invalid']}
    for task in spec['protocol']['tasks']:
        if task['id'] == 'memory-target':
            task['limits']['provider_tokens'] = 200000
    inputs['instructions'] = pilot.instructions(task_plan['data'])
    if args.arm == 'A':
        inputs['instructions'] += """
When persistent memory is offered and the visible qualified feedback contains a failed schedule,
first retain ONE concise counterexample note that is not yet stored, citing the exact eligible evaluator
report ID from the evidence. Explain the violated constraint using the actual schedule and verdict.
Use the offered update_memory protocol; do not invent a source or a successful measurement.
If no eligible source exists, do not assert a sourced fact. After the note is stored, propose improvements
using current evidence and any retained note. Do not repeat the same note or manufacture extra notes.
"""
    inputs['instructions'] += """
Follow the genealogy contract using exact eligible IDs. Emit bare JSON without surrounding prose.
Keep domain_request.payload.candidate exactly "baseline"; this is a fixed interface field, not a strategy name.
"""
    request = {'inputs': inputs, 'protocol': spec['protocol'], 'runtime': runtime,
               'slot': {'task_id': 'memory-target', 'seed': 17, 'arm': args.arm, 'repetition': args.repetition}}
    cfg = _configuration(request, root)
    config = root/'orze.yaml'; config.write_text(yaml.safe_dump(cfg))
    rules = root/'instructions.md'; rules.write_text(inputs['instructions'])
    (root/'ideas.md').write_text('# Ideas\n')
    pilot.write(root/'plan.json', {'request': request, 'budget': budget,
        'heads': {name: subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
                  for name,path in [('core',pilot.CORE),('pro',pilot.PRO)]},
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'effective_prompt_source_sha256': prompt_sources,
        'predeclared_plan_sha256': hashlib.sha256((pilot.ROOT/'model_capacity_plan_20260917.json').read_bytes()).hexdigest(),
        'protocol_scope': 'Inherited task metadata used for configuration; this direct diagnostic is not a formal CLI protocol audit.'})
    run = {'root': str(root), 'cfg': cfg, 'calls': [], 'admissions': [], 'snapshots': [],
           'campaign': {'consumption': [], 'research': [], 'retained_duplicates': []}}
    outcomes = []
    traces = []
    error = None
    try:
        _native(run, 'initialize')
        _initialize_memory(run, request)
        seeds = {
            'valid': task_plan['seed_valid'],
            'overload': task_plan['seed_invalid'],
        }
        for label, schedule in seeds.items():
            candidate = {'instance_id': inputs['data']['instance_id'],
                         'schedule': [{'job_id': job, 'start': tick} for job,tick in schedule]}
            task = 'idea-seed-'+label
            _admit(run, task, task, scheduling.make_request('produce', artifact_utf8=json.dumps(candidate)))
            _native(run, task)
            _finish(run)
            artifact = next(a for a in run['artifacts'] if a['producer']['task_id']==task and a['logical_name']=='candidate')
            evaluator = 'idea-evaluate-seed-'+label
            _admit(run, evaluator, evaluator, scheduling.make_request('evaluate', protocol='schedule-feasibility-v1',source_id=artifact['artifact_id']))
            _native(run, evaluator)
            verdict = _consume_evaluation(run, request, evaluator)
            assert (verdict['status']=='valid' and verdict['scheduled_value']==task_plan['seed_score']) if label=='valid' else verdict['status']=='invalid'
        cfg['_config_path'] = str(config)
        cfg['_research_config_sha256'] = hashlib.sha256(config.read_bytes()).hexdigest()
        from orze.core.research_interfaces import register_domain
        register_domain('schedule_holdout', 'acceptance.schedule.v1', scheduling.SchedulingDomain)
        from orze_pro.agents.memory_sources import capture_report_bindings
        bindings = capture_report_bindings(root/'results', cfg,
            ['idea-evaluate-seed-valid','idea-evaluate-seed-overload'], rules_sha256=hashlib.sha256(rules.read_bytes()).hexdigest())
        pilot.write(root/'seed-source-bindings.json', bindings)
        assert bindings['availability'] == 'available'
        assert all(v['availability'] == 'qualified' for v in bindings['bindings'].values()), bindings
        from orze_pro.agents import research
        from orze_pro.agents import research_llm
        provider = research_llm.call_anthropic
        def with_response_allowance(*args, **kwargs):
            kwargs['max_tokens'] = plan['requested_response_max_tokens']
            return provider(*args, **kwargs)
        research_llm.call_anthropic = with_response_allowance
        original = research.call_llm
        def observe(prompt, *args, **kwargs):
            call_started = time.monotonic()
            response = original(prompt, *args, **kwargs)
            record = {'prompt': prompt, 'response': response, 'seconds': time.monotonic()-call_started,
                      'result': dict(kwargs.get('result_out') or {})}
            pilot.write(root/('response-'+str(len(traces)+1)+'.json'), record)
            traces.append(record)
            return response
        research.call_llm = observe
        os.environ['ORZE_LLM_USAGE_LOG'] = str(root/'usage.jsonl')
        os.environ['ORZE_LLM_TOKEN_ENVELOPE'] = '200000'
        run['seed_setup_seconds'] = time.monotonic()-started
        for cycle in (1,2):
            outcome = {}
            research.run_research_cycle('anthropic', cycle, root/'ideas.md', root/'results', cfg['report'],
                num_ideas=2, model=plan['model'], endpoint='https://api.anthropic.com/v1',
                lake_db_path=root/'lake.db', project_cfg=cfg, rules_file=str(rules),
                rules_sha256=hashlib.sha256(rules.read_bytes()).hexdigest(), result_out=outcome)
            outcome['cycle'] = cycle
            outcome['observed_seconds'] = time.monotonic()-started
            outcomes.append(outcome)
            _snapshot(run, 'research-'+str(cycle))
            accepted = outcome.get('accepted_ids', [])
            for index in range(len(accepted)):
                _native(run, f'produce-{cycle}-{index}')
            for index, task in enumerate(accepted):
                _finish(run)
                sources = [a for a in run['artifacts'] if a['producer']['task_id']==task and a['logical_name']=='candidate']
                if not sources:
                    owner = _retained_duplicate_owner(_database(root), task, (root/'ideas.md').read_text())
                    run['campaign']['retained_duplicates'].append({'cycle':cycle,'idea_id':task,'duplicate_of':owner})
                    continue
                assert len(sources)==1
                evaluator = f'idea-evaluate-diagnostic-{cycle}-{index}'
                _admit(run,evaluator,evaluator,scheduling.make_request('evaluate',protocol='schedule-feasibility-v1',source_id=sources[0]['artifact_id']))
                _native(run,evaluator)
                _consume_evaluation(run,request,evaluator)
                run['campaign']['consumption'][-1]['observed_seconds'] = time.monotonic()-started
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        _finish(run)
        pilot.write(root/'capture.json',run)
        pilot.write(root/'result.json', {'outcomes':outcomes, 'responses':len(traces),
            'elapsed_seconds':time.monotonic()-started, 'memory_rows':_database(root)['research_memory_current_v1'],
            'evaluations':run['campaign']['consumption'], 'scope':budget['scope'], 'error':error,
            'arm':args.arm, 'repetition':args.repetition, 'seed_score':task_plan['seed_score']})
    print(json.dumps({'output':str(root),'outcomes':[x.get('reason') for x in outcomes],
                      'responses':len(traces),'memory_revision':_database(root)['research_memory_current_v1'][0]['revision']}))


if __name__ == '__main__':
    main()
