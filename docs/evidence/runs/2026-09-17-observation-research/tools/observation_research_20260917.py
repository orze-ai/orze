"""Fixed fresh-task comparison of full versus compact native observation views."""
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
    parser.add_argument('repetition', type=int, choices=range(4))
    parser.add_argument('arm', choices=['A', 'B'])
    args = parser.parse_args()
    started = time.monotonic()
    pilot.credentials()
    root = pilot.ROOT / 'observation-research-20260917' / f'pair-{args.repetition}-{args.arm}'
    root.mkdir(mode=0o700)
    plan = json.loads((pilot.ROOT/'observation_research_plan_20260917.json').read_bytes())
    budget = {'authorized_total_usd': 5000, 'reserved_for_slot_usd': 3,
              'whole_batch_reserved_usd': 24, 'cumulative_reserved_usd': 371,
              'bound': plan['budget_bound'], 'scope': plan['scope']}
    assert budget['cumulative_reserved_usd'] <= budget['authorized_total_usd']
    task_index = plan['task_index_by_repetition'][args.repetition]
    task_plan = plan['tasks'][task_index]
    prompt_sources = {name: hashlib.sha256((pilot.PRO/'src/orze_pro/agents'/name).read_bytes()).hexdigest()
                      for name in plan['source_sha256']}
    assert prompt_sources == plan['source_sha256']
    pilot.write(root/'authorization-and-budget.json', budget)
    spec = json.loads((pilot.ROOT/'real-memory-pilot-20260916-admission/haiku45-specification.json').read_bytes())
    runtime = campaign.describe()
    inputs = {**spec['shared'], **spec['tasks']['memory-target'], 'treatment': spec['arms']['B']['treatment']}
    inputs['data'] = task_plan['data']
    inputs['environment'] = runtime['environment']
    inputs['tools'] = {**inputs['tools'], 'rounds': plan['cycles']}
    inputs['treatment']['research_evidence'].update(page_size=32, proposal_page_size=32)
    inputs['initial_history'] = task_plan['history']
    for task in spec['protocol']['tasks']:
        if task['id'] == 'memory-target':
            task['limits']['provider_tokens'] = plan['shared_process_token_envelope']
            task['limits']['reserved_seconds'] = 120
    inputs['instructions'] = pilot.instructions(task_plan['data'])
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
        'predeclared_plan_sha256': hashlib.sha256((pilot.ROOT/'observation_research_plan_20260917.json').read_bytes()).hexdigest(),
        'protocol_scope': 'Inherited task metadata used for configuration; this direct diagnostic is not a formal CLI protocol audit.'})
    run = {'root': str(root), 'cfg': cfg, 'calls': [], 'admissions': [], 'snapshots': [],
           'campaign': {'consumption': [], 'research': [], 'retained_duplicates': []}}
    outcomes = []
    traces = []
    error = None
    try:
        _native(run, 'initialize')
        _initialize_memory(run, request)
        for history in task_plan['history']:
            label = history['id']
            candidate = {'instance_id': inputs['data']['instance_id'], 'schedule': history['schedule']}
            task = 'idea-seed-'+label
            _admit(run, task, task, scheduling.make_request('produce', artifact_utf8=json.dumps(candidate)))
            _native(run, task)
            _finish(run)
            artifact = next(a for a in run['artifacts'] if a['producer']['task_id']==task and a['logical_name']=='candidate')
            evaluator = 'idea-evaluate-seed-'+label
            _admit(run, evaluator, evaluator, scheduling.make_request('evaluate', protocol='schedule-feasibility-v1',source_id=artifact['artifact_id']))
            _native(run, evaluator)
            verdict = _consume_evaluation(run, request, evaluator)
            assert verdict == history['expected_verdict']
        cfg['_config_path'] = str(config)
        cfg['_research_config_sha256'] = hashlib.sha256(config.read_bytes()).hexdigest()
        from orze.core.research_interfaces import register_domain
        register_domain('schedule_holdout', 'acceptance.schedule.v1', scheduling.SchedulingDomain)
        from orze_pro.agents.memory_sources import capture_report_bindings
        bindings = capture_report_bindings(root/'results', cfg,
            ['idea-evaluate-seed-'+h['id'] for h in task_plan['history']], rules_sha256=hashlib.sha256(rules.read_bytes()).hexdigest())
        pilot.write(root/'seed-source-bindings.json', bindings)
        assert bindings['availability'] == 'available'
        assert all(v['availability'] == 'qualified' for v in bindings['bindings'].values()), bindings
        from orze_pro.agents import research
        from orze_pro.agents import observation_prompt
        if args.arm == 'A':
            from copy import deepcopy
            observation_prompt.native_record = deepcopy
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
        os.environ['ORZE_LLM_TOKEN_ENVELOPE'] = str(plan['shared_process_token_envelope'])
        os.environ['ORZE_ANTHROPIC_MAX_TOKENS'] = str(plan['output_tokens'])
        os.environ['ORZE_ANTHROPIC_EFFORT'] = ''
        run['seed_setup_seconds'] = time.monotonic()-started
        for cycle in range(1, plan['cycles']+1):
            outcome = {}
            research.run_research_cycle('anthropic', cycle, root/'ideas.md', root/'results', cfg['report'],
                num_ideas=2, model='claude-haiku-4-5-20251001', endpoint='https://api.anthropic.com/v1',
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
            'arm':args.arm, 'repetition':args.repetition, 'seed_score':task_plan['seed_score'],
            'task_index':task_index,'quality_target':task_plan['quality_target']})
    print(json.dumps({'output':str(root),'outcomes':[x.get('reason') for x in outcomes],
                      'responses':len(traces),'memory_revision':_database(root)['research_memory_current_v1'][0]['revision']}))


if __name__ == '__main__':
    main()
