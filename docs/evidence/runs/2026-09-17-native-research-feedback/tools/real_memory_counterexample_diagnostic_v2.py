"""Two real research cycles after independently evaluated common seed examples.

This prompted mechanism diagnostic is not an A/B research-benefit experiment.
It observes the real provider and normal memory implementation without replacing
responses or licensing. Every new file stays under the current work directory.
"""
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
    pilot.credentials()
    root = pilot.ROOT / 'real-memory-counterexample-diagnostic-20260916-v2'
    root.mkdir(mode=0o700)
    budget = {'authorized_total_usd': 5000, 'prior_reserved_upper_bound_usd': 247,
              'additional_upper_bound_usd': 2, 'total_reserved_upper_bound_usd': 249,
              'bound': 'Two research cycles, each conservatively 100000 tokens at $5/M; reserve $2. No fallback.',
              'scope': 'Prompted real-model memory mechanism diagnostic, not randomized research efficacy.'}
    assert budget['total_reserved_upper_bound_usd'] < budget['authorized_total_usd']
    pilot.write(root/'authorization-and-budget.json', budget)
    spec = json.loads((pilot.ROOT/'real-memory-pilot-20260916-admission/haiku45-specification.json').read_bytes())
    runtime = campaign.describe()
    inputs = {**spec['shared'], **spec['tasks']['memory-target'], 'treatment': spec['arms']['B']['treatment']}
    inputs['environment'] = runtime['environment']
    inputs['instructions'] += '''
This is a memory mechanism diagnostic after actual evaluator feedback.
When persistent memory is offered and the visible qualified feedback contains a failed schedule,
first retain ONE concise counterexample note that is not yet stored, citing the exact eligible evaluator
report ID from the evidence. Explain the violated constraint using the actual schedule and verdict.
Use the offered update_memory protocol; do not invent a source or a successful measurement.
If no eligible source exists, do not assert a sourced fact. After the note is stored, propose improvements
using current evidence and any retained note. Do not repeat the same note or manufacture extra notes.
Follow the genealogy contract using exact eligible IDs. Emit bare JSON without surrounding prose.
Keep domain_request.payload.candidate exactly "baseline"; this is a fixed interface field, not a strategy name.
'''
    request = {'inputs': inputs, 'protocol': spec['protocol'], 'runtime': runtime,
               'slot': {'task_id': 'memory-target', 'seed': 17, 'arm': 'B', 'repetition': 0}}
    cfg = _configuration(request, root)
    config = root/'orze.yaml'; config.write_text(yaml.safe_dump(cfg))
    rules = root/'instructions.md'; rules.write_text(inputs['instructions'])
    (root/'ideas.md').write_text('# Ideas\n')
    pilot.write(root/'plan.json', {'request': request, 'budget': budget,
        'heads': {name: subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
                  for name,path in [('core',pilot.CORE),('pro',pilot.PRO)]},
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    run = {'root': str(root), 'cfg': cfg, 'calls': [], 'admissions': [], 'snapshots': [],
           'campaign': {'consumption': [], 'research': [], 'retained_duplicates': []}}
    started = time.monotonic()
    try:
        _native(run, 'initialize')
        _initialize_memory(run, request)
        seeds = {
            'valid': [('prepare',0),('gate',2),('early',3),('middle',5),('late',7),('finish',9)],
            'overload': [('prepare',0),('gate',2),('long',3),('early',4),('finish',9)],
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
            _consume_evaluation(run, request, evaluator)
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
        original = research.call_llm
        traces = []
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
        os.environ['ORZE_LLM_TOKEN_ENVELOPE'] = '100000'
        outcomes = []
        for cycle in (1,2):
            outcome = {}
            research.run_research_cycle('anthropic', cycle, root/'ideas.md', root/'results', cfg['report'],
                num_ideas=2, model='claude-haiku-4-5-20251001', endpoint='https://api.anthropic.com/v1',
                lake_db_path=root/'lake.db', project_cfg=cfg, rules_file=str(rules),
                rules_sha256=hashlib.sha256(rules.read_bytes()).hexdigest(), result_out=outcome)
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
        pilot.write(root/'result.json', {'outcomes':outcomes, 'responses':len(traces),
            'elapsed_seconds':time.monotonic()-started, 'memory_rows':_database(root)['research_memory_current_v1'],
            'evaluations':run['campaign']['consumption'], 'scope':budget['scope']})
    finally:
        _finish(run)
        pilot.write(root/'capture.json',run)
    print(json.dumps({'output':str(root),'outcomes':[x.get('reason') for x in outcomes],
                      'responses':len(traces),'memory_revision':_database(root)['research_memory_current_v1'][0]['revision']}))


if __name__ == '__main__':
    main()
