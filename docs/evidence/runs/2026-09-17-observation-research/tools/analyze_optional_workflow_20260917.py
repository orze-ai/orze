"""Read every fixed slot; recompute every seed and new candidate verdict."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import analyze_real_memory_v2 as prior
from examples.holdout import scheduling

ROOT=Path(__file__).resolve().parent
BATCH=ROOT/'optional-memory-workflow-20260917'

def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def classify(response):
    try:
        lines=response.strip().splitlines()
        if len(lines)>=3 and lines[0] in ('```json','```') and lines[-1]=='```':response='\n'.join(lines[1:-1])
        value=json.loads(response)
        if isinstance(value,dict) and 'update_memory' in value:
            return 'update_memory'
        if isinstance(value,dict) and any(k in value for k in ('read_evidence','select_evidence','read_proposals','select_proposals')):
            return 'read'
        return 'proposal_or_other'
    except (ValueError,TypeError,AttributeError):return 'unparsed'

def summarize(root, process):
    result=read(root/'result.json'); capture=read(root/'capture.json'); plan=read(root/'plan.json')
    instance=plan['request']['inputs']['data']
    evaluations=[]; seen_invalid=set(); repeated_invalid=0
    for observation in result['evaluations']:
        e=observation['evaluation']; raw=capture['artifact_contents'][e['candidate_artifact_id']]
        verdict=scheduling.evaluate(instance,raw.encode(),e['protocol'])
        assert verdict==e['verdict'], 'independent verdict mismatch'
        seed=e['ref']['task_id'].startswith('idea-evaluate-seed-')
        schedule=json.loads(raw)['schedule']
        canonical=json.dumps(sorted((s['job_id'],s['start']) for s in schedule))
        if verdict['status']=='invalid':
            repeated_invalid+=int(not seed and canonical in seen_invalid)
            seen_invalid.add(canonical)
        evaluations.append({'seed':seed,'task_id':e['ref']['task_id'],'candidate_sha256':e['candidate_sha256'],
            'verdict':verdict,'observed_seconds':observation.get('observed_seconds'),'schedule':schedule})
    new=[e for e in evaluations if not e['seed']]
    valid=[e for e in new if e['verdict']['status']=='valid']
    improved=[e for e in valid if e['verdict']['scheduled_value']>result['seed_score']]
    prompts=[]
    for path in sorted(root.glob('response-*.json'),key=lambda p:int(p.stem.split('-')[-1])):
        r=read(path)
        memory=json.loads(re.search(r'<stored_research_memory>\s*(.*?)\s*</stored_research_memory>',r['prompt'],re.S).group(1))
        view=json.loads(re.search(r'<research_snapshot>\s*(.*?)\s*</research_snapshot>',r['prompt'],re.S).group(1))
        prompts.append({'path':str(path.relative_to(BATCH)),'sha256':sha(path),'seconds':r['seconds'],
            'prompt_bytes':len(r['prompt'].encode()),'memory_revision':memory['revision'],
            'updates':memory['updates'],'memory_entries':memory['entries'],
            'write_template_offered':'Memory update protocol v2:' in r['prompt'],
            'ending_offers_memory': ('- Memory update:' in r['prompt'].split('## Your Task',1)[1] or 'memory-update request' in r['prompt'].split('## Your Task',1)[1]),
            'response_kind':classify(r['response']),
            'feedback': [{'idea_id':v['idea_id'],'availability':v['availability'],'observations':v.get('observations',[])}
                         for v in view['report_evidence']['records']]})
    memory=result['memory_rows']
    entries=json.loads(memory[0]['document_json'])['entries'] if len(memory)==1 else None
    return {'arm':result['arm'],'repetition':result['repetition'],'error':result['error'],**process,
        'elapsed_seconds':result['elapsed_seconds'],'seed_setup_seconds':capture['seed_setup_seconds'],
        'provider_seconds':sum(p['seconds'] for p in prompts),'prompt_bytes':sum(p['prompt_bytes'] for p in prompts),
        'outcomes':result['outcomes'],'usage':prior.usage(root/'usage.jsonl'),
        'new_evaluated':len(new),'new_valid':len(valid),'new_invalid':len(new)-len(valid),
        'improved':bool(improved),'new_best_score':max((e['verdict']['scheduled_value'] for e in valid),default=None),
        'best_including_seed':max([result['seed_score']]+[e['verdict']['scheduled_value'] for e in valid]),
        'first_improvement_seconds':min((e['observed_seconds'] for e in improved),default=None),
        'seed_score':result['seed_score'],
        'repeated_invalid_schedules':repeated_invalid,
        'retained_duplicate_proposals':capture['campaign']['retained_duplicates'],
        'memory_revision':memory[0]['revision'] if len(memory)==1 else None,
        'final_entries':entries,'prompts':prompts,'evaluations':evaluations}

def main():
    plan=read(BATCH/'plan.json')
    processes=[json.loads(line) for line in (BATCH/'runs.jsonl').read_text().splitlines()]
    assert [(p['repetition'],p['arm']) for p in processes]==[tuple(p) for p in plan['order']]
    rows=[summarize(BATCH/f"pair-{p['repetition']}-{p['arm']}",p) for p in processes]
    plans=[read(BATCH/f"pair-{p['repetition']}-{p['arm']}"/'plan.json') for p in processes]
    for repetition in (0,1):
        selected=[p for p in plans if p['request']['slot']['repetition']==repetition]
        for field in ('data','model','tools','initial_history','initial_memory','treatment'):
            assert len({json.dumps(p['request']['inputs'][field],sort_keys=True) for p in selected})==1,field
    assert len({p['request']['runtime']['artifact_sha256'] for p in plans})==1
    assert all(p['effective_prompt_source_sha256']==plan['code_sha256'] for p in plans)
    assert all(p['predeclared_plan_sha256']==sha(ROOT/'optional_memory_workflow_plan_20260917.json') for p in plans)
    for row in rows:
        assert len([e for e in row['evaluations'] if e['seed']])==2
        for p in row['prompts']:
            assert p['write_template_offered']==(p['updates']['used']<p['updates']['maximum'])
            assert p['ending_offers_memory']==(p['updates']['used']<p['updates']['maximum'])
    arms=[]
    for arm in ('A','B'):
        group=[r for r in rows if r['arm']==arm]
        counters=Counter(o['reason'] for r in group for o in r['outcomes'])
        arms.append({'arm':arm,'planned':2,'completed':len(group),'projects_with_improvement':sum(r['improved'] for r in group),
            'new_valid_candidates':sum(r['new_valid'] for r in group),'new_invalid_candidates':sum(r['new_invalid'] for r in group),
            'repeated_invalid_schedules':sum(r['repeated_invalid_schedules'] for r in group),
            'retained_duplicate_proposals':sum(len(r['retained_duplicate_proposals']) for r in group),
            'memory_entries':sum(len(r['final_entries']) for r in group),
            'memory_update_requests':sum(p['response_kind']=='update_memory' for r in group for p in r['prompts']),
            'exhausted_update_requests':sum(p['response_kind']=='update_memory' and p['updates']['used']>=p['updates']['maximum'] for r in group for p in r['prompts']),
            'outcomes':dict(counters),
            'process_wall_seconds':sum(r['process_wall_seconds'] for r in group),
            'provider_seconds':sum(r['provider_seconds'] for r in group),
            'prompt_bytes':sum(r['prompt_bytes'] for r in group),
            'usage':{k:sum(r['usage'][k] for r in group) for k in group[0]['usage']}})
    pairs=[]
    for rep in range(2):
        a,b=[next(r for r in rows if r['arm']==arm and r['repetition']==rep) for arm in ('A','B')]
        equal=a['improved'] and b['improved'] and a['new_best_score']==b['new_best_score']
        pairs.append({'repetition':rep,'A':{k:a[k] for k in ('new_best_score','process_wall_seconds','first_improvement_seconds','seed_score','usage')},
            'B':{k:b[k] for k in ('new_best_score','process_wall_seconds','first_improvement_seconds','seed_score','usage')},
            'equal_improved_quality':equal,'equal_quality_B_minus_A':{
                'process_wall_seconds':b['process_wall_seconds']-a['process_wall_seconds'],
                'usage_usd':b['usage']['priced_usage_usd_upper']-a['usage']['priced_usage_usd_upper']} if equal else None})
    previous=read(ROOT/'memory-next-step-pairs-20260917/summary.json')['cumulative_usage']
    total={k:previous[k]+sum(r['usage'][k] for r in rows) for k in previous}
    result={'schema':1,'scope':plan['scope'],'arms':arms,'pairs':pairs,'runs':rows,
        'cumulative_usage':total,'cumulative_reserve_usd':331,'authorized_cap_usd':5000,
        'limitations':['Two fresh authored nine-job tasks, one pair each, remote sampling uncontrolled; exploration rather than broad validation.',
            'Both arms have optional memory available. A additionally requires one counterexample note before proposals; B does not. This compares workflow rules, not code versions or memory versus no-memory.',
            'Direct research function diagnostics with actual native evaluation, not formal CLI/production acceptance.',
            'First-improvement time is independent collector consumption, not proof the model has read that result.',
            'Usage price estimates are not invoices; historical unknown requests retained; full infrastructure cost unknown.']}
    with (BATCH/'summary.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,sort_keys=True)
    print(json.dumps({'arms':arms,'pairs':pairs,'cumulative_usage':total},indent=2))

if __name__=='__main__':main()
