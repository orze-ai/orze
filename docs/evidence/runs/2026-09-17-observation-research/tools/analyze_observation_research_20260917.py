"""Audit all fixed slots from original candidate bytes and provider journals."""
from pathlib import Path
from collections import Counter
import hashlib,json,re
import analyze_optional_workflow_20260917 as prior
ROOT=Path(__file__).resolve().parent
BATCH=ROOT/'observation-research-20260917'
prior.BATCH=BATCH
read=prior.read

def prompt_audit(root):
    rows=[]
    for path in sorted(root.glob('response-*.json'),key=lambda p:int(p.stem.split('-')[-1])):
        response=read(path)
        snapshot=json.loads(re.search(r'<research_snapshot>\s*(.*?)\s*</research_snapshot>',response['prompt'],re.S).group(1))
        reports=snapshot['report_evidence']['records'];proposals=snapshot['proposals']['records']
        qualified=[r for r in reports if r['availability']=='qualified']
        observations=[o for r in qualified for o in r.get('observations',[])]
        rows.append(dict(path=path.name,prompt_bytes=len(response['prompt'].encode()),
            snapshot_bytes=len(json.dumps(snapshot,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()),
            qualified_reports=len(qualified),qualified_report_ids=[r['idea_id'] for r in qualified],
            compact_reports=sum(r.get('presentation')=='native_observation_summary_v1' for r in qualified),
            valid_observations=sum(o['validation']['status']=='valid' for o in observations),
            invalid_observations=sum(o['validation']['status']=='invalid' for o in observations),
            visible_valid_max=max((o['values'].get('value',0) for o in observations if o['validation']['status']=='valid'),default=None),
            unavailable_report_reasons=dict(Counter(r['reason'] for r in reports if r['availability']=='unavailable')),
            visible_proposal_definitions=sum(r['availability']=='metadata' for r in proposals)))
        assert rows[-1]['snapshot_bytes']<=32768
        # Check every displayed result against the actual native observation.
        captured=read(root/'capture.json');by_id={o['observation_id']:o for o in captured['observations']}
        for row in qualified:
            for observation in row.get('observations',[]):
                original=by_id[observation['observation_id']]
                for key in ('values','validation','protocol_fingerprint','comparison_scope','result_artifact_ids'):
                    assert observation[key]==original[key],(path.name,key)
                if 'evaluated_inputs' in observation:
                    assert observation['evaluated_inputs']==[{'artifact_id':key,'source_idea_id':original['input_artifact_bindings'][key]['producer']['task_id']} for key in original['input_artifact_ids']]
    return rows

def main():
    plan=read(BATCH/'plan.json')
    processes=[json.loads(line) for line in (BATCH/'runs.jsonl').read_text().splitlines()]
    assert [(p['repetition'],p['arm']) for p in processes]==[tuple(p) for p in plan['order']]
    rows=[]
    for process in processes:
        root=BATCH/f"pair-{process['repetition']}-{process['arm']}"
        row=prior.summarize(root,process)
        task=plan['tasks'][plan['task_index_by_repetition'][row['repetition']]]
        frozen=read(root/'plan.json')
        assert frozen['effective_prompt_source_sha256']==plan['source_sha256']
        assert frozen['predeclared_plan_sha256']==hashlib.sha256((ROOT/'observation_research_plan_20260917.json').read_bytes()).hexdigest()
        assert len([e for e in row['evaluations'] if e['seed']])==12
        assert row['seed_score']==task['seed_score']<task['quality_target']
        target=[e for e in row['evaluations'] if not e['seed'] and e['verdict'].get('status')=='valid' and e['verdict'].get('scheduled_value',0)>=task['quality_target']]
        row.update(task_index=plan['task_index_by_repetition'][row['repetition']],quality_target=task['quality_target'],
            target_reached=bool(target),first_target_seconds=min((e['observed_seconds'] for e in target),default=None),
            prompt_audit=prompt_audit(root))
        assert all(p['compact_reports']==(p['qualified_reports'] if row['arm']=='B' else 0) for p in row['prompt_audit'])
        rows.append(row)
    pairs=[]
    for repetition in range(4):
        a,b=[next(r for r in rows if r['repetition']==repetition and r['arm']==arm) for arm in ('A','B')]
        inputs=[read(BATCH/f'pair-{repetition}-{arm}'/'plan.json')['request']['inputs'] for arm in ('A','B')]
        assert inputs[0]==inputs[1]
        fields=('new_best_score','best_including_seed','target_reached','first_target_seconds','process_wall_seconds','seed_setup_seconds','first_improvement_seconds','usage','new_valid','new_invalid','repeated_invalid_schedules')
        pairs.append(dict(repetition=repetition,task_index=a['task_index'],A={k:a[k] for k in fields},B={k:b[k] for k in fields}))
    arms=[]
    for arm in ('A','B'):
        selected=[r for r in rows if r['arm']==arm]
        arms.append(dict(arm=arm,projects=len(selected),improved=sum(r['improved'] for r in selected),target_reached=sum(r['target_reached'] for r in selected),
            new_valid=sum(r['new_valid'] for r in selected),new_invalid=sum(r['new_invalid'] for r in selected),
            repeated_invalid=sum(r['repeated_invalid_schedules'] for r in selected),retained_duplicates=sum(len(r['retained_duplicate_proposals']) for r in selected),
            full_wall_seconds=sum(r['process_wall_seconds'] for r in selected),seed_setup_seconds=sum(r['seed_setup_seconds'] for r in selected),
            provider_seconds=sum(r['provider_seconds'] for r in selected),prompt_bytes=sum(r['prompt_bytes'] for r in selected),
            initial_prompt_bytes=[r['prompt_audit'][0]['prompt_bytes'] for r in selected],initial_qualified_reports=[r['prompt_audit'][0]['qualified_reports'] for r in selected],
            outcomes=dict(Counter(o['reason'] for r in selected for o in r['outcomes'])),memory_entries=sum(len(r['final_entries']) for r in selected),
            usage={k:sum(r['usage'][k] for r in selected) for k in selected[0]['usage']}))
    previous=read(ROOT/'model-effort-20260917/summary.json')['cumulative_usage']
    cumulative={k:previous[k]+sum(r['usage'][k] for r in rows) for k in previous}
    result=dict(schema=1,scope=plan['scope'],arms=arms,pairs=pairs,runs=rows,cumulative_usage=cumulative,cumulative_reserve_usd=371,authorized_cap_usd=5000,limitations=plan['limitations'],
        cost_scope='Returned usage at published prices, not invoices. Unknown costs remain unknown. Process wall includes native historical setup and all failures. First-target times start at runner initialization and denote collector consumption, not model comprehension.')
    with (BATCH/'summary.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
    print(json.dumps({k:result[k] for k in ('arms','pairs','cumulative_usage')},indent=2))
if __name__=='__main__':main()
