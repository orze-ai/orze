"""Close the explicit model-setting experiment against retained controls and bounds."""
import json,hashlib
from pathlib import Path
import analyze_optional_workflow_20260917 as observation
ROOT=Path(__file__).resolve().parent
BATCH=ROOT/'model-effort-20260917'
observation.BATCH=BATCH
read=observation.read

def main():
    plan=read(BATCH/'plan.json')
    processes=[json.loads(line) for line in (BATCH/'runs.jsonl').read_text().splitlines()]
    assert [(p['repetition'],p['arm']) for p in processes]==[tuple(v) for v in plan['order']]
    rows=[observation.summarize(BATCH/f"pair-{p['repetition']}-{p['arm']}",p) for p in processes]
    high=read(ROOT/'model-capacity-20260917/summary.json')
    haiku=read(ROOT/'optional-memory-workflow-20260917/summary.json')
    bounds=read(ROOT/'model-capacity-20260917/upper-bound-audit.json')['tasks']
    comparisons=[]
    for medium in rows:
        rep=medium['repetition'];root=BATCH/f'pair-{rep}-B'
        parameters=[json.loads(line) for line in (root/'request-parameters.jsonl').read_text().splitlines()]
        assert len(parameters)==medium['usage']['prepared_attempts']
        prompts={hashlib.sha256(read(root/Path(p['path']).name)['prompt'].encode()).hexdigest() for p in medium['prompts']}
        for p in parameters:
            assert p['model']=='claude-sonnet-5' and p['output_config']=={'effort':'medium'}
            assert 0<p['max_tokens']<=32768 and p['http_timeout_seconds']==120 and p['prompt_sha256'] in prompts
        medium['actual_request_parameters']=parameters
        high_row=next(r for r in high['runs'] if r['repetition']==rep)
        haiku_row=next(r for r in haiku['runs'] if r['repetition']==rep and r['arm']=='B')
        current_plan=read(root/'plan.json')
        high_plan=read(ROOT/'model-capacity-20260917'/f'pair-{rep}-B'/'plan.json')
        for field in ('data','model','tools','instructions','initial_history','initial_memory','treatment'):
            assert current_plan['request']['inputs'][field]==high_plan['request']['inputs'][field],field
        data=current_plan['request']['inputs']['data']
        bound=next(b for b in bounds if b['instance_id']==data['instance_id'])
        assert hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()==bound['data_sha256']
        treatments={}
        for name,r in [('haiku_8192',haiku_row),('sonnet_high_32768',high_row),('sonnet_medium_32768',medium)]:
            assert r['best_including_seed']<=bound['score_upper_bound']
            optimal=[e for e in r['evaluations'] if not e['seed'] and e['verdict'].get('scheduled_value')==bound['score_upper_bound']]
            fields=('new_best_score','best_including_seed','new_valid','new_invalid','process_wall_seconds','first_improvement_seconds','usage','memory_revision','outcomes')
            treatments[name]={**{k:r[k] for k in fields},'attains_proven_upper_bound':bool(optimal),
                'first_optimal_seconds':min((e['observed_seconds'] for e in optimal),default=None)}
        comparisons.append({'task':rep,'instance_id':data['instance_id'],'upper_bound':bound,**treatments})
    total={k:high['cumulative_usage'][k]+sum(r['usage'][k] for r in rows) for k in high['cumulative_usage']}
    result={'schema':1,'scope':plan['scope'],'runs':rows,'comparisons':comparisons,'cumulative_usage':total,
        'cumulative_reserve_usd':347,'authorized_cap_usd':5000,'limitations':plan['limitations'],
        'money_scope':'Returned usage at public prices, not invoices; missing response usage retained as unknown. Full fixed-round costs include failures after an optimal solution.',
        'bound_scope':'An auditor-only capacity relaxation enumerates every optional subset. Equality with a separately verified feasible candidate proves these finite task optima; bounds were never in provider prompts.'}
    with (BATCH/'summary.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,sort_keys=True)
    compact=[{'task':p['task'],'bound':p['upper_bound']['score_upper_bound'],**{k:{n:v[n] for n in ('new_best_score','process_wall_seconds','first_optimal_seconds','usage','attains_proven_upper_bound')} for k,v in p.items() if k.startswith(('haiku_','sonnet_'))}} for p in comparisons]
    print(json.dumps({'comparisons':compact,'cumulative_usage':total},indent=2))

if __name__=='__main__':main()
