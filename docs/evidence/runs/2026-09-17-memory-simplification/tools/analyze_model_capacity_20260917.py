"""Compare model configurations with all outcomes, including truncation and timeout."""
import json
from pathlib import Path
from collections import Counter
import analyze_optional_workflow_20260917 as observation

ROOT=Path(__file__).resolve().parent
BATCH=ROOT/'model-capacity-20260917'
observation.BATCH=BATCH
read=observation.read

def main():
    plan=read(BATCH/'plan.json')
    processes=[json.loads(line) for line in (BATCH/'runs.jsonl').read_text().splitlines()]
    assert [(r['repetition'],r['arm']) for r in processes]==[tuple(v) for v in plan['order']]
    rows=[observation.summarize(BATCH/f"pair-{p['repetition']}-{p['arm']}",p) for p in processes]
    prior=read(ROOT/'optional-memory-workflow-20260917/summary.json')
    pairs=[]
    for sonnet in rows:
        haiku=next(r for r in prior['runs'] if r['repetition']==sonnet['repetition'] and r['arm']=='B')
        new_plan=read(BATCH/f"pair-{sonnet['repetition']}-B"/'plan.json')
        old_plan=read(ROOT/'optional-memory-workflow-20260917'/f"pair-{sonnet['repetition']}-B"/'plan.json')
        for name in ('data','tools','instructions','initial_history','initial_memory','treatment'):
            assert new_plan['request']['inputs'][name]==old_plan['request']['inputs'][name],name
        assert new_plan['effective_prompt_source_sha256']==old_plan['effective_prompt_source_sha256']
        fields=('new_best_score','best_including_seed','new_valid','new_invalid','process_wall_seconds','first_improvement_seconds','usage','memory_revision','outcomes')
        pairs.append({'task':sonnet['repetition'],'haiku_8192':{k:haiku[k] for k in fields},'sonnet_32768':{k:sonnet[k] for k in fields}})
    total={k:prior['cumulative_usage'][k]+sum(r['usage'][k] for r in rows) for k in prior['cumulative_usage']}
    result={'schema':1,'scope':plan['scope'],'runs':rows,'comparisons':pairs,
        'cumulative_usage':total,'cumulative_reserve_usd':339,'authorized_cap_usd':5000,
        'model_configuration':{'model':'claude-sonnet-5','requested_max_tokens':32768,'provider_http_timeout_seconds':120,'effort':'provider default high'},
        'limitations':plan['limitations'],
        'official_references':['https://platform.claude.com/docs/en/models/sonnet-5/migration-guide','https://platform.claude.com/docs/en/about-claude/pricing'],
        'money_scope':'Returned usage at public prices, not invoices; missing response usage remains unknown.'}
    with (BATCH/'summary.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,sort_keys=True)
    print(json.dumps({'comparisons':pairs,'cumulative_usage':total},indent=2))

if __name__=='__main__':main()
