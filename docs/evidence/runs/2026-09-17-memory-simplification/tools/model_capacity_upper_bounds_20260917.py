"""Independent relaxed-capacity upper bounds; never supplied to model prompts."""
from pathlib import Path
import itertools,json,hashlib
ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'model_capacity_plan_20260917.json').read_bytes())
rows=[]
for task in plan['tasks']:
    data=task['data'];jobs={j['id']:j for j in data['jobs']}
    assert set(j['id'] for j in jobs.values() if j['required'])=={'prepare','gate','finish'}
    assert jobs['prepare']['after']==[] and jobs['gate']['after']==['prepare'] and jobs['finish']['after']==['gate']
    optional=[j for j in jobs.values() if not j['required']]
    assert all(j['after']==['gate'] for j in optional)
    earliest_prepare=jobs['prepare']['release']
    earliest_gate=max(jobs['gate']['release'],earliest_prepare+jobs['prepare']['duration'])
    earliest_optional=earliest_gate+jobs['gate']['duration']
    # Every optional job and the required finish must run after the gate finishes.
    # This ignores deadlines, placement and simultaneous-demand constraints, hence
    # overestimates attainable optional value and supplies a valid upper bound.
    capacity=data['capacity']*(data['horizon']-earliest_optional)-jobs['finish']['duration']*jobs['finish']['demand']
    best=-1;subsets=[]
    for flags in itertools.product((False,True),repeat=len(optional)):
        selected=[j for j,yes in zip(optional,flags) if yes]
        weight=sum(j['duration']*j['demand'] for j in selected)
        if weight>capacity:continue
        value=sum(j['value'] for j in selected)
        if value>best:best=value;subsets=[]
        if value==best:subsets.append([j['id'] for j in selected])
    required=sum(j['value'] for j in jobs.values() if j['required'])
    rows.append({'instance_id':data['instance_id'],'earliest_optional_start':earliest_optional,
        'optional_capacity_upper_bound_unit_ticks':capacity,'relaxed_best_optional_value':best,
        'required_value':required,'score_upper_bound':best+required,'relaxed_maximizing_subsets':subsets,
        'all_subsets_enumerated':2**len(optional),
        'proof':'All optional work follows gate; the mandatory finish also consumes capacity after gate. Enumerating every optional subset below this relaxed work bound can only overestimate feasible value. A separately verified feasible schedule attaining this bound proves optimality.',
        'data_sha256':hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()})
value={'scope':'Auditor-only deterministic upper bounds computed from task declarations; not exposed to providers and not a generated solution.',
       'tasks':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (ROOT/'model-capacity-20260917/upper-bound-audit.json').open('x') as f:json.dump(value,f,indent=2,sort_keys=True)
print(json.dumps(value,indent=2))
