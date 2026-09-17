"""Freeze fresh task inputs before any provider calls; no solution is supplied."""
from pathlib import Path
import hashlib,json,datetime,sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'orze'),str(ROOT/'orze/src')]
from examples.holdout import scheduling

def job(name,release,duration,deadline,demand,value,after=('gate',),required=False):
    return dict(id=name,release=release,duration=duration,deadline=deadline,demand=demand,value=value,after=list(after),required=required)

tasks=[]
for number,capacity,horizon in [(0,2,18),(1,3,20)]:
    optional=( [('a',3,4,9,2,14),('b',3,3,8,1,9),('c',4,4,11,1,12),('d',7,3,12,2,13),('e',8,2,14,1,8),('f',10,4,17,1,15),('g',12,3,17,1,11),('h',5,2,10,1,6),('i',14,2,18,2,12),('j',3,6,15,1,14)] if number==0 else
               [('a',3,5,10,3,19),('b',3,4,10,2,15),('c',4,4,12,1,11),('d',8,4,16,2,17),('e',7,3,14,1,12),('f',11,5,19,1,16),('g',14,4,19,2,20),('h',5,2,11,1,7),('i',15,3,19,3,18),('j',3,6,17,1,18)] )
    jobs=[job('prepare',0,2,2,capacity,1,(),True),job('gate',2,1,4,1,1,('prepare',),True),job('finish',horizon-1,1,horizon,capacity,1,('gate',),True)]
    jobs += [job(*p,after=('gate','h') if p[0]=='j' else ('gate',)) for p in optional]
    data=dict(instance_id=f'observation-history-thirteen-{number}-v1',horizon=horizon,capacity=capacity,jobs=jobs)
    scheduling.validate_instance(data)
    base=[('prepare',0),('gate',2),('finish',horizon-1)]
    extras=[[],[('a',3)],[('b',3)],[('c',4)],[('h',5)],[('h',5),('j',7)],
            [('a',3),('b',3)],[('c',3)],[('j',7)],[('h',6),('j',6)],[('f',horizon-2)],[('finish',horizon-1)]]
    history=[]
    for i,extra in enumerate(extras):
        schedule=[dict(job_id=name,start=tick) for name,tick in base+extra]
        raw=json.dumps(dict(instance_id=data['instance_id'],schedule=schedule)).encode()
        verdict=scheduling.evaluate(data,raw,'schedule-feasibility-v1')
        assert verdict['status']==('valid' if i<6 else 'invalid')
        history.append(dict(id=f'h{i:02}',schedule=schedule,expected_verdict=verdict))
    tasks.append(dict(data=data,history=history,seed_score=max(h['expected_verdict'].get('scheduled_value',0) for h in history),quality_target=45 if number==0 else 55))
files=['observation_prompt.py','report_paging.py','joint_snapshot.py','research.py','research_llm.py']
plan=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    question='Does a compact, source-linked view of exactly the same qualified native observations improve subsequent valid research decisions under an equal model and budget?',
    scope='Exploratory real research consumer with native CPU candidate production and evaluation. Two fresh 13-job tasks, 12 real historical evaluations each; two paired repetitions each. Not cross-domain or production acceptance.',
    arms={'A':'Full native audit metadata, same code and instructions; native_record is an identity copy.', 'B':'Compact native observation view; original host qualification and memory sources unchanged.'},
    model='claude-haiku-4-5-20251001',cycles=3,candidates_per_cycle=2,max_requests_per_cycle=3,max_memory_updates_per_cycle=1,
    shared_process_token_envelope=300000,output_tokens=8192,http_timeout_seconds=120,
    order=[[0,'A'],[0,'B'],[1,'B'],[1,'A'],[2,'B'],[2,'A'],[3,'A'],[3,'B']],
    tasks=tasks,task_index_by_repetition=[0,1,0,1],
    primary_metrics=['independently verified quality target attainment','best new valid score versus best initial history','new valid and invalid candidates','repeated invalid schedules and duplicate proposals','first target attainment time','full elapsed time and returned API usage'],
    prior_cumulative_reserve_usd=347,additional_reserve_usd=24,cumulative_reserve_usd=371,authorized_cap_usd=5000,
    budget_bound='8 processes x 300000 shared token envelope x $5/M maximum token rate = $12; reserve $24. All retries and memory reads/writes share the same process envelope. No fallback or extra samples.',
    source_sha256={name:hashlib.sha256((ROOT/'pro/src/orze_pro/agents'/name).read_bytes()).hexdigest() for name in files},
    limitations=['Authored finite tasks in one domain; no broad scientific capability claim.','Provider sampling is uncontrolled; all fixed slots, failures and unknown costs retained.','Historical candidates are actually run in each project, never injected as fake successful reports.','Both views offer the same pages and read APIs; visible coverage can differ within the same 32768-byte cap.','Quality thresholds are fixed targets, not claimed mathematical optima.','No extending the sample count based on favorable or unfavorable outcomes.'])
p=ROOT/'observation_research_plan_20260917.json'
with p.open('x') as f:json.dump(plan,f,ensure_ascii=False,indent=2,sort_keys=True)
print(json.dumps({'plan':str(p),'initial_best':[t['seed_score'] for t in tasks],'history_verdicts':[[h['expected_verdict']['reason_code'] for h in t['history']] for t in tasks],'reserve':371}))
