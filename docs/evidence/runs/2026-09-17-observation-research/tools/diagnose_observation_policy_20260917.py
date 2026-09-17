"""Read-only reconstruction of the rejected CPU policy boundary."""
from pathlib import Path
import json,sys,traceback,yaml,time
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'orze'),str(ROOT/'orze/src')]
from orze.idea_lake import IdeaLake
from orze.core import research_interfaces as api
from orze.core.cpu_execution import cpu_execution
from orze.core.cpu_budget_observation import observe_cpu_budget
from orze.engine.cpu_phase import QueuePolicy
from orze.engine.cpu_policy_evidence import recorded_evidence
from orze.engine.cpu_proposals import recorded_proposals
from examples.holdout.scheduling import SchedulingDomain
api.register_domain('schedule_holdout','acceptance.schedule.v1',SchedulingDomain)
root=ROOT/'observation-research-20260917/pair-0-A'
cfg=yaml.safe_load((root/'orze.yaml').read_text())
lake=IdeaLake(str(root/'lake.db'))
try:
    queue=[]
    for item in lake.get_queue(limit=32):
        r=lake.get(item['idea_id']);request=yaml.safe_load(r['config'])['domain_request']
        queue.append({'idea_id':item['idea_id'],'action':{'timeout_seconds':request['timeout_seconds']},'request':request})
    snapshot={'queue':queue,'active':False,'now':time.time(),'recorded_evidence':recorded_evidence(lake,root/'results'),'recorded_proposals':recorded_proposals(lake,root/'results')}
    budget=observe_cpu_budget(lake,root/'results',cpu_execution(cfg))['accounting']
    ctx=api.capture_interfaces(cfg);policy=api.BoundPolicy(ctx)
    value={'snapshot_bytes':len(json.dumps(snapshot).encode()),'budget_bytes':len(json.dumps(budget).encode()),'queue_ids':[r['idea_id'] for r in queue]}
    try:value['decision']=policy.decide(snapshot,budget)
    except Exception as e:value['error']=traceback.format_exc()
    with (ROOT/'observation-research-20260917/policy-diagnosis.json').open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(value,indent=2))
finally:lake.close()
