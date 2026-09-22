"""The prepared entrypoint delivers through Orze's real CPU execution boundary."""
import json
import sqlite3
import threading

from orze.research.execution import run_prepared
from orze.research.exploration import validate_trace
from test_cpu_product_loop import project
from test_cpu_domain_product import request, submit


def test_delivery_closes_native_cpu_work_and_accounts_for_unused_request(project):
    root,cfg,run=project
    cfg['execution']['wall_budget_seconds']=20
    cfg['action_domain']={'version':1,'kind':'json_observations','config':{}}
    prepared=threading.Barrier(2);delivered=threading.Event();executed=[]
    spec={'problem_id':'native-prepared','protocol_id':'fixed-fixture','score_scale':1,
          'root':{'score':0,'artifact':{},'feedback':{}},
          'plan':{'branches':2,'depth':3,'calls':6}}
    def prepare(context):
        prepared.wait(timeout=5)
        return '''import json, os
from pathlib import Path
context=json.load(os.fdopen(os.dup(int(os.environ['ORZE_ACTION_INPUT_FD']))))
assert context['history']==[]
value=context['action']['branch']+1
Path('result.json').write_text(json.dumps({'version':1,'observations':[
 {'name':'quality','values':{'quality':value},
  'validation':{'status':'valid','reason_code':'checked'},
  'comparison_scope':'native-prepared-v1'}]}))
'''
    def evaluate(context,source):
        task='idea-'+context['action']['id'];executed.append(task)
        submit(root,task,request(source,inputs=context,observation=True,
            outputs={'result':{'path':'result.json','max_bytes':4096}}))
        assert run()==0
        with sqlite3.connect(root/'lake.db') as db:
            terminal=json.loads(db.execute('SELECT terminal_json FROM execution_attempts WHERE task_id=?',(task,)).fetchone()[0])
            assert terminal['outcome']=='completed'
            assert terminal['process_tree']['wait_proof']=='ECHILD_WALL'
            observation=json.loads(db.execute('SELECT record_json FROM research_observations').fetchone()[0])
        (root/'delivered.json').write_text(json.dumps({'task':task,'artifact_ids':terminal['artifact_ids']}))
        delivered.set()
        return {'score':observation['values']['quality'],'status':'ok',
                'artifact':{'artifact_ids':terminal['artifact_ids']},'feedback':{},'cost':1,'seconds':1}
    def unused(context,source):
        return {'score':None,'status':'blocked','artifact':{'prepared_source':source},
                'feedback':{},'cost':1,'seconds':1}
    trace=run_prepared(spec,prepare,evaluate,output=root/'discovery',finished=delivered.is_set,unused=unused)
    assert len(executed)==1 and (root/'delivered.json').exists()
    assert trace['metrics']['calls']==2 and trace['metrics']['cost']==2
    with sqlite3.connect(root/'lake.db') as db:
        assert db.execute('SELECT state FROM cpu_action_reservations').fetchall()==[('SETTLED',)]
    nodes=trace['rounds'][0]['observations']
    assert sum(n['feedback'].get('unused_after_delivery',False) for n in nodes)==1
    validate_trace(trace)
