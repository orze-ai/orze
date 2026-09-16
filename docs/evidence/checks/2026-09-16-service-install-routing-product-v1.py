"""Saved install routes versus real isolated CPU controllers; manager actions are replaced."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import yaml

CORE=Path(__file__).resolve().parents[3]
BASE=CORE/'docs/evidence/runs/2026-09-16-service-install-routing/baseline-inputs/src/orze/service/install.py'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')

INSTALL=r'''
import importlib.util,json,sys
from pathlib import Path
from unittest.mock import patch
sys.modules['orze_pro']=None
from orze.service import install
if sys.argv[4]=='old':
    spec=importlib.util.spec_from_file_location('routing_baseline_install',sys.argv[5])
    install=importlib.util.module_from_spec(spec);spec.loader.exec_module(install)
service=Path(sys.argv[2]);calls=[]
def forbidden(*args,**kwargs):
    raise AssertionError('routing fixture reached a real service subprocess')
error=None
with patch.object(install,'SERVICE_CONFIG_PATH',service), patch.object(
        install,'_install_systemd',lambda cfg:calls.append(['systemd',cfg])), patch.object(
        install,'_install_crontab',lambda cfg:calls.append(['crontab',cfg])), patch.object(
        install.subprocess,'run',forbidden), patch.object(install.subprocess,'Popen',forbidden):
    try:install.install(sys.argv[1],method=sys.argv[3])
    except RuntimeError as exc:error=str(exc)
value={'module':install.__file__,'error':error,'calls':calls,'cwd_after':str(Path.cwd()),
    'saved':json.loads(service.read_text()) if service.exists() else None,'manager_calls':0}
Path(sys.argv[6]).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
'''

def install(config,service,method,variant,caller,name,env):
    output=caller/(name+'.json')
    command=[sys.executable,'-c',INSTALL,str(config),str(service),method,variant,str(BASE),str(output)]
    with (caller/(name+'.stdout.log')).open('xb') as out,(caller/(name+'.stderr.log')).open('xb') as err:
        done=subprocess.run(command,cwd=caller,env=env,stdout=out,stderr=err,timeout=30)
    assert done.returncode==0,(caller/(name+'.stderr.log')).read_text()
    value=json.loads(output.read_text());assert value['manager_calls']==0
    assert value['module']==str(BASE if variant=='old' else CORE/'src/orze/service/install.py')
    return {'command':command,'exit_code':done.returncode,'result':value}

def tables(database):
    connection=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True);connection.row_factory=sqlite3.Row
    try:return {name:[dict(row) for row in connection.execute('SELECT * FROM '+name)]
        for name in ('ideas','execution_attempts','cpu_action_reservations','research_artifacts')}
    finally:connection.close()

def native(database):
    value=tables(database);attempt,=value['execution_attempts'];terminal=json.loads(attempt['terminal_json'])
    assert attempt['state']=='TERMINAL' and terminal['outcome']=='completed'
    tree=terminal['process_tree'];assert tree['event']=='TREE_CLOSED' and tree['wait_proof']=='ECHILD_WALL'
    ref=tree['binding']['identity']['attempt_ref'];assert all(attempt[key]==v for key,v in ref.items())
    reservation,=value['cpu_action_reservations']
    assert reservation['state']=='SETTLED' and json.loads(reservation['ref_json'])==ref
    assert reservation['terminal_sha256']==sha(attempt['terminal_json'].encode())
    artifact_row,=value['research_artifacts'];artifact=json.loads(artifact_row['record_json'])
    assert artifact['producer']==ref and artifact['artifact_id'] in terminal['artifact_ids']
    raw=Path(artifact['path']).read_bytes();assert sha(raw)==artifact['content_sha256']
    return value,artifact,json.loads(raw)
def main(root,variant='new',absolute_only=False):
    root.mkdir()
    report={'script_sha256':sha(Path(__file__).read_bytes()),'baseline_sha256':sha(BASE.read_bytes()),
        'variant':variant,'absolute_only':absolute_only,'cases':[],'limits':[
        'Real isolated Core CPU controllers and native workers; systemd/crontab manager operations are replaced, never run.',
        'Only the pinned install module is replaced for baseline cases; all other Core source is unchanged.',
        'Core package identity is captured; no Pro package, model, GPU, existing service or old owner is used.',
        'This validates saved results/log/sentinel routes, not effective systemd quoting, runtime environment drift, ownership recovery or deployment.']}
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(CORE/'src'))
    for method in ('systemd','crontab'):
        for route in (('absolute',) if absolute_only else ('relative','absolute')):
            case=root/(method+'-'+route);project=case/'project';caller=case/'installer-cwd'
            for folder in (project,caller):
                folder.mkdir(parents=True);(folder/'.env').write_text('');(folder/'results').mkdir()
            results=project/'results' if route=='relative' else case/'external-results'
            results.mkdir(exist_ok=True)
            database=project/'lake.db';config=project/'orze.yaml';service=case/'service.json';ideas=project/'ideas.md'
            expected={'status':'COMPLETED','route':route,'method':method,'scope':'isolated service routing fixture'}
            command=[sys.executable,'-c',"from pathlib import Path; Path('route-proof.json').write_text("+repr(json.dumps(expected))+')']
            action={'version':1,'adapter':'command','purpose':'record a saved-service controller routing fixture',
                'inputs':{},'command':command,'timeout_seconds':4,
                'outputs':{'route':{'path':'route-proof.json','max_bytes':4096}}}
            cfg={'results_dir':'results' if route=='relative' else str(results),
                'ideas_file':str(ideas),'idea_lake_db':str(database),'min_disk_gb':0,'roles':{},
                'execution':{'version':2,'resource':'cpu','slots':1,'wall_budget_seconds':None},
                'action_policy':{'version':1,'kind':'queue','idle':'wait','wait_seconds':.01},
                'notifications':{'enabled':False},'nested_config_whitelist':['action']}
            config.write_text(yaml.safe_dump(cfg))
            ideas.write_text('## idea-route: Prove saved service route\n```yaml\n'+yaml.safe_dump({'kind':'native_cpu_action','action':action})+'```\n')
            row={'name':case.name,'project':str(project),'results':str(results),'invocations':[]}
            report['cases'].append(row)
            initial=install(config,service,method,variant,caller,'install',env);row['initial_install']=initial
            value=initial['result'];assert value['error'] is None and len(value['calls'])==1
            saved=value['saved'];assert value['calls']==[[method,saved]] and value['cwd_after']==str(caller)
            assert saved['workdir']==str(project) and saved['config_file']==str(config)
            assert [p['name'] for p in saved['runtime_packages']]==['orze']
            assert saved['runtime_packages'][0]['root']==str(CORE/'src/orze')
            assert saved['runtime_packages'][0]['file_count']>0
            row['initial_service_raw_sha256']=sha(service.read_bytes())
            (case/'initial-service.json').write_bytes(service.read_bytes())
            cli=[saved['python'],'-c',"import sys; sys.modules['orze_pro']=None; from orze.cli import main; raise SystemExit(main())",'-c',saved['config_file'],'--once']
            log=Path(saved['log_file'])
            with log.open('xb') as stream:
                completed=subprocess.run(cli,cwd=saved['workdir'],env=env,stdout=stream,stderr=stream,timeout=45)
            row['invocations'].append({'name':'controller','command':cli,'cwd':saved['workdir'],'exit_code':completed.returncode})
            assert completed.returncode==0,log.read_text()[-2500:]
            settled,artifact,body=native(database);assert body==expected
            binding=json.loads(settled['execution_attempts'][0]['binding_json'])
            assert Path(binding['scope'])==results/'idea-route'
            assert binding['artifact_publication']['scope']==str(results)
            assert Path(binding['work_dir']).is_relative_to(results/'idea-route')
            assert json.loads((Path(binding['work_dir'])/'route-proof.json').read_text())==expected
            row.update(tables=settled,artifact=artifact,artifact_body=body,log_path=str(log),
                log_after_controller_sha256=sha(log.read_bytes()),
                routing_observation={'saved_results_dir':saved['results_dir'],'controller_results_dir':str(results),
                    'work_inside_saved_results':Path(binding['work_dir']).is_relative_to(Path(saved['results_dir'])),
                    'same_route':Path(saved['results_dir'])==results,'same_log':log==results/'orze.log'})
            save(case/'after-controller.json',row)
            with (root/'progress.json').open('w') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
            assert row['routing_observation']['same_route'] and row['routing_observation']['same_log']
            assert row['routing_observation']['work_inside_saved_results']
            with log.open('ab') as stream:
                replay=subprocess.run(cli,cwd=saved['workdir'],env=env,stdout=stream,stderr=stream,timeout=45)
            row['invocations'].append({'name':'replay','command':cli,'cwd':saved['workdir'],'exit_code':replay.returncode})
            assert replay.returncode==0 and tables(database)==settled
            row['log_after_replay_sha256']=sha(log.read_bytes())
            saved['review_witness']='existing service metadata must remain'
            service.write_text(json.dumps(saved,indent=2)+'\n');before=service.read_bytes()
            (results/'.orze_stop_all').write_text('target project stop')
            stopped=install(config,service,method,variant,caller,'target-stop',env)
            row['target_stop']=stopped;row['stopped_metadata_unchanged']=service.read_bytes()==before
            (case/'target-stop-before.json').write_bytes(before);(case/'target-stop-after.json').write_bytes(service.read_bytes())
            assert stopped['result']['error'] and 'stop latch' in stopped['result']['error']
            assert stopped['result']['calls']==[] and service.read_bytes()==before
            (results/'.orze_stop_all').unlink()
            unrelated=caller/'results/.orze_shutdown';unrelated.write_text('unrelated project stop')
            accepted=install(config,service,method,variant,caller,'unrelated-stop',env)
            row['unrelated_stop']=accepted
            assert accepted['result']['error'] is None and len(accepted['result']['calls'])==1
            assert accepted['result']['saved']['results_dir']==str(results)
            assert unrelated.read_text()=='unrelated project stop' and tables(database)==settled
            row['final_service_raw_sha256']=sha(service.read_bytes());row['passed']=True
            save(case/'report.json',row)
            with (root/'progress.json').open('w') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    report['passed']=True;save(root/'report.json',report)

if __name__=='__main__':
    main(Path(sys.argv[1]),'old' if '--baseline' in sys.argv else 'new','--absolute-only' in sys.argv)
