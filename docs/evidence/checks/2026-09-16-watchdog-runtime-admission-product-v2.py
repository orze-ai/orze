"""Real CPU controllers remain running during watchdog runtime admission checks."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch
import yaml

sys.modules['orze_pro']=None
CORE=Path(__file__).resolve().parents[3]
BASE=CORE/'docs/evidence/runs/2026-09-16-watchdog-runtime-admission/baseline-inputs/src/orze/service/watchdog.py'
HELPER=CORE/'docs/evidence/checks/2026-09-16-service-install-routing-product-v1.py'
spec=importlib.util.spec_from_file_location('prior_routing_fixture',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')

def main(root,baseline=False,matching_only=False):
    from orze.service import watchdog,runtime_contract
    from orze.engine.supervisor_worker import process_identity
    if baseline:
        spec=importlib.util.spec_from_file_location('baseline_watchdog',BASE)
        watchdog=importlib.util.module_from_spec(spec);spec.loader.exec_module(watchdog)
    root.mkdir()
    report={'script_sha256':sha(Path(__file__).read_bytes()),'helper_sha256':sha(HELPER.read_bytes()),
        'baseline_watchdog_sha256':sha(BASE.read_bytes()),'baseline':baseline,'matching_only':matching_only,'cases':[],
        'limits':['Real Core CLI controllers and CPU worker output, manager install operations replaced.',
                  'All watchdog signals forbidden except kill(pid,0) for the exact controller child created by this script.',
                  'No existing process, provider, Pro account, GPU or deployment.',
                  'The fixture authors the legacy PID marker for its exact CPU controller; native CPU execution does not publish that marker. This is not a complete CPU service integration test.',
                  'Runtime admission ordering only; no PID ownership, scoped service or installation-after-drift guarantee.']}
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(CORE/'src'))
    for mode in (('matching',) if matching_only else ('drift','matching')):
        case=root/mode;project=case/'project';project.mkdir(parents=True);(project/'.env').write_text('')
        results=project/'results';results.mkdir();config=project/'orze.yaml';database=project/'lake.db'
        marker=project/'worker-started';release=project/'release-worker';ideas=project/'ideas.md'
        expected={'status':'COMPLETED','runtime_case':mode,'scope':'isolated watchdog admission'}
        worker="""import json,time
from pathlib import Path
marker=Path(%r);release=Path(%r)
marker.write_text('worker started')
deadline=time.monotonic()+8
while not release.exists() and time.monotonic()<deadline:time.sleep(.02)
assert release.exists(), 'fixture parent did not release worker'
Path('proof.json').write_text(%r)
""" % (str(marker),str(release),json.dumps(expected))
        action={'version':1,'adapter':'command','purpose':'keep an actual CPU controller alive during runtime admission',
            'inputs':{},'command':[sys.executable,'-c',worker],'timeout_seconds':10,
            'outputs':{'proof':{'path':'proof.json','max_bytes':4096}}}
        cfg={'results_dir':str(results),'ideas_file':str(ideas),'idea_lake_db':str(database),'min_disk_gb':0,
            'roles':{},'notifications':{'enabled':False},'execution':{'version':2,'resource':'cpu','slots':1,'wall_budget_seconds':None},
            'action_policy':{'version':1,'kind':'queue','idle':'wait','wait_seconds':.01},'nested_config_whitelist':['action']}
        config.write_text(yaml.safe_dump(cfg))
        ideas.write_text('## idea-route: Watchdog runtime admission\n```yaml\n'+yaml.safe_dump({'kind':'native_cpu_action','action':action})+'```\n')
        initial=helper.install(config,case/'service.json','crontab','new',project,'install',env)
        assert initial['result']['error'] is None and initial['result']['manager_calls']==0
        saved=initial['result']['saved'];observed_cfg=copy.deepcopy(saved)
        if mode=='drift':observed_cfg['runtime_packages'][0]['sha256']='0'*64
        save(case/'watchdog-service.json',observed_cfg)
        row={'mode':mode,'project':str(project),'results':str(results),'initial_install':initial,'watchdog_service':observed_cfg,'invocations':[]}
        report['cases'].append(row)
        command=[saved['python'],'-c',"import sys;sys.modules['orze_pro']=None;from orze.cli import main;raise SystemExit(main())",'-c',saved['config_file'],'--once']
        stdout=project/'controller.stdout.log';stderr=project/'controller.stderr.log'
        events=[];caught=None
        with stdout.open('xb') as out,stderr.open('xb') as err:
            controller=subprocess.Popen(command,cwd=saved['workdir'],env=env,stdout=out,stderr=err,start_new_session=True)
            try:
                deadline=time.monotonic()+15
                while not marker.exists():
                    assert controller.poll() is None,stderr.read_text()[-2000:]
                    assert time.monotonic()<deadline,'worker did not start'
                    time.sleep(.02)
                identity=process_identity(controller.pid)
                assert identity[1]==os.getpid() and controller.poll() is None
                pidfile=results/('.orze.pid.'+socket.gethostname())
                assert not pidfile.exists()
                pidfile.write_text(str(controller.pid))
                save(case/'pid-marker-origin.json',{'origin':'fixture-authored legacy service PID marker','controller_identity':identity,'path':str(pidfile)})
                assert int(pidfile.read_text())==controller.pid
                raw_pid=pidfile.read_bytes()
                original_read=watchdog._read_pid;original_kill=os.kill
                def read_pid(*args):
                    events.append('read_pid');return original_read(*args)
                def own_probe(pid,sig):
                    assert pid==controller.pid and sig==0 and process_identity(pid)==identity
                    events.append('own_controller_kill0');return original_kill(pid,sig)
                def forbidden(*args,**kwargs):raise AssertionError('watchdog attempted a forbidden process or manager operation')
                with patch.object(watchdog,'_read_pid',read_pid),patch.object(watchdog,'_is_orze_running',forbidden),patch.object(watchdog.os,'kill',own_probe),patch.object(watchdog.os,'killpg',forbidden),patch.object(watchdog.subprocess,'run',forbidden),patch.object(watchdog.subprocess,'Popen',forbidden),patch.object(watchdog,'_notify_failure_loop',lambda *args:None):
                    audit=runtime_contract.audit_runtime_contract(observed_cfg)
                    assert audit['startup_allowed']==(mode=='matching')
                    try:watchdog.check_and_restart(observed_cfg)
                    except watchdog.WatchdogLaunchError as exc:caught=exc.code
                alive=controller.poll() is None and process_identity(controller.pid)==identity
                row.update(controller_identity=identity,watchdog_events=events,watchdog_error=caught,
                    audit=audit,controller_alive_after_watchdog=alive,pid_file_unchanged=pidfile.read_bytes()==raw_pid)
                save(case/'after-watchdog.json',row)
            finally:
                release.write_text('fixture releases its own worker')
                code=controller.wait(timeout=30)
                save(case/'controller-reaped.json',{'pid':controller.pid,'exit_code':code,'reaped':True})
        row['invocations'].append({'name':'controller','command':command,'cwd':saved['workdir'],'exit_code':code})
        assert code==0,stderr.read_text()[-2000:]
        tables,artifact,body=helper.native(database);assert body==expected
        row.update(tables=tables,artifact=artifact,artifact_body=body)
        with (project/'replay.stdout.log').open('xb') as out,(project/'replay.stderr.log').open('xb') as err:
            replay=subprocess.run(command,cwd=saved['workdir'],env=env,stdout=out,stderr=err,timeout=30)
        row['invocations'].append({'name':'replay','command':command,'cwd':saved['workdir'],'exit_code':replay.returncode})
        assert replay.returncode==0 and helper.tables(database)==tables
        save(case/'after-replay.json',row)
        with (root/'progress.json').open('w') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
        assert row['controller_alive_after_watchdog'] and row['pid_file_unchanged']
        if mode=='drift':assert caught=='runtime_contract_rejected' and events==[], 'runtime drift was not rejected before controller inspection'
        else:assert caught is None and events==['read_pid','own_controller_kill0']
        row['passed']=True;save(case/'report.json',row)
    report['passed']=True;save(root/'report.json',report)
    print(json.dumps({'passed':True,'cpu_workers':len(report['cases']),'cli_invocations':sum(len(c['invocations']) for c in report['cases'])}))

if __name__=='__main__':main(Path(sys.argv[1]),'--baseline' in sys.argv,'--matching-only' in sys.argv)
