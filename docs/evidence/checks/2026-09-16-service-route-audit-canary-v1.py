"""Own disposable child only: changed results route must not terminate an unrelated live process."""
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
sys.modules['orze_pro']=None
from orze.service import runtime_contract,watchdog
root=Path(sys.argv[1]);root.mkdir();project=root/'project';project.mkdir()
results=project/'results';results.mkdir();(project/'.env').write_text('')
config=project/'orze.yaml';config.write_text('results_dir: results\nnotifications:\n  enabled: false\n')
packages=runtime_contract.capture_runtime_packages()
cfg={'method':'crontab','python':sys.executable,'workdir':str(project),'results_dir':str(results),
     'config_file':str(config),'runtime_contract_version':runtime_contract.CONTRACT_VERSION,
     'runtime_packages':packages,'log_file':str(results/'watchdog.log'),'stall_threshold':1}
target=project/'other-results';target.mkdir();(target/'.orze_stop_all').write_text('operator stop')
config.write_text('results_dir: other-results\nnotifications:\n  enabled: false\n')
(root/'service.json').write_text(json.dumps(cfg,indent=2)+'\n')
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],start_new_session=True)
kill,killpg=os.kill,os.killpg
signals=[];observed=None;error=None
try:
    from orze.engine.supervisor_worker import process_identity
    identity=process_identity(child.pid)
    assert child.poll() is None and os.getpgid(child.pid)==child.pid and child.pid!=os.getpgrp()
    hostname=socket.gethostname()
    (results/('.orze.pid.'+hostname)).write_text(str(child.pid))
    (results/('_host_'+hostname+'_probe.json')).write_text(json.dumps({'epoch':time.time()-3600}))
    def check_owner(pid):
        assert pid==child.pid and process_identity(pid)==identity
    def own_kill(pid,sig):
        check_owner(pid)
        if sig:signals.append({'target':'pid','pid':pid,'signal':int(sig)})
        return kill(pid,sig)
    def own_killpg(pid,sig):
        check_owner(pid);assert os.getpgid(pid)==pid
        signals.append({'target':'own_isolated_pgid','pid':pid,'signal':int(sig)})
        return killpg(pid,sig)
    def forbidden(*a,**k):raise AssertionError('canary reached a manager or controller subprocess')
    with patch.object(watchdog.os,'kill',own_kill),patch.object(watchdog.os,'killpg',own_killpg),patch.object(watchdog,'_is_orze_running',lambda:False),patch.object(watchdog,'_notify_failure_loop',lambda *a:None),patch.object(watchdog.subprocess,'run',forbidden),patch.object(watchdog.subprocess,'Popen',forbidden):
        before=runtime_contract.audit_runtime_contract(cfg);assert not before['startup_allowed']
        try:watchdog.check_and_restart(cfg)
        except watchdog.WatchdogLaunchError as exc:error=exc.code
    observed=child.poll()
    value={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'own_canary_identity':identity,'canary_exit_before_cleanup':observed,'signals_before_cleanup':signals,
        'audit_before':before,'current_results':str(target),'watchdog_error':error,'actual_controllers':0,'actual_managers':0,
        'limits':'One child created by this script in its own session; all signals restricted to its exact birth identity and group. No existing process or service touched.'}
    (root/'observation.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps(value),flush=True)
finally:
    if child.poll() is None:child.terminate()
    code=child.wait(timeout=10)
    (root/'cleanup.json').write_text(json.dumps({'child_pid':child.pid,'reaped':True,'exit_code':code})+'\n')
assert error=='runtime_contract_rejected'
assert observed is None and signals==[], 'watchdog terminated the unrelated own canary before runtime rejection'
