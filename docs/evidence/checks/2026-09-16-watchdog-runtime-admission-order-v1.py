"""Instrumented stale branch: actual runtime rejection, no real PID lookup or signal."""
import hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
sys.modules['orze_pro']=None
from orze.service import runtime_contract,watchdog
root=Path(sys.argv[1]);root.mkdir();project=root/'project';project.mkdir()
results=project/'results';results.mkdir();(project/'.env').write_text('')
config=project/'orze.yaml';config.write_text('results_dir: results\nnotifications:\n  enabled: false\n')
packages=runtime_contract.capture_runtime_packages();packages[0]['sha256']='0'*64
cfg={'method':'crontab','python':sys.executable,'workdir':str(project),'results_dir':str(results),
     'config_file':str(config),'runtime_contract_version':runtime_contract.CONTRACT_VERSION,
     'runtime_packages':packages,'log_file':str(results/'watchdog.log'),'stall_threshold':10}
(root/'service.json').write_text(json.dumps(cfg,indent=2)+'\n')
actual=runtime_contract.audit_runtime_contract(cfg);assert not actual['startup_allowed']
events=[]
def forbidden(*args,**kwargs):raise AssertionError('probe crossed an actual process or manager boundary')
error=None
with patch.object(watchdog,'_read_pid',lambda *a:424242),patch.object(watchdog,'_is_pid_alive',lambda *a:True),patch.object(watchdog,'_is_heartbeat_stale',lambda *a:(True,99)),patch.object(watchdog,'_kill_stale',lambda pid:events.append(['requested_stale_kill',pid])),patch.object(watchdog,'_is_orze_running',lambda:False),patch.object(watchdog.time,'sleep',lambda *a:None),patch.object(watchdog,'_notify_failure_loop',lambda *a:events.append(['notification_replaced'])),patch.object(watchdog.os,'kill',forbidden),patch.object(watchdog.os,'killpg',forbidden),patch.object(watchdog.subprocess,'run',forbidden),patch.object(watchdog.subprocess,'Popen',forbidden):
    try:watchdog.check_and_restart(cfg)
    except watchdog.WatchdogLaunchError as exc:error=exc.code
value={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'actual_runtime_audit':actual,
    'events':events,'watchdog_error':error,'actual_controllers':0,'actual_signals':0,'actual_managers':0,
    'limits':'PID and heartbeat are synthetic, kill primitive replaced by recorder; this proves call ordering, not actual termination.'}
(root/'observation.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps(value))
assert error=='runtime_contract_rejected'
assert not any(e[0]=='requested_stale_kill' for e in events),'watchdog requested stale kill before rejecting runtime drift'
