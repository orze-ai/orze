"""Exploratory route drift counterexample; temporary metadata, no controller or manager."""
import hashlib,json,os,sys
from pathlib import Path
from unittest.mock import patch
sys.modules['orze_pro']=None
from orze.service import install,runtime_contract
from orze.core.config import load_project_config
root=Path(sys.argv[1]);root.mkdir()
project=root/'project';project.mkdir();(project/'.env').write_text('')
for name in ('results-a','results-b'):(project/name).mkdir()
config=project/'orze.yaml';config.write_text('results_dir: results-a\nnotifications:\n  enabled: false\n')
service=root/'service.json';calls=[]
def forbidden(*args,**kwargs):raise AssertionError('exploratory route audit reached service subprocess')
os.chdir(project)
with patch.object(install,'SERVICE_CONFIG_PATH',service),patch.object(install,'_install_crontab',lambda cfg:calls.append(cfg)),patch.object(install.subprocess,'run',forbidden),patch.object(install.subprocess,'Popen',forbidden):
    install.install(str(config),method='crontab')
    saved=json.loads(service.read_text());initial=runtime_contract.audit_runtime_contract(saved)
    assert initial['startup_allowed'] and len(calls)==1
    (root/'installed-config.yaml').write_bytes(config.read_bytes())
    config.write_text('results_dir: results-b\nnotifications:\n  enabled: false\n')
    target=project/'results-b';(target/'.orze_stop_all').write_text('target operator stop')
    actual=Path(load_project_config(str(config))['results_dir']).resolve()
    assert actual==target
    observed=runtime_contract.audit_runtime_contract(saved)
    value={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'initial_audit':initial,'after_config_change_audit':observed,
        'saved_results':saved['results_dir'],'current_config_results':str(actual),
        'target_stop_present':(actual/'.orze_stop_all').exists(),'service_metadata_unchanged':json.loads(service.read_text())==saved,
        'actual_controllers':0,'actual_managers':0,'limits':'Contract read-only decision only, no signal, owner or controller launch claim.'}
    (root/'observation.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps(value))
    assert not observed['startup_allowed'], 'service audit admitted changed results authority with a stop marker in the actual target'
