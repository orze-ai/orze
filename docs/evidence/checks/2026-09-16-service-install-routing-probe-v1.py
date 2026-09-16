"""Installation route probe: temporary metadata only, no real service-manager calls."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

root=Path(sys.argv[1]);root.mkdir()
caller=root/'installer-cwd';project=root/'project'
for folder in (caller,project):
    folder.mkdir();(folder/'results').mkdir();(folder/'.env').write_text('')
config=project/'orze.yaml';config.write_text('results_dir: results\n')
from orze.core.config import load_project_config
from orze.service import install
os.chdir(caller)
rendered=[]
def forbidden_manager(*args,**kwargs):
    raise AssertionError('isolated routing probe reached a real subprocess boundary')
# Use the real install entry point. Replace its service-manager action and
# runtime capture only; do not duplicate its path construction in the probe.
with patch.object(install,'SERVICE_CONFIG_PATH',root/'service.json'), patch(
        'orze.service.runtime_contract.capture_runtime_packages',lambda:[]), patch.object(
        install,'_install_systemd',lambda value:rendered.append(value)), patch.object(
        install.subprocess,'run',forbidden_manager), patch.object(
        install.subprocess,'Popen',forbidden_manager):
    install.install(str(config),method='systemd')
saved=json.loads((root/'service.json').read_text())
assert rendered==[saved]
os.chdir(project)
started=load_project_config(str(config))
actual=Path(started['results_dir']).resolve()
import hashlib
value={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'installer_cwd':str(caller),'service_workdir':saved['workdir'],
    'saved_results_dir':saved['results_dir'],'controller_results_dir':str(actual),
    'same_route':saved['results_dir']==str(actual),'service_manager_calls':0,
    'actual_controller_processes':0,'runtime_package_capture':'isolated stub; no runtime identity claim'}
(root/'observation.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
print(json.dumps(value))
assert value['same_route'], 'saved watchdog route differs from controller workdir-relative route'
