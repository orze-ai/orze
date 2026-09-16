"""Old/new complete read-only service audit costs; no service operations."""
import gc, hashlib, importlib.util, json, os, statistics, sys, time, tracemalloc
from pathlib import Path
from unittest.mock import patch
sys.modules['orze_pro']=None
from orze.service import runtime_contract as current
CORE=Path(__file__).resolve().parents[3]
BASE=CORE/'docs/evidence/runs/2026-09-16-service-route-audit/baseline-inputs/src/orze/service/runtime_contract.py'
spec=importlib.util.spec_from_file_location('route_audit_baseline',BASE)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
root=Path(sys.argv[1]);root.mkdir()
report={'schema':1,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'baseline_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),'cases':[],
        'scope':'Same process alternating old/new complete audit, actual Core package hashing; no manager, controller, provider or signal. Python allocations are not RSS.'}
def forbidden(*args,**kwargs):raise AssertionError('read-only cost audit reached a subprocess')
with patch.object(current.subprocess,'run',forbidden),patch.object(current.subprocess,'Popen',forbidden):
    for padding in (0,65536,262144):
        project=root/str(padding);project.mkdir();(project/'.env').write_text('')
        results=project/'results';results.mkdir();config=project/'orze.yaml'
        config.write_text('results_dir: results\nnotes: "'+('x'*padding)+'"\n')
        svc={'method':'crontab','python':sys.executable,'workdir':str(project),'results_dir':str(results),
             'config_file':str(config),'runtime_contract_version':1,'runtime_packages':current.capture_runtime_packages()}
        (project/'service.json').write_text(json.dumps(svc,indent=2)+'\n')
        samples=[];saved=[]
        for variant in ('old','new','new','old','new','old','old','new'):
            module=old if variant=='old' else current
            start=time.perf_counter_ns();cpu=time.process_time_ns();value=module.audit_runtime_contract(svc)
            samples.append({'variant':variant,'wall_ns':time.perf_counter_ns()-start,'cpu_ns':time.process_time_ns()-cpu})
            assert value['startup_allowed'];saved.append(value)
        peaks={}
        for variant,module in [('old',old),('new',current)]:
            gc.collect();tracemalloc.start();value=module.audit_runtime_contract(svc)
            peaks[variant]=tracemalloc.get_traced_memory()[1];tracemalloc.stop()
            assert value==saved[0]
        assert all(value==saved[0] for value in saved)
        row={'config_bytes':config.stat().st_size,'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
             'samples':samples,'python_peak_bytes':peaks,'outputs_identical':True,
             'median':{v:{k:statistics.median(s[k] for s in samples if s['variant']==v) for k in ('wall_ns','cpu_ns')} for v in ('old','new')}}
        report['cases'].append(row)
        with (project/'cost.json').open('x') as f:json.dump(row,f,indent=2);f.write('\n')
report['passed']=True
with (root/'report.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(report))
