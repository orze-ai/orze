"""Archive all five fixed batches, including failures; baseline Pro code stays private."""
import hashlib
import json
from pathlib import Path
import tarfile
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parent
REL=Path('docs/evidence/runs/2026-09-17-memory-simplification')
PUBLIC,PRIVATE=ROOT/'orze'/REL,ROOT/'pro'/REL

def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True)

def main():
    sources=[ROOT/'memory-action-pairs-20260917', ROOT/'memory-next-step-pairs-20260917', ROOT/'optional-memory-workflow-20260917', ROOT/'model-capacity-20260917', ROOT/'model-effort-20260917']
    private_sources=[ROOT/'memory-action-baseline-20260917', ROOT/'memory-next-step-baseline-20260917']
    scripts=['model_capacity_plan_20260917.json','model_capacity_diagnostic_20260917.py','model_capacity_batch_20260917.py','model_capacity_batch_20260917_initial.py','model_capacity_launcher_failure_20260917.json','model_capacity_upper_bounds_20260917.py','analyze_model_capacity_20260917.py','model_effort_plan_20260917.json','model_effort_plan_20260917_initial.json','model_effort_diagnostic_20260917.py','model_effort_batch_20260917.py','analyze_model_effort_20260917.py','model_settings_checks_20260917.py','model_settings_checks_20260917_v2.py','optional_memory_workflow_plan_20260917.json','optional_memory_workflow_20260917.py','optional_memory_workflow_batch_20260917.py','analyze_optional_workflow_20260917.py','memory_action_plan_20260917.json','memory_next_step_plan_20260917.json',
        'memory_action_diagnostic_20260917.py','memory_next_step_diagnostic_20260917.py',
        'memory_action_batch_20260917.py','memory_next_step_batch_20260917.py',
        'analyze_memory_actions_20260917.py','analyze_memory_next_step_20260917.py',
        'memory_action_checks_20260917.py','memory_next_step_checks_20260917.py',
        'real_memory_pilot_v2.py','analyze_real_memory_v2.py','package_memory_simplification_20260917.py']
    checks=['memory-action-check-20260917','memory-next-step-check-20260917','model-settings-check-20260917','model-settings-check-20260917-v2']
    files=[p for s in sources+private_sources for p in s.rglob('*') if p.is_file()]
    files += [ROOT/s for s in scripts]
    files += [ROOT/s/n for s in checks for n in ('stdout.log','junit.xml','result.json')]
    secrets=[v.encode() for v in dotenv_values(ROOT.parent/'auto-research-1.7b/.env').values() if v and len(v)>=12]
    for p in files:
        assert not p.is_symlink() and not any(part in ('.env','.git','__pycache__') for part in p.parts),p
        assert not any(v in p.read_bytes() for v in secrets),'credential detected in selected evidence'
    PUBLIC.mkdir(parents=True,exist_ok=False);PRIVATE.mkdir(parents=True,exist_ok=False)
    archives=[]
    for source in sources+private_sources:
        dest=(PUBLIC if source in sources else PRIVATE)/'raw'/(source.name+'.tar.gz')
        dest.parent.mkdir(exist_ok=True)
        with tarfile.open(dest,'x:gz') as t:t.add(source,arcname=source.name)
        entries=[]
        with tarfile.open(dest,'r:gz') as t:
            for m in t:
                if not m.isfile():
                    assert m.isdir(),m.name
                    continue
                raw=t.extractfile(m).read();assert raw==(ROOT/m.name).read_bytes()
                entries.append({'path':m.name,'bytes':len(raw),'sha256':sha(raw)})
        write(dest.with_suffix('.manifest.json'),entries)
        if source in sources:
            archives.append({'path':str(dest.relative_to(PUBLIC)),'files':len(entries),'bytes':dest.stat().st_size,'sha256':sha(dest.read_bytes())})
    for name in scripts:
        p=PUBLIC/'tools'/name;p.parent.mkdir(exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    for name in checks:
        for filename in ('stdout.log','junit.xml','result.json'):
            p=PRIVATE/'checks'/name/filename;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes((ROOT/name/filename).read_bytes())
        value=json.loads((ROOT/name/'result.json').read_bytes())
        write(PUBLIC/'checks'/(name+'.json'),{'exit_code':value['exit_code'],'seconds':value['seconds'],
            'sources_unchanged':value['before']==value['after'],
            'private_log_sha256':sha((ROOT/name/'stdout.log').read_bytes()),
            'summary':(ROOT/name/'stdout.log').read_text().splitlines()[-1],
            'scope':'Offline provider fixtures and test-only license patch; actual research consumer and durable-memory paths.' if name=='memory-next-step-check-20260917' else value['scope'],
            'scope_note':None if name!='memory-next-step-check-20260917' else 'The original generic launcher scope mentions native CPU workers; this second test selection does not include the native CPU feedback module. The full original metadata and selected test list are retained privately.'})
    for source in sources:(PUBLIC/(source.name+'-summary.json')).write_bytes((source/'summary.json').read_bytes())
    write(PUBLIC/'archives.json',archives)
    write(PRIVATE/'public-link.json',{p.name:sha(p.read_bytes()) for p in PUBLIC.glob('*-summary.json')})
    for dest in (PUBLIC,PRIVATE):
        write(dest/'files.json',{str(p.relative_to(dest)):{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}
              for p in sorted(dest.rglob('*')) if p.is_file()})
    print(json.dumps({'credential_scan_files':len(files),'public_archives':archives}))

if __name__=='__main__':main()
