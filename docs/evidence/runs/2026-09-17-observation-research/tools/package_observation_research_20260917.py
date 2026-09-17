"""Preserve complete and aborted experiments; project audit and Pro logs stay private."""
from pathlib import Path
import json,hashlib,tarfile
from dotenv import dotenv_values
ROOT=Path(__file__).resolve().parent
REL=Path('docs/evidence/runs/2026-09-17-observation-research')
PUBLIC=ROOT/'orze'/REL;PRIVATE=ROOT/'pro'/REL
SOURCES=['observation-research-20260917','observation-research-20260917-v2']
CHECKS=['observation-prompt-check-20260917','observation-prompt-check-20260917-v2','observation-prompt-check-20260917-v3','observation-prompt-check-20260917-v4','observation-queue-check-20260917','observation-restored-check-20260917']
SCRIPTS=['prepare_observation_research_20260917.py','observation_research_20260917.py','observation_research_20260917_v2.py','observation_research_batch_20260917.py','observation_research_batch_20260917_v2.py','analyze_observation_research_20260917.py','analyze_observation_research_20260917_v2.py','observation_research_plan_20260917_preflight.json','observation_research_plan_20260917.json','observation_research_plan_20260917_v2.json','diagnose_observation_policy_20260917.py','observation_prompt_checks_20260917.py','observation_prompt_checks_20260917_v2.py','observation_prompt_checks_20260917_v3.py','observation_prompt_checks_20260917_v4.py','observation_queue_checks_20260917.py','observation_restored_checks_20260917.py','package_observation_research_20260917.py','real_memory_pilot_v2.py','analyze_optional_workflow_20260917.py','analyze_real_memory_v2.py']
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True)
def copy(p,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==p.read_bytes(),target
    else:
        with target.open('xb') as f:f.write(p.read_bytes())
def main():
    assert (ROOT/SOURCES[1]/'summary.json').is_file()
    paths=[p for name in SOURCES+['asr-research-case-20260917'] for p in (ROOT/name).rglob('*') if p.is_file()]
    paths += [ROOT/p for p in SCRIPTS]+[ROOT/'audit_asr_research_case_20260917.py',ROOT/'rescore_asr_ted_case_20260917.py']
    paths += [ROOT/c/n for c in CHECKS for n in ('result.json','junit.xml','stdout.log')]
    secrets=[v.encode() for v in dotenv_values(ROOT.parent/'auto-research-1.7b/.env').values() if v and len(v)>=12]
    for p in paths:
        assert not p.is_symlink() and not any(v in ('.env','.git','__pycache__') for v in p.parts),p
        assert not any(secret in p.read_bytes() for secret in secrets),'Credential value detected in selected evidence'
    PRIVATE.mkdir(parents=True,exist_ok=False)
    archives=[]
    for name in SOURCES:
        target=PUBLIC/'raw'/(name+'.tar.gz');target.parent.mkdir(exist_ok=True)
        with tarfile.open(target,'x:gz') as tf:tf.add(ROOT/name,arcname=name)
        manifest=[]
        with tarfile.open(target,'r:gz') as tf:
            for m in tf:
                if not m.isfile():assert m.isdir();continue
                raw=tf.extractfile(m).read();assert raw==(ROOT/m.name).read_bytes()
                manifest.append({'path':m.name,'bytes':len(raw),'sha256':sha(raw)})
        write(target.with_suffix('.manifest.json'),manifest)
        archives.append({'path':str(target.relative_to(PUBLIC)),'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'files':len(manifest)})
        copy(ROOT/name/'summary.json',PUBLIC/(name+'-summary.json'))
    for name in SCRIPTS:copy(ROOT/name,PUBLIC/'tools'/name)
    for name in CHECKS:
        destination=PUBLIC if name.startswith('observation-queue') else PRIVATE
        for item in ('result.json','junit.xml','stdout.log'):copy(ROOT/name/item,destination/'checks'/name/item)
        value=json.loads((ROOT/name/'result.json').read_bytes())
        write(PUBLIC/'checks'/(name+'.json'),{'exit_code':value['exit_code'],'seconds':value['seconds'],'sources_unchanged':value['before']==value['after'],'scope':value['scope'],'summary':(ROOT/name/'stdout.log').read_text().splitlines()[-1],'log_sha256':sha((ROOT/name/'stdout.log').read_bytes())})
    for p in (ROOT/'asr-research-case-20260917').iterdir():copy(p,PRIVATE/'asr-case'/p.name)
    copy(ROOT/'audit_asr_research_case_20260917.py',PRIVATE/'asr-case/audit_asr_research_case_20260917.py')
    copy(ROOT/'rescore_asr_ted_case_20260917.py',PRIVATE/'asr-case/rescore_asr_ted_case_20260917.py')
    write(PUBLIC/'archives.json',archives)
    write(PRIVATE/'public-link.json',{p.name:sha(p.read_bytes()) for p in PUBLIC.glob('*-summary.json')})
    for dest in (PUBLIC,PRIVATE):write(dest/'files.json',{str(p.relative_to(dest)):{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(dest.rglob('*')) if p.is_file()})
    print(json.dumps({'credential_scan_files':len(paths),'archives':archives}))
if __name__=='__main__':main()
