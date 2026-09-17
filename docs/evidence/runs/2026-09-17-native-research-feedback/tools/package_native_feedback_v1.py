"""Preserve raw research outputs publicly; keep private implementation checks in Pro."""
import hashlib
import json
from pathlib import Path
import tarfile
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parent
REL=Path('docs/evidence/runs/2026-09-17-native-research-feedback')
PUBLIC,PRIVATE=ROOT/'orze'/REL,ROOT/'pro'/REL


def sha(raw): return hashlib.sha256(raw).hexdigest()


def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True)


def main():
 secrets=[v.encode() for v in dotenv_values(ROOT.parent/'auto-research-1.7b/.env').values() if v and len(v)>=12]
 sources=[ROOT/f'real-memory-counterexample-diagnostic-20260916-v{v}' for v in (1,2,3,4,5,6)]+[ROOT/'real-memory-pilot-20260916-feedback']
 scripts=[f'real_memory_counterexample_diagnostic_v{v}.py' for v in (1,2,3,4,5,6)]+['real_memory_after_feedback_v1.py','analyze_native_feedback_v1.py','package_native_feedback_v1.py','real_memory_pilot_v2.py','analyze_real_memory_v2.py','prompt_choice_plan_20260916.json','memory_wrapper_plan_20260917.json']
 checks=['feedback-check-20260916-v1','feedback-check-20260916-v2','feedback-check-20260916-v3','prompt-choice-check-20260916','prompt-format-check-20260917','memory-wrapper-check-20260917']
 files=[p for s in sources for p in s.rglob('*') if p.is_file()]+[ROOT/s for s in scripts]+[ROOT/s/'check.log' for s in checks]
 for p in files:
  assert not p.is_symlink() and not any(part in ('.env','.git','__pycache__','orze_pro') for part in p.parts),p
  assert not any(secret in p.read_bytes() for secret in secrets),'credential found in '+str(p.relative_to(ROOT))
 PUBLIC.mkdir(parents=True,exist_ok=False)
 PRIVATE.mkdir(parents=True,exist_ok=False)
 archives=[]
 for s in sources:
  dest=PUBLIC/'raw'/(s.name+'.tar.gz');dest.parent.mkdir(exist_ok=True)
  with tarfile.open(dest,'x:gz') as t:t.add(s,arcname=s.name)
  entries=[]
  with tarfile.open(dest,'r:gz') as t:
   for m in t:
    if not m.isfile():
     assert m.isdir(),m.name
     continue
    raw=t.extractfile(m).read();assert raw==(ROOT/m.name).read_bytes()
    entries.append({'path':m.name,'bytes':len(raw),'sha256':sha(raw)})
  write(dest.with_suffix('.manifest.json'),entries)
  archives.append({'path':str(dest.relative_to(PUBLIC)),'files':len(entries),'bytes':dest.stat().st_size,'sha256':sha(dest.read_bytes())})
 for name in scripts:
  p=PUBLIC/'tools'/name;p.parent.mkdir(exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
 for name in checks:
  p=PRIVATE/'checks'/(name+'.log');p.parent.mkdir(exist_ok=True);p.write_bytes((ROOT/name/'check.log').read_bytes())
  write(PUBLIC/'checks'/(name+'.json'),{'private_log_sha256':sha(p.read_bytes()),'summary':p.read_text().splitlines()[-1],
       'scope':'Offline test providers and test-only license fixture; actual CPU workers. No real model calls.'})
 (PUBLIC/'summary.json').write_bytes((sources[-1]/'summary.json').read_bytes())
 write(PUBLIC/'archives.json',archives)
 write(PRIVATE/'public-link.json',{'summary_sha256':sha((PUBLIC/'summary.json').read_bytes()),'public_path':str(REL)})
 for dest in (PUBLIC,PRIVATE):
  write(dest/'files.json',{str(p.relative_to(dest)):{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(dest.rglob('*')) if p.is_file()})
 print(json.dumps({'credential_scan_files':len(files),'archives':archives,'public':str(PUBLIC),'private':str(PRIVATE)}))


if __name__=='__main__':main()
