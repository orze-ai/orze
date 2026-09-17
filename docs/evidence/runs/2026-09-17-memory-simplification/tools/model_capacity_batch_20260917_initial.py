"""Run every predeclared slot in order, retaining all outcomes."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
checks=json.loads((ROOT/'memory-next-step-check-20260917/result.json').read_bytes())
assert (ROOT/'model-capacity-20260917/summary.json').is_file()
assert checks['exit_code']==0 and checks['before']==checks['after']
plan=json.loads((ROOT/'model_capacity_plan_20260917.json').read_bytes())
root=ROOT/'model-capacity-20260917'
root.mkdir(mode=0o700)
(root/'tmp').mkdir()
(root/'plan.json').write_bytes((ROOT/'model_capacity_plan_20260917.json').read_bytes())
runner=ROOT/'model_capacity_diagnostic_20260917.py'
pinned=hashlib.sha256(runner.read_bytes()).hexdigest()
rows=[]
for repetition,arm in plan['order']:
    assert hashlib.sha256(runner.read_bytes()).hexdigest()==pinned
    label=f'pair-{repetition}-{arm}'
    print('Starting '+label,flush=True)
    started=time.monotonic()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',TMPDIR=str(root/'tmp'))
    with (root/(label+'.log')).open('x') as log:
        result=subprocess.run([sys.executable,str(runner),str(repetition),arm],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    row={'repetition':repetition,'arm':arm,'exit_code':result.returncode,'process_wall_seconds':time.monotonic()-started,'runner_sha256':pinned}
    rows.append(row)
    with (root/'runs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
print(json.dumps({'planned':2,'completed':len(rows),'successful_processes':sum(r['exit_code']==0 for r in rows)}),flush=True)
