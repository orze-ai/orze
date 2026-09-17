"""Offline regression only; actual experiments use the normal license."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
out=ROOT/'observation-prompt-check-20260917-v2'
out.mkdir(mode=0o700)
(out/'tmp').mkdir()
names=['src/orze_pro/agents/observation_prompt.py','src/orze_pro/agents/report_paging.py','src/orze_pro/agents/joint_snapshot.py','tests/test_native_cpu_research_feedback.py']
def hashes():return {n:hashlib.sha256((ROOT/'pro'/n).read_bytes()).hexdigest() for n in names}
before=hashes()
args=['-q','tests/test_native_cpu_research_feedback.py','tests/test_research_evidence_paging.py','tests/test_joint_research_snapshot.py','tests/test_research_memory_sources.py','--tb=short','-p','no:cacheprovider','--basetemp='+str(out/'pytest'),'--junitxml='+str(out/'junit.xml')]
code='from unittest.mock import patch; import pytest; patch("orze_pro._gate.require_license",lambda:None).start(); raise SystemExit(pytest.main('+repr(args)+'))'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',TMPDIR=str(out/'tmp'),PYTHONPATH=os.pathsep.join(map(str,[ROOT/'orze/src',ROOT/'orze',ROOT/'pro/src'])))
started=time.monotonic()
with (out/'stdout.log').open('x') as f:result=subprocess.run([sys.executable,'-c',code],cwd=ROOT/'pro',env=env,stdout=f,stderr=subprocess.STDOUT)
value={'exit_code':result.returncode,'seconds':time.monotonic()-started,'before':before,'after':hashes(),'args':args,'scope':'Offline provider fixtures and test-only license patch; actual native CPU workers.'}
(out/'result.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value),flush=True)
raise SystemExit(result.returncode)
