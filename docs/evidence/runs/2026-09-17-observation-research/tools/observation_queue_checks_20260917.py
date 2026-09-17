"""Real Core CPU loops and admission checks for the unused-history fix."""
from pathlib import Path
import os,sys,subprocess,json,hashlib,time
ROOT=Path(__file__).resolve().parent
out=ROOT/'observation-queue-check-20260917';out.mkdir(mode=0o700);(out/'tmp').mkdir()
names=['src/orze/engine/cpu_phase.py','tests/test_cpu_domain_product.py']
def hashes():return {p:hashlib.sha256((ROOT/'orze'/p).read_bytes()).hexdigest() for p in names}
before=hashes()
args=['-q','tests/test_cpu_domain_product.py','tests/test_cpu_action_sources.py','tests/test_cpu_policy_audit_boundary.py','tests/test_cpu_evidence_paging_policy.py','tests/test_cpu_proposal_paging_policy.py','--tb=short','-p','no:cacheprovider','--basetemp='+str(out/'pytest'),'--junitxml='+str(out/'junit.xml')]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',TMPDIR=str(out/'tmp'),PYTHONPATH=str(ROOT/'orze/src'))
start=time.monotonic()
with (out/'stdout.log').open('x') as f:r=subprocess.run([sys.executable,'-m','pytest',*args],cwd=ROOT/'orze',env=env,stdout=f,stderr=subprocess.STDOUT)
value={'exit_code':r.returncode,'seconds':time.monotonic()-start,'before':before,'after':hashes(),'args':args,'scope':'Actual native CPU workers and source admission; offline fault fixtures, no provider calls.'}
(out/'result.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
raise SystemExit(r.returncode)
