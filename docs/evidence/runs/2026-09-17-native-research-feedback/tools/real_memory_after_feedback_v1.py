"""Replicate the same Haiku A/B design after native evaluation feedback is exposed to the model."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import real_memory_pilot_v2 as pilot
from examples.research_comparison import batch, campaign
from examples.research_comparison.protocol import digest

pilot.credentials()
output=pilot.ROOT/'real-memory-pilot-20260916-feedback'
output.mkdir(mode=0o700)
budget={'authorized_total_usd':5000,'prior_reserved_upper_bound_usd':259,'new_upper_bound_usd':32,
        'total_reserved_upper_bound_usd':291,
        'arithmetic':'8 projects x 4 processes x 100000-token envelope x $5/M maximum rate = $16; reserve $32',
        'scope':'same 4 paired Haiku tasks with native CPU feedback; optional memory unchanged; no post-result resampling',
        'price_source':'https://platform.claude.com/docs/en/about-claude/pricing'}
assert budget['total_reserved_upper_bound_usd'] <= budget['authorized_total_usd']
pilot.write(output/'authorization-and-budget.json',budget)
previous=pilot.ROOT/'real-memory-pilot-20260916-admission/haiku45-specification.json'
spec=json.loads(previous.read_bytes())
runtime=campaign.describe()
spec['shared']['environment']=runtime['environment']
plan=spec['protocol']
plan['comparison_id']='real-memory-haiku45-nativefeedback'
plan['shared']['environment']=digest(runtime['environment'])
plan['verifier_sha256']=runtime['verifier_sha256']
for arm in ('A','B'):
    spec['arms'][arm]['runtime']=runtime
    plan['arms'][arm]['artifact_sha256']=runtime['artifact_sha256']
pilot.write(output/'predecessor.json',{'path':str(previous),'sha256':hashlib.sha256(previous.read_bytes()).hexdigest(),
    'unchanged_inputs':['model','tools','task data','instructions','initial history','initial memory','treatment','ordering','repetitions','seeds','limits'],
    'core_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=pilot.CORE,text=True).strip(),
    'pro_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=pilot.PRO,text=True).strip(),
    'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
batch.validate(spec)
pilot.write(output/'haiku45-specification.json',spec)
receipt=batch.execute(spec,output/'haiku45',env=dict(os.environ))
pilot.write(output/'haiku45-receipt.json',receipt)
report=batch.audit(output/'haiku45',receipt['specification_sha256'])
pilot.write(output/'haiku45-report.json',report)
print(json.dumps({'receipt':receipt,'counts':report['counts']}))
