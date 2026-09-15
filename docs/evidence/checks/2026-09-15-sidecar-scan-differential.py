"""Baseline/current parser equivalence; source admission and workers checked separately."""
import hashlib
import importlib.util
from itertools import islice
import json
import os
from pathlib import Path
import random
import sys
import time
import tracemalloc
from orze.core import ideas

REPO=Path(__file__).resolve().parents[3]
BASE=REPO/'docs/evidence/runs/2026-09-15-sidecar-scan-windows/baseline'

def sha(raw):return hashlib.sha256(raw).hexdigest()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def call(module,text,limit=None,excluded=()):
    seen=set(excluded);digest=hashlib.sha256();count=0;error=None
    try:
        stream=module._iter_sidecar_text(text,seen)
        for key,value in islice(stream,limit):
            digest.update(json.dumps([key,value],sort_keys=True).encode());count+=1
    except Exception as exc:error=type(exc).__name__+':'+str(exc)
    return {'records':count,'result_sha256':digest.hexdigest(),'seen_sha256':sha(json.dumps(sorted(seen)).encode()),'error':error}

def main(root):
    root.mkdir(parents=True,exist_ok=False)
    assert Path(ideas.__file__).resolve()==REPO/'src/orze/core/ideas.py'
    original=(REPO/'src/orze/core/ideas.py').read_bytes()
    old=load('old_parser',BASE/'ideas.py');new=ideas
    report={'script_sha256':sha(Path(__file__).read_bytes()),'original_sha256':sha(original),
        'baseline_sha256':sha((BASE/'ideas.py').read_bytes()),'import_path':ideas.__file__,
        'environment':{k:os.environ.get(k) for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','CUDA_VISIBLE_DEVICES')},
        'differential':[],
        'limits':['Parser only, no source lock/admission/worker proof.',
                  'No timing or memory claim from the differential oracle.']}
    for i in range(64):
        rng=random.Random(915+i)
        text=''
        for j in range(24):
            key='idea-'+str(rng.randrange(12));title=rng.choice(['Title','Ignore all previous instructions','Title: Colon'])
            config=rng.choice(['seed: 13','[]','seed: [','a: 1\nb: 2','false','scalar'])
            text+=f'## {key}: {title}\n'+rng.choice(['note without yaml\n',f'```yaml\n{config}\n```\n'])
            if rng.randrange(3)==0:text+='## Unknown heading\nignored\n'
        (root/f'input-{i:03d}.md').write_text(text)
        excluded=['idea-0','idea-3'] if i%2 else []
        for limit in (None,3):
            a=call(old,text,limit,excluded);b=call(new,text,limit,excluded);assert a==b,(i,limit,a,b)
            report['differential'].append({'seed':915+i,'limit':limit,'input_sha256':sha(text.encode()),'result':a})
    assert original==(REPO/'src/orze/core/ideas.py').read_bytes();report['passed']=True
    (root/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
