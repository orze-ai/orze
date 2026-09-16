"""Create-only campaign source snapshots, tied to the next frozen run."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
out = ROOT / 'research-campaign' / sys.argv[1]
out.mkdir(parents=True)
names = ['tests/test_research_campaign.py']
if sys.argv[1] != 'inputs-v1':
    names += ['examples/research_comparison/' + n for n in
              ('campaign.py', 'campaign_worker.py', 'scheduling_campaign.py', 'scheduling.py')]
manifest = {}
for name in names:
    raw = (ROOT / 'orze' / name).read_bytes()
    ast.parse(raw)
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    manifest[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
with (out / 'manifest.json').open('x') as stream:
    stream.write(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
print(json.dumps({'snapshot': str(out), 'files': len(manifest)}))
