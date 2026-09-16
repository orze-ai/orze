"""Create-only source snapshots for closed-service recovery validation."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
out = ROOT / 'service-recovery' / sys.argv[1]
out.mkdir(parents=True)
names = ['tests/test_service_recovery.py', 'tests/test_service_host_product.py',
         'src/orze/engine/controller_handoff.py', 'src/orze/service/recovery.py',
         'src/orze/service/host.py', 'src/orze/service/closed_state.py', 'src/orze/cli.py',
         'tests/test_runtime_lease_publication.py']
manifest = {}
for name in names:
    raw = (ROOT / 'orze' / name).read_bytes()
    ast.parse(raw)
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    manifest[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
with (out / 'manifest.json').open('x') as stream:
    stream.write(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
print(json.dumps({'snapshot': str(out), 'files': len(manifest)}))
