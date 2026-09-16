"""Create-only paired changed-source snapshots, before each recorded run."""
import ast
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
output = root / 'gc-guarded' / sys.argv[1]
output.mkdir(parents=True, exist_ok=False)
names = ['src/orze/engine/' + name + '.py' for name in (
    'gc_guarded', 'gc_retirement', 'gc_safety', 'attempt_effect_lock', 'storage_preflight')]
names += ['tests/test_gc_guarded.py']
manifest = {}
for name in names:
    source = root / 'orze' / name
    if not source.exists():
        continue
    raw = source.read_bytes()
    ast.parse(raw)
    target = output / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(raw)
    manifest[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
with (output / 'manifest.json').open('x') as stream:
    stream.write(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
print(json.dumps({'files': len(manifest), 'output': str(output)}))
