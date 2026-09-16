"""Create-only actual GC probe: three modes, fresh owned CephFS paths only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from orze.engine.gc_safety import collect, gc_scope

base = Path(sys.argv[1]).absolute()
base.mkdir(parents=True, exist_ok=False)
records = []
for mode in ('checkpoints', 'results', 'archive'):
    root = base / mode
    results, checkpoints, archive = [root / name for name in ('results', 'checkpoints', 'archive')]
    task = results / 'idea-disposable'
    task.mkdir(parents=True)
    (task / 'metrics.json').write_text('{"status":"FAILED"}')
    checkpoint = checkpoints / task.name
    checkpoint.mkdir(parents=True)
    (checkpoint / 'weights.pt').write_bytes(b'owned checkpoint')
    (task / 'scratch.pt').write_bytes(b'owned result')
    if mode == 'archive':
        (task / 'overlays/nested').mkdir(parents=True)
        (task / 'overlays/nested/data').write_bytes(b'owned nested archive')
    cfg = {'_project_root': str(root), 'results_dir': str(results), 'gc': {'storage_mode': 'guarded'}}
    scope = gc_scope(results, cfg, checkpoints_dir=checkpoints, archive_dir=archive)
    stats = collect(scope, checkpoints if mode == 'checkpoints' else results, set(), mode=mode)
    remaining = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(root.rglob('*')) if p.is_file()}
    records.append({'mode': mode, 'stats': stats, 'remaining': remaining})
report = {'filesystem': subprocess.check_output(['stat','-f','-c','%T',str(base)], text=True).strip(),
          'records': records, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (base / 'result.json').open('x') as output:
    output.write(json.dumps(report, sort_keys=True, indent=2) + '\n')
print(json.dumps({'filesystem': report['filesystem'], 'operations': [(r['mode'],r['stats']) for r in records]}))
raise SystemExit(0 if all(r['stats']['errors'] == 0 for r in records) else 1)
