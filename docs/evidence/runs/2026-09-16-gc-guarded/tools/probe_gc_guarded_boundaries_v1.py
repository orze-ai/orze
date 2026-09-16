"""Isolated real failure paths: archive FD cleanup and bounded history scanning.

Only new local fixture files and this process's descriptors are inspected.
No process control, existing scope, manager, model or production service.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

from orze.engine import gc_guarded, gc_retirement, gc_safety
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt

base = Path(sys.argv[1]).absolute()
base.mkdir(parents=True, exist_ok=False)
project = base / 'archive-failure'
task = project / 'results/idea-owned'
task.mkdir(parents=True)
(task / 'metrics.json').write_text('{"status":"FAILED"}')
(task / 'scratch.pt').write_bytes(b'owned archive input')
destination = project / 'archive/idea-owned/scratch.pt'
cfg = {'_project_root':str(project), 'results_dir':str(task.parent), 'gc':{'storage_mode':'guarded'}}
scope = gc_safety.gc_scope(task.parent, cfg, archive_dir=project / 'archive')


def descriptors():
    values = {}
    for name in os.listdir('/proc/self/fd'):
        try:
            values[int(name)] = os.readlink('/proc/self/fd/' + name)
        except FileNotFoundError:
            pass
    return values


actual_open = gc_guarded._open_directory
def denied(path, *args, **kwargs):
    if Path(path) == destination.parent and not kwargs.get('create', False):
        raise PermissionError('owned destination parent became unavailable')
    return actual_open(path, *args, **kwargs)

before = descriptors()
gc_guarded._open_directory = denied
try:
    stats = gc_safety.collect(scope, task.parent, set(), mode='archive')
finally:
    gc_guarded._open_directory = actual_open
after = descriptors()
leaked = {fd:path for fd,path in after.items() if fd not in before}
assert stats['errors'] == 1 and stats['archived_files'] == 0
assert (task / 'scratch.pt').read_bytes() == b'owned archive input'
assert all(path == str(task) for path in leaked.values()), leaked
for fd in leaked:
    os.close(fd)  # This probe opened them; do not retain diagnostic FD leaks.

reader = base / 'reader-only'
history = reader / '_gc_retirements'
history.mkdir(parents=True)
for number in range(2000):
    (history / f'{number:032x}').mkdir()
actual_listdir = os.listdir
materialized = []
def observed(path):
    result = actual_listdir(path)
    if Path(path) == history:
        materialized.append(len(result))
    return result
os.listdir = observed
try:
    try:
        gc_retirement.require_quiet(reader)
    except AttemptEffectInDoubt:
        refused = True
    else:
        refused = False
finally:
    os.listdir = actual_listdir
assert refused
report = {'archive_errors':stats['errors'], 'leaked_descriptors':len(leaked),
          'leaked_owned_paths':list(leaked.values()), 'reader_entries':2000,
          'eager_directory_entries':materialized, 'reader_refused':refused,
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope':'Actual isolated error mechanics; not a research experiment'}
with (base / 'result.json').open('x') as out:
    out.write(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(report))
raise SystemExit(0 if not leaked and max(materialized, default=0) <= 1025 else 1)
