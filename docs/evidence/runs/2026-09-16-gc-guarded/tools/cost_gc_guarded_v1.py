"""Finite CPU/I/O cost measurement; synthetic payloads, no research-gain claim.

Compare atomic/guarded GC on local storage; measure actual CephFS guarded GC.
Separate fabricated, read-only retirement history fixtures measure reader cost
only and are never used as execution authority or real completion evidence.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc

from orze.engine import gc_safety, gc_retirement

BASE = Path(sys.argv[1]).absolute()
LOCAL = Path(sys.argv[2]).absolute()
BASE.mkdir(parents=True, exist_ok=False)
LOCAL.mkdir(parents=True, exist_ok=False)
BLOCK = bytes(range(256)) * 2048  # 512 KiB; hash-known synthetic I/O payload.
report = {'gc': [], 'reader': [], 'files_per_tree': 64, 'bytes_per_file': len(BLOCK),
          'scope': 'Synthetic storage/reader cost; no models, real research conclusions, or production durability claim',
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def save(path, value):
    with path.open('x') as out:
        out.write(json.dumps(value, sort_keys=True, indent=2) + '\n')


def one(root, backend, archive):
    root.mkdir()
    results, checkpoints, cold = (root / x for x in ('results', 'checkpoints', 'archive'))
    task = results / 'idea-cost'
    task.mkdir(parents=True)
    (task / 'metrics.json').write_text('{"status":"FAILED"}')
    source = task / 'overlays' if archive else checkpoints / task.name
    source.mkdir(parents=True)
    for i in range(64):
        (source / str(i)).write_bytes(BLOCK)
    cfg = {'_project_root': str(root), 'results_dir': str(results), 'gc': {'storage_mode': backend}}
    scope = gc_safety.gc_scope(results, cfg, checkpoints_dir=checkpoints, archive_dir=cold)
    actual = gc_safety.attempt_effect_lock
    guards = []
    @contextmanager
    def observed(*args, **kwargs):
        start = time.perf_counter()
        try:
            with actual(*args, **kwargs) as lease:
                yield lease
        finally:
            guards.append(time.perf_counter() - start)
    gc_safety.attempt_effect_lock = observed
    try:
        tracemalloc.start()
        start = time.perf_counter()
        stats = gc_safety.collect(scope, results if archive else checkpoints, set(),
                                  mode='archive' if archive else 'checkpoints')
        seconds = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        gc_safety.attempt_effect_lock = actual
    assert stats['errors'] == 0 and stats['archived_files' if archive else 'deleted'] == 1, stats
    assert not source.exists()
    if archive:
        destination = cold / task.name / 'overlays'
        assert len(list(destination.iterdir())) == 64
        assert all(hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256(BLOCK).digest()
                   for p in destination.iterdir())
    result = {'root': str(root), 'filesystem': subprocess.check_output(['stat','-f','-c','%T',str(root)],text=True).strip(),
              'backend': backend, 'operation': 'archive' if archive else 'delete', 'seconds': seconds,
              'peak_traced_bytes': peak, 'retained_traced_bytes': current, 'guard_seconds': guards, 'stats': stats}
    save(root / 'measurement.json', result)
    report['gc'].append(result)


for repetition in range(2):
    for archive in (False, True):
        for backend in (('atomic','guarded') if repetition == 0 else ('guarded','atomic')):
            one(LOCAL / f'run-{repetition}-{int(archive)}-{backend}', backend, archive)
        one(BASE / f'run-{repetition}-{int(archive)}-guarded', 'guarded', archive)

for parent, label in ((LOCAL, 'local'), (BASE, 'ceph')):
    for count in (0, 1, 128, 1024):
        task = parent / ('synthetic-reader-' + str(count))
        task.mkdir()
        for i in range(count):
            token = f'{i:032x}'
            record = task / '_gc_retirements' / token
            record.mkdir(parents=True)
            intent = json.dumps({'version':1, 'task':str(task), 'token':token,
                'journal':str(task / 'not-executed-journal'), 'plan_sha256':'0'*64},
                sort_keys=True, separators=(',', ':')).encode()
            (record / 'intent.json').write_bytes(intent)
            (record / 'completed.json').write_text(json.dumps({'version':1,
                'intent_sha256':hashlib.sha256(intent).hexdigest()},sort_keys=True,separators=(',', ':')))
        samples = []
        for _ in range(3):
            tracemalloc.start()
            start = time.perf_counter()
            gc_retirement.require_quiet(task)
            seconds = time.perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            samples.append({'seconds':seconds,'peak_traced_bytes':peak})
        report['reader'].append({'fixture':str(task), 'storage':label, 'records':count,
            'samples':samples, 'median_seconds':statistics.median(s['seconds'] for s in samples),
            'scope':'Fabricated read-only metadata, never execution/completion evidence'})

save(BASE / 'result.json', report)
print(json.dumps({'gc_runs':len(report['gc']), 'reader_fixtures':len(report['reader']),
                  'reader_medians':[(r['storage'],r['records'],r['median_seconds']) for r in report['reader']]}))
