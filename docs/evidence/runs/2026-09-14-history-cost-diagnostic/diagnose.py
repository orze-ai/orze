"""Metadata-only SQL/canonical-parser cost diagnostic; never executes tasks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import orze
from orze.idea_lake import IdeaLake
from orze.core import cpu_action_budget as budget
from orze.core.integrity import hash_config, load_hashes
from orze.engine.idea_ingress import ingest_ideas_source

ROOT = Path(__file__).resolve().parent
REPO = Path('/hot-data/fsx/workspace/erik/orze-production-validation-2026-09-12.UgS3uV/orze')
assert Path(orze.__file__).is_relative_to(REPO / 'src')
assert importlib.util.find_spec('orze_pro') is None
FILES = ['src/orze/core/cpu_action_budget.py', 'src/orze/engine/idea_ingress.py',
         'src/orze/idea_lake.py', 'src/orze/core/integrity.py',
         'src/orze/core/proposal_admission.py', 'src/orze/core/ideas.py',
         'src/orze/engine/cpu_phase.py', 'src/orze/engine/native_cpu_action.py']
hashes = lambda: {p: hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in FILES}
before = hashes()

def timed(fn, repetitions=3):
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return {'seconds': times, 'median_seconds': statistics.median(times)}

results = []
for n in (100, 1000, 5000):
    project = ROOT / str(n)
    project.mkdir()
    results_dir = project / 'results'
    results_dir.mkdir()
    other_dir = project / 'other-results'
    other_dir.mkdir()
    lake = IdeaLake(project / 'lake.db')
    declaration = {'version': 2, 'resource': 'cpu', 'slots': 1, 'wall_budget_seconds': None}
    scope = budget.initialize(lake, results_dir, declaration)
    other = budget.initialize(lake, other_dir, declaration)
    # DIRECT METADATA FIXTURE ONLY: schema-valid SETTLED-shaped historical
    # entries, but no native attempts, TREE, effect, claims or real settlement.
    rows = []
    for target, prefix in ((scope, 'main'), (other, 'other')):
        for i in range(n):
            task = 'diagnostic-' + prefix + '-' + str(i).zfill(6)
            identity = hashlib.sha256(task.encode()).hexdigest()[:48]
            permit = {'schema': 1, 'budget_scope': target, 'reservation_id': identity,
                      'task_id': task, 'slot': 0, 'wall_limit_seconds': 2,
                      'reserved_nanoseconds': '2000000000'}
            ref = {'task_id': task, 'phase': 'action', 'attempt_id': identity, 'generation': 1}
            rows.append((identity, target['results_dir'], task, 0, budget._json(permit),
                         budget._json(ref), 'SETTLED', '0' * 64))
    lake.conn.executemany('INSERT INTO cpu_action_reservations VALUES (?,?,?,?,?,?,?,?)', rows)
    idea_rows = []
    cache = {}
    for i in range(n):
        task = 'idea-fixture-' + str(i).zfill(6)
        raw = json.dumps({'seed': i})
        fingerprint = hash_config({'seed': i})
        cache[fingerprint] = task
        idea_rows.append((task, 'metadata-only', raw, fingerprint, hashlib.sha256(raw.encode()).hexdigest(), '', 'completed', 'native_cpu_action'))
    lake.conn.executemany('INSERT INTO ideas(idea_id,title,config,config_hash,config_source_sha256,raw_markdown,status,kind) VALUES (?,?,?,?,?,?,?,?)', idea_rows)
    lake.conn.commit()
    cfg = {'ideas_file': str(project / 'ideas.md'), '_orze_dir': str(project / '.orze'),
           '_env_ORZE_RESULTS_DIR': str(results_dir), 'results_dir': str(results_dir)}
    Path(cfg['ideas_file']).write_text('# Ideas\n')
    cache_path = project / '.orze/state/config_hashes.json'
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps(cache, sort_keys=True))
    engine = SimpleNamespace(results_dir=results_dir, lake=lake, active_roles={},
                             _config_override_hash=hash_config,
                             _load_config_hashes=lambda: load_hashes(results_dir, cfg))
    sql = ('SELECT CASE WHEN length(CAST(permit_json AS BLOB))<=16384 THEN permit_json END '
           'FROM main.cpu_action_reservations WHERE scope=? ORDER BY reservation_id')
    query_plan = [list(r) for r in lake.conn.execute('EXPLAIN QUERY PLAN ' + sql, (scope['results_dir'],))]
    measured = {
        'metadata_rows_in_requested_scope': n, 'metadata_rows_in_other_scope': n,
        'idea_history_rows': n, 'config_cache_bytes': cache_path.stat().st_size,
        'totals': timed(lambda: budget._totals(lake.conn, scope)),
        'snapshot': timed(lambda: budget.snapshot(lake, scope)),
        'get_all_ids': timed(lake.get_all_ids),
        'load_config_hashes': timed(lambda: load_hashes(results_dir, cfg)),
        'empty_primary_ingress': timed(lambda: ingest_ideas_source(engine, cfg)),
        'totals_query_plan': query_plan}
    statements = []
    decode_calls = [0]
    original_decode = budget._decode
    def counted(raw):
        decode_calls[0] += 1
        return original_decode(raw)
    lake.conn.set_trace_callback(statements.append)
    budget._decode = counted
    try:
        charged, active = budget._totals(lake.conn, scope)
    finally:
        budget._decode = original_decode
        lake.conn.set_trace_callback(None)
    assert charged == 2_000_000_000 * n and active == {}
    measured['totals_instrumented_sql_count'] = len(statements)
    measured['totals_actual_decode_calls'] = decode_calls[0]
    statements = []
    lake.conn.set_trace_callback(statements.append)
    try:
        assert ingest_ideas_source(engine, cfg) == ({}, [])
    finally:
        lake.conn.set_trace_callback(None)
    measured['empty_ingress_actual_sql'] = statements
    # Scratch-only index changes neither code nor historical ledger bytes.
    lake.conn.execute('CREATE INDEX diagnostic_scope_reservation ON cpu_action_reservations(scope,reservation_id)')
    lake.conn.commit()
    measured['private_scope_index_query_plan'] = [list(r) for r in lake.conn.execute('EXPLAIN QUERY PLAN ' + sql, (scope['results_dir'],))]
    measured['totals_with_private_index'] = timed(lambda: budget._totals(lake.conn, scope))
    # Real targeted API with a missing hash; all rows already have identities.
    measured['indexed_no_match_lookup'] = timed(lambda: lake.find_admitted_config_hashes({'f' * 64}))
    lake.close()
    results.append(measured)

report = {'schema': 1, 'source_root': str(REPO), 'fixture_root': str(ROOT),
          'before': before, 'after': hashes(), 'rows': results,
          'limits': ['Only newly created /tmp SQLite and text fixtures; no real experiment history.',
                     'Synthetic SETTLED-shaped rows measure metadata readers only; no TREE/effect/settlement proof fabricated or claimed.',
                     'Times are warm local /tmp microdiagnostics, not deployed filesystem throughput or scientific benefit.',
                     'Transparent _decode wrapper counts real parser calls and delegates unchanged body; timing runs do not use that wrapper.',
                     'Temporary index exists only in scratch DB and does not establish production migration or compatibility.',
                     'No worker, CLI, GPU, provider, Pro runtime, source edit or test edit.']}
assert report['before'] == report['after']
(ROOT/'report.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
print(json.dumps(report, sort_keys=True))
