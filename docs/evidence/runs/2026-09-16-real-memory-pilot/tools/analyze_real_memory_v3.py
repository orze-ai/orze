"""Compare the fixed, completed rerun with the unmodified baseline summary."""
import hashlib
import json
from collections import Counter
from statistics import median
import analyze_real_memory_v2 as before

FIXED = before.ROOT / 'real-memory-pilot-20260916-fixed'


def main():
    baseline = before.read(before.CURRENT / 'independent-summary-v2.json')
    report = before.read(FIXED / 'haiku45-report.json')
    assert report['counts'] == {'missing_runs': 0, 'planned_runs': 8, 'provided_runs': 8}
    rows = []
    for line in (FIXED / 'haiku45/runs.jsonl').read_text().splitlines():
        pointer = json.loads(line)
        directory = (FIXED / 'haiku45' / pointer['path']).parent
        measured = next(r for r in report['runs'] if r['run_id'] == pointer['run_id'])
        row = before.summarize_run(directory, measured)
        capture_path = directory / 'capture.json'
        if not capture_path.exists():
            capture_path = directory / 'project/partial-capture.json'
        capture = before.read(capture_path)
        row['round_outcomes'] = dict(Counter((r.get('outcome') or {}).get('reason', 'unavailable')
                                            for r in capture['campaign']['research']))
        rows.append(row)
    arms = []
    for arm in ('A', 'B'):
        group = [r for r in rows if r['arm'] == arm]
        arms.append({'arm': arm, 'planned': 4, 'provided': len(group),
            'confirmed': sum(bool(r['quality']['confirmed']) for r in group),
            'optimal': sum(bool(r['quality']['confirmed']) and r['quality']['score'] ==
                           (27 if r['task_id'] == 'memory-target' else 10) for r in group),
            'median_wall_seconds_all_runs': median(r['wall_seconds'] for r in group),
            'prepared_attempts': sum(r['usage']['prepared_attempts'] for r in group),
            'priced_usage_usd_upper': sum(r['usage']['priced_usage_usd_upper'] for r in group),
            'unknown_cost_attempts': sum(r['usage']['unknown_cost_attempts'] for r in group),
            'repeated_invalid_schedules': sum(r['repeated_invalid_schedules'] for r in group),
            'projects_with_memory_updates': sum(r['memory_revision'] > 1 for r in group)})
    pairs = []
    for task in ('memory-target', 'memory-negative-control'):
        for repetition in (0, 1):
            a, b = [next(r for r in rows if (r['task_id'], r['repetition'], r['arm']) ==
                         (task, repetition, arm)) for arm in ('A', 'B')]
            equal_quality = all(r['quality']['confirmed'] for r in (a, b)) and a['quality']['score'] == b['quality']['score']
            item = {'task_id': task, 'repetition': repetition, 'equal_confirmed_quality': equal_quality,
                    'A': {k: a[k] for k in ('run_id', 'quality', 'wall_seconds', 'metrics', 'usage')},
                    'B': {k: b[k] for k in ('run_id', 'quality', 'wall_seconds', 'metrics', 'usage')}}
            # This descriptive subset is not overall qualification: billed fees,
            # remote GPU and some worker measurements are still unavailable.
            if equal_quality:
                item['descriptive_B_minus_A'] = {
                    'wall_seconds': b['wall_seconds'] - a['wall_seconds'],
                    'estimated_model_cost_usd': b['usage']['priced_usage_usd_upper'] - a['usage']['priced_usage_usd_upper']}
            pairs.append(item)
    journals = sorted(p for root in (before.FIRST, before.CURRENT, FIXED) for p in root.rglob('usage.jsonl'))
    quantities = [before.usage(p) for p in journals]
    result = {'scope': baseline['scope'], 'money_scope': baseline['money_scope'],
        'total_budget_usd': 5000, 'conservative_reserved_upper_bound_usd': 213,
        'baseline_summary_sha256': hashlib.sha256((before.CURRENT/'independent-summary-v2.json').read_bytes()).hexdigest(),
        'baseline_models': baseline['models'],
        'after_fix_haiku45': {'counts': report['counts'], 'all_pairs_qualified': report['all_pairs_qualified'],
            'arms': arms, 'runs': rows, 'pairs': pairs,
            'report_sha256': hashlib.sha256((FIXED/'haiku45-report.json').read_bytes()).hexdigest()},
        'all_trials_and_probes_usage': {k: sum(q[k] for q in quantities) for k in quantities[0]},
        'journals': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in journals]}
    with (FIXED/'independent-summary-v3.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'after_fix_arms': arms, 'all_usage': result['all_trials_and_probes_usage'],
                      'pairs': [{k: v for k, v in p.items() if k not in ('A', 'B')} for p in pairs]}, indent=2))


if __name__ == '__main__':
    main()
