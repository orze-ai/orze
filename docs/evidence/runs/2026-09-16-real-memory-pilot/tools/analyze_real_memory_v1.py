"""Read-only pilot summary; billed invoices and missing usage remain unknown."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path('/hot-data/fsx/workspace/erik/orze-p1-continuation-20260914')
FIRST = ROOT / 'real-memory-pilot-20260916'
CURRENT = ROOT / 'real-memory-pilot-20260916-v2'
RATES = {'claude-sonnet-5': (2, 10, .2, 2.5, 4), 'claude-haiku-4-5-20251001': (1, 5, .1, 1.25, 2)}


def read(path):
    return json.loads(path.read_bytes())


def usage(path):
    if not path.exists():
        return {'prepared_attempts': 0, 'returned_attempts': 0, 'unknown_cost_attempts': 0,
                'priced_usage_usd_lower': 0, 'priced_usage_usd_upper': 0, 'input_tokens': 0, 'output_tokens': 0}
    events = [json.loads(line) for line in path.read_text().splitlines()]
    prepared = {r['attempt_id'] for r in events if r['event'] == 'attempt_prepared'}
    finished = [r for r in events if r['event'] == 'attempt_finished']
    result = {'prepared_attempts': len(prepared), 'returned_attempts': sum(r['transport_returned'] for r in finished),
              'unknown_cost_attempts': len(prepared - {r['attempt_id'] for r in finished}),
              'priced_usage_usd_lower': 0, 'priced_usage_usd_upper': 0, 'input_tokens': 0, 'output_tokens': 0}
    for row in finished:
        fields = [row.get(key) for key in ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens')]
        model = row.get('response_model')
        if model not in RATES or any(type(v) is not int or v < 0 for v in fields):
            result['unknown_cost_attempts'] += 1
            continue
        i, o, cr, cw = fields
        ri, ro, rr, rw_low, rw_high = RATES[model]
        result['priced_usage_usd_lower'] += (i*ri + o*ro + cr*rr + cw*rw_low)/1e6
        result['priced_usage_usd_upper'] += (i*ri + o*ro + cr*rr + cw*rw_high)/1e6
        result['input_tokens'] += i + cr + cw
        result['output_tokens'] += o
    return result


def summarize_run(directory, measured):
    record = read(directory/'run.json')
    complete = (directory/'capture.json').exists()
    path = directory/'capture.json' if complete else directory/'project/partial-capture.json'
    captured = read(path) if path.exists() else None
    stats = {'run_id': record['run_id'], 'arm': record['arm'], 'task_id': record['task_id'],
             'repetition': record['repetition'], 'exit_code': record['exit_code'],
             'wall_seconds': record['wall_seconds'], 'status': measured.get('status'),
             'quality': measured.get('quality'), 'metrics': measured.get('metrics'),
             'usage': usage(directory/'project/usage.jsonl')}
    if captured is None:
        return stats
    research = captured['campaign']['research']
    rejects = Counter()
    for row in research:
        rejects.update((row.get('outcome') or {}).get('rejection_reasons', {}))
    observations = captured['campaign'].get('consumption', [])
    duplicate_failures, invalid_seen, valid_scores = 0, set(), []
    for observation in observations:
        evaluated = observation['evaluation']
        raw = captured['artifact_contents'][evaluated['candidate_artifact_id']]
        try:
            candidate = json.loads(raw)
            canonical = json.dumps(sorted((r['job_id'], r['start']) for r in candidate['schedule']))
        except (ValueError, TypeError, KeyError):
            canonical = raw
        verdict = evaluated['verdict']
        if verdict['status'] == 'invalid':
            duplicate_failures += canonical in invalid_seen
            invalid_seen.add(canonical)
        else:
            valid_scores.append(verdict['scheduled_value'])
    memory = captured['database'].get('research_memory_current_v1', [])
    entries = json.loads(memory[0]['document_json'])['entries'] if len(memory) == 1 else None
    stats.update(rounds_started=len(research), accepted=sum((r.get('outcome') or {}).get('accepted_count', 0) for r in research),
                 rejected=sum((r.get('outcome') or {}).get('rejected_count', 0) for r in research),
                 rejection_reasons=dict(rejects), evaluated_candidates=len(observations), valid_candidates=len(valid_scores),
                 best_observed_score=max(valid_scores) if valid_scores else None, repeated_invalid_schedules=duplicate_failures,
                 memory_revision=memory[0]['revision'] if len(memory)==1 else None,
                 memory_entries=len(entries) if entries is not None else None,
                 terminal_step=captured['calls'][-1]['label'] if captured['calls'] else None,
                 terminal_step_error=captured['calls'][-1]['error'] if captured['calls'] else None)
    return stats


def main(publish):
    result = {'scope': 'Two small authored scheduling tasks; exploratory real-model pilot, not cross-domain or long-memory validation.',
              'money_scope': 'Public-price estimates from returned token usage, not provider invoices. Unknown attempts retained separately.',
              'total_budget_usd': 5000, 'conservative_reserved_upper_bound_usd': 179, 'models': {}}
    for model in ('sonnet5','deepseek_flash','haiku45'):
        path = CURRENT/(model+'-report.json')
        if not path.exists():
            if publish: raise RuntimeError('model batch has not completed: '+model)
            continue
        report = read(path)
        rows=[]
        for pointer in (CURRENT/model/'runs.jsonl').read_text().splitlines():
            pointer=json.loads(pointer)
            measured=next(r for r in report['runs'] if r['run_id']==pointer['run_id'])
            rows.append(summarize_run((CURRENT/model/pointer['path']).parent,measured))
        groups=[]
        for arm in ('A','B'):
            group=[r for r in rows if r['arm']==arm]
            groups.append({'arm':arm,'planned':4,'provided':len(group),
                           'confirmed':sum(bool((r['quality'] or {}).get('confirmed')) for r in group),
                           'optimal':sum(bool((r['quality'] or {}).get('confirmed')) and r['quality']['score']==(27 if r['task_id']=='memory-target' else 10) for r in group),
                           'median_wall_seconds_all_runs':median(r['wall_seconds'] for r in group),
                           'prepared_attempts':sum(r['usage']['prepared_attempts'] for r in group),
                           'priced_usage_usd_upper':sum(r['usage']['priced_usage_usd_upper'] for r in group),
                           'unknown_cost_attempts':sum(r['usage']['unknown_cost_attempts'] for r in group)})
        result['models'][model]={'counts':report['counts'],'all_pairs_qualified':report['all_pairs_qualified'],'arms':groups,'runs':rows,
                                 'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    journals=list(FIRST.rglob('usage.jsonl'))+list(CURRENT.rglob('usage.jsonl'))
    all_usage=[usage(path) for path in journals]
    result['all_trials_and_probes_usage']={key:sum(row[key] for row in all_usage) for key in all_usage[0]} if all_usage else {}
    result['journals']=[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in journals]
    if publish:
        with (CURRENT/'independent-summary-v1.json').open('x') as stream:
            json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({**{k:v for k,v in result.items() if k not in ('models','journals')},
                      'models':{k:{'counts':v['counts'],'arms':v['arms']} for k,v in result['models'].items()}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true')
    main(parser.parse_args().publish)
