"""Account for all feedback diagnostics and fixed paired reruns, including failures."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import analyze_real_memory_v2 as prior

ROOT = Path(__file__).resolve().parent
BATCH = ROOT/'real-memory-pilot-20260916-feedback'


def read(path):
    return json.loads(path.read_bytes())


def diagnostic(root):
    result = read(root/'result.json')
    prompts = []
    for path in sorted(root.glob('response-*.json')):
        record = read(path)
        snapshot = json.loads(re.search(r'<research_snapshot>\s*(.*?)\s*</research_snapshot>', record['prompt'], re.S).group(1))
        memory = json.loads(re.search(r'<stored_research_memory>\s*(.*?)\s*</stored_research_memory>', record['prompt'], re.S).group(1))
        prompts.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'provider_wall_seconds': record['seconds'], 'memory': memory,
            'feedback': [{'idea_id': r['idea_id'], 'availability': r['availability'], 'reason': r.get('reason'),
                          'observations': r.get('observations', [])} for r in snapshot['report_evidence']['records']]})
    return {'directory': root.name, 'scope': result['scope'], 'responses': result['responses'],
            'elapsed_seconds': result['elapsed_seconds'], 'outcomes': result['outcomes'],
            'final_memory': result['memory_rows'], 'usage': prior.usage(root/'usage.jsonl'), 'prompts': prompts}


def main():
    report = read(BATCH/'haiku45-report.json')
    assert report['counts'] == {'missing_runs': 0, 'planned_runs': 8, 'provided_runs': 8}
    rows = []
    for line in (BATCH/'haiku45/runs.jsonl').read_text().splitlines():
        pointer = json.loads(line)
        directory = (BATCH/'haiku45'/pointer['path']).parent
        measured = next(r for r in report['runs'] if r['run_id'] == pointer['run_id'])
        row = prior.summarize_run(directory, measured)
        path = directory/'capture.json'
        if not path.exists(): path = directory/'project/partial-capture.json'
        capture = read(path)
        row['retained_duplicate_proposals'] = len(capture['campaign'].get('retained_duplicates', []))
        row['round_outcomes'] = dict(Counter((r.get('outcome') or {}).get('reason', 'unavailable') for r in capture['campaign']['research']))
        row['invalid_reasons'] = dict(Counter(r['evaluation']['verdict']['reason_code'] for r in capture['campaign']['consumption'] if r['evaluation']['verdict']['status'] == 'invalid'))
        row['memory_documents'] = [json.loads(r['document_json']) for r in capture['database'].get('research_memory_current_v1', [])]
        rows.append(row)
    arms = []
    for arm in 'AB':
        group = [r for r in rows if r['arm'] == arm]
        arms.append({'arm': arm, 'planned': 4, 'provided': len(group),
            'confirmed': sum(bool(r['quality']['confirmed']) for r in group),
            'optimal': sum(bool(r['quality']['confirmed']) and r['quality']['score'] == (27 if r['task_id'] == 'memory-target' else 10) for r in group),
            'complete_wall_seconds': sum(r['wall_seconds'] for r in group),
            'priced_usage_usd': sum(r['usage']['priced_usage_usd_upper'] for r in group),
            'prepared_attempts': sum(r['usage']['prepared_attempts'] for r in group),
            'unknown_cost_attempts': sum(r['usage']['unknown_cost_attempts'] for r in group),
            'invalid_candidates': sum(r['evaluated_candidates']-r['valid_candidates'] for r in group),
            'repeated_invalid_schedules': sum(r['repeated_invalid_schedules'] for r in group),
            'retained_duplicate_proposals': sum(r['retained_duplicate_proposals'] for r in group),
            'projects_with_memory_updates': sum((r['memory_revision'] or 0) > 1 for r in group)})
    pairs = []
    for task in ('memory-target','memory-negative-control'):
        for repetition in (0,1):
            a,b=[next(r for r in rows if (r['task_id'],r['repetition'],r['arm']) == (task,repetition,arm)) for arm in 'AB']
            equal=all(r['quality']['confirmed'] for r in (a,b)) and a['quality']['score']==b['quality']['score']
            pairs.append({'task_id':task,'repetition':repetition,'equal_confirmed_quality':equal,
                'A':{k:a[k] for k in ('quality','wall_seconds','usage','memory_revision')},
                'B':{k:b[k] for k in ('quality','wall_seconds','usage','memory_revision')},
                'equal_quality_B_minus_A':{'wall_seconds':b['wall_seconds']-a['wall_seconds'],
                    'priced_usage_usd':b['usage']['priced_usage_usd_upper']-a['usage']['priced_usage_usd_upper']} if equal else None})
    diagnostic_roots=[ROOT/f'real-memory-counterexample-diagnostic-20260916-v{v}' for v in (1,2,3,4,5,6)]
    diagnostics=[diagnostic(p) for p in diagnostic_roots]
    previous=read(ROOT/'real-memory-pilot-20260916-admission/independent-summary-v4.json')['all_trials_and_probes_usage']
    additional=[prior.usage(p/'usage.jsonl') for p in diagnostic_roots]+[r['usage'] for r in rows]
    total={k:previous[k]+sum(r[k] for r in additional) for k in previous}
    result={'schema':1,'scope':'Diagnostic mechanism checks and one fixed four-pair Haiku replication; not long-horizon or cross-domain research efficacy.',
        'diagnostics':diagnostics,'batch':{'arms':arms,'runs':rows,'pairs':pairs,'all_pairs_qualified':report['all_pairs_qualified']},
        'cumulative_usage':total,'cumulative_reserve_usd':291,'authorized_cap_usd':5000,
        'money_scope':'Returned usage priced using public rates, not invoices; historical unknown attempts retained.',
        'timing_scope':'Full fixed-round batch costs include every failure; consumption timestamps refer to the collector, not model reads.'}
    with (BATCH/'summary.json').open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2,sort_keys=True)
    print(json.dumps({'arms':arms,'pairs':pairs,'cumulative_usage':total},ensure_ascii=False,indent=2))


if __name__ == '__main__':main()
