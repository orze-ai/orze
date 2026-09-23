"""Read completed episodes; independently rescore stored predictions only."""
from pathlib import Path
import ast
import argparse
import hashlib
import json
import statistics
import time

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--study', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
BASE = args.study.resolve()
OUT = args.output.resolve()
assert OUT != BASE and BASE not in OUT.parents, 'Keep analysis outside the frozen study'
LABELS = ['abalone-1-control', 'abalone-1-candidate',
          'agnews-1-control', 'agnews-1-candidate',
          'sarcos-0-control', 'sarcos-0-candidate',
          'sarcos-1-control', 'sarcos-1-candidate']
hashes = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    raw = path.read_bytes()
    hashes[str(path.relative_to(BASE))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


rows, checks = [], []
for episode in LABELS:
    directory = BASE / 'episodes' / episode
    complete = read(directory / 'completed.json')
    read(directory / 'goal-delivered.json')
    action = complete['state']['delivered']['action']
    source = BASE / 'calls' / episode / action / 'candidate.py'
    hashes[str(source.relative_to(BASE))] = sha(source)
    assert sha(source) == complete['state']['delivered']['source_sha256']
    world = f"{complete['task']}-{complete['repetition']}"
    reference = read(BASE / 'references' / f'{world}.json')
    task = read(BASE / 'data' / world / 'task.json')
    row = {'episode': episode, 'selected_action': action, 'completed': complete,
           'source_sha256': sha(source),
           'model_authored_rationale': ast.get_docstring(ast.parse(source.read_text())),
           'stages': {}}
    for split, suffix in [('development', ''), ('confirmation', '-confirmation'),
                          ('audit', '-audit')]:
        stage_dir = directory / (action + suffix)
        if not (stage_dir / 'result.json').exists():
            continue
        result = read(stage_dir / 'result.json')
        prediction = np.asarray(result['prediction'], dtype=np.float64)
        label_path = BASE / 'data' / world / split / 'labels.npy'
        hashes[str(label_path.relative_to(BASE))] = sha(label_path)
        labels = np.load(label_path, allow_pickle=False).astype(np.float64)
        baseline = np.asarray(reference['predictions'][split], dtype=np.float64)
        assert labels.shape == prediction.shape == baseline.shape
        assert np.isfinite(prediction).all() and np.isfinite(baseline).all()
        if task['metric'] == 'mse':
            loss = float(np.mean((prediction - labels) ** 2))
            base = float(np.mean((baseline - labels) ** 2))
        elif task['metric'] == 'error_rate':
            assert np.equal(prediction, np.rint(prediction)).all()
            loss = float(np.mean(prediction != labels))
            base = float(np.mean(baseline != labels))
        else:
            raise ValueError(task['metric'])
        stage = {'findings': result.get('findings', {}), 'candidate_loss': loss,
                 'baseline_loss': base, 'relative_gain': 1 - loss / base}
        for name in ['timing.json', 'closure.json', 'comparison.json']:
            if (stage_dir / name).exists():
                stage[name] = read(stage_dir / name)
        if split == 'development':
            score = read(stage_dir / 'development-score.json')
            assert abs(score['loss'] - loss) <= 1e-10
        else:
            comparison = stage['comparison.json']
            assert abs(comparison['candidate_loss'] - loss) <= 1e-10
            assert abs(comparison['baseline_loss'] - base) <= 1e-10
        checks.append({'episode': episode, 'split': split, 'rows': len(labels),
                       'candidate_loss': loss, 'baseline_loss': base,
                       'relative_gain': 1 - loss / base,
                       'meets_effect_floor': 1 - loss / base >= task['goal_relative_reduction']})
        row['stages'][split] = stage
    row['attempts'] = []
    for path in sorted((BASE / 'calls' / episode).glob('*/outcome.json')):
        outcome = read(path)
        row['attempts'].append({
            'action': path.parent.name, 'status': outcome.get('status'),
            'cost': outcome.get('cost'), 'seconds': outcome.get('seconds'),
            'feedback': {k: v for k, v in outcome.get('feedback', {}).items()
                         if k in ['valid', 'error', 'reason', 'unused_after_delivery',
                                  'source_sha256', 'score']}})
    rows.append(row)

all_rows = [json.loads(p.read_text()) for p in (BASE / 'episodes').glob('*/completed.json')]
aggregates = {}
for tasks, title in [({'abalone'}, 'abalone'), ({'sarcos'}, 'sarcos'),
                     ({'abalone', 'sarcos'}, 'regression'), ({'agnews'}, 'agnews')]:
    aggregates[title] = {}
    for arm in ['control', 'candidate']:
        subset = [r for r in all_rows if r['task'] in tasks and r['arm'] == arm]
        assert len(subset) == 2 * len(tasks)
        for item in subset:
            read(BASE / 'episodes' / item['label'] / 'completed.json')
        aggregates[title][arm] = {
            'closed': len(subset), 'accepted': sum(r['goal_reached'] for r in subset),
            'mean_capped_goal_minutes': statistics.mean(r['capped_goal_seconds'] for r in subset) / 60}

report = {'created': time.time(),
          'scope': 'Eight completed episodes; selected source and stored predictions only. '
                   'No candidate code executed/imported, no model requests, no GPU executions. '
                   'Rationale is model-authored, not causal proof.',
          'frozen_plan_sha256': sha(BASE / 'plan.json'), 'rows': rows,
          'complete_task_or_family_aggregates': aggregates, 'evidence_sha256': hashes,
          'whole_campaign_verified': False, 'default_promotion_authorized': False}
review_path = OUT / 'review-003.json'
review_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
check = {'created': time.time(), 'valid': True,
         'scope': 'Independent NumPy point-loss rescoring; bootstrap intervals, all attempts '
                  'and whole campaign closure are not reverified here.',
         'review_sha256': sha(review_path), 'checker_sha256': sha(Path(__file__)),
         'rows': checks, 'whole_campaign_verified': False,
         'research_requests': 0, 'gpu_executions': 0}
(OUT / 'point-loss-check-003.json').write_text(json.dumps(check, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'reviewed': len(rows), 'prediction_sets_checked': len(checks),
                  'aggregates': aggregates}, ensure_ascii=False))
