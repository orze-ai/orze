"""Recompute acceptance and time-to-goal means from published audit counts."""
from pathlib import Path
import itertools
import json
import math
import statistics
from scipy.stats import t
from metrics import compare

BASE=Path(__file__).resolve().parent
read=lambda name:json.loads((BASE/name).read_text())
rows=read('audit-counts.json');summary=read('summary.json');verification=read('verification.json')
assert len(rows)==16 and len({r['label'] for r in rows})==16
tasks=sorted({r['task'] for r in rows});assert len(tasks)==4
for row in rows:
    for kind in ['baseline','candidate']:
        score=row[kind]
        assert score['words']==sum(r['reference_words'] for r in score['rows'])
        assert score['wer']==sum(r['errors'] for r in score['rows'])/score['words']
    result=compare(row['baseline'],row['candidate'])
    elapsed=row['delivered']-row['started']
    accepted=result['qualified'] and row['final_audit_executed'] and elapsed<=7200
    assert accepted==row['goal_reached']
    assert row['capped_goal_seconds']==(elapsed if accepted else 7200)
problem_means={}
for task in tasks:
    problem_means[task]={}
    for arm in ['control','combined']:
        values=[r for r in rows if r['task']==task and r['arm']==arm]
        assert len(values)==2 and {r['repetition'] for r in values}=={0,1}
        problem_means[task][arm]=statistics.fmean(r['capped_goal_seconds'] for r in values)
for arm in ['control','combined']:
    actual=statistics.fmean(g[arm] for g in problem_means.values())
    assert math.isclose(actual,summary['means'][arm]['capped_goal_seconds'],abs_tol=1e-9)
    rate=statistics.fmean(r['goal_reached'] for r in rows if r['arm']==arm)
    assert rate==summary['means'][arm]['success_rate']
differences=[g['combined']-g['control'] for g in problem_means.values()]
mean=statistics.fmean(differences);half=float(t.ppf(.975,3))*statistics.stdev(differences)/2
p=sum(abs(sum(v*s for v,s in zip(differences,signs))/4)>=abs(mean)-1e-9 for signs in itertools.product([-1,1],repeat=4))/16
expected=verification['combined_minus_control_capped_seconds']
assert math.isclose(mean,expected['mean'],abs_tol=1e-9)
assert all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip([mean-half,mean+half],expected['problem_level_95_t_interval']))
assert p==expected['two_sided_exact_problem_sign_flip_p']
print(json.dumps({'verified_episodes':16,'means':summary['means'],'difference':expected}))
