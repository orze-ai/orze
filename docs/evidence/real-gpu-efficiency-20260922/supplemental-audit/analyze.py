"""Read-only audit of completed episodes; overlapping clocks are not added."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import statistics
import time

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'real-gpu-efficiency-20260922'
metrics=None


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clock_partition(start, end, model_intervals, gpu_intervals):
    """Partition real elapsed seconds; overlapping requests count only once."""
    assert end>=start
    clipped=[]
    for kind,intervals in [('model',model_intervals),('gpu',gpu_intervals)]:
        for left,right in intervals:
            assert right>=left
            left=max(left,start);right=min(right,end)
            if right>left:clipped.append((kind,left,right))
    boundaries=sorted({start,end}|{v for _,left,right in clipped for v in (left,right)})
    totals={key:0.0 for key in ['model_only','gpu_only','overlap','other']}
    for left,right in zip(boundaries,boundaries[1:]):
        midpoint=(left+right)/2
        active={kind for kind,a,b in clipped if a<=midpoint<b}
        key='overlap' if len(active)==2 else ('model_only' if 'model' in active else 'gpu_only' if 'gpu' in active else 'other')
        totals[key]+=right-left
    assert abs(sum(totals.values())-(end-start))<1e-5
    return totals


def audit_episode(folder):
    row=read(folder/'completed.json');start=row['started'];delivered=row['state']['delivered'];end=delivered['finished']
    assert read(folder/'goal-delivered.json')==delivered and end<=row['finished']
    world=f"{row['task']}-{row['repetition']}";data=BASE/'features'/world
    reference=read(BASE/'references-v2'/f'{world}.json')
    trace=read(folder/'online/trace.json');nodes=[n for batch in trace['rounds'] for n in batch['observations']]
    assert len(nodes)==row['metrics']['calls']
    comparisons=[];development_scores=0
    for node in nodes:
        if node['status']=='ok' and node['score'] is not None:
            p=folder/node['id'];prediction=read(p/'result.json')['prediction']
            actual=metrics.score(read(data/'development/records.json'),prediction)
            assert actual==read(p/'development-score.json') and -actual['wer']==node['score']
            development_scores+=1
    for p in folder.glob('*/comparison.json'):
        split='audit' if p.parent.name.endswith('audit') else 'confirmation_'+p.parent.name.rsplit('-',1)[1]
        records=read(data/split/'records.json');prediction=read(p.with_name('result.json'))['prediction']
        actual=metrics.compare(metrics.score(records,reference['predictions'][split]),metrics.score(records,prediction))
        assert actual==read(p)
        comparisons.append((split,actual))
    final=[value for split,value in comparisons if split=='audit'];assert len(final)<=1
    accepted=bool(final and final[0]['qualified']);assert accepted==delivered['accepted']
    reached=bool(accepted and end-start<=7200);assert reached==row['goal_reached']
    assert row['capped_goal_seconds']==(end-start if reached else 7200)
    if delivered['action']!='baseline':
        assert digest(folder/delivered['action']/'weights.pt')==delivered['checkpoint_sha256']
        assert digest(BASE/'calls'/f"{row['label']}-{delivered['action']}"/'candidate.py')==delivered['source_sha256']
    model_intervals=[];gpu_intervals=[];errors=Counter();unused=[];provider_seconds=0
    for node in nodes:
        call=BASE/'calls'/f"{row['label']}-{node['id']}";out=read(call/'outcome.json')
        for key in ['artifact','cost','feedback','score','seconds','status']:assert node[key]==out[key]
        closed=read(call/'process-closed.json');assert closed['exit_code']==0
        response=read(call/'response.json');reservation=read(call/'reservation.json')
        model_intervals.append((reservation['created'],response['finished']))
        provider_seconds+=response['finished']-response['started']
        if out['feedback'].get('unused_after_delivery'):
            assert not (folder/node['id']).exists()
            unused.append({'action':node['id'],'usage_estimate_usd':out['cost'],
                           'request_seconds':response['finished']-response['started']})
        elif out['status']!='ok':errors[out['feedback'].get('error',out['status'])]+=1
    for p in folder.glob('*/dispatch.json'):
        dispatch=read(p);closed=read(p.with_name('closure.json'))
        assert not closed['state']['Running']
        gpu_intervals.append((dispatch['started'],closed['finished']))
    return {'label':row['label'],'task':row['task'],'repetition':row['repetition'],'arm':row['arm'],
        'goal_reached':reached,'capped_goal_seconds':row['capped_goal_seconds'],
        'delivery_elapsed_seconds':end-start,'cleanup_seconds':row['finished']-end,
        'time_partition_seconds':clock_partition(start,end,model_intervals,gpu_intervals),
        'final_relative_wer_gain':final[0]['relative_wer_gain'] if final else None,
        'final_wer':final[0]['candidate_wer'] if final else None,
        'calls':len(nodes),'errors':dict(errors),'unused_proposals':unused,
        'cumulative_provider_seconds':provider_seconds,'rescored_predictions':development_scores+len(comparisons),
        'completed_sha256':digest(folder/'completed.json')}


def main():
    global BASE,metrics
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--study',type=Path,default=BASE);args=parser.parse_args()
    BASE=args.study.resolve()
    spec=importlib.util.spec_from_file_location('frozen_efficiency_metrics',BASE/'metrics.py')
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    # Snapshot only complete, immutable episode results. Never infer a full mean
    # from whichever episodes happen to finish first.
    folders=sorted(p.parent for p in (BASE/'episodes').glob('*/completed.json'))
    rows=[audit_episode(folder) for folder in folders]
    final_verified=False
    if len(rows)==16 and (BASE/'verification-process-closed.json').exists():
        final_verified=read(BASE/'verification-process-closed.json')['exit_code']==0 and read(BASE/'verification.json')['valid']
    arms={}
    for arm in ['control','improved']:
        group=[r for r in rows if r['arm']==arm]
        errors=Counter()
        for r in group:errors.update(r['errors'])
        arms[arm]={'completed':len(group),'accepted':sum(r['goal_reached'] for r in group),
            'invalid_attempts':sum(errors.values()),'errors':dict(errors),
            'unused_proposals':sum(len(r['unused_proposals']) for r in group),
            'time_partition_seconds':{key:sum(r['time_partition_seconds'][key] for r in group)
                for key in ['model_only','gpu_only','overlap','other']}}
        if final_verified:
            tasks=read(BASE/'plan.json')['tasks']
            arms[arm]['mean_capped_goal_seconds']=statistics.fmean(statistics.fmean(r['capped_goal_seconds'] for r in group if r['task']==task) for task in tasks)
            assert arms[arm]['mean_capped_goal_seconds']==read(BASE/'summary.json')['means'][arm]['capped_goal_seconds']
    result={'created':time.time(),'completed_snapshot':len(rows),'full_campaign_verified':bool(final_verified),
        'independently_rescored_predictions':sum(r['rescored_predictions'] for r in rows),
        'arms':arms,'episodes':rows,
        'limits':['Partial snapshots omit ongoing episodes and must not select a default or compare aggregate means.',
            'Time categories partition actual delivery wall time per episode; summing episodes is not campaign duration.',
            'Model intervals measure outstanding requests, not server inference utilization.',
            'GPU intervals measure container lifetime, not active GPU kernels.',
            'This supplementary audit does not replace the frozen full input, process, and scientific verification.']}
    with args.output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ['episodes','limits']},ensure_ascii=False))

if __name__=='__main__':main()
