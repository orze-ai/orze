import json
import threading

import pytest

from orze.research.execution import run_prepared
from orze.research.exploration import validate_trace


def spec(**plan):
    return {'problem_id':'fixture','protocol_id':'fixed','score_scale':1,
            'root':{'score':0,'artifact':{},'feedback':{}},
            'plan':dict(branches=2,depth=3,calls=6,**plan)}


def outcome(score=1, cost=1):
    return {'score':score,'status':'ok' if score is not None else 'blocked',
            'artifact':{},'feedback':{},'cost':cost,'seconds':1}


def test_ready_delivery_precedes_other_request_and_preserves_charge(tmp_path):
    slow_started=threading.Event();release=threading.Event();delivered=threading.Event()
    calls=[];evaluated=[];settled=[]
    def prepare(context):
        aid=context['action']['id'];calls.append(aid)
        assert context['history']==[]
        if aid=='b0-s0':
            slow_started.set();assert release.wait(5)
        else:assert slow_started.wait(5)
        settled.append(aid)
        return {'request':aid,'cost':1.25}
    def evaluate(context,value):
        assert not release.is_set()
        assert context['action']['id']=='b1-s0'
        evaluated.append(context['action']['id'])
        (tmp_path/'delivered.json').write_text(json.dumps({'action':value['request']}))
        delivered.set();release.set()
        return outcome(cost=value['cost'])
    def unused(context,value):
        assert (tmp_path/'delivered.json').exists()
        return dict(outcome(None,value['cost']),artifact={'request':value['request']})
    original=spec()
    try:
        trace=run_prepared(original,prepare,evaluate,output=tmp_path/'online',
                           finished=delivered.is_set,unused=unused)
    finally:release.set()
    assert 'workers' not in original['plan']
    assert trace['spec']['plan']['workers']==2
    assert set(calls)==set(settled)=={'b0-s0','b1-s0'}
    assert evaluated==['b1-s0']
    assert trace['metrics']['calls']==2 and trace['metrics']['cost']==2.5
    nodes=trace['rounds'][0]['observations']
    assert nodes[0]['feedback']['unused_after_delivery'] and nodes[0]['score'] is None
    assert trace['stop_reason']=='Verified research delivery completed'
    validate_trace(trace)


@pytest.mark.parametrize('workers',[1,2])
def test_explicit_capacity_call_budget_and_branch_isolation(tmp_path,workers):
    calls=[];evaluations=[]
    plan=spec(workers=workers);plan['plan']['calls']=3
    def prepare(context):
        calls.append(context['action']['id'])
        assert all(node['branch']==context['action']['branch'] for node in context['history'])
        assert len(context['history'])==context['action']['step']
        return context['action']['step']+1
    def evaluate(context,value):
        evaluations.append(context['action']['id'])
        assert threading.current_thread() is threading.main_thread()
        return outcome(value)
    trace=run_prepared(plan,prepare,evaluate,output=tmp_path/'online')
    assert len(calls)==len(evaluations)==trace['metrics']['calls']==3
    assert trace['spec']['plan']['workers']==workers
    assert all(len(row['observations'])<=workers for row in trace['rounds'])
    validate_trace(trace)


def test_failed_provider_is_not_retried_and_other_request_settles(tmp_path):
    both_started=threading.Barrier(2);calls=[];settled=[]
    def prepare(context):
        aid=context['action']['id'];calls.append(aid);both_started.wait(timeout=5)
        if aid=='b0-s0':raise RuntimeError('uncertain provider')
        settled.append(aid);return aid
    with pytest.raises(RuntimeError,match='uncertain provider'):
        run_prepared(spec(),prepare,lambda *_:outcome(),output=tmp_path/'online')
    assert set(calls)=={'b0-s0','b1-s0'} and settled==['b1-s0']
    assert json.loads((tmp_path/'online/failure.json').read_text())['retry_allowed'] is False
    assert not (tmp_path/'online/trace.json').exists()


def test_terminal_before_launch_does_not_issue_a_request(tmp_path):
    def forbidden(*_):raise AssertionError('must not be called')
    trace=run_prepared(spec(),forbidden,forbidden,output=tmp_path/'online',
                       finished=lambda:True,unused=forbidden)
    assert trace['metrics']['calls']==0


def test_finished_requires_cost_preserving_unused_handler(tmp_path):
    with pytest.raises(ValueError,match='together'):
        run_prepared(spec(),lambda c:c,lambda *_:outcome(),output=tmp_path/'online',finished=lambda:False)
    assert not (tmp_path/'online').exists()


def test_invalid_python_consumes_one_attempt_and_feedback_reaches_its_branch(tmp_path):
    from orze.research.source import parse_python_proposal
    plan=spec(workers=1);plan['plan']['calls']=3
    source='def build(api): pass\ndef train(api, model): pass\ndef predict(api, model, features): pass'
    seen=[]
    def prepare(context):
        seen.append(context)
        return {'text':'' if context['action']['id']=='b0-s0' else source,'cost':.5}
    def evaluate(context,response):
        try:
            parsed=parse_python_proposal(response['text'],complete=True)
        except (ValueError,SyntaxError) as exc:
            return dict(outcome(None,response['cost']),status='repairable',feedback={'error':str(exc)})
        return dict(outcome(cost=response['cost']),artifact=parsed)
    trace=run_prepared(plan,prepare,evaluate,output=tmp_path/'online')
    assert trace['metrics']['calls']==3 and trace['metrics']['cost']==1.5
    assert seen[1]['action']['id']=='b1-s0' and seen[1]['history']==[]
    assert seen[2]['action']['id']=='b0-s1'
    assert seen[2]['history'][0]['status']=='repairable'
    assert trace['metrics']['best_score']==1
