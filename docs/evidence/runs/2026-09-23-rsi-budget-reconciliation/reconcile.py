"""Retire the still-reserved parent study; retain every unknown dollar bound.

No new pricing assumptions: all known tokens retain the old $50/M ceiling.
The one unknown invocation retains its complete original $10 allowance.
"""
from decimal import Decimal as D
from pathlib import Path
import hashlib
import json
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
PROOFS = {}


def read(relative):
    p = ROOT/relative
    PROOFS[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def same(a, b): assert abs(D(str(a))-D(str(b))) < D('0.00000001'), (a,b)


def main():
    original = read('research-parent-20260918/budget.json')
    delivered = read('research-parent-final-delivery-20260918.json')
    assert original['new_reserve_usd'] == 280
    assert original['new_release_usd'] == 0 and delivered['new_release_usd'] == 0
    assert not delivered['live_research_execution_handles'] and not delivered['live_execution_handles']
    assert delivered['new_unknown_cost'] == 1
    # Follow every subsequent budget transition. All retirements concern OTHER
    # named pools; the parent $280 has remained in the running global balance.
    context = read('research-context-20260918/budget.json')
    assert context['new_release_usd'] == 0
    same(context['effective_reserved_usd'], D(str(original['effective_reserved_usd']))+D(str(context['new_reserve_usd'])))
    long = read('dream-rsi-long-20260918/budget.json')
    assert Path(long['retired_study']).name == 'research-ml-20260917-v2'
    same(long['prior_effective_reserved_usd'],context['effective_reserved_usd'])
    same(long['effective_reserved_usd'],D(str(long['prior_effective_reserved_usd']))-D(str(long['released_usd']))+D(str(long['new_study_reserved_usd'])))
    current = long['effective_reserved_usd']
    for directory, previous_pool in [('dream-rsi-combined-20260920',975),('real-gpu-research-20260921',600),('real-gpu-efficiency-20260922',300)]:
        row = read(directory+'/budget.json')
        same(row['prior_global_reserved_usd'],current)
        assert row.get('previous_pool_usd',row.get('prior_closed_pool_usd')) == previous_pool
        same(row['effective_global_reserved_usd'],D(str(current))-D(str(row['released_usd']))+D(str(row['new_pool_usd'])))
        current = row['effective_global_reserved_usd']
    cross = read('cross-domain-research-20260922/budget.json')
    assert cross['previous_pool_usd'] == 250
    same(cross['prior_effective_reserved_usd'],D(str(current))-D(str(cross['released_usd'])))
    same(cross['effective_global_reserved_usd'],D(str(cross['prior_effective_reserved_usd']))+D(str(cross['new_pool_usd'])))
    current = cross['effective_global_reserved_usd']
    for directory, pool in [('rsi-efficiency-20260922',18),('rsi-effort-20260922',12.5),('rsi-evidence-20260922',7.5)]:
        allocation=read(directory+'/budget.json');release=read(directory+'/budget-release.json')
        same(allocation.get('previous_effective_global_reserved_usd',allocation.get('prior_effective_reserved_usd')),current)
        assert allocation['new_pool_usd'] == pool
        same(allocation['effective_global_reserved_usd'],D(str(current))+D(str(pool)))
        same(release['previous_effective_global_reserved_usd'],allocation['effective_global_reserved_usd'])
        assert release['released_pool_usd'] == pool
        same(release['released_unused_usd'],D(str(pool))-D(str(release['retained_upper_usd'])))
        same(release['effective_global_reserved_usd'],D(str(allocation['effective_global_reserved_usd']))-D(str(release['released_unused_usd'])))
        current=release['effective_global_reserved_usd']
    previous_path = 'cross-domain-research-20260922/budget-release.json'
    previous = read(previous_path)
    assert previous['released_pool_usd'] == 160
    same(previous['previous_effective_global_reserved_usd'],current)
    same(previous['effective_global_reserved_usd'],D(str(current))-D(str(previous['released_unused_usd'])))
    current = D(str(previous['effective_global_reserved_usd']))
    records=[];known_tokens=0;unknown=0;call_ids=set()
    for arm in ('opus-A','opus-B','fable-A','fable-B'):
        folder=ROOT/'research-parent-20260918'/arm
        assert len(list(folder.glob('usage-*.jsonl'))) == 7
        for step in range(1,8):
            usage=folder/f'usage-{step}.jsonl'
            PROOFS[str(usage.relative_to(ROOT))]=hashlib.sha256(usage.read_bytes()).hexdigest()
            rows=[json.loads(line) for line in usage.read_text().splitlines()]
            starts=[r for r in rows if r['event']=='call_started']
            ends=[r for r in rows if r['event']=='call_finished']
            attempts=[r for r in rows if r['event']=='attempt_finished']
            assert len(starts)==len(ends)==len(attempts)==1
            event=attempts[0];call=starts[0]['call_id']
            assert call not in call_ids;call_ids.add(call)
            assert all(r['call_id']==call and r['pid']==starts[0]['pid'] for r in rows)
            assert not Path(f"/proc/{starts[0]['pid']}").exists()
            assert event['attempt']==1 and ends[0]['attempts']==1
            response_files=list(folder.glob(f'response-{step}-*.json'))
            assert len(response_files)==1
            response=read(str(response_files[0].relative_to(ROOT)))
            token_count=event.get('total_tokens')
            if token_count is None:
                unknown+=1;retained=D('10')
            else:
                assert event['status']=='complete' and event['transport_returned']
                assert ends[0]['status']=='complete'
                assert response['provider_result']['status']=='complete' and response['response']
                assert event['response_model']==event['model']==starts[0]['model']
                assert event['input_total_tokens']==event['input_tokens']+event['cache_read_tokens']+event['cache_write_tokens']
                assert token_count==event['input_total_tokens']+event['output_tokens']
                assert 0 <= token_count <= 200000
                known_tokens+=token_count;retained=D(token_count)*D(50)/D(1000000)
            records.append({'project':arm,'step':step,'call_id':call,'model':event['model'],
                'known_tokens':token_count,'retained_upper_usd':float(retained),'process_absent':True})
    assert len(records)==28 and unknown==1
    retained=sum(D(str(r['retained_upper_usd'])) for r in records)
    released=D(280)-retained;effective=current-released
    result={'created':time.time(),'retired_pool':'research-parent-20260918',
        'retired_pool_must_not_resume':True,'previous_ledger_path':str(ROOT/previous_path),
        'previous_ledger_sha256':PROOFS[previous_path],
        'previous_effective_global_reserved_usd':float(current),'released_pool_usd':280,
        'closed_known_calls':27,'known_tokens':known_tokens,'all_token_upper_usd_per_million':50,
        'unknown_invocations':unknown,'unknown_full_original_bound_retained_usd':10,
        'retained_upper_usd':float(retained),'released_unused_usd':float(released),
        'effective_global_reserved_usd':float(effective),'unallocated_authorized_usd':float(D(5000)-effective),
        'authorized_cap_usd':5000,'unknown_historical_records_not_released':25,
        'research_model_requests':0,'records':records,'proofs':PROOFS,
        'reconcile_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (BASE/'budget-release.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','proofs')},indent=2))


if __name__=='__main__': main()
