"""Audit actual native artifacts; producer/model labels cannot certify quality."""
import copy
import hashlib
import json

import pytest

from examples.holdout.testing import TASKS
from examples.research_comparison import scheduling

pytest_plugins = ['examples.holdout.testing']


def ref(row):
    return {k: row[k] for k in ('task_id','phase','attempt_id','generation')}


def inputs(run, task=None, confirmation=True, protocol='schedule-feasibility-v1'):
    attempts=run['database']['execution_attempts']
    selected=next(row for row in attempts if row['task_id']==(task or TASKS['recovered_v1']))
    replica=json.loads(run['database']['replication_requests'][0]['record_json']) if confirmation else None
    return dict(instance=run['cfg']['action_domain']['config']['instance'],protocol=protocol,
        expected_attempt_refs=[ref(row) for row in attempts],selected_ref=ref(selected),
        confirmation_ref=ref(next(row for row in attempts if row['task_id']==replica['task_id'])) if replica else None)


def test_raw_candidate_quality_failed_cost_and_protocol_coverage(holdout_runs):
    run=holdout_runs['workflow'];result=scheduling.audit_scheduling(run,**inputs(run))
    measured=result['measurement']
    assert measured['status']=='completed'
    assert measured['quality']['valid'] and measured['quality']['confirmed']
    assert measured['quality']['score']>0
    assert measured['metrics']['native_actions']==7
    assert measured['metrics']['analysis_actions']==5
    assert measured['metrics']['reserved_seconds']==14
    assert measured['native_outcomes']=={'completed':6,'failed':1,'interrupted':0}
    assert measured['observations']=={'valid':3,'invalid':1,'unknown':0}
    for key in ('provider_calls','provider_tokens','provider_cost_usd','gpu_seconds',
                'worker_cpu_seconds','worker_wall_seconds','first_valid_consumed_seconds','confirmed_selection_seconds'):
        assert measured['metrics'][key] is None
    assert result['confirmation_scope']=='independent_evaluator_attempt_same_candidate'
    assert result['new_research_evidence'] is False and result['campaign_identity_verified'] is False


def test_valid_zero_remains_valid_without_confirmation(holdout_runs):
    run=holdout_runs['workflow']
    result=scheduling.audit_scheduling(run,**inputs(run,TASKS['baseline_v1'],False))['measurement']
    assert result['quality']['valid'] and not result['quality']['confirmed']
    assert result['quality']['score']==0 and result['metrics']['reserved_seconds']==14


def test_protocol_change_retains_invalid_quality_and_all_cost(holdout_runs):
    run=holdout_runs['workflow']
    result=scheduling.audit_scheduling(run,**inputs(run,TASKS['challenger_v2'],False,'schedule-feasibility-v2'))['measurement']
    assert result['status']=='failed' and not result['quality']['valid']
    assert result['quality']['score'] is None and result['metrics']['native_actions']==7


def test_failed_evaluator_has_no_scientific_score(holdout_runs):
    run=holdout_runs['workflow']
    result=scheduling.audit_scheduling(run,**inputs(run,TASKS['failed_v1'],False))['measurement']
    assert result['status']=='failed' and not result['quality']['valid']
    assert result['quality']['score'] is None and result['metrics']['reserved_seconds']==14


def test_same_evaluator_is_not_independent_confirmation(holdout_runs):
    run=holdout_runs['workflow'];scope=inputs(run);scope['confirmation_ref']=scope['selected_ref']
    with pytest.raises(ValueError):scheduling.audit_scheduling(run,**scope)


@pytest.mark.parametrize('change', ['candidate_bytes','declared_score','result_score','source_binding',
    'omit_failure','duplicate_attempt','terminal_charge','unsettled','native_closure','controller_closure',
    'clock_order','domain_identity','artifact_membership','observation_identity'])
def test_corrupted_evidence_cannot_qualify(holdout_runs,change):
    original=holdout_runs['workflow'];scope=inputs(original);run=copy.deepcopy(original)
    db=run['database']
    if change=='candidate_bytes':
        artifact=next(a for a in run['artifacts'] if a['logical_name']=='candidate')
        run['artifact_contents'][artifact['artifact_id']]+=' '
    elif change in ('declared_score','source_binding','observation_identity'):
        row=db['research_observations'][0];value=json.loads(row['record_json'])
        if change=='declared_score':value['values']['value']=999999
        elif change=='source_binding':next(iter(value['input_artifact_bindings'].values()))['content_sha256']='0'*64
        else:value['evaluator']['generation']+=1
        row['record_json']=json.dumps(value)
    elif change=='result_score':
        row=db['research_artifacts'][1];artifact=json.loads(row['record_json'])
        assert artifact['logical_name']=='evaluation'
        value=json.loads(run['artifact_contents'][artifact['artifact_id']]);value['verdict']['scheduled_value']=999999
        raw=json.dumps(value);run['artifact_contents'][artifact['artifact_id']]=raw
        artifact['content_sha256']=hashlib.sha256(raw.encode()).hexdigest();artifact['size_bytes']=len(raw.encode())
        row['record_json']=json.dumps(artifact)
    elif change=='omit_failure':
        db['execution_attempts']=[r for r in db['execution_attempts'] if r['task_id']!=TASKS['failed_v1']]
    elif change=='duplicate_attempt':db['execution_attempts'].append(copy.deepcopy(db['execution_attempts'][0]))
    elif change=='terminal_charge':db['cpu_action_reservations'][0]['terminal_sha256']='0'*64
    elif change=='unsettled':db['cpu_action_reservations'][0]['state']='ACTIVE'
    elif change=='native_closure':
        row=db['execution_attempts'][0];value=json.loads(row['terminal_json']);value['process_tree']['wait_proof']='unknown';row['terminal_json']=json.dumps(value)
    elif change=='controller_closure':run['calls'][0]['controller_supervision']['closure']['forced_cleanup']=True
    elif change=='clock_order':run['calls'].reverse()
    elif change=='domain_identity':
        row=db['execution_attempts'][0];value=json.loads(row['binding_json']);value['domain_run']['domain_config_sha256']='0'*64;row['binding_json']=json.dumps(value)
    elif change=='artifact_membership':
        row=db['execution_attempts'][0];value=json.loads(row['terminal_json']);value['artifact_ids']=[];row['terminal_json']=json.dumps(value)
    with pytest.raises(ValueError):scheduling.audit_scheduling(run,**scope)


def test_expected_instance_and_attempt_inventory_are_independent_inputs(holdout_runs):
    run=holdout_runs['workflow'];scope=inputs(run)
    scope['instance']=copy.deepcopy(scope['instance']);scope['instance']['capacity']+=1
    with pytest.raises(ValueError):scheduling.audit_scheduling(run,**scope)
    scope=inputs(run);scope['expected_attempt_refs'].pop()
    with pytest.raises(ValueError):scheduling.audit_scheduling(run,**scope)


def test_auditor_does_not_mutate_its_callers(holdout_runs):
    run=holdout_runs['workflow'];scope=inputs(run);before=copy.deepcopy((run,scope))
    result=scheduling.audit_scheduling(run,**scope)
    assert (run,scope)==before
    result['measurement']['quality']['comparison_key']['instance_sha256']='changed'
    assert (run,scope)==before
