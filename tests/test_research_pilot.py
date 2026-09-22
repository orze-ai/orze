"""Pilots preserve observations and never turn rejection into a scored method."""
from types import SimpleNamespace
import pytest
from orze.research.pilot import run_training
from orze.research.source import parse_python_proposal


def forbidden(*args):
    pytest.fail('rejected pilot must not build or train')


def test_rejected_pilot_keeps_findings_without_model_or_score():
    candidate = SimpleNamespace(pilot=lambda _: {'continue': False, 'findings': {'sample_loss': .4}, 'model': object()},
                                build=forbidden, train=forbidden)
    result = run_training(candidate, object(), allow_pilot=True)
    assert result.kind == 'analysis' and result.model is None
    assert result.findings == {'pilot': {'sample_loss': .4}, 'continued': False}
    assert not hasattr(result, 'score')


def test_continuation_reuses_partial_training_without_rewriting_pilot_evidence():
    partial = {'steps': 2}; findings = {'steps': 2}; calls = []
    def train(api, model):
        assert model is partial
        model['steps'] += 3; findings['steps'] = 999; calls.append('train')
        return {'steps': model['steps']}
    candidate = SimpleNamespace(pilot=lambda _: {'continue': True, 'findings': findings, 'model': partial},
                                build=forbidden, train=train)
    result = run_training(candidate, object(), allow_pilot=True)
    assert result.kind == 'method' and result.model is partial and calls == ['train']
    assert result.findings['pilot']['steps'] == 2 and result.findings['training']['steps'] == 5


def test_feature_is_opt_in_and_legacy_findings_are_unchanged():
    model = {}; candidate = SimpleNamespace(pilot=forbidden, build=lambda _: model, train=lambda a,m:{'loss': 1.})
    result = run_training(candidate, object())
    assert result.model is model and result.findings == {'loss': 1.}


def test_no_partial_model_builds_and_trains_once():
    events = []
    candidate = SimpleNamespace(pilot=lambda _: {'continue': True, 'findings': {}},
        build=lambda _:events.append('build') or {}, train=lambda a,m:events.append('train') or {})
    assert run_training(candidate, object(), allow_pilot=True).kind == 'method'
    assert events == ['build', 'train']


@pytest.mark.parametrize('decision', [
    {'continue': 1, 'findings': {}}, {'continue': True},
    {'continue': False, 'findings': {'x': float('nan')}},
    {'continue': False, 'findings': {}, 'score': 1.},
])
def test_invalid_pilot_cannot_become_negative_research_evidence(decision):
    candidate = SimpleNamespace(pilot=lambda _:decision, build=forbidden, train=forbidden)
    with pytest.raises(ValueError): run_training(candidate, object(), allow_pilot=True)


def test_execution_failure_propagates_without_automatic_retry():
    calls = []
    def pilot(api):
        calls.append(1); raise RuntimeError('measurement implementation broke')
    with pytest.raises(RuntimeError, match='implementation broke'):
        run_training(SimpleNamespace(pilot=pilot), object(), allow_pilot=True)
    assert calls == [1]


def test_pilot_does_not_relax_the_training_findings_contract():
    candidate = SimpleNamespace(pilot=lambda _: {'continue': True, 'findings': {}},
        build=lambda _: {}, train=lambda a,m:['not a findings object'])
    with pytest.raises(ValueError, match='findings must be an object'):
        run_training(candidate, object(), allow_pilot=True)


def test_findings_limit_covers_combined_pilot_and_training():
    candidate = SimpleNamespace(pilot=lambda _: {'continue': True, 'findings': {'data': 'a'*60}},
        build=lambda _: {}, train=lambda a,m:{'data': 'b'*60})
    with pytest.raises(ValueError, match='byte limit'):
        run_training(candidate, object(), allow_pilot=True, max_findings_bytes=100)


def test_existing_source_contract_accepts_optional_pilot_without_execution():
    source = 'raise RuntimeError("must not execute here")\ndef pilot(api): pass\ndef build(api): pass\ndef train(api,model): pass\ndef predict(api,model,features): pass'
    assert parse_python_proposal(source, complete=True)['kind'] == 'method'
