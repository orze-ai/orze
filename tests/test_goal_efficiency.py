"""Goal audits retain failures, batch visibility and pre-confirmation selection."""
import pytest

from orze.research.exploration import run_online
from orze.research.goal_efficiency import incumbent_checkpoints, time_to_goal


def make_trace(tmp_path, scores, workers=1):
    spec = {'problem_id': 'goal', 'protocol_id': 'fixed', 'score_scale': 1,
            'root': {'score': 0, 'artifact': {'source': 'baseline'}, 'feedback': {}},
            'plan': {'branches': 2, 'depth': 3, 'workers': workers, 'calls': len(scores)}}
    values = iter(scores)
    def execute(contexts):
        out = {}
        for context in contexts:
            score = next(values)
            out[context['action']['id']] = {
                'score': score, 'status': 'ok' if score is not None else 'repairable',
                'artifact': {'source': context['action']['id']}, 'feedback': {},
                'cost': None, 'seconds': 1}
        return out
    return run_online(spec, execute_batch=execute, output=tmp_path/'run')


def test_failure_time_and_future_scores_cannot_change_prefix_selection(tmp_path):
    trace = make_trace(tmp_path, [None, .4, .3, .4, .8])
    points = incumbent_checkpoints(trace, [20, 70, 80, 90, 200])
    assert [(p['id'], p['seconds']) for p in points] == [('baseline', 0), ('b1-s0', 70), ('b0-s2', 200)]
    # An intermediate non-incumbent cannot be cherry-picked by its hidden score.
    result = time_to_goal(points, {'baseline': 0, 'b1-s0': .6, 'b0-s2': .2}, target=.5, horizon_seconds=100)
    assert result['achieved'] and result['discovery_seconds'] == result['capped_seconds'] == 70


def test_batch_only_becomes_visible_at_measured_completion(tmp_path):
    trace = make_trace(tmp_path, [.3, .7, .8, .8], workers=2)
    points = incumbent_checkpoints(trace, [110, 250])
    assert [(p['id'], p['seconds']) for p in points] == [('baseline', 0), ('b1-s0', 110), ('b0-s1', 250)]


@pytest.mark.parametrize('score', [None, .49])
def test_unsolved_runs_contribute_full_horizon(score):
    points = [{'id': 'baseline', 'seconds': 0}, {'id': 'attempt', 'seconds': 12}]
    result = time_to_goal(points, {'baseline': 0, 'attempt': score}, target=.5, horizon_seconds=3600)
    assert not result['achieved'] and result['capped_seconds'] == 3600
    assert result['discovery_seconds'] is None


def test_exact_boundary_and_late_success():
    points = [{'id': 'a', 'seconds': 60}]
    assert time_to_goal(points, {'a': .5}, target=.5, horizon_seconds=60)['achieved']
    assert not time_to_goal(points, {'a': .5}, target=.5, horizon_seconds=59)['achieved']


@pytest.mark.parametrize('times', [[1], [20, 19], [1, float('nan')], [1, True]])
def test_missing_or_invalid_clock_is_not_inferred_from_worker_seconds(tmp_path, times):
    with pytest.raises(ValueError):
        incumbent_checkpoints(make_trace(tmp_path, [.3, .6]), times)


def test_missing_confirmation_is_not_silently_dropped():
    with pytest.raises(ValueError):
        time_to_goal([{'id': 'a', 'seconds': 1}], {}, target=.5, horizon_seconds=60)
