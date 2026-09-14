"""Optional proposal-history reads are queries, never admission authority."""
import copy

import pytest

from orze.core import cpu_execution as execution
from orze.core import research_interfaces as api
from test_cpu_evidence_paging_policy import policy


@pytest.mark.parametrize('size', [1, 32])
def test_explicit_proposal_page_size_is_accepted_and_fingerprinted(policy, size):
    cfg, _ = policy
    original = execution.execution_fingerprint(cfg)
    cfg['action_policy']['proposal_page_size'] = size
    assert execution.action_policy(cfg) == cfg['action_policy']
    assert execution.execution_fingerprint(cfg) != original


@pytest.mark.parametrize('version', [1, 2])
def test_legacy_policy_shape_does_not_gain_proposal_fields(policy, version):
    cfg, _ = policy
    cfg['action_policy']['version'] = version
    if version == 1:
        cfg['action_policy'].pop('evidence_page_size')
    original = copy.deepcopy(cfg['action_policy'])
    assert execution.action_policy(cfg) == original
    assert 'proposal_page_size' not in api.BoundPolicy(api.capture_interfaces(cfg)).declaration


@pytest.mark.parametrize('size', [None, True, 0, 33, 1.0, '1'])
def test_proposal_page_size_is_exact_and_bounded(policy, size):
    cfg, _ = policy
    cfg['action_policy']['proposal_page_size'] = size
    with pytest.raises(ValueError):
        execution.action_policy(cfg)


@pytest.mark.parametrize('kind', ['ReadProposals', 'SelectProposals'])
def test_omitted_opt_in_cannot_gain_query_permission(policy, kind):
    cfg, selected = policy
    selected['decision'] = ({'kind': kind, 'cursor': 'next'} if kind == 'ReadProposals'
                            else {'kind': kind, 'request_ids': ['r-0']})
    with pytest.raises(api.ResearchInterfaceError):
        api.BoundPolicy(api.capture_interfaces(cfg)).decide(
            {'queue': [], 'proposal_page': {'next_cursor': 'next'}}, {})


def test_cursor_is_checked_against_private_snapshot(policy, monkeypatch):
    cfg, selected = policy
    cfg['action_policy']['proposal_page_size'] = 2
    selected['decision'] = {'kind': 'ReadProposals', 'cursor': 'real'}
    bound = api.BoundPolicy(api.capture_interfaces(cfg))
    snapshot = {'queue': [], 'proposal_page': {'next_cursor': 'real'}}
    assert bound.decide(snapshot, {}) == selected['decision']
    state = api._context(bound._context)
    def forge(view, budget):
        view['proposal_page']['next_cursor'] = 'forged'
        return {'kind': 'ReadProposals', 'cursor': 'forged'}
    monkeypatch.setitem(state, 'decide', forge)
    with pytest.raises(api.ResearchInterfaceError):
        bound.decide(snapshot, {})
    assert snapshot['proposal_page']['next_cursor'] == 'real'


@pytest.mark.parametrize('ids', [[], ['r', 'r'], [True], ['bad/name'], ['r'] * 33])
def test_selection_requires_unique_bounded_request_ids(policy, ids):
    cfg, selected = policy
    cfg['action_policy']['proposal_page_size'] = 1
    selected['decision'] = {'kind': 'SelectProposals', 'request_ids': ids}
    with pytest.raises(api.ResearchInterfaceError):
        api.BoundPolicy(api.capture_interfaces(cfg)).decide({'queue': []}, {})


def test_selection_accepts_ids_not_supplied_outcomes(policy):
    cfg, selected = policy
    cfg['action_policy']['proposal_page_size'] = 1
    selected['decision'] = {'kind': 'SelectProposals', 'request_ids': ['r-2', 'r-1']}
    bound = api.BoundPolicy(api.capture_interfaces(cfg))
    assert bound.decide({'queue': []}, {}) == selected['decision']
    selected['decision']['results'] = [{'status': 'inserted'}]
    with pytest.raises(api.ResearchInterfaceError):
        bound.decide({'queue': []}, {})
