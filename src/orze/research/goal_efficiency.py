"""Time to a declared research goal, assessed on independently scored incumbents.

This is an audit, not a stopping policy: confirmation scores must never feed
back into the exploration that produced these checkpoints. Callers supply
measured completion times, including failed attempts and scheduler overhead.
"""
from __future__ import annotations

import math

from .exploration import validate_trace


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def incumbent_checkpoints(trace, completion_seconds):
    """Choose by development score at each completed batch; ties keep the earlier.

    ``completion_seconds`` contains one measured elapsed time per round. Do not
    pass summed worker durations: parallel workers and policy overhead differ.
    Artifacts are selected without consulting their confirmation performance.
    """
    trace = validate_trace(trace)
    if len(completion_seconds) != len(trace['rounds']):
        raise ValueError('one measured completion time per round is required')
    root = trace['spec']['root']
    best = {'id': 'baseline', 'seconds': 0.0,
            'development_score': root['score'], 'artifact': root['artifact']}
    checkpoints = [best]
    previous = 0.0
    for row, seconds in zip(trace['rounds'], completion_seconds):
        if not _finite(seconds) or seconds < previous:
            raise ValueError('completion times must be finite and nondecreasing')
        previous = seconds
        selected = best
        for node in row['observations']:
            if node['score'] is not None and node['score'] > selected['development_score']:
                selected = {'id': node['id'], 'seconds': seconds,
                            'development_score': node['score'], 'artifact': node['artifact']}
        if selected is not best:
            checkpoints.append(selected)
            best = selected
    return checkpoints


def time_to_goal(checkpoints, confirmation_scores, *, target, horizon_seconds):
    """First selected incumbent meeting a fixed higher-is-better target.

    None denotes failed confirmation. Unsolved runs contribute the full fixed
    horizon to the capped mean, never zero and never a missing observation.
    Returned discovery time excludes subsequent audit/confirmation latency;
    report that separately. This cannot establish an online stopping time.
    """
    if not _finite(target) or not _finite(horizon_seconds) or horizon_seconds <= 0:
        raise ValueError('finite target and positive horizon required')
    if not checkpoints or len({c['id'] for c in checkpoints}) != len(checkpoints):
        raise ValueError('nonempty distinct checkpoints required')
    ids = {c['id'] for c in checkpoints}
    if set(confirmation_scores) != ids:
        raise ValueError('every selected checkpoint needs a confirmation result')
    previous = 0.0
    first = None
    for point in checkpoints:
        seconds = point['seconds']
        score = confirmation_scores[point['id']]
        if (not _finite(seconds) or seconds < previous
                or score is not None and not _finite(score)):
            raise ValueError('invalid checkpoint time or confirmation score')
        previous = seconds
        if first is None and seconds <= horizon_seconds and score is not None and score >= target:
            first = point
    return {'achieved': first is not None,
            'first_id': first['id'] if first is not None else None,
            'discovery_seconds': first['seconds'] if first is not None else None,
            'capped_seconds': first['seconds'] if first is not None else horizon_seconds,
            'target': target, 'horizon_seconds': horizon_seconds}
