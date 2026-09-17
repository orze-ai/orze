"""Search two-model convex mixtures using aligned, verified development predictions.

This solves a finite pool's pairwise quadratic objectives, not general model
selection or future performance. Callers bind row identities and source data.
"""
import math
from itertools import combinations


def mix(left, right, left_weight):
    """Preserve endpoint predictions exactly; interpolate aligned vectors."""
    if type(left_weight) not in (int, float) or not math.isfinite(left_weight) or not 0 <= left_weight <= 1:
        raise ValueError("finite convex weight required")
    if len(left) != len(right):
        raise ValueError("aligned prediction lengths required")
    if left_weight == 1:
        return list(left)
    if left_weight == 0:
        return list(right)
    return [left_weight * a + (1 - left_weight) * b for a, b in zip(left, right)]


def best_pair(row_ids, targets, records):
    """Include all single models and every pair; strict improvement replaces best.

    Records have ``id``, ``row_ids`` and ``prediction``. Supplied order breaks
    exact ties. No fitting or numerical optimizer is needed. Grouped evaluation
    and independent confirmation remain the caller's responsibility.
    """
    n = len(targets)
    if not n or len(row_ids) != n or len(set(row_ids)) != n or not records:
        raise ValueError("nonempty unique evaluation rows and predictions required")
    seen = set()
    for record in records:
        if type(record['id']) is not str or not record['id'] or record['id'] in seen:
            raise ValueError("unique candidate identities required")
        seen.add(record['id'])
        if record['row_ids'] != row_ids or len(record['prediction']) != n:
            raise ValueError("prediction row identities must match exactly")
    for values in [targets] + [r['prediction'] for r in records]:
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("finite numeric targets and predictions required")

    def loss(prediction):
        value = math.fsum((p - y) ** 2 for p, y in zip(prediction, targets)) / n
        if not math.isfinite(value):
            raise ValueError("nonfinite squared error")
        return value

    single = min(records, key=lambda r: loss(r['prediction']))
    best = {'left_id': single['id'], 'right_id': single['id'], 'left_weight': 1.0,
            'loss': loss(single['prediction']), 'prediction': list(single['prediction'])}
    for left, right in combinations(records, 2):
        delta = [a - b for a, b in zip(left['prediction'], right['prediction'])]
        denominator = math.fsum(d * d for d in delta)
        numerator = math.fsum(d * (y - b) for d, y, b in zip(delta, targets, right['prediction']))
        if not math.isfinite(denominator) or not math.isfinite(numerator):
            raise ValueError("nonfinite mixture statistics")
        weight = max(0.0, min(1.0, numerator / denominator)) if denominator else 1.0
        prediction = mix(left['prediction'], right['prediction'], weight)
        candidate_loss = loss(prediction)
        if candidate_loss < best['loss']:
            best = {'left_id': left['id'], 'right_id': right['id'], 'left_weight': weight,
                    'loss': candidate_loss, 'prediction': prediction}
    return {**best, 'unique_candidates': len(records),
            'pairs_evaluated': len(records) * (len(records) - 1) // 2,
            'scope': 'Best measured convex pair in this supplied development pool. '
                     'Reused labels select the mixture; they do not confirm generalization.'}
