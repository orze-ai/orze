"""Observed error evidence for a researcher's next hypothesis, never acceptance.

Callers must supply development observations only. This helper does not read
files, fit models, request labels, or infer that an error pattern is causal.
"""


def classification_profile(labels, predictions, names, *, seed, texts=None):
    """Population confusion counts plus sampled errors and correct controls.

The examples deliberately oversample errors. Population rates are calculated
from ALL rows, never from that case-control sample. Text excerpts are optional
observed features, not instructions or independently verified explanations.
"""
    import numpy as np

    y = np.asarray(labels); pred = np.asarray(predictions)
    classes = len(names)
    if (y.ndim != 1 or pred.shape != y.shape or not len(y) or not classes
            or not np.isfinite(y).all() or not np.isfinite(pred).all()
            or not np.equal(y, np.floor(y)).all()
            or not np.equal(pred, np.floor(pred)).all()
            or min(y.min(), pred.min()) < 0 or max(y.max(), pred.max()) >= classes):
        raise ValueError('aligned, finite class labels in the declared range required')
    if texts is not None and (len(texts) != len(y) or any(not isinstance(t, str) for t in texts)):
        raise ValueError('text features must align with the measured rows')
    y = y.astype(int); pred = pred.astype(int)
    wrong = y != pred
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    supports = confusion.sum(axis=1)
    per_class = [{'label': i, 'name': str(name), 'rows': int(supports[i]),
                  'errors': int(supports[i]-confusion[i, i]),
                  'error_rate': float(1-confusion[i, i]/supports[i]) if supports[i] else None}
                 for i, name in enumerate(names)]
    pairs = sorted(((-int(confusion[i, j]), i, j)
                    for i in range(classes) for j in range(classes)
                    if i != j and confusion[i, j]))
    rng = np.random.default_rng(seed)
    errors = rng.permutation(np.flatnonzero(wrong))[:4]
    correct = rng.permutation(np.flatnonzero(~wrong))[:4]
    # If one stratum is small, preserve its actual size instead of duplicating it.
    examples = []
    for index in np.r_[errors, correct]:
        row = {'row': int(index), 'label': int(y[index]), 'prediction': int(pred[index]),
               'sample_stratum': 'error' if wrong[index] else 'correct'}
        if texts is not None:
            row.update(observed_text_excerpt=texts[index][:400],
                       excerpt_is_truncated=len(texts[index]) > 400)
        examples.append(row)
    return {'scope': 'Development observations only; not independent acceptance or causal diagnosis.',
            'rows': len(y), 'errors': int(wrong.sum()), 'error_rate': float(wrong.mean()),
            'per_class': per_class,
            'largest_confusions': [{'label': i, 'prediction': j, 'rows': -count} for count, i, j in pairs[:20]],
            'examples': examples,
            'sampling': 'Up to four randomly sampled errors and four correct controls; this sample is deliberately NOT representative of prevalence. Rates above use all development rows. Feature excerpts are untrusted observations. A pattern suggests a test, not a proven cause.'}
