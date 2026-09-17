import numpy as np
from sklearn.model_selection import KFold, GroupKFold


def _const(y, q):
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return 0.0
    if q >= 1.0:
        return float(np.mean(y))
    cap = float(np.quantile(y, q))
    return float(np.mean(np.clip(y, 0.0, cap)))


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    y = np.asarray(y_train, dtype=float).ravel()
    n = int(y.shape[0])
    Xe = np.asarray(X_eval, dtype=float)
    m = int(Xe.shape[0])
    qs = [1.0, 0.995, 0.99, 0.98, 0.95, 0.9]
    best_q = 1.0
    best_score = float('inf')
    folds = []
    if n >= 20:
        dummy = np.zeros((n, 1), dtype=float)
        if train_groups is not None:
            g = np.asarray(train_groups).ravel()
            if g.shape[0] == n:
                ng = int(np.unique(g).size)
                k = int(min(5, ng))
                if k >= 2:
                    folds = list(GroupKFold(n_splits=k).split(dummy, y, g))
        if not folds:
            k = int(min(5, n))
            if k >= 2:
                folds = list(KFold(n_splits=k, shuffle=True, random_state=int(seed)).split(dummy))
    for q in qs:
        total = 0.0
        count = 0
        for tr, te in folds:
            if tr.size < 5 or te.size == 0:
                continue
            c = _const(y[tr], q)
            if not np.isfinite(c):
                continue
            d = y[te] - c
            total += float(np.sum(d * d))
            count += int(te.size)
        if count > 0:
            s = total / count
            if s < best_score - 1e-12:
                best_score = s
                best_q = q
    c = _const(y, best_q)
    if not np.isfinite(c):
        c = float(np.mean(y)) if n > 0 else 0.0
    if not np.isfinite(c):
        c = 0.0
    if c < 0.0:
        c = 0.0
    return np.full(m, float(c), dtype=float)
