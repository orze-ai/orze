import numpy as np
from sklearn.model_selection import KFold, GroupKFold


def _const(y, q):
    if q >= 1.0:
        return float(np.mean(y))
    cap = float(np.quantile(y, q))
    return float(np.mean(np.minimum(y, cap)))


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    m = int(Xe.shape[0]) if Xe.ndim > 0 else 0
    n = int(y.shape[0])
    if n == 0:
        return np.zeros(m, dtype=float)
    qs = [1.0, 0.99, 0.975, 0.95, 0.9, 0.85, 0.8]
    best = 1.0
    if n >= 20:
        splits = None
        if train_groups is not None:
            g = np.asarray(train_groups).ravel()
            if g.shape[0] == n:
                ng = int(np.unique(g).shape[0])
                if ng >= 5:
                    try:
                        splits = list(GroupKFold(n_splits=5).split(y, y, g))
                    except Exception:
                        splits = None
        if splits is None:
            splits = list(KFold(n_splits=5, shuffle=True,
                                random_state=int(seed)).split(y))
        scores = np.zeros(len(qs), dtype=float)
        ok = True
        for tr, va in splits:
            ytr = y[tr]
            yva = y[va]
            if ytr.shape[0] < 4 or yva.shape[0] < 1:
                ok = False
                break
            for j, q in enumerate(qs):
                c = _const(ytr, q)
                if not np.isfinite(c):
                    ok = False
                    break
                d = yva - c
                scores[j] += float(np.dot(d, d))
            if not ok:
                break
        if ok and np.all(np.isfinite(scores)):
            best = qs[int(np.argmin(scores))]
    c = _const(y, best)
    if not np.isfinite(c):
        c = float(np.mean(y))
    if not np.isfinite(c):
        c = 0.0
    if c < 0.0:
        c = 0.0
    return np.full(m, float(c), dtype=float)
