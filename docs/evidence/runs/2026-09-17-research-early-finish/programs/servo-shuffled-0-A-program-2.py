import numpy as np
from sklearn.model_selection import GroupKFold, KFold


def _cands(y):
    m = float(np.mean(y))
    med = float(np.median(y))
    out = [("mean", m)]
    for q in (0.02, 0.05, 0.10):
        lo = float(np.quantile(y, q))
        hi = float(np.quantile(y, 1.0 - q))
        out.append(("win_%.2f" % q, float(np.mean(np.clip(y, lo, hi)))))
    for lam in (0.10, 0.25):
        out.append(("shr_%.2f" % lam, (1.0 - lam) * m + lam * med))
    return out


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    if Xe.ndim == 1:
        Xe = Xe.reshape(1, -1)
    n_eval = int(Xe.shape[0])
    n = int(y.shape[0])
    if n_eval == 0:
        return np.zeros(0, dtype=float)
    if n == 0 or not np.all(np.isfinite(y)):
        return np.zeros(n_eval, dtype=float)
    base = float(np.mean(y))
    fallback = np.full(n_eval, base, dtype=float)
    if n < 20:
        return fallback
    if train_groups is not None:
        g = np.asarray(train_groups).ravel()
        if g.shape[0] != n:
            g = np.arange(n)
    else:
        g = np.arange(n)
    names = [c[0] for c in _cands(y)]
    err = dict((k, 0.0) for k in names)
    Y2 = y.reshape(-1, 1)
    try:
        uniq = np.unique(g)
        if uniq.shape[0] >= 4:
            k = int(min(5, uniq.shape[0]))
            folds = list(GroupKFold(n_splits=k).split(Y2, y, groups=g))
        else:
            folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(Y2))
    except Exception:
        return fallback
    used = 0
    for tr, te in folds:
        if tr.shape[0] < 10 or te.shape[0] < 1:
            continue
        yte = y[te]
        for name, val in _cands(y[tr]):
            err[name] += float(np.sum((yte - val) ** 2))
        used += int(te.shape[0])
    if used < 15 or err["mean"] <= 0.0:
        return fallback
    best = min(names, key=lambda k: err[k])
    if err[best] >= 0.98 * err["mean"]:
        return fallback
    full = dict(_cands(y))
    val = float(full[best])
    if not np.isfinite(val):
        return fallback
    lo = float(np.min(y))
    hi = float(np.max(y))
    if val < lo:
        val = lo
    if val > hi:
        val = hi
    return np.full(n_eval, val, dtype=float)
