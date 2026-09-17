import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.ensemble import RandomForestRegressor

S_GRID = (0.4, 0.55, 0.7, 0.85, 1.0)
W_GRID = (0.0, 0.2, 0.4)


def _signal(Xtr, ytr, Xev, seed):
    ylog = np.log1p(np.maximum(ytr, 0.0))
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=8,
                               max_features=0.5, random_state=seed, n_jobs=1)
    rf.fit(Xtr, ylog)
    pe = np.expm1(rf.predict(Xev))
    pt = np.expm1(rf.predict(Xtr))
    return pe - float(np.mean(pt))


def _splits(n, groups, seed):
    if groups is not None:
        g = np.asarray(groups).ravel()
        if g.shape[0] == n:
            k = min(5, int(np.unique(g).shape[0]))
            if k >= 2:
                return list(GroupKFold(n_splits=k).split(np.zeros((n, 1)), None, g))
    k = min(5, n)
    if k < 2:
        return []
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros((n, 1))))


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    n = X.shape[0]
    mu = float(np.mean(y)) if n > 0 else 0.0
    s, w = 1.0, 0.0
    try:
        err = np.zeros((len(S_GRID), len(W_GRID)), dtype=float)
        folds = 0
        for tr, va in _splits(n, train_groups, seed):
            if tr.shape[0] < 10 or va.shape[0] < 1:
                continue
            m = float(np.mean(y[tr]))
            sig = _signal(X[tr], y[tr], X[va], seed)
            if not np.all(np.isfinite(sig)):
                sig = np.zeros(va.shape[0], dtype=float)
            for i, sc in enumerate(S_GRID):
                for j, wc in enumerate(W_GRID):
                    p = np.maximum(sc * m + wc * sig, 0.0)
                    err[i, j] += float(np.mean((p - y[va]) ** 2))
            folds += 1
        if folds > 0 and np.all(np.isfinite(err)):
            bi, bj = np.unravel_index(int(np.argmin(err)), err.shape)
            s, w = S_GRID[bi], W_GRID[bj]
    except Exception:
        s, w = 1.0, 0.0
    base = max(s * mu, 0.0)
    pred = np.full(Xe.shape[0], base, dtype=float)
    if w > 0.0:
        try:
            add = _signal(X, y, Xe, seed)
            if np.all(np.isfinite(add)):
                pred = pred + w * add
        except Exception:
            pass
    pred = np.asarray(pred, dtype=float).ravel()
    pred = np.maximum(pred, 0.0)
    bad = ~np.isfinite(pred)
    if np.any(bad):
        pred[bad] = base
    return pred
