import numpy as np
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor


def _folds(n, groups, seed, k, reps):
    rng = np.random.RandomState(int(seed) % 2147483647)
    idx = np.arange(n)
    out = []
    g = None
    uniq = None
    if groups is not None:
        gg = np.asarray(groups).ravel()
        if gg.shape[0] == n:
            u = np.unique(gg)
            if u.shape[0] >= k:
                g = gg
                uniq = u
    for r in range(reps):
        if g is not None:
            perm = rng.permutation(uniq.shape[0])
            lab = np.empty(uniq.shape[0], dtype=int)
            lab[perm] = np.arange(uniq.shape[0]) % k
            row = lab[np.searchsorted(uniq, g)]
        else:
            perm = rng.permutation(n)
            row = np.empty(n, dtype=int)
            row[perm] = np.arange(n) % k
        for f in range(k):
            te = idx[row == f]
            tr = idx[row != f]
            if te.shape[0] > 0 and tr.shape[0] >= 20:
                out.append((tr, te))
    return out


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    mu = float(np.mean(y))
    fallback = np.full(Xe.shape[0], mu, dtype=float)
    n = X.shape[0]
    if n < 50 or Xe.shape[0] == 0:
        return fallback
    folds = _folds(n, train_groups, seed, 5, 5)
    if len(folds) < 10:
        return fallback
    models = [make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
              make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=25, weights="uniform", n_jobs=1)),
              make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=40, weights="uniform", n_jobs=1))]
    base = np.zeros(len(folds), dtype=float)
    for i in range(len(folds)):
        tr, te = folds[i]
        base[i] = float(np.mean((y[te] - np.mean(y[tr])) ** 2))
    mbase = float(np.mean(base))
    if not np.isfinite(mbase) or mbase <= 0.0:
        return fallback
    best = None
    for mdl in models:
        err = np.zeros(len(folds), dtype=float)
        ok = True
        for i in range(len(folds)):
            tr, te = folds[i]
            m = clone(mdl)
            m.fit(X[tr], y[tr])
            p = np.asarray(m.predict(X[te]), dtype=float).ravel()
            if not np.all(np.isfinite(p)):
                ok = False
                break
            err[i] = float(np.mean((y[te] - p) ** 2))
        if not ok:
            continue
        wins = float(np.mean(err < base))
        rel = 1.0 - float(np.mean(err)) / mbase
        if wins >= 0.7 and rel >= 0.03:
            if best is None or rel > best[0]:
                best = (rel, mdl)
    if best is None:
        return fallback
    s = float(min(max(best[0], 0.0), 1.0))
    if s <= 0.0:
        return fallback
    final = clone(best[1])
    final.fit(X, y)
    p = np.asarray(final.predict(Xe), dtype=float).ravel()
    p = np.where(np.isfinite(p), p, mu)
    pred = mu + s * (p - mu)
    return np.clip(pred, float(np.min(y)), float(np.max(y)))
