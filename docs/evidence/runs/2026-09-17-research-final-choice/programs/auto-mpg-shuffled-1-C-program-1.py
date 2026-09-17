import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    mu = float(np.mean(y))
    out = np.full(Xe.shape[0], mu, dtype=float)
    n = X.shape[0]
    if n < 40:
        return out
    k = 5
    splits = None
    if train_groups is not None:
        g = np.asarray(train_groups).ravel()
        if np.unique(g).size >= k:
            splits = list(GroupKFold(n_splits=k).split(X, y, g))
    if splits is None:
        splits = list(KFold(n_splits=k, shuffle=True, random_state=int(seed)).split(X))
    cands = [make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
             make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=25, weights='uniform', n_jobs=1)),
             make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=40, weights='uniform', n_jobs=1))]
    base_err = 0.0
    for tr, te in splits:
        base_err += float(np.sum((y[te] - np.mean(y[tr])) ** 2))
    base_err = base_err / float(n)
    if not np.isfinite(base_err) or base_err <= 0.0:
        return out
    best = None
    for mdl in cands:
        err = 0.0
        good = True
        for tr, te in splits:
            m = clone(mdl)
            m.fit(X[tr], y[tr])
            p = np.asarray(m.predict(X[te]), dtype=float).ravel()
            if not np.all(np.isfinite(p)):
                good = False
                break
            err += float(np.sum((y[te] - p) ** 2))
        if not good:
            continue
        err = err / float(n)
        if best is None or err < best[0]:
            best = (err, mdl)
    if best is None or best[0] >= base_err:
        return out
    shrink = 1.0 - best[0] / base_err
    shrink = float(min(max(shrink, 0.0), 1.0))
    if shrink <= 0.0:
        return out
    final = clone(best[1])
    final.fit(X, y)
    p = np.asarray(final.predict(Xe), dtype=float).ravel()
    p = np.where(np.isfinite(p), p, mu)
    pred = mu + shrink * (p - mu)
    return np.clip(pred, float(np.min(y)), float(np.max(y)))
