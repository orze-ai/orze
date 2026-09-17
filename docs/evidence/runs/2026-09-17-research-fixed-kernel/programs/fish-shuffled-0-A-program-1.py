import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold

NAMES = ["ridge", "trees", "knn"]


def _make(name, seed):
    if name == "ridge":
        return Ridge(alpha=10.0)
    if name == "trees":
        return ExtraTreesRegressor(n_estimators=160, min_samples_leaf=4,
                                   max_features=0.5, max_depth=8,
                                   random_state=seed, n_jobs=1)
    return KNeighborsRegressor(n_neighbors=25, weights="uniform")


def _pred(name, seed, Xtr, ytr, Xte):
    if name in ("ridge", "knn"):
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr)
        Xte = sc.transform(Xte)
    m = _make(name, seed)
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte), dtype=float)


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    n = int(X.shape[0])
    mu = float(np.mean(y)) if n > 0 else 0.0
    out = np.full(int(Xe.shape[0]), mu, dtype=float)
    if n < 30 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(X)):
        return out
    sd = int(seed)
    if train_groups is not None:
        g = np.asarray(train_groups).ravel()
        k = int(min(5, len(np.unique(g))))
        if k >= 2 and g.shape[0] == n:
            splits = list(GroupKFold(n_splits=k).split(X, y, g))
        else:
            splits = list(KFold(n_splits=5, shuffle=True, random_state=sd).split(X))
    else:
        splits = list(KFold(n_splits=5, shuffle=True, random_state=sd).split(X))
    oof = np.zeros((n, len(NAMES)), dtype=float)
    for tr, te in splits:
        if tr.shape[0] < 10 or te.shape[0] < 1:
            return out
        for j, nm in enumerate(NAMES):
            oof[te, j] = _pred(nm, sd, X[tr], y[tr], X[te])
    if not np.all(np.isfinite(oof)):
        return out
    base = float(np.mean((y - mu) ** 2))
    cen = oof.mean(axis=0)
    D = oof - cen
    a = np.zeros(len(NAMES), dtype=float)
    for j in range(len(NAMES)):
        v = float(np.dot(D[:, j], D[:, j]))
        if v <= 1e-12:
            continue
        aj = float(np.dot(D[:, j], y - mu) / v)
        aj = min(1.0, max(0.0, aj))
        if aj > 0.0:
            cv = float(np.mean((y - (mu + aj * D[:, j])) ** 2))
            if cv < base - 1e-9:
                a[j] = aj
    if not np.any(a > 0.0):
        return out
    sig = (D * a).mean(axis=1)
    vv = float(np.dot(sig, sig))
    if vv <= 1e-12:
        return out
    beta = float(np.dot(sig, y - mu) / vv)
    beta = min(1.0, max(0.0, beta))
    if beta <= 0.0:
        return out
    if float(np.mean((y - (mu + beta * sig)) ** 2)) >= base - 1e-9:
        return out
    full = np.zeros((int(Xe.shape[0]), len(NAMES)), dtype=float)
    for j, nm in enumerate(NAMES):
        if a[j] <= 0.0:
            continue
        full[:, j] = _pred(nm, sd, X, y, Xe)
    if not np.all(np.isfinite(full)):
        return out
    pred = mu + beta * ((full - cen) * a).mean(axis=1)
    lo = float(np.min(y))
    hi = float(np.max(y))
    pred = np.clip(pred, lo, hi)
    bad = ~np.isfinite(pred)
    if np.any(bad):
        pred[bad] = mu
    return pred
