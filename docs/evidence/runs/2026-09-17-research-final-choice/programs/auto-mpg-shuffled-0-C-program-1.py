import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold


def _feats(X):
    X = np.asarray(X, dtype=float)
    o = X[:, 6]
    oh = np.column_stack([(o == 1).astype(float), (o == 2).astype(float), (o == 3).astype(float)])
    return np.column_stack([X[:, :6], oh])


def _models():
    return [make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=25, weights="uniform", p=2, n_jobs=1))]


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    Xtr = _feats(X_train)
    Xev = _feats(X_eval)
    y = np.asarray(y_train, dtype=float).ravel()
    n = Xtr.shape[0]
    mu = float(np.mean(y))
    lo = float(np.min(y))
    hi = float(np.max(y))
    if n < 25:
        return np.full(Xev.shape[0], mu, dtype=float)
    splits = []
    if train_groups is not None:
        g = np.asarray(train_groups).ravel()
        ng = int(np.unique(g).shape[0])
        k = int(min(5, ng))
        if k >= 2:
            splits = list(GroupKFold(n_splits=k).split(Xtr, y, groups=g))
    if not splits:
        splits = list(KFold(n_splits=5, shuffle=True, random_state=int(seed)).split(Xtr))
    if not splits:
        return np.full(Xev.shape[0], mu, dtype=float)
    m = 2
    oof = np.zeros((n, m), dtype=float)
    for tr, te in splits:
        if tr.shape[0] < 5 or te.shape[0] < 1:
            continue
        mu_f = float(np.mean(y[tr]))
        mdls = _models()
        for j in range(m):
            mdl = mdls[j]
            try:
                mdl.fit(Xtr[tr], y[tr])
                p = np.asarray(mdl.predict(Xtr[te]), dtype=float).ravel()
            except Exception:
                p = np.full(te.shape[0], mu_f, dtype=float)
            if p.shape[0] != te.shape[0] or not np.all(np.isfinite(p)):
                p = np.full(te.shape[0], mu_f, dtype=float)
            oof[te, j] = p
    cands = [oof[:, 0], oof[:, 1], 0.5 * (oof[:, 0] + oof[:, 1])]
    grid = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    best_score = float(np.mean((y - mu) ** 2))
    best_ci = -1
    best_w = 0.0
    for ci in range(len(cands)):
        c = cands[ci]
        for w in grid:
            pred = mu + w * (c - mu)
            sc = float(np.mean((y - pred) ** 2))
            if sc < best_score - 1e-9:
                best_score = sc
                best_ci = ci
                best_w = w
    if best_ci < 0 or best_w <= 0.0:
        return np.full(Xev.shape[0], mu, dtype=float)
    mdls = _models()
    preds = []
    for j in range(m):
        mdl = mdls[j]
        try:
            mdl.fit(Xtr, y)
            p = np.asarray(mdl.predict(Xev), dtype=float).ravel()
        except Exception:
            p = np.full(Xev.shape[0], mu, dtype=float)
        if p.shape[0] != Xev.shape[0] or not np.all(np.isfinite(p)):
            p = np.full(Xev.shape[0], mu, dtype=float)
        preds.append(p)
    if best_ci == 0:
        base = preds[0]
    elif best_ci == 1:
        base = preds[1]
    else:
        base = 0.5 * (preds[0] + preds[1])
    out = mu + best_w * (base - mu)
    out = np.asarray(out, dtype=float).ravel()
    if not np.all(np.isfinite(out)):
        out = np.full(Xev.shape[0], mu, dtype=float)
    return np.clip(out, lo, hi)
