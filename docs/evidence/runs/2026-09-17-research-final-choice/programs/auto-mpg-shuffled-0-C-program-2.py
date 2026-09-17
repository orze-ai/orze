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
    return [make_pipeline(StandardScaler(), Ridge(alpha=50.0)),
            make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=25, weights="uniform", p=2, n_jobs=1))]


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    Xtr = _feats(X_train)
    Xev = _feats(X_eval)
    y = np.asarray(y_train, dtype=float).ravel()
    n = Xtr.shape[0]
    mu = float(np.mean(y))
    out_mean = np.full(Xev.shape[0], mu, dtype=float)
    if n < 40:
        return out_mean
    splits = []
    if train_groups is not None:
        g = np.asarray(train_groups).ravel()
        if g.shape[0] == n:
            ng = int(np.unique(g).shape[0])
            k = int(min(5, ng))
            if k >= 3:
                try:
                    splits = list(GroupKFold(n_splits=k).split(Xtr, y, groups=g))
                except Exception:
                    splits = []
    if not splits:
        try:
            splits = list(KFold(n_splits=5, shuffle=True, random_state=int(seed)).split(Xtr))
        except Exception:
            return out_mean
    if len(splits) < 3:
        return out_mean
    m = 2
    gains = np.zeros((len(splits), m), dtype=float)
    for fi in range(len(splits)):
        tr, te = splits[fi]
        if tr.shape[0] < 10 or te.shape[0] < 2:
            return out_mean
        mu_f = float(np.mean(y[tr]))
        base = float(np.mean((y[te] - mu_f) ** 2))
        mdls = _models()
        for j in range(m):
            try:
                mdls[j].fit(Xtr[tr], y[tr])
                p = np.asarray(mdls[j].predict(Xtr[te]), dtype=float).ravel()
            except Exception:
                p = np.full(te.shape[0], mu_f, dtype=float)
            if p.shape[0] != te.shape[0] or not np.all(np.isfinite(p)):
                p = np.full(te.shape[0], mu_f, dtype=float)
            gains[fi, j] = base - float(np.mean((y[te] - p) ** 2))
    mean_gain = gains.mean(axis=0)
    if gains.shape[0] > 1:
        se = gains.std(axis=0, ddof=1) / np.sqrt(float(gains.shape[0]))
    else:
        se = np.full(m, np.inf)
    j = int(np.argmax(mean_gain))
    if not np.isfinite(mean_gain[j]) or not np.isfinite(se[j]):
        return out_mean
    if not (mean_gain[j] > 0.0 and mean_gain[j] > se[j]):
        return out_mean
    mdl = _models()[j]
    try:
        mdl.fit(Xtr, y)
        p = np.asarray(mdl.predict(Xev), dtype=float).ravel()
    except Exception:
        return out_mean
    if p.shape[0] != Xev.shape[0] or not np.all(np.isfinite(p)):
        return out_mean
    out = mu + 0.5 * (p - mu)
    out = np.asarray(out, dtype=float).ravel()
    if not np.all(np.isfinite(out)):
        return out_mean
    return np.clip(out, float(np.min(y)), float(np.max(y)))
