import math
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold

SEED = 1729


def _models():
    return [
        ExtraTreesRegressor(n_estimators=600, max_features=None, min_samples_leaf=1,
                            random_state=SEED, n_jobs=1),
        RandomForestRegressor(n_estimators=600, max_features=0.5, min_samples_leaf=1,
                              random_state=SEED, n_jobs=1),
        GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                  random_state=SEED),
    ]


def _fit_log(X, y):
    yl = np.log(np.maximum(y, 1e-6))
    ms = _models()
    for m in ms:
        m.fit(X, yl)
    return ms


def _mean_log_pred(ms, X):
    P = np.column_stack([m.predict(X) for m in ms])
    return np.mean(P, axis=1)


def _splits(X, y, groups, n):
    if groups is not None:
        try:
            g = np.asarray(groups).ravel()
            ng = int(np.unique(g).shape[0])
            if ng >= 3 and g.shape[0] == n:
                return list(GroupKFold(n_splits=min(5, ng)).split(X, y, g))
        except Exception:
            pass
    k = min(5, max(2, n))
    return list(KFold(n_splits=k, shuffle=True, random_state=SEED).split(X))


def fit_predict(X_train, y_train, X_eval, seed=1729, train_groups=None):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    n = X.shape[0]

    scale = 1.0
    try:
        oof = np.full(n, np.nan)
        for tr, va in _splits(X, y, train_groups, n):
            if len(tr) < 10 or len(va) < 1:
                continue
            ms = _fit_log(X[tr], y[tr])
            oof[va] = _mean_log_pred(ms, X[va])
        ok = np.isfinite(oof)
        if int(ok.sum()) >= 20:
            base = np.exp(np.clip(oof[ok], -20.0, 20.0))
            denom = float(np.sum(base * base))
            if denom > 1e-12:
                s = float(np.sum(base * y[ok]) / denom)
                if np.isfinite(s):
                    scale = float(min(max(s, 0.7), 1.3))
    except Exception:
        scale = 1.0

    ms = _fit_log(X, y)
    pl = _mean_log_pred(ms, Xe)
    pred = scale * np.exp(np.clip(pl, -20.0, 20.0))
    pred = np.asarray(pred, dtype=float).ravel()
    lo = float(np.min(y)) * 0.5
    hi = float(np.max(y)) * 1.5
    pred = np.clip(pred, lo, hi)
    bad = ~np.isfinite(pred)
    if np.any(bad):
        pred[bad] = float(np.mean(y))
    return pred
