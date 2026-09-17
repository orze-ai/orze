import math
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold


def _models(seed):
    return [
        ExtraTreesRegressor(n_estimators=500, max_features=None, min_samples_leaf=1,
                            random_state=seed, n_jobs=1),
        GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                  random_state=seed),
        RandomForestRegressor(n_estimators=500, max_features=0.5, min_samples_leaf=1,
                              random_state=seed, n_jobs=1),
    ]


def _fit_log(X, y, seed):
    yl = np.log(np.maximum(y, 1e-6))
    ms = _models(seed)
    for m in ms:
        m.fit(X, yl)
    return ms


def _pred_log(ms, X):
    return np.column_stack([m.predict(X) for m in ms])


def fit_predict(X_train, y_train, X_eval, seed=1729, train_groups=None):
    seed = 1729
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    n = X.shape[0]

    grid = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        for b in (0.0, 0.25, 0.5, 0.75, 1.0):
            c = 1.0 - a - b
            if c > -1e-9:
                grid.append((a, b, max(c, 0.0)))

    oof = np.full((n, 3), np.nan)
    try:
        if train_groups is not None:
            g = np.asarray(train_groups).ravel()
            ng = int(np.unique(g).shape[0])
            if ng >= 3:
                splits = GroupKFold(n_splits=min(5, ng)).split(X, y, g)
            else:
                splits = KFold(n_splits=min(5, n), shuffle=True,
                               random_state=seed).split(X)
        else:
            splits = KFold(n_splits=min(5, n), shuffle=True,
                           random_state=seed).split(X)
        for tr, va in splits:
            if len(tr) < 5 or len(va) < 1:
                continue
            ms = _fit_log(X[tr], y[tr], seed)
            oof[va] = _pred_log(ms, X[va])
    except Exception:
        oof = np.full((n, 3), np.nan)

    best_w = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    best_s = 1.0
    ok = np.all(np.isfinite(oof), axis=1)
    if int(ok.sum()) >= 10:
        yv = y[ok]
        P = oof[ok]
        best = None
        for w in grid:
            wl = P.dot(np.array(w, dtype=float))
            base = np.exp(np.clip(wl, -20.0, 20.0))
            for s in (1.0, 1.02, 1.05, 1.1, 1.15):
                pr = s * base
                mse = float(np.mean((pr - yv) ** 2))
                if best is None or mse < best:
                    best = mse
                    best_w = w
                    best_s = s

    ms = _fit_log(X, y, seed)
    Pe = _pred_log(ms, Xe)
    pred = best_s * np.exp(np.clip(Pe.dot(np.array(best_w, dtype=float)), -20.0, 20.0))
    pred = np.asarray(pred, dtype=float).ravel()
    lo = float(np.min(y)) * 0.5
    hi = float(np.max(y)) * 1.5
    pred = np.clip(pred, lo, hi)
    bad = ~np.isfinite(pred)
    if np.any(bad):
        pred[bad] = float(np.mean(y))
    return pred
