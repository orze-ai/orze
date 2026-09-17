import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

SEED = 1729


def _aug(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 12:
        return X
    p = X[:, 10]
    v = X[:, 11]
    ratio = p / np.maximum(np.abs(v), 1e-9)
    extra = np.column_stack([
        p - v,
        p + v,
        p * v,
        ratio,
        np.log1p(np.maximum(p, 0.0)),
        np.log1p(np.maximum(v, 0.0)),
    ])
    return np.column_stack([X, extra])


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    Xtr = _aug(X_train)
    Xe = _aug(X_eval)
    ytr = np.asarray(y_train, dtype=float).ravel()
    fallback = float(np.mean(ytr)) if ytr.size > 0 else 0.0
    models = [
        ExtraTreesRegressor(n_estimators=800, max_features=0.5,
                            min_samples_leaf=1, random_state=SEED, n_jobs=1),
        RandomForestRegressor(n_estimators=800, max_features=0.5,
                              min_samples_leaf=1, random_state=SEED, n_jobs=1),
    ]
    preds = []
    for m in models:
        try:
            m.fit(Xtr, ytr)
            p = np.asarray(m.predict(Xe), dtype=float).ravel()
            if p.shape[0] == Xe.shape[0] and np.all(np.isfinite(p)):
                preds.append(p)
        except Exception:
            continue
    if len(preds) == 0:
        return np.full(Xe.shape[0], fallback, dtype=float)
    out = np.mean(np.vstack(preds), axis=0)
    out = np.asarray(out, dtype=float).ravel()
    out = np.clip(out, 0.0, None)
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = fallback
    return out
